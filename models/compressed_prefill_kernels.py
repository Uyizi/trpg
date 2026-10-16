"""Tiled causal attention over native affine Q4/Q8 KV plus exact residual.

Online softmax follows Triton's official fused-attention tutorial:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
Storage follows this project's KIVI-inspired layout, not KIVI's kernels:
https://arxiv.org/abs/2402.02750
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _online(q, key, value, valid, accum, maximum, denom,
            SCALE: tl.constexpr, DTYPE: tl.constexpr):
    logits = tl.dot(q, key) * (SCALE * 1.4426950408889634)
    logits = tl.where(valid, logits, -float('inf'))
    new_max = tl.maximum(maximum, tl.max(logits, 1))
    # Invalid padded query rows must not produce inf-inf or NaNs.
    new_max = tl.where(new_max == -float('inf'), 0., new_max)
    alpha = tl.exp2(maximum - new_max)
    prob = tl.exp2(logits - new_max[:, None])
    accum = accum * alpha[:, None]
    accum = tl.dot(prob.to(DTYPE), value, accum)
    denom = denom * alpha + tl.sum(prob, 1)
    return accum, new_max, denom


@triton.jit(do_not_specialize=['QLEN', 'PACKED', 'TOTAL', 'QHS', 'QTS',
    'KDS1', 'KDS2', 'KDS3', 'KMS1', 'KMS2', 'KMS3',
    'VDS1', 'VDS2', 'VDS3', 'VMS1', 'VMS2', 'VMS3',
    'RKS1', 'RKS2', 'RVS1', 'RVS2'])
def _prefill(Q, KD, KM, VD, VM, RK, RV, OUT,
             QHS, QTS,
             KDS1, KDS2, KDS3, KMS1, KMS2, KMS3,
             VDS1, VDS2, VDS3, VMS1, VMS2, VMS3,
             RKS1, RKS2, RVS1, RVS2,
             QLEN, PACKED, TOTAL, D: tl.constexpr, GROUPS: tl.constexpr,
             BITS: tl.constexpr, SCALE: tl.constexpr,
             BM: tl.constexpr, BT: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    kvhead = head // GROUPS
    rows = block * BM + tl.arange(0, BM)
    channel = tl.arange(0, D)
    times = tl.arange(0, BT)
    q = tl.load(Q + head * QHS + rows[:, None] * QTS + channel[None, :],
                rows[:, None] < QLEN, 0)
    accum = tl.full((BM, D), 0, tl.float32)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    # Packed history precedes every current query. No causal test is needed here.
    for start in range(0, PACKED, BT):
        t = start + times
        timeblock, sample = t // 128, t % 128
        code = tl.load(KD + kvhead*KDS1 + timeblock[None, :]*KDS2
                       + channel[:, None]*KDS3 + sample[None, :]//(2 if BITS == 4 else 1),
                       t[None, :] < PACKED, 0).to(tl.int32)
        if BITS == 4:
            code = (code >> ((sample[None, :] % 2)*4)) & 15
        scale = tl.load(KM + kvhead*KMS1 + timeblock[None, :]*KMS2 + channel[:, None]*KMS3,
                        t[None, :] < PACKED, 0)
        low = tl.load(KM + kvhead*KMS1 + timeblock[None, :]*KMS2 + channel[:, None]*KMS3 + 1,
                      t[None, :] < PACKED, 0)
        # Explicit FMA and dtype rounding reproduce quantized_kv_kernels.unpack.
        key = tl.fma(code.to(tl.float32), scale, low).to(Q.dtype.element_ty)
        vcode = tl.load(VD + kvhead*VDS1 + t[:, None]*VDS2 + (channel[None, :]//64)*VDS3
                        + (channel[None, :]%64)//(2 if BITS == 4 else 1),
                        t[:, None] < PACKED, 0).to(tl.int32)
        if BITS == 4:
            vcode = (vcode >> ((channel[None, :]%2)*4)) & 15
        vscale = tl.load(VM + kvhead*VMS1 + t[:, None]*VMS2 + (channel[None, :]//64)*VMS3,
                         t[:, None] < PACKED, 0)
        vlow = tl.load(VM + kvhead*VMS1 + t[:, None]*VMS2 + (channel[None, :]//64)*VMS3 + 1,
                       t[:, None] < PACKED, 0)
        value = tl.fma(vcode.to(tl.float32), vscale, vlow).to(Q.dtype.element_ty)
        valid = (rows[:, None] < QLEN) & (t[None, :] < PACKED)
        accum, maximum, denom = _online(q, key, value, valid, accum, maximum, denom,
                                        SCALE, Q.dtype.element_ty)
    # The exact residual includes this entire query chunk. Stop at this CTA's
    # final causal key and mask the diagonal, with lower-right alignment.
    stop = tl.minimum(TOTAL - PACKED, TOTAL - QLEN - PACKED + (block+1)*BM)
    for start in range(0, stop, BT):
        t = start + times
        valid_t = t < TOTAL - PACKED
        key = tl.load(RK + kvhead*RKS1 + t[None, :]*RKS2 + channel[:, None],
                      valid_t[None, :], 0)
        value = tl.load(RV + kvhead*RVS1 + t[:, None]*RVS2 + channel[None, :],
                        valid_t[:, None], 0)
        valid = ((rows[:, None] < QLEN) & valid_t[None, :]
                 & (PACKED + t[None, :] <= TOTAL - QLEN + rows[:, None]))
        accum, maximum, denom = _online(q, key, value, valid, accum, maximum, denom,
                                        SCALE, Q.dtype.element_ty)
    output = accum / denom[:, None]
    tl.store(OUT + (head*QLEN + rows[:, None])*D + channel[None, :],
             output, rows[:, None] < QLEN)


def prefill(query, view, scale):
    """Launch after the public eligibility checks; no full-history decode."""
    heads, qlen, dimension = query.shape[1:]
    kd, km, vd, vm = view.key_data, view.key_meta, view.value_data, view.value_meta
    result = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    _prefill[(triton.cdiv(qlen, 32), heads)](
        query, kd, km, vd, vm, view.keys, view.values, result,
        query.stride(1), query.stride(2), *kd.stride()[1:4], *km.stride()[1:4],
        *vd.stride()[1:4], *vm.stride()[1:4],
        *view.keys.stride()[1:3], *view.values.stride()[1:3],
        qlen, view.packed_tokens, view.length, dimension,
        heads//view.keys.shape[1], view.bits, scale, 32, 64,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return result
