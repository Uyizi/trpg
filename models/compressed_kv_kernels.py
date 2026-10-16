"""Native KIVI-style affine Q4/Q8 split-K decode with online softmax.

Algorithm references: KIVI (2402.02750) and Triton's fused-attention tutorial.
Unlike a dense dequantization path, only one tile exists in registers at a time.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['PACKED','TOTAL','RKS1','RVS1','NS','SPAN'])
def _partial(Q, KD, KM, VD, VM, RK, RV, PART, MAX, DEN,
             QH_STRIDE: tl.constexpr, KH: tl.constexpr,
             KDS1: tl.constexpr, KDS2: tl.constexpr, KDS3: tl.constexpr,
             KMS1: tl.constexpr, KMS2: tl.constexpr, KMS3: tl.constexpr,
             VDS1: tl.constexpr, VDS2: tl.constexpr, VDS3: tl.constexpr,
             VMS1: tl.constexpr, VMS2: tl.constexpr, VMS3: tl.constexpr,
             RKS1, RKS2: tl.constexpr, RVS1, RVS2: tl.constexpr,
             PACKED, TOTAL, D: tl.constexpr,
             GROUPS: tl.constexpr, BITS: tl.constexpr, NS,
             SPAN, SCALE: tl.constexpr, BT: tl.constexpr, BG: tl.constexpr):
    kvhead, split = tl.program_id(0), tl.program_id(1)
    group = tl.arange(0, BG)
    heads = kvhead*GROUPS+group
    channel = tl.arange(0, D)
    offsets = tl.arange(0, BT)
    q = tl.load(Q + heads[:,None]*QH_STRIDE + channel[None,:], group[:,None] < GROUPS, 0)
    maximum = tl.full((BG,), -float('inf'), tl.float32)
    denom = tl.full((BG,), 0, tl.float32)
    accum = tl.full((BG,D), 0, tl.float32)
    for block in range(SPAN // BT):
        times = split * SPAN + block * BT + offsets
        packed_mask = times < PACKED
        residual_mask = (times >= PACKED) & (times < TOTAL)
        timeblock, sample = times // 128, times % 128
        # K is stored channel-major. Load [channel,time] once for the GQA group.
        code = tl.load(KD + kvhead*KDS1 + timeblock[None,:]*KDS2
                       + channel[:,None]*KDS3 + (sample[None,:] // (2 if BITS == 4 else 1)),
                       packed_mask[None,:], 0).to(tl.int32)
        if BITS == 4:
            code = (code >> ((sample[None,:] % 2)*4)) & 15
        scale = tl.load(KM + kvhead*KMS1 + timeblock[None,:]*KMS2 + channel[:,None]*KMS3,
                        packed_mask[None,:], 0)
        low = tl.load(KM + kvhead*KMS1 + timeblock[None,:]*KMS2 + channel[:,None]*KMS3 + 1,
                      packed_mask[None,:], 0)
        key = tl.fma(code.to(tl.float32),scale,low).to(Q.dtype.element_ty)
        recent = tl.load(RK + kvhead*RKS1 + (times[None,:]-PACKED)*RKS2 + channel[:,None],
                         residual_mask[None,:], 0)
        key = key + recent
        logits = tl.dot(q,key) * (SCALE * 1.4426950408889634)
        logits = tl.where((times[None,:] < TOTAL) & (group[:,None] < GROUPS), logits, -float('inf'))
        block_max = tl.max(logits, 1)
        new_max = tl.maximum(maximum, block_max)
        # A final split can have whole trailing tiles outside TOTAL.
        new_max = tl.where(new_max == -float('inf'), 0., new_max)
        alpha = tl.exp2(maximum-new_max)
        probs = tl.exp2(logits-new_max[:,None])
        vcode = tl.load(VD + kvhead*VDS1 + times[:,None]*VDS2
                        + (channel[None,:]//64)*VDS3 + ((channel[None,:]%64)//(2 if BITS == 4 else 1)),
                        packed_mask[:,None], 0).to(tl.int32)
        if BITS == 4:
            vcode = (vcode >> ((channel[None,:]%2)*4)) & 15
        vscale = tl.load(VM + kvhead*VMS1 + times[:,None]*VMS2 + (channel[None,:]//64)*VMS3,
                         packed_mask[:,None], 0)
        vlow = tl.load(VM + kvhead*VMS1 + times[:,None]*VMS2 + (channel[None,:]//64)*VMS3 + 1,
                       packed_mask[:,None], 0)
        value = tl.fma(vcode.to(tl.float32),vscale,vlow).to(Q.dtype.element_ty)
        recent_v = tl.load(RV + kvhead*RVS1 + (times[:,None]-PACKED)*RVS2 + channel[None,:],
                           residual_mask[:,None], 0)
        value = value + recent_v
        accum = accum*alpha[:,None] + tl.dot(probs.to(Q.dtype.element_ty),value)
        denom = denom*alpha + tl.sum(probs, 1)
        maximum = new_max
    tl.store(PART + (heads[:,None]*NS+split)*D + channel[None,:], accum, group[:,None] < GROUPS)
    tl.store(MAX + heads*NS+split, maximum, group < GROUPS)
    tl.store(DEN + heads*NS+split, denom, group < GROUPS)


@triton.jit(do_not_specialize=['NS'])
def _merge(PART, MAX, DEN, OUT, NS, D: tl.constexpr, BS: tl.constexpr):
    head = tl.program_id(0)
    splits = tl.arange(0, BS)
    channel = tl.arange(0, D)
    maximum = tl.load(MAX + head*NS+splits, splits < NS, -float('inf'))
    weight = tl.exp2(maximum - tl.max(maximum, 0))
    denom = tl.load(DEN + head*NS+splits, splits < NS, 0)
    partial = tl.load(PART + (head*NS+splits[:,None])*D + channel[None,:], splits[:,None] < NS, 0)
    output = tl.sum(partial*weight[:,None], 0) / tl.sum(denom*weight, 0)
    tl.store(OUT + head*D+channel, output)


def decode(query, view, scale):
    heads, dimension = query.shape[1], query.shape[-1]
    span = max(128, triton.cdiv(view.length, 128*128)*128)
    splits = triton.cdiv(view.length, span)
    partial = torch.empty((heads, splits, dimension), device=query.device, dtype=torch.float32)
    maximum = torch.empty((heads, splits), device=query.device, dtype=torch.float32)
    denom = torch.empty_like(maximum)
    result = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    kd, km, vd, vm = view.key_data, view.key_meta, view.value_data, view.value_meta
    _partial[(view.keys.shape[1], splits)](
        query, kd, km, vd, vm, view.keys, view.values, partial, maximum, denom,
        query.stride(1), view.keys.shape[1], *kd.stride()[1:4], *km.stride()[1:4],
        *vd.stride()[1:4], *vm.stride()[1:4], *view.keys.stride()[1:3], *view.values.stride()[1:3],
        view.packed_tokens, view.length, dimension, heads//view.keys.shape[1], view.bits,
        splits, span, scale, 64, 16, num_warps=8, num_stages=1, enable_fp_fusion=False)
    _merge[(heads,)](partial, maximum, denom, result, splits, dimension,
                    triton.next_power_of_2(splits), num_warps=8, enable_fp_fusion=False)
    return result
