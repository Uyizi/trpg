"""Fused affine pack/unpack kernels; called only for CUDA KV tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(X, Q, M, G: tl.constexpr, BITS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, G)
    x = tl.load(X + row * G + i).to(tl.float32)
    low = tl.min(x, 0)
    scale = tl.maximum((tl.max(x, 0) - low) / ((1 << BITS) - 1), 1e-8)
    codes = tl.minimum(tl.maximum(tl.floor((x - low) / scale + 0.5), 0), (1 << BITS) - 1).to(tl.uint8)
    if BITS == 4:
        j = tl.arange(0, G // 2)
        packed = tl.gather(codes, 2 * j, 0) | (tl.gather(codes, 2 * j + 1, 0) << 4)
        tl.store(Q + row * (G // 2) + j, packed)
    else:
        tl.store(Q + row * G + i, codes)
    tl.store(M + row * 2, scale)
    tl.store(M + row * 2 + 1, low)


@triton.jit
def _unpack_kernel(Q, M, Y, N, G: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = i // G, i % G
    if BITS == 4:
        code = tl.load(Q + row * (G // 2) + column // 2, i < N, 0).to(tl.int32)
        code = (code >> ((column % 2) * 4)) & 15
    else:
        code = tl.load(Q + i, i < N, 0).to(tl.int32)
    scale = tl.load(M + row * 2, i < N, 0)
    low = tl.load(M + row * 2 + 1, i < N, 0)
    tl.store(Y + i, code * scale + low, i < N)


def pack(rows, bits):
    shape = list(rows.shape)
    group = shape[-1]
    shape[-1] = group // (2 if bits == 4 else 1)
    packed = torch.empty(shape, dtype=torch.uint8, device=rows.device)
    shape[-1] = 2
    meta = torch.empty(shape, dtype=torch.float32, device=rows.device)
    _pack_kernel[(rows.numel() // group,)](rows, packed, meta, group, bits, num_warps=4)
    return packed, meta


def unpack(data, meta, bits, dtype):
    shape = list(data.shape)
    shape[-1] *= 2 if bits == 4 else 1
    output = torch.empty(shape, device=data.device, dtype=dtype)
    _unpack_kernel[(triton.cdiv(output.numel(), 1024),)](
        data, meta, output, output.numel(), shape[-1], bits, 1024)
    return output
