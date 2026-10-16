"""One launch and one output allocation for zero-centered FP32 RMSNorm."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(X, WEIGHT, OUT, H: tl.constexpr, ROW_STRIDE: tl.constexpr,
              EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    x = tl.load(X + row * ROW_STRIDE + columns, columns < H, 0).to(tl.float32)
    weight = tl.load(WEIGHT + columns, columns < H, 0).to(tl.float32)
    variance = tl.sum(x * x, 0) / H
    normalized = x * tl.rsqrt(variance + EPS)
    # Upstream rounds only after both FP32 multiplications. In particular, no
    # low precision (1+weight), no cast before multiplying, no FMA contraction.
    result = normalized * (1.0 + weight)
    tl.store(OUT + row * H + columns, result, columns < H)


def zero_centered_rms_norm(x, weight, eps, row_stride):
    """Internal launch helper; all eligibility is checked in text_norm_fusion."""
    hidden = x.shape[-1]
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    block = triton.next_power_of_2(hidden)
    with torch.cuda.device(x.device):
        _rms_norm[(x.numel() // hidden,)](
            x, weight, output, hidden, row_stride, eps, block,
            num_warps=8 if block >= 4096 else 4, enable_fp_fusion=False)
    return output
