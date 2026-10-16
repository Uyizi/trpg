"""Ordinary adapter GEMV with PEFT NF4's cast-before-add boundary."""
import torch
import triton
import triton.language as tl


@triton.jit
def _ordinary_a(X, A, LOW, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    weights = tl.load(A + row * H + columns, columns < H, 0)
    x = tl.load(X + columns, columns < H, 0).to(tl.float32)
    tl.store(LOW + row, tl.sum(weights * x, 0))


@triton.jit
def _ordinary_b_add(LOW, B, BASE, OUT, N: tl.constexpr, R: tl.constexpr,
                    SCALE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ranks = tl.arange(0, BLOCK_R)
    weights = tl.load(B + rows[:, None] * R + ranks[None, :],
                      (rows[:, None] < N) & (ranks[None, :] < R), 0)
    low = tl.load(LOW + ranks, ranks < R, 0)
    delta = tl.sum(weights * low[None, :], 1) * SCALE
    delta = delta.to(BASE.dtype.element_ty).to(tl.float32)
    base = tl.load(BASE + rows, rows < N, 0).to(tl.float32)
    tl.store(OUT + rows, base + delta, rows < N)


def ordinary_lora_add(x, base, a, b, scale):
    rank, hidden = a.shape
    out = base.shape[-1]
    low = torch.empty((rank,), device=x.device, dtype=torch.float32)
    result = torch.empty_like(base)
    block = triton.next_power_of_2(hidden)
    _ordinary_a[(rank,)](x, a, low, hidden, block,
                         num_warps=8 if block > 8192 else 4, enable_fp_fusion=False)
    _ordinary_b_add[(triton.cdiv(out, 128),)](
        low, b, base, result, out, rank, scale, 128, triton.next_power_of_2(rank),
        num_warps=4, enable_fp_fusion=False)
    return result
