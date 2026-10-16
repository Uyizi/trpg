"""One/two-token selected-expert GEMV; no copies of trained adapter weights.

Uses the gather-plus-low-rank-GEMV idea from Punica's SGMV. This small local
implementation is limited to two routed tokens and ordinary FP32 adapters.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _selected_a(X, A0, A1, SELECTED, LOW, H: tl.constexpr, BLOCK: tl.constexpr):
    rank_row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    expert = tl.load(SELECTED)
    if expert == 0:
        weights = tl.load(A0 + rank_row * H + columns, columns < H, 0)
    else:
        weights = tl.load(A1 + rank_row * H + columns, columns < H, 0)
    x = tl.load(X + columns, columns < H, 0).to(tl.float32)
    tl.store(LOW + rank_row, tl.sum(weights * x, 0))


@triton.jit
def _selected_b_add(LOW, B0, B1, BASE, SELECTED, OUT,
                    N: tl.constexpr, R: tl.constexpr,
                    SCALE0: tl.constexpr, SCALE1: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ranks = tl.arange(0, BLOCK_R)
    expert = tl.load(SELECTED)
    if expert == 0:
        weights = tl.load(B0 + rows[:, None] * R + ranks[None, :],
                          (rows[:, None] < N) & (ranks[None, :] < R), 0)
        scale = SCALE0
    else:
        weights = tl.load(B1 + rows[:, None] * R + ranks[None, :],
                          (rows[:, None] < N) & (ranks[None, :] < R), 0)
        scale = SCALE1
    low = tl.load(LOW + ranks, ranks < R, 0)
    delta = tl.sum(weights * low[None, :], 1) * scale
    # Existing PEFT path rounds delta before adding it to the NF4 base output.
    delta = delta.to(BASE.dtype.element_ty).to(tl.float32)
    base = tl.load(BASE + rows, rows < N, 0).to(tl.float32)
    tl.store(OUT + rows, base + delta, rows < N)


@triton.jit
def _selected_a_two(X, A0, A1, SELECTED, LOW,
                    H: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    rank_row = tl.program_id(0)
    token = tl.program_id(1)
    columns = tl.arange(0, BLOCK)
    expert = tl.load(SELECTED + token)
    if expert == 0:
        weights = tl.load(A0 + rank_row * H + columns, columns < H, 0)
    else:
        weights = tl.load(A1 + rank_row * H + columns, columns < H, 0)
    x = tl.load(X + token * H + columns, columns < H, 0).to(tl.float32)
    tl.store(LOW + token * R + rank_row, tl.sum(weights * x, 0))


@triton.jit
def _selected_b_add_two(LOW, B0, B1, BASE, SELECTED, OUT,
                        N: tl.constexpr, R: tl.constexpr,
                        SCALE0: tl.constexpr, SCALE1: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    token = tl.program_id(1)
    ranks = tl.arange(0, BLOCK_R)
    expert = tl.load(SELECTED + token)
    if expert == 0:
        weights = tl.load(B0 + rows[:, None] * R + ranks[None, :],
                          (rows[:, None] < N) & (ranks[None, :] < R), 0)
        scale = SCALE0
    else:
        weights = tl.load(B1 + rows[:, None] * R + ranks[None, :],
                          (rows[:, None] < N) & (ranks[None, :] < R), 0)
        scale = SCALE1
    low = tl.load(LOW + token * R + ranks, ranks < R, 0)
    delta = tl.sum(weights * low[None, :], 1) * scale
    delta = delta.to(BASE.dtype.element_ty).to(tl.float32)
    base = tl.load(BASE + token * N + rows, rows < N, 0).to(tl.float32)
    tl.store(OUT + token * N + rows, base + delta, rows < N)


def selected_lora_add(x, base, selected, a0, b0, a1, b1, scale0, scale1):
    rank, hidden = a0.shape
    out = base.shape[1]
    if x.shape[0] == 2:
        low = torch.empty((2, rank), device=x.device, dtype=torch.float32)
        result = torch.empty_like(base)
        block = triton.next_power_of_2(hidden)
        _selected_a_two[(rank, 2)](x, a0, a1, selected, low, hidden, rank, block,
                                  num_warps=8 if block > 8192 else 4, enable_fp_fusion=False)
        _selected_b_add_two[(triton.cdiv(out, 128), 2)](
            low, b0, b1, base, selected, result, out, rank, scale0, scale1,
            128, triton.next_power_of_2(rank), num_warps=4, enable_fp_fusion=False)
        return result
    low = torch.empty((rank,), device=x.device, dtype=torch.float32)
    result = torch.empty_like(base)
    block = triton.next_power_of_2(hidden)
    _selected_a[(rank,)](x, a0, a1, selected, low, hidden, block,
                        num_warps=8 if block > 8192 else 4, enable_fp_fusion=False)
    _selected_b_add[(triton.cdiv(out, 128),)](
        low, b0, b1, base, selected, result, out, rank, scale0, scale1,
        128, triton.next_power_of_2(rank), num_warps=4, enable_fp_fusion=False)
    return result
