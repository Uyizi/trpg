"""Standalone inference-only NVFP4 packing and native cuBLAS GEMM experiment.

No monkey patches, model integration, downloads, CUDA initialization, or writes at
import. E2M1 data and E4M3 block-16 scales follow the cuBLAS documented format:
https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
The eager packer is a correctness reference, not an optimized serving kernel.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PackedNVFP4:
    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    shape: tuple[int, int]

    def to(self, device):
        # Move bytes; torch's float4 copy kernels are intentionally never needed.
        return PackedNVFP4(self.packed.to(device), self.scales.to(device),
                           self.global_scale.to(device), self.shape)


def pack_e2m1(x):
    """Encode finite values using nearest-even E2M1, even element in low nibble."""
    if x.shape[-1] % 2:
        raise ValueError('E2M1 packing requires an even last dimension')
    a = x.abs()
    # At each midpoint choose the level whose integer code is even.
    code = ((a > .25).to(torch.uint8) + (a >= .75).to(torch.uint8)
            + (a > 1.25).to(torch.uint8) + (a >= 1.75).to(torch.uint8)
            + (a > 2.5).to(torch.uint8) + (a >= 3.5).to(torch.uint8)
            + (a > 5.).to(torch.uint8))
    code = code | (torch.signbit(x).to(torch.uint8) << 3)
    return (code[..., ::2] | (code[..., 1::2] << 4)).contiguous()


def unpack_e2m1(packed):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device)
    return levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)


def swizzle_scales(scales):
    """Map scales to cuBLAS 128-row/4-column tiles, with 32x4x4 ordering."""
    rows, cols = scales.shape
    padded_rows, padded_cols = ((rows+127)//128)*128, ((cols+3)//4)*4
    # Byte permutations support float8 on CPU as well as CUDA.
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales.view(torch.uint8)
    tiles = padded.reshape(padded_rows//128, 4, 32, padded_cols//4, 4)
    return tiles.permute(0, 3, 2, 1, 4).contiguous().flatten().view(scales.dtype)


def unswizzle_scales(scales, rows, cols):
    rb, cb = (rows+127)//128, (cols+3)//4
    tiled = scales.view(torch.uint8).reshape(rb, cb, 32, 4, 4)
    linear = tiled.permute(0, 3, 2, 1, 4).contiguous().reshape(rb*128, cb*4)
    return linear[:rows, :cols].contiguous().view(scales.dtype)


@torch.inference_mode()
def quantize_nvfp4(x, *, check_finite=True):
    """Two-level global-FP32/block16-E4M3 quantization; no training or files."""
    if x.ndim != 2 or x.shape[1] % 16 or min(x.shape) <= 0:
        raise ValueError('NVFP4 expects a nonempty 2D matrix with K divisible by 16')
    if check_finite and not torch.isfinite(x).all().item():
        raise ValueError('NVFP4 input must contain finite values')
    rows, k = x.shape
    values = x.float().reshape(rows, k//16, 16)
    global_scale = (values.abs().amax() / (448.*6.)).clamp_min(torch.finfo(torch.float32).tiny)
    blocks = (values.abs().amax(-1) / (6.*global_scale)).clamp(2.**-9, 448.).to(torch.float8_e4m3fn)
    normalized = values / (blocks.float().unsqueeze(-1)*global_scale)
    return PackedNVFP4(pack_e2m1(normalized.reshape(rows, k)), swizzle_scales(blocks),
                       global_scale, (rows, k))


def dequantize_nvfp4(q):
    rows, k = q.shape
    scales = unswizzle_scales(q.scales, rows, k//16).float()
    values = unpack_e2m1(q.packed).reshape(rows, k//16, 16)
    return (values * scales.unsqueeze(-1) * q.global_scale).reshape(rows, k)


def native_scaled_mm(a, weight, *, api='public'):
    """Execute native FP4 GEMM against weight stored as [N,K], without fallback."""
    if a.shape[1] != weight.shape[1]:
        raise ValueError('A and weight must share K')
    if a.packed.device.type != 'cuda' or weight.packed.device.type != 'cuda':
        raise ValueError('Native NVFP4 GEMM requires CUDA packed operands')
    aq = a.packed.view(torch.float4_e2m1fn_x2)
    bq = weight.packed.view(torch.float4_e2m1fn_x2).t()
    if api == 'public':
        # Both scale levels stay inside cuBLAS, avoiding BF16 intermediate overflow.
        return F.scaled_mm(aq, bq,
            scale_a=[a.scales, a.global_scale],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[weight.scales, weight.global_scale],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16)
    if api == 'legacy':
        # Legacy TorchAO recipe: FP32 output then tensor dequantization; kept
        # explicit in report because it incurs an additional multiply/cast.
        result = torch._scaled_mm(aq, bq, a.scales, weight.scales, out_dtype=torch.float32)
        return (result * (a.global_scale * weight.global_scale)).to(torch.bfloat16)
    raise ValueError('api must be public or legacy')


def error_summary(reference, actual):
    ref, value = reference.float(), actual.float()
    finite = bool(torch.isfinite(value).all().item())
    if not finite:
        return {'finite': False, 'relative_l2': None, 'max_abs_error': None, 'cosine': None}
    delta = value-ref
    tiny = torch.finfo(torch.float32).tiny
    return {'finite': True,
            'relative_l2': (delta.norm()/ref.norm().clamp_min(tiny)).item(),
            'max_abs_error': delta.abs().max().item(),
            'cosine': F.cosine_similarity(ref.flatten(), value.flatten(), dim=0).item()}
