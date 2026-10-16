"""Guarded inference-only LoRA fusion for the existing two-expert decode path."""
import math
import torch
from torch import nn


def routed_lora_add(x, base, selected, a0, b0, a1, b1, scale0, scale1):
    """Return the fused result, or None when ordinary PyTorch must handle it.

    `selected` is the existing GPU argmax, so its value is known to be 0 or 1.
    Only FP32 adapters are supported, matching PEFT's current inference load.
    """
    if (not x.is_cuda or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
        or x.ndim != 2 or x.shape[0] not in (1, 2) or base.ndim != 2 or base.shape[0] != x.shape[0]
        or x.dtype not in (torch.float16, torch.bfloat16) or base.dtype != x.dtype
        or selected.numel() != x.shape[0] or selected.dtype not in (torch.int32, torch.int64)):
        return None
    weights = (a0, b0, a1, b1)
    tensors = (x, base, selected, *weights)
    if (any(t.device != x.device or not t.is_contiguous() or t.requires_grad for t in tensors)
        or any(w.dtype != torch.float32 or w.ndim != 2 for w in weights)):
        return None
    rank, hidden = a0.shape
    if (not 1 <= rank <= 64 or not 1 <= hidden <= 16384 or not 1 <= base.shape[1] <= 65536
        or x.shape[1] != hidden or a1.shape != a0.shape
        or b0.shape != (base.shape[1], rank) or b1.shape != b0.shape
        or not math.isfinite(scale0) or not math.isfinite(scale1)):
        return None
    try:
        from .routed_lora_kernels import selected_lora_add
    except ImportError:
        return None
    return selected_lora_add(x, base, selected, *weights, scale0, scale1)


def try_routed_lora(x, base, selected, layer0, layer1):
    """Inspect live adapter state; do not cache matrices or adapter selection."""
    weights, scales = [], []
    for layer in (layer0, layer1):
        if (layer.training or layer.disable_adapters or layer.merged or layer.lora_variant
            or len(layer.active_adapters) != 1):
            return None
        adapter = layer.active_adapters[0]
        if adapter not in layer.lora_A or adapter not in layer.lora_B:
            return None
        a, b = layer.lora_A[adapter], layer.lora_B[adapter]
        dropout = layer.lora_dropout[adapter]
        if (type(a) is not nn.Linear or type(b) is not nn.Linear
            or a.bias is not None or b.bias is not None
            or type(dropout) not in (nn.Dropout, nn.Identity) or dropout.training):
            return None
        weights.extend((a.weight, b.weight))
        scales.append(layer.scaling[adapter])
    return routed_lora_add(x, base, selected, *weights, *scales)
