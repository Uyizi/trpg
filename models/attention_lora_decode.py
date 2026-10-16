"""Frozen single-token NF4 attention LoRA acceleration with live PEFT fallback.

This only replaces the adapter arithmetic. The existing bitsandbytes base
projection, its quantized weights, and the PEFT adapter tensors stay in place.
"""
import math
import re
from types import MethodType

import torch
from torch import nn


_ATTENTION = re.compile(
    r'(?:^|\.)(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)'
    r'|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj))$')


def ordinary_lora_weights(layer):
    """Read the current ordinary adapter, returning None for unsupported state."""
    if (layer.training or layer.disable_adapters or layer.merged or layer.lora_variant
        or len(layer.active_adapters) != 1
        or getattr(layer, 'cast_input_dtype_enabled', True) is False
        or nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks):
        return None
    adapter = layer.active_adapters[0]
    if (adapter not in layer.lora_A or adapter not in layer.lora_B
        or layer.use_dora.get(adapter, False)):
        return None
    a, b, dropout = layer.lora_A[adapter], layer.lora_B[adapter], layer.lora_dropout[adapter]
    if (type(a) is not nn.Linear or type(b) is not nn.Linear
        or a.bias is not None or b.bias is not None
        or type(dropout) not in (nn.Dropout, nn.Identity) or dropout.training
        or any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
               for m in (a, b, dropout))):
        return None
    scale = layer.scaling[adapter]
    if (type(scale) not in (int, float) or not math.isfinite(scale)
        or any(w.dtype != torch.float32 or w.requires_grad or not w.is_contiguous()
               for w in (a.weight, b.weight))):
        return None
    return a.weight, b.weight, float(scale)


def _eligible_lora_input(x, a, b, scale):
    """Preflight everything knowable before executing the NF4 base projection."""
    if (not x.is_cuda or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
        or x.ndim not in (2, 3) or x.numel() != x.shape[-1]
        or x.dtype not in (torch.float16, torch.bfloat16)
        or a.ndim != 2 or b.ndim != 2
        or a.dtype != torch.float32 or b.dtype != torch.float32
        or any(t.device != x.device or t.requires_grad or not t.is_contiguous()
               for t in (x, a, b))
        or type(scale) not in (int, float) or not math.isfinite(scale)):
        return False
    rank, hidden = a.shape
    return (1 <= rank <= 64 and 1 <= hidden <= 16384
            and 1 <= b.shape[0] <= 65536 and hidden == x.shape[-1]
            and b.shape[1] == rank)


def attention_lora_add(x, base, a, b, scale):
    """Add one ordinary FP32 adapter; unsupported inputs return None."""
    if (not _eligible_lora_input(x, a, b, scale)
        or base.ndim != x.ndim or base.shape[:-1] != x.shape[:-1]
        or base.shape[-1] != b.shape[0] or base.dtype != x.dtype
        or base.device != x.device or base.requires_grad or not base.is_contiguous()):
        return None
    try:
        from .attention_lora_decode_kernels import ordinary_lora_add
    except ImportError:
        return None
    return ordinary_lora_add(x, base, a, b, scale)


def _attention_forward(self, x, *args, **kwargs):
    # Check the cheap decode eligibility before doing any base projection work.
    if (args or kwargs or not isinstance(x, torch.Tensor) or not x.is_cuda
        or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
        or x.ndim not in (2, 3) or x.numel() != x.shape[-1]):
        return self._attention_lora_original_forward(x, *args, **kwargs)
    weights = ordinary_lora_weights(self)
    if (weights is None or not _eligible_lora_input(x, *weights)
        # Base hooks can alter adapter state as well as the returned tensor.
        # Preserve their complete PEFT call ordering through the original forward.
        or self.base_layer._forward_hooks or self.base_layer._forward_pre_hooks):
        return self._attention_lora_original_forward(x, *args, **kwargs)
    # PEFT's argument validation and the original NF4 base are preserved.
    self._check_forward_args(x)
    base = self.base_layer(x)
    result = attention_lora_add(x, base, *weights)
    if result is None:
        # An unexpected output layout/dtype or unavailable Triton must not run
        # NF4 (and any side effects) a second time. Continue PEFT's ordinary
        # adapter calculation from the base that has already been computed.
        adapter = self.active_adapters[0]
        a, b = self.lora_A[adapter], self.lora_B[adapter]
        cast_x = self._cast_input_dtype(x, a.weight.dtype)
        delta = b(a(self.lora_dropout[adapter](cast_x))) * self.scaling[adapter]
        return base.clone() + delta.to(base.dtype)
    self._attention_lora_fused_calls += 1
    return result


def configure_attention_lora_decode(model, enabled=True):
    """Install/restore only existing PEFT NF4 attention projection forwards.

    Idempotent, with no adapter copies, module replacements, or state_dict edits.
    Calling with False restores the exact previously bound forward.
    """
    if not enabled:
        for module in model.modules():
            if getattr(module, '_attention_lora_decode_enabled', False):
                module.forward = module._attention_lora_original_forward
                module._attention_lora_decode_enabled = False
        return attention_lora_decode_info(model)
    try:
        import bitsandbytes as bnb
        from peft.tuners.lora.bnb import Linear4bit
    except ImportError:
        return attention_lora_decode_info(model)
    for name, module in model.named_modules():
        if (not _ATTENTION.search(name) or type(module) is not Linear4bit
            or type(module.base_layer) is not bnb.nn.Linear4bit
            or getattr(module.base_layer.weight, 'quant_type', None) != 'nf4'
            or getattr(module, '_attention_lora_decode_enabled', False)):
            continue
        module._attention_lora_original_forward = module.forward
        module._attention_lora_fused_calls = getattr(module, '_attention_lora_fused_calls', 0)
        module._attention_lora_decode_enabled = True
        module.forward = MethodType(_attention_forward, module)
    return attention_lora_decode_info(model)


def attention_lora_decode_info(model):
    modules = [m for m in model.modules() if hasattr(m, '_attention_lora_decode_enabled')]
    return dict(backend='ordinary-attention-lora-triton',
                configured_layers=sum(m._attention_lora_decode_enabled for m in modules),
                fused_projection_calls=sum(m._attention_lora_fused_calls for m in modules))
