"""Reversible, inference-only zero-centered Qwen3.5 text RMSNorm fusion.

Only the verified upstream class/methods are supported. This adapter does not
replace modules, parameters, the FLA gated norm, or load kernels from the Hub.
"""
from functools import lru_cache
import hashlib
import inspect
import math
from types import MethodType

import torch
from torch import nn


# Exact locally inspected Transformers 5.17 source. An already replaced/fused
# method, or an upstream arithmetic change, must remain on its own forward.
_REFERENCE_SOURCE = {
    'forward': '350041feaa40063352594579e212741fdff004594990c36162458d778dae1a9e',
    '_norm': 'a1c27ed5fac56db440a5ff38ad6b04ef79ea341b7100b9e22a6b58e1f1d18aa6',
}


@lru_cache(maxsize=16)
def _reference_method(method, name):
    try:
        return (method.__module__ == 'transformers.models.qwen3_5.modeling_qwen3_5'
                and hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
                == _REFERENCE_SOURCE[name])
    except (AttributeError, OSError, TypeError):
        return False


def _row_stride(x):
    """Return a constant stride between flattened rows, without copying x.

    The strided query half of [batch, sequence, heads, 2*head_dim] is supported.
    Leading transposes, overlaps, zero strides and strided last axes fall back.
    """
    if x.ndim < 1 or x.numel() == 0 or x.stride(-1) != 1:
        return None
    hidden = x.shape[-1]
    row_stride = next((stride for size, stride in zip(reversed(x.shape[:-1]),
                                                     reversed(x.stride()[:-1])) if size > 1), hidden)
    if row_stride < hidden:
        return None
    expected = row_stride
    for size, stride in zip(reversed(x.shape[:-1]), reversed(x.stride()[:-1])):
        if size > 1 and stride != expected:
            return None
        expected *= size
    return row_stride


def _fallback_reason(module, x):
    # Keep CPU eligibility entirely independent of the CUDA runtime.
    if type(x) is not torch.Tensor or not x.is_cuda:
        return 'cpu_or_tensor_subclass'
    if module.training or torch.is_grad_enabled() or x.requires_grad or module.weight.requires_grad:
        return 'training_or_gradients'
    if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
        or module._backward_pre_hooks or nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks or nn.modules.module._global_backward_hooks
        or nn.modules.module._global_backward_pre_hooks):
        return 'hooks'
    if torch.is_autocast_enabled('cuda'):
        return 'autocast'
    if (type(module.weight) is not nn.Parameter or module.weight.layout != torch.strided
        or module.weight.device != x.device or module.weight.ndim != 1
        or not module.weight.is_contiguous() or x.layout != torch.strided
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or module.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or x.ndim < 1 or not 1 <= x.shape[-1] <= 8192
        or x.shape[-1] != module.weight.numel() or _row_stride(x) is None):
        return 'shape_layout_device_or_dtype'
    if type(module.eps) not in (int, float) or not math.isfinite(module.eps) or module.eps < 0:
        return 'epsilon'
    if (getattr(module._norm, '__func__', None) is not module._text_norm_original_norm
        or type(module) is not module._text_norm_class):
        return 'custom_method'
    return None


def _text_norm_forward(self, x, *args, **kwargs):
    reason = 'arguments' if args or kwargs else _fallback_reason(self, x)
    if reason is None:
        try:
            from .text_norm_fusion_kernels import zero_centered_rms_norm
        except ImportError:
            reason = 'triton_unavailable'
        else:
            result = zero_centered_rms_norm(x, self.weight, self.eps, _row_stride(x))
            self._text_norm_counts['fused_calls'] += 1
            return result
    counts = self._text_norm_counts
    counts['fallback_calls'] += 1
    counts['fallback_reasons'][reason] = counts['fallback_reasons'].get(reason, 0) + 1
    return self._text_norm_original_forward(x, *args, **kwargs)


def configure_text_norm_fusion(model, enabled=True):
    """Enable/restore supported RMSNorm forwards in-place; safe to call repeatedly.

    Source guards deliberately skip already accelerated or customized forwards.
    Disabling restores descriptor lookup when forward had no instance override.
    Counters are Python launch counters: they never synchronize CUDA.
    """
    if not enabled:
        for module in model.modules():
            if getattr(module, '_text_norm_enabled', False):
                if getattr(module.forward, '__func__', None) is _text_norm_forward:
                    if module._text_norm_had_forward:
                        module.forward = module._text_norm_original_forward
                    else:
                        del module.forward
                module._text_norm_enabled = False
        return text_norm_fusion_info(model)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    for module in model.modules():
        if type(module) is not Qwen3_5RMSNorm or getattr(module, '_text_norm_enabled', False):
            continue
        forward = getattr(module.forward, '__func__', None)
        norm = getattr(module._norm, '__func__', None)
        if (forward is not Qwen3_5RMSNorm.forward or norm is not Qwen3_5RMSNorm._norm
            or not _reference_method(forward, 'forward') or not _reference_method(norm, '_norm')):
            continue
        module._text_norm_had_forward = 'forward' in module.__dict__
        module._text_norm_original_forward = module.forward
        module._text_norm_original_norm = norm
        module._text_norm_class = Qwen3_5RMSNorm
        module._text_norm_counts = getattr(module, '_text_norm_counts',
                                           dict(fused_calls=0, fallback_calls=0, fallback_reasons={}))
        module._text_norm_enabled = True
        module.forward = MethodType(_text_norm_forward, module)
    return text_norm_fusion_info(model)


def text_norm_fusion_info(model):
    """Describe configuration and dispatch without reading any CUDA tensors."""
    modules = [m for m in model.modules() if hasattr(m, '_text_norm_enabled')]
    reasons = {}
    for module in modules:
        for reason, count in module._text_norm_counts['fallback_reasons'].items():
            reasons[reason] = reasons.get(reason, 0) + count
    return dict(backend='zero-centered-text-rmsnorm-triton',
                configured_layers=sum(m._text_norm_enabled for m in modules),
                fused_calls=sum(m._text_norm_counts['fused_calls'] for m in modules),
                fallback_calls=sum(m._text_norm_counts['fallback_calls'] for m in modules),
                fallback_reasons=reasons)
