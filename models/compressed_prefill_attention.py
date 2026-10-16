"""Guarded native packed prefill API; callers own mask dispatch and compaction."""
import math
import torch


def compressed_kv_prefill(query, view, scale):
    """Return lower-right causal attention, or None for unsupported inputs.

    Call after appending the current chunk and before compact_residual().
    Custom attention masks must be rejected by the caller before using this API.
    """
    if (query.ndim != 4 or not query.is_cuda or torch.is_grad_enabled()
        or torch.is_autocast_enabled('cuda') or query.requires_grad
        or query.shape[0] != 1 or not 16 <= query.shape[2] <= 4096
        or query.shape[-1] not in (64, 128, 256) or query.stride(-1) != 1
        or query.dtype not in (torch.bfloat16, torch.float16)
        or view.bits not in (4, 8) or view.packed_tokens <= 0
        or view.packed_tokens % 128 or not math.isfinite(scale)):
        return None
    tensors = (view.keys, view.values, view.key_data, view.key_meta, view.value_data, view.value_meta)
    if any(t is None or t.device != query.device or t.requires_grad for t in tensors):
        return None
    keys, values = view.keys, view.values
    kd, km, vd, vm = tensors[2:]
    d, packed = query.shape[-1], view.packed_tokens
    if (keys.ndim != 4 or values.shape != keys.shape or keys.shape[0] != 1
        or keys.shape[1] == 0 or keys.shape[-1] != d
        or keys.dtype != query.dtype or values.dtype != query.dtype
        or keys.stride(-1) != 1 or values.stride(-1) != 1
        or keys.shape[2] < query.shape[2]
        or view.length != packed + keys.shape[2]
        or query.shape[1] == 0 or query.shape[1] % keys.shape[1]
        or query.shape[1]//keys.shape[1] > 16):
        return None
    h, samples = keys.shape[1], 128//(2 if view.bits == 4 else 1)
    if (kd.ndim != 5 or km.ndim != 5 or vd.ndim != 5 or vm.ndim != 5
        or kd.shape[:2] != (1, h) or km.shape[:2] != (1, h)
        or vd.shape[:2] != (1, h) or vm.shape[:2] != (1, h)
        or kd.shape[2] < packed//128 or km.shape[2] < packed//128
        or vd.shape[2] < packed or vm.shape[2] < packed
        or kd.shape[3:] != (d, samples) or km.shape[3:] != (d, 2)
        or vd.shape[3:] != (d//64, 64//(2 if view.bits == 4 else 1))
        or vm.shape[3:] != (d//64, 2)
        or kd.dtype != torch.uint8 or vd.dtype != torch.uint8
        or km.dtype != torch.float32 or vm.dtype != torch.float32
        or any(t.stride(-1) != 1 for t in (kd, km, vd, vm))):
        return None
    try:
        from .compressed_prefill_kernels import prefill
    except ImportError:
        return None
    return prefill(query, view, scale)
