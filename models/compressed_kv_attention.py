"""Direct single-query decode of native affine Q4/Q8 KV, with SDPA fallback."""
import math
from types import MethodType
import torch
from .quantized_kv import QuantizedAttentionLayer


def compressed_kv_decode(query, view, scale):
    if (not query.is_cuda or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
        or query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != 1
        or query.dtype not in (torch.bfloat16, torch.float16) or query.stride(-1) != 1
        or view.bits not in (4,8) or view.packed_tokens < 128
        or query.shape[-1] not in (64,128,256) or not math.isfinite(scale)):
        return None
    tensors = (view.keys, view.values, view.key_data, view.key_meta, view.value_data, view.value_meta)
    if any(t is None or t.device != query.device or t.requires_grad for t in tensors):
        return None
    if (view.keys.ndim != 4 or view.values.shape != view.keys.shape or view.keys.shape[0] != 1
        or view.keys.dtype != query.dtype or view.values.dtype != query.dtype
        or view.keys.shape[-1] != query.shape[-1] or query.shape[1] % view.keys.shape[1]
        or query.shape[1] // view.keys.shape[1] > 16
        or view.length != view.packed_tokens + view.keys.shape[2]
        or view.keys.stride(-1) != 1 or view.values.stride(-1) != 1):
        return None
    try:
        from .compressed_kv_kernels import decode
    except ImportError:
        return None
    return decode(query, view, scale)


def _supported_mask(mask, length):
    if mask is None:
        return True
    from torch.nn.attention.bias import CausalBias, CausalVariant
    return (type(mask) is CausalBias and mask.variant == CausalVariant.LOWER_RIGHT
            and mask.seq_len_q == 1 and mask.seq_len_kv == length)


def _forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    layer = None
    if past_key_values is not None and self.layer_idx < len(past_key_values.layers):
        layer = past_key_values.layers[self.layer_idx]
    eligible = (self._direct_kv_enabled and not self.training and not torch.is_grad_enabled()
                and not torch.is_autocast_enabled('cuda') and hidden_states.is_cuda
                and hidden_states.shape[:2] == (1,1) and hidden_states.dtype in (torch.bfloat16,torch.float16)
                and isinstance(layer, QuantizedAttentionLayer) and layer.key_bank.units > 0
                and self.head_dim in (64,128,256) and self.config._attn_implementation == 'sdpa'
                and not kwargs.get('output_attentions', False) and kwargs.get('head_mask') is None
                # Upstream decoder also forwards position_ids. RoPE has already
                # consumed the computed embeddings; SDPA ignores this metadata.
                and all(value is None or name in ('is_causal','output_attentions','use_cache','position_ids')
                        for name,value in kwargs.items())
                and _supported_mask(attention_mask, layer.get_seq_length()+1))
    if not eligible:
        return self._direct_kv_original(hidden_states, position_embeddings, attention_mask,
                                        past_key_values=past_key_values, **kwargs)
    # Keep the installed upstream projection/norm/RoPE/gating calculation.
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, self.head_dim)
    query, gate = torch.chunk(self.q_proj(hidden_states).view(*shape, -1, self.head_dim*2), 2, -1)
    gate = gate.reshape(*shape, -1)
    query = self.q_norm(query.reshape(head_shape)).transpose(1,2)
    key = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1,2)
    value = self.v_proj(hidden_states).view(head_shape).transpose(1,2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    view = layer.append_for_attention(key, value)
    output = compressed_kv_decode(query, view, self.scaling)
    if output is None:
        # State was appended once: complete the same operation via SDPA,
        # rather than invoking upstream and appending this token a second time.
        keys = torch.cat((layer.key_bank.unpack(layer.dtype), view.keys), -2)
        values = torch.cat((layer.value_bank.unpack(layer.dtype), view.values), -2)
        output = torch.nn.functional.scaled_dot_product_attention(query, keys, values,
                    enable_gqa=True, scale=self.scaling)
    else:
        self._direct_kv_calls += 1
        self._direct_kv_last_length = view.length
    layer.compact_residual()
    output = output.transpose(1,2).reshape(*shape, -1).contiguous()
    return self.o_proj(output * torch.sigmoid(gate)), None


def configure_compressed_kv_attention(model, enabled=True):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    for module in model.modules():
        if type(module) is not Qwen3_5Attention:
            continue
        if not hasattr(module, '_direct_kv_original'):
            module._direct_kv_original = module.forward
            module._direct_kv_calls = module._direct_kv_last_length = 0
            module.forward = MethodType(_forward, module)
        module._direct_kv_enabled = bool(enabled)
    return compressed_kv_info(model)


def compressed_kv_info(model):
    modules = [m for m in model.modules() if hasattr(m, '_direct_kv_enabled')]
    return dict(backend='native-affine-q4-q8-triton', configured_layers=sum(m._direct_kv_enabled for m in modules),
                direct_decode_calls=sum(m._direct_kv_calls for m in modules),
                longest_decode=max((m._direct_kv_last_length for m in modules), default=0),
                fallback='SDPA for prefill, arbitrary masks, unsupported shapes or precision')
