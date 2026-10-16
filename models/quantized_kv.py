"""Hybrid Qwen cache: blockwise affine Q4/Q8 KV plus unmodified recurrent state.

KIVI (https://arxiv.org/abs/2402.02750) motivates K per-channel over time,
V per-token over channels and a recent high-precision residual. This is a
local 4/8-bit implementation, not the paper's kernels or its benchmark claim.
Prefill/reference attention dequantizes one layer. Direct decode can consume
the packed view without materializing a full-history BF16 KV tensor.
"""
from typing import NamedTuple
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


def _pack(rows, bits):
    if rows.is_cuda:
        from .quantized_kv_kernels import pack
        return pack(rows, bits)
    rows = rows.float()
    low = rows.amin(-1)
    scale = ((rows.amax(-1) - low) / ((1 << bits) - 1)).clamp_min(1e-8)
    codes = ((rows - low[..., None]) / scale[..., None]).round().clamp(0, (1 << bits) - 1).to(torch.uint8)
    if bits == 4:
        codes = codes[..., ::2] | (codes[..., 1::2] << 4)
    return codes, torch.stack((scale, low), -1)


def _unpack(data, meta, bits, dtype):
    if data.is_cuda:
        from .quantized_kv_kernels import unpack
        return unpack(data, meta, bits, dtype)
    codes = data if bits == 8 else torch.stack((data & 15, data >> 4), -1).flatten(-2)
    return (codes.float() * meta[..., :1] + meta[..., 1:]).to(dtype)


class _PackedBank:
    """Grow packed storage in slabs, never re-quantize historical blocks."""
    def __init__(self, bits, kind):
        self.bits, self.kind = bits, kind
        self.data = self.meta = None
        self.units = 0

    def append(self, x):
        b, h, t, d = x.shape
        if self.kind == 'key':
            # [B,H,time blocks,channel,128 time samples]
            rows = x.reshape(b, h, t // 128, 128, d).transpose(-1, -2).contiguous()
            slab = 32
        else:
            rows = x.reshape(b, h, t, d // 64, 64).contiguous()
            slab = 4096
        data, meta = _pack(rows, self.bits)
        end = self.units + data.shape[2]
        if self.data is None or end > self.data.shape[2]:
            capacity = ((end + slab - 1) // slab) * slab
            shape = list(data.shape); shape[2] = capacity
            new_data = torch.empty(shape, device=x.device, dtype=torch.uint8)
            shape[-1] = 2
            new_meta = torch.empty(shape, device=x.device, dtype=torch.float32)
            if self.units:
                new_data[:, :, :self.units].copy_(self.data[:, :, :self.units])
                new_meta[:, :, :self.units].copy_(self.meta[:, :, :self.units])
            self.data, self.meta = new_data, new_meta
        self.data[:, :, self.units:end].copy_(data)
        self.meta[:, :, self.units:end].copy_(meta)
        self.units = end

    def unpack(self, dtype):
        # The slicing may have slab padding; kernels consume contiguous rows.
        rows = _unpack(self.data[:, :, :self.units].contiguous(),
                       self.meta[:, :, :self.units].contiguous(), self.bits, dtype)
        b, h = rows.shape[:2]
        if self.kind == 'key':
            return rows.transpose(-1, -2).reshape(b, h, -1, rows.shape[-2])
        return rows.flatten(-2)

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.data, self.meta) if t is not None)


class QuantizedKVView(NamedTuple):
    key_data: torch.Tensor | None
    key_meta: torch.Tensor | None
    value_data: torch.Tensor | None
    value_meta: torch.Tensor | None
    packed_tokens: int
    keys: torch.Tensor
    values: torch.Tensor
    bits: int
    length: int


class QuantizedAttentionLayer(DynamicLayer):
    is_croppable = False

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.cumulative_length = 0
        self.key_bank = _PackedBank(bits, 'key')
        self.value_bank = _PackedBank(bits, 'value')

    def append_for_attention(self, key_states, value_states):
        if key_states.shape[-1] % 64 or value_states.shape[-1] % 64:
            raise ValueError('量化 KV 要求 head_dim 为 64 的倍数')
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        self.keys = torch.cat((self.keys, key_states), dim=-2)
        self.values = torch.cat((self.values, value_states), dim=-2)
        return QuantizedKVView(self.key_bank.data, self.key_bank.meta,
                               self.value_bank.data, self.value_bank.meta,
                               self.key_bank.units * 128, self.keys, self.values,
                               self.bits, self.cumulative_length)

    def compact_residual(self):
        # Compact after current attention: that call still sees the old exact
        # residual, matching the original cache update's rounding boundary.
        count = max(0, (self.keys.shape[-2] - 128) // 128 * 128)
        if count:
            self.key_bank.append(self.keys[..., :count, :])
            self.value_bank.append(self.values[..., :count, :])
            self.keys = self.keys[..., count:, :].clone()
            self.values = self.values[..., count:, :].clone()

    def update(self, key_states, value_states, *args, **kwargs):
        self.append_for_attention(key_states, value_states)
        if self.key_bank.units:
            keys = torch.cat((self.key_bank.unpack(self.dtype), self.keys), dim=-2)
            values = torch.cat((self.value_bank.unpack(self.dtype), self.values), dim=-2)
        else:
            keys, values = self.keys, self.values
        self.compact_residual()
        return keys, values

    def get_seq_length(self):
        return self.cumulative_length

    def bytes(self):
        return self.key_bank.bytes() + self.value_bank.bytes() + sum(
            t.numel() * t.element_size() for t in (self.keys, self.values) if t is not None)

    def reset(self):
        self.__init__(self.bits)

    def reorder_cache(self, beam_idx):
        raise NotImplementedError('量化混合缓存仅支持单会话，不能进行 beam reorder')

    def offload(self):
        raise NotImplementedError('此量化缓存不支持 CPU 卸载')

    def crop(self, *args, **kwargs):
        raise NotImplementedError('混合状态必须整体重建，不能截断 KV 后继续复用')

    def batch_repeat_interleave(self, *args, **kwargs):
        raise NotImplementedError('量化混合缓存仅支持单会话')

    def batch_select_indices(self, *args, **kwargs):
        raise NotImplementedError('量化混合缓存仅支持单会话')


def make_cache(config, mode='q8'):
    if mode not in ('bf16', 'q8', 'q4'):
        raise ValueError('KV 模式需为 bf16、q8 或 q4')
    cache = DynamicCache(config=config)
    if mode != 'bf16':
        for index, kind in enumerate(config.layer_types):
            if kind == 'full_attention':
                cache.layers[index] = QuantizedAttentionLayer(int(mode[1:]))
    return cache


def cache_bytes(cache):
    # Newer Transformers holds linear-attention state in dictionaries. Include
    # those states and entire retained storage, counting shared views only once.
    seen = set()
    def retained(value):
        if isinstance(value, torch.Tensor):
            storage=value.untyped_storage()
            key=(value.device,storage.data_ptr())
            if key in seen:return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, dict):return sum(retained(v) for v in value.values())
        if isinstance(value, (list,tuple)):return sum(retained(v) for v in value)
        if isinstance(value, _PackedBank):return sum(retained(v) for v in vars(value).values())
        return 0
    return sum(retained(vars(layer)) for layer in cache.layers)
