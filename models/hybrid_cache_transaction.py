"""One-shot rollback for inference appends to the current hybrid DynamicCache.

Dynamic attention replaces its KV tensors on append. Quantized attention also
replaces its residual and only writes unused packed-bank rows. Saving those
references and logical lengths therefore costs no history copies. Only linear
attention's fixed convolution and recurrent tensors need cloning.

This is deliberately an append transaction, not a general cache undo facility:
reset, crop, beam operations, offloading, training, concurrent cache access and
in-place edits to historical KV or packed prefixes are outside its contract.
"""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

from .quantized_kv import QuantizedAttentionLayer, _PackedBank


class HybridCacheTransaction:
    """Snapshot a supported cache; consume the snapshot with commit or rollback.

    Both completion methods return the original cache. A completed transaction
    cannot be reused. ``copied_bytes`` counts the fixed-state tensor payload
    cloned at construction, excluding the small Python metadata snapshot.
    """

    def __init__(self, cache):
        if type(cache) is not DynamicCache:
            raise TypeError('Hybrid cache transactions require an exact DynamicCache')
        if cache.offloading:
            raise ValueError('Hybrid cache transactions do not support offloading')
        if cache.layer_class_to_replicate not in (None, DynamicLayer):
            raise TypeError('Unsupported dynamically replicated cache layer')
        for layer in cache.layers:
            if type(layer) not in (DynamicLayer, LinearAttentionLayer, QuantizedAttentionLayer):
                raise TypeError(f'Unsupported hybrid cache layer: {type(layer).__name__}')
            if type(layer) is LinearAttentionLayer and layer.record_past:
                raise ValueError('Recorded convolution history is not a bounded cache state')
            if type(layer) is QuantizedAttentionLayer:
                if type(layer.key_bank) is not _PackedBank or type(layer.value_bank) is not _PackedBank:
                    raise TypeError('Unsupported quantized packed bank')

        self.copied_bytes = 0
        self._tensor_copies = {}
        self._cache = cache
        self._layers = cache.layers
        self._original_layers = list(cache.layers)
        self._cache_state = self._attributes(cache, exclude={'layers'})
        self._layer_states = []
        self._bank_states = []
        for layer in cache.layers:
            if type(layer) is QuantizedAttentionLayer:
                state = self._attributes(layer, exclude={'key_bank', 'value_bank'}, references={'keys', 'values'})
                state['key_bank'], state['value_bank'] = layer.key_bank, layer.value_bank
                for bank in (layer.key_bank, layer.value_bank):
                    self._bank_states.append((bank, self._attributes(bank, references={'data', 'meta'})))
            else:
                state = self._attributes(layer, references={'keys', 'values'},
                                         clones={'conv_states', 'recurrent_states'})
            self._layer_states.append((layer, state))
        # The snapshots themselves own the clones until completion.
        self._tensor_copies.clear()

    def _attributes(self, obj, *, exclude=(), references=(), clones=()):
        return {name: self._snapshot(value, clone=name in clones, reference=name in references)
                for name, value in vars(obj).items() if name not in exclude}

    def _snapshot(self, value, *, clone=False, reference=False):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise ValueError('Hybrid cache transactions support inference tensors only')
            if clone:
                identity = id(value)
                if identity not in self._tensor_copies:
                    self._tensor_copies[identity] = value.detach().clone()
                    self.copied_bytes += value.numel() * value.element_size()
                return self._tensor_copies[identity]
            if reference:
                return value
            raise TypeError('Unsupported tensor metadata in hybrid cache')
        if isinstance(value, dict):
            return {key: self._snapshot(item, clone=clone, reference=reference)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [self._snapshot(item, clone=clone, reference=reference) for item in value]
        if isinstance(value, tuple):
            return tuple(self._snapshot(item, clone=clone, reference=reference) for item in value)
        if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device, type)):
            return value
        raise TypeError(f'Unsupported metadata in hybrid cache: {type(value).__name__}')

    def _require_active(self):
        if self._cache is None:
            raise RuntimeError('Hybrid cache transaction has already completed')
        return self._cache

    def _release(self):
        self._cache = None
        self._layers = None
        self._original_layers = None
        self._cache_state = None
        self._layer_states = None
        self._bank_states = None

    def rollback(self):
        """Restore the pre-append state, consume the snapshot and return the cache."""
        cache = self._require_active()
        for bank, state in self._bank_states:
            vars(bank).clear()
            vars(bank).update(state)
        for layer, state in self._layer_states:
            vars(layer).clear()
            vars(layer).update(state)
        self._layers[:] = self._original_layers
        vars(cache).clear()
        vars(cache).update(self._cache_state)
        cache.layers = self._layers
        self._release()
        return cache

    def commit(self):
        """Keep the appended state, consume the snapshot and return the cache."""
        cache = self._require_active()
        self._release()
        return cache
