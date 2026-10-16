"""CPU rollback checks for the exact append-only hybrid cache implementations."""
import copy
import importlib.util
import inspect
import unittest
import weakref
from unittest.mock import patch

import torch
from transformers import DynamicCache, Qwen3_5TextConfig
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

from models.quantized_kv import QuantizedAttentionLayer, make_cache


class HybridCacheTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def transaction(self, cache):
        self.assertIsNotNone(importlib.util.find_spec('models.hybrid_cache_transaction'),
                             'The bounded hybrid cache transaction must exist')
        from models.hybrid_cache_transaction import HybridCacheTransaction
        return HybridCacheTransaction(cache)

    def cache(self, mode='bf16', length=3):
        config = Qwen3_5TextConfig(num_hidden_layers=2,
                                  layer_types=['linear_attention', 'full_attention'])
        cache = make_cache(config, mode)
        cache.layers[0] = LinearAttentionLayer(number_of_states=2)
        for index in range(2):
            cache.update_conv_state(torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) + index,
                                    0, state_idx=index)
            cache.update_recurrent_state(torch.full((1, 2, 4, 4), float(index + 1)),
                                         0, state_idx=index)
        if length:
            self.append(cache, length)
        return cache

    def append(self, cache, count):
        x = torch.arange(count * 64, dtype=torch.float32).reshape(1, 1, count, 64)
        cache.update(x.to(torch.bfloat16), (x + 2).to(torch.bfloat16), 1)

    def mutate_linear(self, cache):
        for index in range(2):
            cache.update_conv_state(torch.full((1, 4, 2), 77.), 0, state_idx=index)
            cache.layers[0].recurrent_states[index].add_(33)
        cache.layers[0].has_previous_state[0] = False
        cache.layers[0].conv_kernel_size[1] = 9
        cache.layers[0].temporary_attribute = 'discard me'

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_state_equal(left, right)
        elif hasattr(expected, '__dict__') and not isinstance(expected, type):
            self.assertIs(type(actual), type(expected))
            self.assert_state_equal(vars(actual), vars(expected))
        else:
            self.assertEqual(actual, expected)

    def test_rollback_restores_in_place_states_attributes_and_kv_references(self):
        cache = self.cache()
        cache.step_counts = {'target': 3}
        expected = copy.deepcopy(cache)
        layers = cache.layers
        original_layers = list(layers)
        original_keys, original_values = cache.layers[1].keys, cache.layers[1].values
        transaction = self.transaction(cache)
        self.append(cache, 2)
        self.mutate_linear(cache)
        cache.step_counts['target'] = 5
        cache.extra_attribute = 1
        self.assertIs(transaction.rollback(), cache)
        self.assertIs(cache.layers, layers)
        for actual, original in zip(cache.layers, original_layers):
            self.assertIs(actual, original)
        self.assertIs(cache.layers[1].keys, original_keys)
        self.assertIs(cache.layers[1].values, original_values)
        self.assert_state_equal(cache, expected)
        # Consuming a snapshot must allow a fresh transaction after further mutation.
        cache.layers[0].recurrent_states[0].add_(1)
        second = self.transaction(cache)
        cache.layers[0].recurrent_states[0].zero_()
        second.rollback()
        self.assertTrue(torch.all(cache.layers[0].recurrent_states[0] == 2))

    def test_quantized_compaction_and_slab_growth_restore_prefix_and_replay(self):
        for mode in ('q4', 'q8'):
            for length in (255, 256, 4351):
                with self.subTest(mode=mode, length=length):
                    cache = self.cache(mode, length)
                    layer = cache.layers[1]
                    expected = copy.deepcopy(cache)
                    old_keys, old_values = layer.keys, layer.values
                    banks = (layer.key_bank, layer.value_bank)
                    saved = [(bank.data, bank.meta, bank.units) for bank in banks]
                    prefix = [(None if bank.data is None else bank.data[:, :, :bank.units].clone(),
                               None if bank.meta is None else bank.meta[:, :, :bank.units].clone())
                              for bank in banks]
                    transaction = self.transaction(cache)
                    self.append(cache, 2)
                    self.mutate_linear(cache)
                    if length == 4351:
                        for bank, (data, _, _) in zip(banks, saved):
                            self.assertIsNot(bank.data, data)
                    transaction.rollback()
                    self.assertIs(layer.keys, old_keys)
                    self.assertIs(layer.values, old_values)
                    for bank, original, state, packed in zip(
                            (layer.key_bank, layer.value_bank), banks, saved, prefix):
                        self.assertIs(bank, original)
                        self.assertIs(bank.data, state[0])
                        self.assertIs(bank.meta, state[1])
                        self.assertEqual(bank.units, state[2])
                        if packed[0] is not None:
                            torch.testing.assert_close(bank.data[:, :, :bank.units], packed[0])
                            torch.testing.assert_close(bank.meta[:, :, :bank.units], packed[1])
                    # Unused slab rows may change, so compare the visible prefix rather than padding.
                    self.assert_state_equal(vars(cache.layers[0]), vars(expected.layers[0]))
                    self.assertEqual(cache.get_seq_length(), length)
                    actual = cache.update(torch.ones(1, 1, 1, 64), torch.full((1, 1, 1, 64), 3.), 1)
                    wanted = expected.update(torch.ones(1, 1, 1, 64), torch.full((1, 1, 1, 64), 3.), 1)
                    for left, right in zip(actual, wanted):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_snapshot_copies_only_fixed_states_regardless_of_history_length(self):
        for mode in ('bf16', 'q4', 'q8'):
            for length in (3, 4351):
                with self.subTest(mode=mode, length=length):
                    cache = self.cache(mode, length)
                    cloned = []
                    clone = torch.Tensor.clone

                    def track(tensor, *args, **kwargs):
                        cloned.append(tensor.numel() * tensor.element_size())
                        return clone(tensor, *args, **kwargs)

                    with patch.object(torch.Tensor, 'clone', track):
                        transaction = self.transaction(cache)
                    # Two 12-float convolution and two 32-float recurrent states.
                    self.assertEqual(sorted(cloned), [48, 48, 128, 128])
                    self.assertEqual(transaction.copied_bytes, 352)
                    transaction.commit()

    def test_commit_keeps_new_state_and_all_completion_actions_are_one_shot(self):
        for action in ('commit', 'rollback'):
            with self.subTest(action=action):
                cache = self.cache()
                transaction = self.transaction(cache)
                self.append(cache, 2)
                self.mutate_linear(cache)
                self.assertIs(getattr(transaction, action)(), cache)
                self.assertEqual(cache.get_seq_length(), 5 if action == 'commit' else 3)
                for repeated in ('commit', 'rollback'):
                    with self.assertRaises(RuntimeError):
                        getattr(transaction, repeated)()

    def test_commit_releases_old_history_and_fixed_state_clones(self):
        cache = self.cache()
        old_keys = weakref.ref(cache.layers[1].keys)
        clones = []
        clone = torch.Tensor.clone

        def track(tensor, *args, **kwargs):
            result = clone(tensor, *args, **kwargs)
            clones.append(weakref.ref(result))
            return result

        with patch.object(torch.Tensor, 'clone', track):
            transaction = self.transaction(cache)
        self.append(cache, 2)
        self.assertIsNotNone(old_keys())
        self.assertTrue(all(reference() is not None for reference in clones))
        transaction.commit()
        self.assertIsNone(old_keys())
        self.assertTrue(all(reference() is None for reference in clones))

    def test_uninitialized_and_dynamically_added_layers_are_restored(self):
        cache = DynamicCache()
        transaction = self.transaction(cache)
        x = torch.ones(1, 1, 2, 64)
        cache.update(x, x, 0)
        transaction.rollback()
        self.assertEqual(cache.layers, [])
        cache = make_cache(Qwen3_5TextConfig(num_hidden_layers=2,
                           layer_types=['linear_attention', 'full_attention']), 'q4')
        expected = copy.deepcopy(cache)
        transaction = self.transaction(cache)
        cache.update_conv_state(torch.ones(1, 4, 3), 0)
        cache.update_recurrent_state(torch.ones(1, 2, 4, 4), 0)
        self.append(cache, 256)
        transaction.rollback()
        self.assert_state_equal(cache, expected)

    def test_unknown_cache_layers_and_unsafe_modes_rejected_without_mutation(self):
        class CacheSubtype(DynamicCache):
            pass

        class LayerSubtype(DynamicLayer):
            pass

        class LinearSubtype(LinearAttentionLayer):
            pass

        class QuantizedSubtype(QuantizedAttentionLayer):
            pass

        class UnsupportedLayer:
            def __init__(self):
                self.touched = False

        cases = [CacheSubtype()] + [self.cache() for _ in range(6)]
        cases[1].layers[1] = LayerSubtype()
        cases[2].offloading = True
        cases[3].layers[0].record_past = True
        cases[4].layers[0] = LinearSubtype()
        cases[5].layers[1] = QuantizedSubtype(8)
        cases[6].layers[1] = UnsupportedLayer()
        for cache in cases:
            with self.subTest(cache=type(cache), layers=[type(layer) for layer in cache.layers]):
                expected = copy.deepcopy(cache)
                with self.assertRaises((TypeError, ValueError)):
                    self.transaction(cache)
                self.assert_state_equal(cache, expected)

    def test_grad_bearing_snapshots_and_unknown_packed_banks_are_rejected(self):
        cache = self.cache()
        cache.layers[0].recurrent_states[0].requires_grad_(True)
        expected = copy.deepcopy(cache)
        with self.assertRaisesRegex(ValueError, 'inference'):
            self.transaction(cache)
        self.assert_state_equal(cache, expected)
        cache = self.cache('q8', 256)
        cache.layers[1].key_bank = object()
        with self.assertRaisesRegex(TypeError, 'bank'):
            self.transaction(cache)

    @torch.inference_mode()
    def test_real_tiny_model_exception_restores_cache_and_exact_next_logits(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

        # Installed FLA wrappers select CUDA kernels for CPU tensors; use the
        # upstream PyTorch reference math, as in the existing CPU model tests.
        with patch.object(upstream, 'torch_chunk_gated_delta_rule',
                          inspect.unwrap(upstream.torch_chunk_gated_delta_rule)), \
             patch.object(upstream, 'torch_recurrent_gated_delta_rule',
                          inspect.unwrap(upstream.torch_recurrent_gated_delta_rule)):
            torch.manual_seed(27)
            config = tiny_config()
            config.text_config.head_dim = 64
            config.text_config.max_position_embeddings = 512
            config.text_config.rope_parameters['mrope_section'] = [16, 8, 8]
            model = TextOnlyQwen3_5ForCausalLM(config).eval()
            for mode in ('bf16', 'q4', 'q8'):
                with self.subTest(mode=mode):
                    cache = make_cache(config.text_config, mode)
                    model(torch.full((1, 255), 5), past_key_values=cache, use_cache=True)
                    expected_cache = copy.deepcopy(cache)
                    transaction = self.transaction(cache)

                    def fail_after_cache_update(*args):
                        raise RuntimeError('forced forward failure')

                    hook = model.model.language_model.layers[-1].register_forward_hook(fail_after_cache_update)
                    try:
                        with self.assertRaisesRegex(RuntimeError, 'forced forward failure'):
                            model(torch.tensor([[9, 11]]), past_key_values=cache, use_cache=True)
                    finally:
                        hook.remove()
                        transaction.rollback()
                    self.assertEqual(cache.get_seq_length(), 255)
                    actual = model(torch.tensor([[9]]), past_key_values=cache, use_cache=True)
                    expected = model(torch.tensor([[9]]), past_key_values=expected_cache, use_cache=True)
                    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                    self.assert_state_equal(vars(cache.layers[0]), vars(expected_cache.layers[0]))
                    torch.testing.assert_close(cache.layers[1].keys, expected_cache.layers[1].keys,
                                               rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
