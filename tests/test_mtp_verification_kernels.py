"""CPU-only contracts for scoped cached two-token DeltaNet dispatch."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import importlib
import inspect
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

from models.qwen35_kernels import _delta_adapter


def fixture(length=2):
    generator = torch.Generator().manual_seed(1881)
    query = torch.randn(1, length, 2, 8, generator=generator)
    key = torch.randn(query.shape, generator=generator) / 8
    value = torch.randn(1, length, 2, 4, generator=generator)
    gate = -torch.rand(1, length, 2, generator=generator)
    beta = torch.rand(1, length, 2, generator=generator)
    initial = torch.randn(1, 2, 8, 4, generator=generator) / 8
    return (query, key, value, gate, beta), initial


def stats():
    return dict(fast=0, fallback=0, fallback_reasons={})


class MTPVerificationKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        try:
            self.scope = importlib.import_module('models.mtp_verification_kernels')
        except ModuleNotFoundError:
            self.fail('MTP verification kernel scope is not implemented')

    def dispatch(self, tensors, initial, **kwargs):
        # Distinct CPU backends make accidental default dispatch observable.
        def chunk_reference(*args, **options):
            return ('chunk', options)

        def recurrent_reference(*args, **options):
            return ('recurrent', options)

        def forbidden_kernel(*args, **options):
            raise AssertionError('CPU must not enter FLA')

        chunk_stats, recurrent_stats = stats(), stats()
        chunk = _delta_adapter(chunk_reference, forbidden_kernel, True, chunk_stats)
        recurrent = _delta_adapter(recurrent_reference, forbidden_kernel, False, recurrent_stats)
        with patch.object(modeling, 'torch_recurrent_gated_delta_rule', recurrent):
            result = chunk(*tensors, initial_state=initial, **kwargs)
        return result, chunk_stats, recurrent_stats

    def test_only_enabled_cached_length_two_selects_recurrent(self):
        tensors, initial = fixture()
        self.assertEqual(self.dispatch(tensors, initial)[0][0], 'chunk')
        with self.scope.mtp_verification_scope():
            result, chunk_counts, recurrent_counts = self.dispatch(tensors, initial)
            self.assertEqual(result[0], 'recurrent')
            self.assertEqual(recurrent_counts['fallback_reasons'], {'cpu': 1})
            self.assertEqual(chunk_counts.get('mtp_recurrent', 0), 1)
            self.assertEqual(self.dispatch(tensors, None)[0][0], 'chunk')
            for length in (1, 3):
                inputs, state = fixture(length)
                self.assertEqual(self.dispatch(inputs, state)[0][0], 'chunk')
        self.assertEqual(self.dispatch(tensors, initial)[0][0], 'chunk')

    def test_nested_disabled_scope_and_exception_restore_parent(self):
        tensors, initial = fixture()
        with self.scope.mtp_verification_scope():
            with self.scope.mtp_verification_scope(False):
                self.assertEqual(self.dispatch(tensors, initial)[0][0], 'chunk')
            self.assertEqual(self.dispatch(tensors, initial)[0][0], 'recurrent')
            with self.assertRaisesRegex(RuntimeError, 'abort'):
                with self.scope.mtp_verification_scope(False):
                    raise RuntimeError('abort')
            self.assertEqual(self.dispatch(tensors, initial)[0][0], 'recurrent')
        self.assertEqual(self.dispatch(tensors, initial)[0][0], 'chunk')

    def test_scope_is_context_and_thread_local(self):
        tensors, initial = fixture()
        other_context = copy_context()
        with self.scope.mtp_verification_scope():
            self.assertEqual(other_context.run(self.dispatch, tensors, initial)[0][0], 'chunk')
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(executor.submit(self.dispatch, tensors, initial).result()[0][0], 'chunk')
            self.assertEqual(self.dispatch(tensors, initial)[0][0], 'recurrent')

    def test_preserves_state_flags_and_metadata(self):
        tensors, initial = fixture()
        with self.scope.mtp_verification_scope():
            result, _, recurrent_counts = self.dispatch(tensors, initial,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                use_cache=True)
        self.assertEqual(result[0], 'recurrent')
        self.assertTrue(result[1]['output_final_state'])
        self.assertTrue(result[1]['use_qk_l2norm_in_kernel'])
        self.assertTrue(result[1]['use_cache'])
        self.assertIs(result[1]['initial_state'], initial)
        self.assertEqual(recurrent_counts['fallback'], 1)

    def test_existing_recurrent_guard_rejects_changed_math(self):
        tensors, initial = fixture()
        with self.scope.mtp_verification_scope():
            for changed in ({'scale': .2}, {'state_v_first': True}, {'use_gate_in_kernel': True}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    self.dispatch(tensors, initial, **changed)
            with self.assertRaises(NotImplementedError):
                self.dispatch(tensors, initial, cp_context=object())

    def test_torch_reference_outputs_state_and_gradients_remain_equivalent(self):
        chunk_reference = inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
        recurrent_reference = inspect.unwrap(modeling.torch_recurrent_gated_delta_rule)

        def forbidden_kernel(*args, **kwargs):
            raise AssertionError('CPU must not enter FLA')

        chunk = _delta_adapter(chunk_reference, forbidden_kernel, True, stats())
        recurrent = _delta_adapter(recurrent_reference, forbidden_kernel, False, stats())
        for normalized, packed in ((False, False), (True, False), (True, True)):
            tensors, initial = fixture()
            tensors[0].requires_grad_()
            saved_initial = initial.clone()
            options = dict(initial_state=initial, output_final_state=True,
                           use_qk_l2norm_in_kernel=normalized)
            expected = chunk_reference(*tensors, **options)
            if packed:
                options.update(cu_seqlens=torch.tensor([0, 2]), cu_seqlens_cpu=torch.tensor([0, 2]))
            with patch.object(modeling, 'torch_recurrent_gated_delta_rule', recurrent):
                with self.scope.mtp_verification_scope():
                    actual = chunk(*tensors, **options)
            for observed, wanted in zip(actual, expected):
                torch.testing.assert_close(observed, wanted, rtol=2e-5, atol=2e-6)
            self.assertEqual(actual[1].dtype, torch.float32)
            torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
            actual_gradient, = torch.autograd.grad(actual[0].sum(), tensors[0])
            expected_gradient, = torch.autograd.grad(expected[0].sum(), tensors[0])
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
