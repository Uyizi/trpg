"""Local, weight-free checks for optional Qwen3.5 FLA acceleration."""
import inspect
import unittest

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

from models.qwen35_kernels import install_convolution_kernels, kernel_metadata


BOUND_ORIGINALS = {
    name: getattr(modeling, name) for name in (
        'causal_conv1d_fn', 'causal_conv1d_update',
        'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
    )
}
ORIGINALS = {name: inspect.unwrap(function) for name, function in BOUND_ORIGINALS.items()}


def inputs(length, device='cpu', dtype=torch.float32, batch=2, heads=3):
    generator = torch.Generator(device=device).manual_seed(37 + length)
    q = torch.randn(batch, length, heads, 32, generator=generator, device=device, dtype=dtype)
    k = torch.randn(q.shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(batch, length, heads, 16, generator=generator, device=device, dtype=dtype)
    g = -torch.rand(batch, length, heads, generator=generator, device=device, dtype=torch.float32)
    beta = torch.rand(batch, length, heads, generator=generator, device=device, dtype=dtype)
    return q, k, v, g, beta


class FastKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        install_convolution_kernels()

    def assert_pair_close(self, actual, expected, rtol=0, atol=0):
        torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
        if expected[1] is None:
            self.assertIsNone(actual[1])
        else:
            torch.testing.assert_close(actual[1], expected[1], rtol=rtol, atol=atol)
            self.assertEqual(actual[1].dtype, torch.float32)

    def test_install_patches_both_delta_rules_and_is_idempotent(self):
        installed = getattr(modeling, '_local_fla_kernels', ())
        self.assertEqual(set(installed), set(ORIGINALS))
        wrappers = {name: getattr(modeling, name) for name in ORIGINALS}
        install_convolution_kernels()
        for name, wrapper in wrappers.items():
            self.assertIs(getattr(modeling, name), wrapper)
            self.assertIs(wrapper.__wrapped__, BOUND_ORIGINALS[name])

    def test_dispatch_metadata_counts_successful_reference_calls(self):
        tensors = inputs(3)
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            before = kernel_metadata().get('dispatch', {}).get(name, {})
            actual = getattr(modeling, name)(*tensors, use_qk_l2norm_in_kernel=True, use_cache=True)
            expected = ORIGINALS[name](*tensors, use_qk_l2norm_in_kernel=True)
            self.assert_pair_close(actual, expected)
            after = kernel_metadata().get('dispatch', {}).get(name, {})
            self.assertEqual(after.get('fallback', 0), before.get('fallback', 0) + 1)
            self.assertEqual(after.get('fast', 0), before.get('fast', 0))
            self.assertEqual(after.get('fallback_reasons', {}).get('cpu', 0),
                             before.get('fallback_reasons', {}).get('cpu', 0) + 1)

    def test_cpu_reference_outputs_final_states_and_backward_are_preserved(self):
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            for normalized in (False, True):
                with self.subTest(name=name, normalized=normalized):
                    tensors = inputs(7)
                    # Stable unnormalized keys keep the recurrence well conditioned.
                    tensors = (tensors[0], tensors[1] / 8, *tensors[2:])
                    q = tensors[0].requires_grad_()
                    initial = torch.randn(2, 3, 32, 16) / 8
                    saved_initial = initial.clone()
                    options = dict(initial_state=initial, output_final_state=True,
                                   use_qk_l2norm_in_kernel=normalized, use_cache=True)
                    expected = ORIGINALS[name](*tensors, **options)
                    actual = getattr(modeling, name)(*tensors, **options)
                    self.assert_pair_close(actual, expected)
                    torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
                    actual_gradient, = torch.autograd.grad(actual[0].sum(), q)
                    expected_gradient, = torch.autograd.grad(expected[0].sum(), q)
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                    self.assert_pair_close(getattr(modeling, name)(*tensors), ORIGINALS[name](*tensors))

    def test_packed_cpu_sequences_have_independent_states(self):
        tensors = inputs(10, batch=1)
        lengths = [0, 3, 10]
        initial = torch.randn(2, 3, 32, 16) / 8
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            options = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
            parts = [ORIGINALS[name](*(x[:, start:end] for x in tensors),
                                     initial_state=initial[index:index + 1], **options)
                     for index, (start, end) in enumerate(zip(lengths, lengths[1:]))]
            expected = (torch.cat([p[0] for p in parts], dim=1), torch.cat([p[1] for p in parts]))
            actual = getattr(modeling, name)(*tensors, cu_seqlens=torch.tensor(lengths),
                                             initial_state=initial, **options)
            self.assert_pair_close(actual, expected)

    def test_malformed_packed_lengths_and_parallel_context_are_rejected(self):
        tensors = inputs(7, batch=1)
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            for lengths in ([1, 7], [0, 8], [0, 5, 4, 7], [0], [0, 0, 7]):
                with self.subTest(name=name, lengths=lengths):
                    with self.assertRaises(ValueError):
                        getattr(modeling, name)(*tensors, cu_seqlens=torch.tensor(lengths))
            with self.assertRaises((NotImplementedError, ValueError)):
                getattr(modeling, name)(*tensors, cp_context=object())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for FLA numerical checks')
    @torch.inference_mode()
    def test_cuda_prefill_and_cached_decode_match_reference(self):
        for dtype in (torch.bfloat16, torch.float16):
            for normalized in (False, True):
                for length in (1, 17, 79):
                    with self.subTest(dtype=dtype, normalized=normalized, length=length):
                        tensors = inputs(length, 'cuda', dtype)
                        if not normalized:
                            tensors = (tensors[0], tensors[1] / 8, *tensors[2:])
                        name = 'torch_recurrent_gated_delta_rule' if length == 1 else 'torch_chunk_gated_delta_rule'
                        options = dict(use_qk_l2norm_in_kernel=normalized, output_final_state=True)
                        initial = torch.randn(2, 3, 32, 16, device='cuda') / 8
                        saved_initial = initial.clone()
                        actual = getattr(modeling, name)(*tensors, initial_state=initial, **options)
                        expected = ORIGINALS[name](*tensors, initial_state=initial, **options)
                        self.assert_pair_close(actual, expected, rtol=0.025, atol=0.003)
                        torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
                        decode = inputs(1, 'cuda', dtype)
                        if not normalized:
                            decode = (decode[0], decode[1] / 8, *decode[2:])
                        actual_next = modeling.torch_recurrent_gated_delta_rule(
                            *decode, initial_state=actual[1], **options)
                        expected_next = ORIGINALS['torch_recurrent_gated_delta_rule'](
                            *decode, initial_state=expected[1], **options)
                        self.assert_pair_close(actual_next, expected_next, rtol=0.025, atol=0.003)
                        no_state = getattr(modeling, name)(*tensors, use_qk_l2norm_in_kernel=normalized)
                        self.assertIsNone(no_state[1])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for FLA numerical checks')
    @torch.inference_mode()
    def test_cuda_packed_sequences_match_independent_reference(self):
        tensors = inputs(31, 'cuda', torch.bfloat16, batch=1)
        lengths = [0, 9, 31]
        initial = torch.randn(2, 3, 32, 16, device='cuda') / 8
        options = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            parts = [ORIGINALS[name](*(x[:, start:end] for x in tensors),
                                     initial_state=initial[index:index + 1], **options)
                     for index, (start, end) in enumerate(zip(lengths, lengths[1:]))]
            expected = (torch.cat([p[0] for p in parts], dim=1), torch.cat([p[1] for p in parts]))
            actual = getattr(modeling, name)(*tensors, cu_seqlens=torch.tensor(lengths, device='cuda'),
                                             initial_state=initial, **options)
            self.assert_pair_close(actual, expected, rtol=0.025, atol=0.003)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for FLA numerical checks')
    def test_cuda_recurrent_training_keeps_working_backward(self):
        tensors = inputs(3, 'cuda', torch.bfloat16)
        tensors[0].requires_grad_()
        options = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
        actual = modeling.torch_recurrent_gated_delta_rule(*tensors, **options)
        expected = ORIGINALS['torch_recurrent_gated_delta_rule'](*tensors, **options)
        self.assert_pair_close(actual, expected)
        actual_gradient, = torch.autograd.grad(actual[0].sum(), tensors[0])
        expected_gradient, = torch.autograd.grad(expected[0].sum(), tensors[0])
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    @torch.inference_mode()
    def test_packed_convolution_resets_at_each_sequence_boundary(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            x = torch.randn(1, 24, 11, device=device, dtype=torch.bfloat16)
            weight = torch.randn(24, 4, device=device, dtype=torch.bfloat16)
            expected = torch.cat([ORIGINALS['causal_conv1d_fn'](x[..., :3], weight, activation='silu'),
                                  ORIGINALS['causal_conv1d_fn'](x[..., 3:], weight, activation='silu')], dim=-1)
            for keyword in ('cu_seqlens', 'cu_seq_lens_q'):
                actual = modeling.causal_conv1d_fn(
                    x, weight, activation='silu', **{keyword: torch.tensor([0, 3, 11], device=device)})
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    @torch.inference_mode()
    def test_convolution_cpu_and_cuda_keep_cache_update_semantics(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            x = torch.randn(2, 24, 9, device=device, dtype=torch.bfloat16)
            weight = torch.randn(24, 4, device=device, dtype=torch.bfloat16)
            bias = torch.randn(24, device=device, dtype=torch.bfloat16)
            for activation in (None, 'silu'):
                expected = ORIGINALS['causal_conv1d_fn'](x, weight, bias, activation)
                actual = modeling.causal_conv1d_fn(x, weight, bias, activation, use_cache=True)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                state = torch.randn(2, 24, 4, device=device, dtype=torch.bfloat16)
                expected_state = state.clone()
                expected = ORIGINALS['causal_conv1d_update'](x[..., :1], expected_state, weight, bias, activation)
                actual = modeling.causal_conv1d_update(x[..., :1], state, weight, bias, activation)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
