"""CPU contract tests; opt-in CUDA coverage needs an explicit serialized GPU lease."""
import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated


class TextNormFusionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('models.text_norm_fusion'),
                             'Text norm fusion adapter is missing')
        self.impl = importlib.import_module('models.text_norm_fusion')

    def model(self):
        model = nn.Module()
        model.input_layernorm = Qwen3_5RMSNorm(16)
        model.post_attention_layernorm = Qwen3_5RMSNorm(16)
        model.q_norm = Qwen3_5RMSNorm(8)
        model.k_norm = Qwen3_5RMSNorm(8)
        model.gated = Qwen3_5RMSNormGated(16)
        return model.eval().requires_grad_(False)

    def test_enable_disable_preserves_modules_parameters_and_state_dict(self):
        model = self.model()
        originals = {n: m.forward for n, m in model.named_modules()}
        modules = dict(model.named_modules())
        parameters = dict(model.named_parameters())
        state = {n: v.clone() for n, v in model.state_dict().items()}
        info = self.impl.configure_text_norm_fusion(model)
        self.assertEqual(info['configured_layers'], 4)
        self.assertEqual(model.gated.forward, originals['gated'])
        self.assertEqual(self.impl.configure_text_norm_fusion(model)['configured_layers'], 4)
        self.assertEqual(dict(model.named_modules()), modules)
        for name, value in model.named_parameters():
            self.assertIs(value, parameters[name])
        self.assertEqual(set(model.state_dict()), set(state))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertEqual(self.impl.configure_text_norm_fusion(model, False)['configured_layers'], 0)
        for name, module in model.named_modules():
            self.assertEqual(module.forward, originals[name])
        self.assertNotIn('forward', model.input_layernorm.__dict__)
        self.assertEqual(self.impl.configure_text_norm_fusion(model)['configured_layers'], 4)

    def test_cpu_fallback_never_queries_cuda_and_reads_live_weights(self):
        model = self.model()
        norm = model.input_layernorm
        reference = norm.forward
        self.impl.configure_text_norm_fusion(model)
        x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        with patch('torch.cuda.is_available', side_effect=AssertionError('CUDA queried')), \
             patch('torch.cuda.get_device_capability', side_effect=AssertionError('CUDA queried')), \
             patch('torch.cuda.current_device', side_effect=AssertionError('CUDA queried')), \
             torch.inference_mode():
            first = norm(x)
            norm.weight.add_(.37)
            torch.testing.assert_close(norm(x), reference(x), rtol=0, atol=0)
            self.assertFalse(torch.equal(first, norm(x)))
        self.assertEqual(self.impl.text_norm_fusion_info(model)['fused_calls'], 0)
        self.assertGreater(self.impl.text_norm_fusion_info(model)['fallback_calls'], 0)

    def test_training_gradients_hooks_and_custom_norm_keep_reference_behavior(self):
        model = self.model()
        norm = model.input_layernorm
        self.impl.configure_text_norm_fusion(model)
        norm.train().requires_grad_(True)
        x = torch.randn(2, 16, requires_grad=True)
        seen = []
        handle = norm.register_forward_hook(lambda module, args, result: seen.append(result))
        result = norm(x)
        result.sum().backward()
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        handle.remove()
        class CustomNorm(Qwen3_5RMSNorm):
            def forward(self, x):
                return super().forward(x) + 3
        custom = CustomNorm(16)
        original = custom.forward
        self.assertEqual(self.impl.configure_text_norm_fusion(custom)['configured_layers'], 0)
        self.assertEqual(custom.forward, original)
        patched = Qwen3_5RMSNorm(16)
        patched.forward = lambda x: x + 2
        original = patched.forward
        self.assertEqual(self.impl.configure_text_norm_fusion(patched)['configured_layers'], 0)
        self.assertIs(patched.forward, original)

    def test_tiny_text_model_prefill_and_cached_decode_are_identical_on_cpu(self):
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        config = Qwen3_5TextConfig(
            vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=['full_attention'], max_position_embeddings=32,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                             'partial_rotary_factor': 1.0, 'mrope_section': [4, 2, 2]})
        model = Qwen3_5TextModel(config).eval().requires_grad_(False)
        tokens = torch.tensor([[3, 7, 11]])
        with torch.inference_mode():
            expected = model(tokens, use_cache=True)
            expected_decode = model(torch.tensor([[5]]), past_key_values=expected.past_key_values,
                                    use_cache=True)
            self.assertEqual(self.impl.configure_text_norm_fusion(model)['configured_layers'], 5)
            actual = model(tokens, use_cache=True)
            actual_decode = model(torch.tensor([[5]]), past_key_values=actual.past_key_values,
                                  use_cache=True)
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(actual_decode.last_hidden_state,
                                   expected_decode.last_hidden_state, rtol=0, atol=0)
        self.assertEqual(self.impl.text_norm_fusion_info(model)['fused_calls'], 0)

    def test_row_layout_accepts_query_gate_view_and_rejects_transposes(self):
        # Qwen query projection chunks the final 2D dimension into query and gate.
        q = torch.randn(2, 3, 4, 16)[..., :8]
        self.assertEqual(self.impl._row_stride(q), 16)
        self.assertIsNone(self.impl._row_stride(q.transpose(1, 2)))
        self.assertIsNone(self.impl._row_stride(q[..., ::2]))
        self.assertEqual(self.impl._row_stride(torch.randn(16)), 16)


@unittest.skipUnless(os.environ.get('QWEN35_TEST_TEXT_NORM_CUDA') == '1',
                     'Set QWEN35_TEST_TEXT_NORM_CUDA=1 only with a serialized GPU lease')
class TextNormFusionCudaTests(unittest.TestCase):
    def test_fused_matches_fp32_zero_centered_reference_and_layouts(self):
        from models.text_norm_fusion import configure_text_norm_fusion, text_norm_fusion_info
        torch.manual_seed(77)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for hidden, shape, stride in ((2560, (1, 1, 2560), 1),
                                          (256, (2, 3, 4, 256), 2),
                                          (63, (3, 63), 1)):
                norm = Qwen3_5RMSNorm(hidden).to(device='cuda', dtype=dtype).eval().requires_grad_(False)
                with torch.inference_mode():
                    norm.weight.copy_(torch.randn_like(norm.weight) * .3)
                    norm.weight[0] = -1
                    x = torch.randn((*shape[:-1], hidden * stride), device='cuda', dtype=dtype)[..., :hidden]
                    reference = norm(x)
                    configure_text_norm_fusion(norm)
                    actual = norm(x)
                self.assertEqual(text_norm_fusion_info(norm)['fused_calls'], 1)
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.shape, x.shape)
                self.assertFalse(torch.isnan(actual).any())
                torch.testing.assert_close(actual, reference,
                                           rtol=2e-6 if dtype == torch.float32 else .008,
                                           atol=2e-6 if dtype == torch.float32 else .008)

    def test_mixed_weight_dtype_and_extreme_finite_values_keep_fp32_order(self):
        from models.text_norm_fusion import configure_text_norm_fusion, text_norm_fusion_info
        for dtype in (torch.float16, torch.bfloat16):
            norm = Qwen3_5RMSNorm(256).cuda().eval().requires_grad_(False)
            x = torch.randn(6, 256, device='cuda', dtype=dtype)
            x[0].zero_()
            x[1].mul_(1e-4)
            x[2].mul_(1000)
            with torch.inference_mode():
                # FP32 weights just above -1 catch any premature low precision
                # cast of (1+weight) or normalized input.
                norm.weight.fill_(-.99993)
                norm.weight[::3] = .00391
                expected = norm(x)
                configure_text_norm_fusion(norm)
                actual = norm(x)
            self.assertEqual(text_norm_fusion_info(norm)['fused_calls'], 1)
            torch.testing.assert_close(actual, expected, rtol=.008, atol=2e-5)

    def test_guards_fall_back_after_installation(self):
        from models.text_norm_fusion import configure_text_norm_fusion, text_norm_fusion_info
        norm = Qwen3_5RMSNorm(16).cuda().eval().requires_grad_(False)
        original = norm.forward
        configure_text_norm_fusion(norm)
        x = torch.randn(2, 3, 16, device='cuda')
        with torch.inference_mode():
            for value in (x.transpose(0, 1), x.double()):
                torch.testing.assert_close(norm(value), original(value), rtol=0, atol=0)
            handle = norm.register_forward_hook(lambda *args: None)
            torch.testing.assert_close(norm(x), original(x), rtol=0, atol=0)
            handle.remove()
            norm.train()
            torch.testing.assert_close(norm(x), original(x), rtol=0, atol=0)
            norm.eval()
            norm.weight.requires_grad_(True)
            torch.testing.assert_close(norm(x), original(x), rtol=0, atol=0)
            norm.weight.requires_grad_(False)
        grad_x = x.detach().requires_grad_(True)
        norm(grad_x).sum().backward()
        self.assertIsNotNone(grad_x.grad)
        self.assertEqual(text_norm_fusion_info(norm)['fused_calls'], 0)


if __name__ == '__main__':
    unittest.main()
