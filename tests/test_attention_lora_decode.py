"""Ordinary attention adapters retain PEFT behavior outside frozen CUDA decode."""
import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn


class AttentionLoraTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('models.attention_lora_decode'),
                             'Ordinary attention LoRA decode fusion is missing')
        self.impl = importlib.import_module('models.attention_lora_decode')

    def layer(self, device='cpu', bias=False, rank=4):
        import bitsandbytes as bnb
        from peft import LoraConfig
        from peft.tuners.lora.bnb import Linear4bit
        base = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                                 quant_type='nf4').to(device)
        config = LoraConfig(r=rank, lora_alpha=6, lora_dropout=.1, lora_bias=bias)
        layer = Linear4bit(base, 'default', config=config, r=rank, lora_alpha=6,
                          lora_dropout=.1, lora_bias=bias).to(device)
        # PeftModel's normal adapter loading promotes these matrices to FP32.
        layer.lora_A['default'].float()
        layer.lora_B['default'].float()
        nn.init.normal_(layer.lora_B['default'].weight, std=.1)
        return layer.eval().requires_grad_(False)

    def model(self, device='cpu'):
        model = nn.Module()
        model.self_attn = nn.Module()
        model.self_attn.q_proj = self.layer(device)
        model.linear_attn = nn.Module()
        model.linear_attn.in_proj_qkv = self.layer(device)
        model.mlp = nn.Module()
        model.mlp.q_proj = self.layer(device)
        return model

    def test_installer_scopes_to_attention_and_restores_original_forward(self):
        model = self.model()
        x = torch.randn(1, 1, 64, dtype=torch.bfloat16)
        original = model.self_attn.q_proj.forward
        excluded = model.mlp.q_proj.forward
        with torch.inference_mode():
            expected = original(x)
            self.impl.configure_attention_lora_decode(model, True)
            self.assertEqual(self.impl.attention_lora_decode_info(model)['configured_layers'], 2)
            self.assertEqual(model.mlp.q_proj.forward, excluded)
            self.assertTrue(torch.equal(model.self_attn.q_proj(x), expected))
            self.impl.configure_attention_lora_decode(model, True)
            self.assertEqual(self.impl.attention_lora_decode_info(model)['configured_layers'], 2)
            self.impl.configure_attention_lora_decode(model, False)
            self.assertEqual(model.self_attn.q_proj.forward, original)
            self.assertEqual(self.impl.attention_lora_decode_info(model)['configured_layers'], 0)

    def test_live_adapter_state_and_unsupported_variants_are_rechecked(self):
        layer = self.layer()
        weights = self.impl.ordinary_lora_weights(layer)
        self.assertIs(weights[0], layer.lora_A['default'].weight)
        self.assertIs(weights[1], layer.lora_B['default'].weight)
        self.assertEqual(weights[2], 1.5)
        layer.scaling['default'] = -.5
        self.assertEqual(self.impl.ordinary_lora_weights(layer)[2], -.5)
        layer.enable_adapters(False)
        self.assertIsNone(self.impl.ordinary_lora_weights(layer))
        layer.enable_adapters(True)
        layer.requires_grad_(False)
        layer.train()
        self.assertIsNone(self.impl.ordinary_lora_weights(layer))
        layer.eval()
        layer.lora_variant['default'] = object()
        self.assertIsNone(self.impl.ordinary_lora_weights(layer))
        layer.lora_variant.clear()
        self.assertIsNone(self.impl.ordinary_lora_weights(self.layer(bias=True)))
        layer.lora_A['default'].register_forward_hook(lambda *args: None)
        self.assertIsNone(self.impl.ordinary_lora_weights(layer))

    def test_cpu_fallback_preserves_live_weights_disabled_adapter_and_gradients(self):
        model = self.model()
        layer = model.self_attn.q_proj
        x = torch.randn(1, 1, 64, dtype=torch.bfloat16)
        original = layer.forward
        self.impl.configure_attention_lora_decode(model, True)
        with torch.inference_mode():
            before = layer(x)
            layer.lora_B['default'].weight.add_(.5)
            torch.testing.assert_close(layer(x), original(x), rtol=0, atol=0)
            self.assertFalse(torch.equal(layer(x), before))
            layer.enable_adapters(False)
            torch.testing.assert_close(layer(x), original(x), rtol=0, atol=0)
        layer.enable_adapters(True)
        layer.train()
        y = layer(x)
        y.float().sum().backward()
        self.assertIsNotNone(layer.lora_A['default'].weight.grad)
        self.assertEqual(self.impl.attention_lora_decode_info(model)['fused_projection_calls'], 0)

    def test_raw_cpu_input_falls_back(self):
        with torch.inference_mode():
            self.assertIsNone(self.impl.attention_lora_add(
                torch.ones(1, 64, dtype=torch.bfloat16),
                torch.ones(1, 64, dtype=torch.bfloat16),
                torch.ones(4, 64), torch.ones(64, 4), 1.5))

    def test_global_hooks_and_live_adapter_replacement_are_preserved(self):
        layer = self.layer()
        replacement = nn.Linear(64, 4, bias=False).requires_grad_(False).eval()
        layer.lora_A['default'] = replacement
        self.assertIs(self.impl.ordinary_lora_weights(layer)[0], replacement.weight)
        handle = nn.modules.module.register_module_forward_hook(lambda *args: None)
        try:
            self.assertIsNone(self.impl.ordinary_lora_weights(layer),
                              'Fusion cannot bypass global hooks on LoRA submodules')
        finally:
            handle.remove()

    def test_ineligible_live_states_fall_back(self):
        for state in ('multiple', 'merged', 'trainable', 'dtype', 'dora', 'scale', 'dropout', 'cast'):
            with self.subTest(state=state):
                layer = self.layer()
                if state == 'multiple': layer.set_adapter(['default', 'other'])
                if state == 'merged': layer.merged_adapters.append('default')
                if state == 'trainable': layer.lora_A['default'].requires_grad_(True)
                if state == 'dtype': layer.lora_A['default'].bfloat16()
                if state == 'dora': layer.use_dora['default'] = True
                if state == 'scale': layer.scaling['default'] = float('nan')
                if state == 'dropout': layer.lora_dropout['default'] = nn.ReLU().eval()
                if state == 'cast': layer.cast_input_dtype_enabled = False
                self.assertIsNone(self.impl.ordinary_lora_weights(layer))

    @unittest.skipUnless(os.environ.get('QWEN_TEST_GPU') == '1', 'Exclusive CUDA window required')
    def test_cuda_fallback_calls_nf4_base_once_and_preserves_base_hooks(self):
        for reason in ('rank128', 'base_dtype', 'input_dtype', 'input_layout'):
            with self.subTest(reason=reason):
                model = nn.Module()
                model.self_attn = nn.Module()
                layer = model.self_attn.q_proj = self.layer('cuda', rank=128 if reason == 'rank128' else 4)
                x = torch.randn(1, 1, 64, device='cuda', dtype=torch.bfloat16)
                if reason == 'input_dtype': x = x.float()
                if reason == 'input_layout':
                    x = torch.randn(1, 1, 128, device='cuda', dtype=torch.bfloat16)[..., ::2]
                base_calls = []
                def base_hook(module, inputs, output):
                    base_calls.append(output)
                    return output.float() if reason == 'base_dtype' else output
                handle = layer.base_layer.register_forward_hook(base_hook)
                try:
                    with torch.inference_mode():
                        expected = layer(x)
                        base_calls.clear()
                        self.impl.configure_attention_lora_decode(model, True)
                        actual = layer(x)
                    self.assertEqual(len(base_calls), 1, 'Fallback must not evaluate the NF4 base twice')
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(self.impl.attention_lora_decode_info(model)['fused_projection_calls'], 0)
                finally:
                    handle.remove()

    @unittest.skipUnless(os.environ.get('QWEN_TEST_GPU') == '1', 'Exclusive CUDA window required')
    def test_cuda_unexpected_postbase_rejection_reuses_computed_base(self):
        model = self.model('cuda')
        layer = model.self_attn.q_proj
        x = torch.randn(1, 1, 64, device='cuda', dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = layer(x)
        base_calls = []
        base_forward = type(layer.base_layer).forward
        def counted_base(module, inputs):
            base_calls.append(module)
            return base_forward(module, inputs)
        self.impl.configure_attention_lora_decode(model, True)
        with (patch.object(type(layer.base_layer), 'forward', counted_base),
              patch.object(self.impl, 'attention_lora_add', return_value=None),
              torch.inference_mode()):
            actual = layer(x)
        self.assertEqual(len(base_calls), 1, 'Unexpected post-base fallback must reuse its NF4 output')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(self.impl.attention_lora_decode_info(model)['fused_projection_calls'], 0)

    @unittest.skipUnless(os.environ.get('QWEN_TEST_GPU') == '1', 'Exclusive CUDA window required')
    def test_real_nf4_fusion_matches_reference_and_rounding(self):
        model = self.model('cuda')
        layer = model.self_attn.q_proj
        x = torch.randn(1, 1, 64, device='cuda', dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = layer(x)
            self.impl.configure_attention_lora_decode(model, True)
            actual = layer(x)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.004)
            self.assertEqual(self.impl.attention_lora_decode_info(model)['fused_projection_calls'], 1)
            layer.lora_B['default'].weight.add_(.1)
            actual = layer(x)
            self.impl.configure_attention_lora_decode(model, False)
            torch.testing.assert_close(actual, layer(x), rtol=.02, atol=.004)
            x = torch.ones(1, 4, device='cuda', dtype=torch.bfloat16)
            base = torch.full((1, 7), -1., device='cuda', dtype=torch.bfloat16)
            a = torch.full((2, 4), .25, device='cuda')
            b = torch.full((7, 2), .502, device='cuda')
            result = self.impl.attention_lora_add(x, base, a, b, 1.)
            self.assertTrue(torch.equal(result, torch.full_like(base, .0078125)))

    @unittest.skipUnless(os.environ.get('QWEN_TEST_GPU') == '1', 'Exclusive CUDA window required')
    def test_cuda_tail_shapes_dtypes_and_guard_fallbacks(self):
        from torch.nn import functional as F
        torch.manual_seed(37)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(1, 1, 79, device='cuda', dtype=dtype)
            base = torch.randn(1, 1, 113, device='cuda', dtype=dtype)
            a = torch.randn(13, 79, device='cuda') * .1
            b = torch.randn(113, 13, device='cuda') * .1
            original = base.clone()
            with torch.inference_mode():
                expected = base + (F.linear(F.linear(x.float(), a), b) * -.75).to(dtype)
                actual = self.impl.attention_lora_add(x, base, a, b, -.75)
                torch.testing.assert_close(actual, expected, rtol=.008, atol=.004)
                self.assertTrue(torch.equal(base, original))
                self.assertIsNone(self.impl.attention_lora_add(x.expand(1, 2, -1), base, a, b, 1.))
                self.assertIsNone(self.impl.attention_lora_add(x, base, a.T.contiguous().T, b, 1.))
                self.assertIsNone(self.impl.attention_lora_add(x, base, a.bfloat16(), b, 1.))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    self.assertIsNone(self.impl.attention_lora_add(x, base, a, b, 1.))
            with torch.enable_grad():
                self.assertIsNone(self.impl.attention_lora_add(x, base, a, b, 1.))


if __name__ == '__main__':
    unittest.main()
