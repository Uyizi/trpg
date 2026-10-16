"""Selected expert math, rounding boundaries and safe inference fallbacks."""
import importlib
import importlib.util
import unittest

import torch
from torch import nn
from torch.nn import functional as F


class RoutedLoraTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('models.routed_lora'),
                             'GPU-selected LoRA decode support is missing')
        self.impl = importlib.import_module('models.routed_lora')

    def tensors(self, device='cpu', dtype=torch.bfloat16, hidden=79, out=113, rank=16):
        generator = torch.Generator(device=device).manual_seed(11)
        def rand(shape, scale, kind):
            return torch.randn(shape, generator=generator, device=device, dtype=kind) * scale
        return (rand((1, hidden), 1, dtype), rand((1, out), .2, dtype),
                torch.tensor([0], device=device),
                rand((rank, hidden), .1, torch.float32), rand((out, rank), .1, torch.float32),
                rand((rank, hidden), .1, torch.float32), rand((out, rank), .1, torch.float32))

    def test_cpu_input_uses_existing_reference_path(self):
        with torch.inference_mode():
            self.assertIsNone(self.impl.routed_lora_add(*self.tensors(), 2., -.5))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_both_routes_and_distinct_scales_match_selected_expert(self):
        for dtype in (torch.bfloat16, torch.float16):
            tensors = self.tensors('cuda', dtype)
            x, base, selected, a0, b0, a1, b1 = tensors
            untouched = base.clone()
            for expert, a, b, scale in ((0, a0, b0, 2.), (1, a1, b1, -.5)):
                with self.subTest(dtype=dtype, expert=expert), torch.inference_mode():
                    selected.fill_(expert)
                    expected = base + (F.linear(F.linear(x.float(), a), b) * scale).to(base.dtype)
                    actual = self.impl.routed_lora_add(*tensors, 2., -.5)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual, expected, rtol=.008, atol=.004)
                    self.assertTrue(torch.equal(base, untouched), 'Fusion must not modify base output')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_delta_rounds_to_base_dtype_before_addition(self):
        # 1.004 rounds to BF16 1.0078125 before being added to -1.
        # Fusing the addition in FP32 would produce a different observable result.
        x = torch.ones((1, 4), device='cuda', dtype=torch.bfloat16)
        base = torch.full((1, 7), -1., device='cuda', dtype=torch.bfloat16)
        a = torch.full((2, 4), .25, device='cuda')
        b = torch.full((7, 2), .502, device='cuda')
        selected = torch.zeros(1, device='cuda', dtype=torch.long)
        with torch.inference_mode():
            actual = self.impl.routed_lora_add(x, base, selected, a, b, a, b, 1., 1.)
        self.assertTrue(torch.equal(actual, torch.full_like(base, .0078125)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gradients_autocast_and_noncontiguous_weights_fall_back(self):
        tensors = self.tensors('cuda')
        with torch.enable_grad():
            self.assertIsNone(self.impl.routed_lora_add(*tensors, 2., -.5))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.assertIsNone(self.impl.routed_lora_add(*tensors, 2., -.5))
        tensors = list(tensors)
        tensors[3] = tensors[3].T.contiguous().T
        with torch.inference_mode():
            self.assertIsNone(self.impl.routed_lora_add(*tensors, 2., -.5))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_prefill_trainable_and_unsupported_dtype_fall_back(self):
        for kind in ('prefill', 'trainable', 'dtype', 'rank'):
            tensors = list(self.tensors('cuda'))
            if kind == 'prefill': tensors[0] = tensors[0].expand(2, -1)
            if kind == 'trainable': tensors[3].requires_grad_(True)
            if kind == 'dtype': tensors[3] = tensors[3].bfloat16()
            if kind == 'rank': tensors[5] = tensors[5][:8].contiguous()
            with self.subTest(kind=kind), torch.inference_mode():
                self.assertIsNone(self.impl.routed_lora_add(*tensors, 2., -.5))


class RealNF4ExpertTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_real_nf4_peft_experts_preserve_routes_and_live_adapter_weights(self):
        import bitsandbytes as bnb
        from peft import get_peft_model, LoraConfig
        from models.qwen35_moe import Top1Experts
        torch.manual_seed(19)
        class Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = bnb.nn.Linear4bit(64, 128, bias=False, compute_dtype=torch.bfloat16,
                                                 quant_type='nf4', compress_statistics=True)
                self.up_proj = bnb.nn.Linear4bit(64, 128, bias=False, compute_dtype=torch.bfloat16,
                                               quant_type='nf4', compress_statistics=True)
                self.down_proj = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=torch.bfloat16,
                                                 quant_type='nf4', compress_statistics=True)
                self.act_fn = F.silu
            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        expert = Top1Experts(Dense().to('cuda'), share_frozen=True)
        model = get_peft_model(expert, LoraConfig(r=8, lora_alpha=12, lora_dropout=.05,
                                                target_modules=['gate_proj', 'up_proj', 'down_proj']))
        for name, parameter in model.named_parameters():
            if 'lora_' in name:
                nn.init.normal_(parameter, std=.07)
        model.eval().requires_grad_(False)
        for sign in (1, -1):
            x = torch.randn(1, 1, 64, device='cuda', dtype=torch.bfloat16)
            x[..., 0] = sign * 2
            with torch.inference_mode():
                expert.router.weight.zero_()
                expert.router.weight[0, 0], expert.router.weight[1, 0] = 1, -1
                expert.fused_lora = False
                expected = model(x)
                expert.fused_lora = True
                actual = model(x)
            torch.testing.assert_close(actual, expected, rtol=.035, atol=.001)
        self.assertGreater(getattr(expert, 'fused_lora_projections', 0), 0,
                           'Numerical parity alone must not hide an unused fused path')
        before = actual.clone()
        with torch.inference_mode():
            expert.experts[1].down_proj.lora_B['default'].weight.add_(.5)
            actual = model(x)
            expert.fused_lora = False
            expected = model(x)
        self.assertFalse(torch.equal(before, actual), 'Live adapter updates must be read')
        torch.testing.assert_close(actual, expected, rtol=.035, atol=.001)
        # Disabling adapters must keep the fallback correct.
        with model.disable_adapter(), torch.inference_mode():
            expert.fused_lora = False
            expected = model(x)
            expert.fused_lora = True
            actual = model(x)
        self.assertTrue(torch.equal(actual, expected))
        # PEFT's disable_adapter context re-enables adapter gradients on exit.
        model.requires_grad_(False)
        from models.routed_lora import try_routed_lora
        class EvalTransform(nn.Module):
            def forward(self, value):
                return value * 2
        layer0, layer1 = (e.gate_proj for e in expert.experts)
        layer0.lora_dropout['default'] = EvalTransform().eval()
        with torch.inference_mode():
            selected = torch.zeros(1, device='cuda', dtype=torch.long)
            self.assertIsNone(try_routed_lora(x.reshape(1, -1), layer0.base_layer(x.reshape(1, -1)),
                                            selected, layer0, layer1),
                              'A custom eval transform must not be silently omitted')


if __name__ == '__main__':
    unittest.main()
