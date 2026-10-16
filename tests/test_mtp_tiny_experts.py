"""CPU math and dispatch checks for the opt-in two-token expert path.

The small linear base models NF4's projection interface without CUDA or weights.
Routing, adapter math, Top1Experts, and scope restoration run as production code.
"""
import importlib
import importlib.util
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from models.qwen35_moe import Top1Experts


PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


class CountingBase(nn.Linear):
    def __init__(self, inputs, outputs, dtype):
        super().__init__(inputs, outputs, bias=False, dtype=dtype)
        self.weight.quant_state = object()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


class Adapter(nn.Module):
    def __init__(self, base, expert):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict({'default': nn.Linear(base.in_features, 2, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(2, base.out_features, bias=False)})
        self.lora_dropout = nn.ModuleDict({'default': nn.Dropout(.2)})
        self.scaling = {'default': (.7, -.4)[expert]}
        self.active_adapters = ['default']
        self.disable_adapters = False
        self.merged = False
        self.lora_variant = {}

    def forward(self, x):
        result = self.base_layer(x)
        if not self.disable_adapters:
            for name in self.active_adapters:
                value = self.lora_B[name](self.lora_A[name](self.lora_dropout[name](x.float())))
                result = result + (value * self.scaling[name]).to(result.dtype)
        return result


class Dense(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.gate_proj = CountingBase(3, 4, dtype)
        self.up_proj = CountingBase(3, 4, dtype)
        self.down_proj = CountingBase(4, 3, dtype)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def fixture(dtype=torch.float32, shared=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(402)
        model = Top1Experts(Dense(dtype), share_frozen=shared)
        for expert_id, expert in enumerate(model.experts):
            for name in PROJECTIONS:
                setattr(expert, name, Adapter(getattr(expert, name), expert_id))
        with torch.no_grad():
            model.router.weight.copy_(torch.tensor([[1., 0., -.5], [-1., 0., .5]]))
    return model.eval().requires_grad_(False)


def reset_calls(model):
    for expert in model.experts:
        for name in PROJECTIONS:
            getattr(expert, name).base_layer.calls = 0


def base_calls(model):
    return [[getattr(expert, name).base_layer.calls for name in PROJECTIONS]
            for expert in model.experts]


class MTPTinyExpertTests(unittest.TestCase):
    def scope(self, model, enabled=True):
        self.assertIsNotNone(importlib.util.find_spec('models.mtp_expert_scope'),
                             'Scoped two-token expert verification is missing')
        return importlib.import_module('models.mtp_expert_scope').expert_verification_scope(model, enabled)

    def test_scoped_mixed_and_same_routes_match_reference_with_one_shared_projection(self):
        for dtype in (torch.float32, torch.bfloat16):
            for signs in ((1., -1.), (1., 1.), (-1., -1.)):
                with self.subTest(dtype=dtype, signs=signs), torch.inference_mode():
                    model = fixture(dtype)
                    x = torch.tensor([[[signs[0], .3, .1], [signs[1], -.2, .15]]], dtype=dtype)
                    expected = model(x)
                    reset_calls(model)
                    with self.scope(model), patch('torch.where', side_effect=AssertionError('route sync')):
                        actual = model(x)
                    torch.testing.assert_close(actual, expected, rtol=.02, atol=.001)
                    self.assertEqual(base_calls(model), [[1, 1, 1], [0, 0, 0]])

    def test_default_and_disabled_scope_preserve_ordinary_two_row_dispatch(self):
        model = fixture()
        x = torch.tensor([[[1., .3, .1], [-1., -.2, .15]]])
        with torch.inference_mode():
            expected = model(x)
            self.assertEqual(base_calls(model), [[1, 1, 1], [1, 1, 1]])
            self.assertFalse(model.tinybatch_verification)
            reset_calls(model)
            with self.scope(model, enabled=False):
                actual = model(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(base_calls(model), [[1, 1, 1], [1, 1, 1]])

    def test_scope_restores_each_expert_after_nesting_and_exception(self):
        first, second = fixture(), fixture()
        second.tinybatch_verification = True
        model = nn.ModuleList([first, second])
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            with self.scope(model):
                self.assertTrue(first.tinybatch_verification)
                self.assertTrue(second.tinybatch_verification)
                with self.scope(model, enabled=False):
                    self.assertFalse(first.tinybatch_verification)
                    self.assertFalse(second.tinybatch_verification)
                self.assertTrue(first.tinybatch_verification)
                self.assertTrue(second.tinybatch_verification)
                raise RuntimeError('cancelled')
        self.assertFalse(first.tinybatch_verification)
        self.assertTrue(second.tinybatch_verification)

    def test_training_and_grad_enabled_eval_keep_routed_gradient_path(self):
        for training in (False, True):
            with self.subTest(training=training), torch.enable_grad():
                model = fixture().train(training).requires_grad_(True)
                x = torch.tensor([[[1., .3, .1], [-1., -.2, .15]]], requires_grad=True)
                with self.scope(model):
                    model(x).sum().backward()
                self.assertEqual(base_calls(model), [[1, 1, 1], [1, 1, 1]])
                self.assertIsNotNone(x.grad)
                self.assertGreater(x.grad.abs().sum().item(), 0)
                if training:
                    self.assertIsNotNone(model.balance_loss)

    def test_no_grad_scope_still_requires_eval_mode(self):
        for training in (False, True):
            with self.subTest(training=training), torch.no_grad():
                model = fixture().train(training)
                x = torch.tensor([[[1., .3, .1], [-1., -.2, .15]]])
                with self.scope(model):
                    model(x)
                self.assertEqual(base_calls(model), [[1, 1, 1],
                                                     [1, 1, 1] if training else [0, 0, 0]])

    def test_scope_keeps_shared_quantized_adapter_guards(self):
        for guard in ('unshared', 'unquantized', 'disabled', 'merged', 'variant'):
            with self.subTest(guard=guard), torch.inference_mode():
                model = fixture(shared=guard != 'unshared')
                layer = model.experts[1].up_proj
                if guard == 'unquantized':
                    del layer.base_layer.weight.quant_state
                elif guard == 'disabled':
                    layer.disable_adapters = True
                elif guard == 'merged':
                    layer.merged = True
                elif guard == 'variant':
                    layer.lora_variant = {'default': object()}
                x = torch.tensor([[[1., .3, .1], [-1., -.2, .15]]])
                expected = model(x)
                reset_calls(model)
                with self.scope(model):
                    actual = model(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(base_calls(model), [[1, 1, 1], [1, 1, 1]])

    def test_scope_does_not_enable_larger_prefill(self):
        model = fixture()
        x = torch.tensor([[[1., .3, .1], [-1., -.2, .15], [1., .1, .2]]])
        with torch.inference_mode():
            expected = model(x)
            reset_calls(model)
            with self.scope(model):
                actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(base_calls(model), [[1, 1, 1], [1, 1, 1]])

    def test_one_row_existing_fast_path_is_unchanged_with_grad_enabled(self):
        model = fixture()
        x = torch.tensor([[[1., .3, .1]]])
        with torch.enable_grad(), patch('torch.where', side_effect=AssertionError('route sync')):
            expected = model(x)
            for enabled in (False, True):
                reset_calls(model)
                with self.scope(model, enabled):
                    actual = model(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(base_calls(model), [[1, 1, 1], [0, 0, 0]])

    def test_live_adapter_updates_and_fused_optin_use_two_row_torch_fallback(self):
        model = fixture(torch.bfloat16)
        model.fused_lora = True
        x = torch.tensor([[[1., .3, .1], [-1., -.2, .15]]], dtype=torch.bfloat16)
        with torch.inference_mode():
            before = model(x)
            model.experts[1].down_proj.lora_B['default'].weight.add_(2.)
            expected = model(x)
            reset_calls(model)
            with self.scope(model), patch('torch.where', side_effect=AssertionError('route sync')):
                actual = model(x)
        self.assertFalse(torch.equal(before, actual))
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.001)
        self.assertEqual(base_calls(model), [[1, 1, 1], [0, 0, 0]])
        self.assertEqual(model.fused_lora_projections, 0)


if __name__ == '__main__':
    unittest.main()
