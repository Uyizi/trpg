"""CPU checks never probe CUDA; real NF4 checks require an explicit GPU slot."""
import importlib
import importlib.util
import os
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.qwen35_moe import Top1Experts


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CudaGraphCpuTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('models.cuda_graph_decode'),
                             'Guarded graph decode is missing')
        self.impl = importlib.import_module('models.cuda_graph_decode')
        torch.manual_seed(81)
        self.expert = Top1Experts(Dense(), share_frozen=True).eval().requires_grad_(False)
        self.model = nn.Sequential(self.expert)

    def test_cpu_output_and_live_weight_changes_use_reference(self):
        x = torch.randn(1, 1, 8)
        with torch.inference_mode():
            expected = self.model(x)
            self.impl.configure_cuda_graph_decode(self.model)
            torch.testing.assert_close(self.model(x), expected, rtol=0, atol=0)
            self.expert.experts[0].down_proj.weight.add_(1)
            changed = self.model(x)
            self.impl.configure_cuda_graph_decode(self.model, enabled=False)
            torch.testing.assert_close(self.model(x), changed, rtol=0, atol=0)
        self.assertFalse(torch.equal(changed, expected))

    def test_enable_is_idempotent_and_disable_restores_forward(self):
        original = self.expert.forward
        self.impl.configure_cuda_graph_decode(self.model)
        wrapped = self.expert.forward
        self.impl.configure_cuda_graph_decode(self.model)
        self.assertIs(self.expert.forward, wrapped)
        self.impl.configure_cuda_graph_decode(self.model, enabled=False)
        self.assertEqual(self.expert.forward, original)
        info = self.impl.cuda_graph_decode_info(self.model)
        self.assertEqual(info['configured_layers'], 0)
        self.assertEqual(info['captured_layers'], 0)

    def test_hooks_are_observed_once_and_keyword_call_survives(self):
        self.impl.configure_cuda_graph_decode(self.model)
        observed = []
        handle = self.expert.experts[0].gate_proj.register_forward_hook(
            lambda module, args, output: observed.append(output.clone()))
        with torch.inference_mode():
            self.expert(hidden=torch.randn(1, 1, 8))
        handle.remove()
        self.assertEqual(len(observed), 1)

    def test_training_and_gradients_are_preserved(self):
        self.impl.configure_cuda_graph_decode(self.model)
        self.expert.train().requires_grad_(True)
        x = torch.randn(2, 3, 8, requires_grad=True)
        self.model(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.expert.balance_loss)
        self.assertIsNotNone(self.expert.router.weight.grad)

    def test_reporting_unconfigured_model_and_limits(self):
        before = self.impl.cuda_graph_decode_info(self.model)
        self.assertEqual(before['configured_layers'], 0)
        self.impl.configure_cuda_graph_decode(self.model, max_graphs=1, max_memory_bytes=1024)
        info = self.impl.cuda_graph_decode_info(self.model)
        self.assertEqual(info['configured_layers'], 1)
        self.assertEqual(info['max_graphs'], 1)
        self.assertEqual(info['max_memory_bytes'], 1024)
        with torch.inference_mode():
            self.model(torch.randn(1, 1, 8))
        self.assertEqual(self.impl.cuda_graph_decode_info(self.model)['replays'], 0)

    def test_invalid_limits_do_not_install_partial_configuration(self):
        for options in ({'max_graphs': 0}, {'max_memory_bytes': -1}):
            with self.assertRaises(ValueError):
                self.impl.configure_cuda_graph_decode(self.model, **options)
        self.assertEqual(self.impl.cuda_graph_decode_info(self.model)['configured_layers'], 0)


@unittest.skipUnless(os.environ.get('RUN_CUDA_GRAPH_TESTS') == '1', 'Explicit exclusive GPU slot required')
class CudaGraphNF4Tests(unittest.TestCase):
    def setUp(self):
        import bitsandbytes as bnb
        from peft import LoraConfig, get_peft_model
        from models.cuda_graph_decode import configure_cuda_graph_decode
        torch.manual_seed(19)

        class QuantDense(nn.Module):
            def __init__(self):
                super().__init__()
                for name, left, right in (('gate_proj', 64, 128), ('up_proj', 64, 128), ('down_proj', 128, 64)):
                    setattr(self, name, bnb.nn.Linear4bit(left, right, bias=False, compute_dtype=torch.bfloat16,
                                                        quant_type='nf4', compress_statistics=True))
                self.act_fn = F.silu

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        self.expert = Top1Experts(QuantDense().to('cuda'), share_frozen=True)
        self.expert.is_loaded_in_4bit = True
        self.model = get_peft_model(self.expert, LoraConfig(r=8, lora_alpha=12, lora_dropout=.05,
                                                          target_modules=['gate_proj', 'up_proj', 'down_proj'],
                                                          modules_to_save=['router']))
        for name, parameter in self.model.named_parameters():
            if 'lora_' in name:
                nn.init.normal_(parameter, std=.07)
        self.model.eval().requires_grad_(False)
        with torch.no_grad():
            self.expert.router.weight.zero_()
            self.expert.router.weight[0, 0], self.expert.router.weight[1, 0] = 1, -1
        self.expert.fused_lora = True
        self.reference = self.expert.forward
        configure_cuda_graph_decode(self.model)

    def tearDown(self):
        from models.cuda_graph_decode import configure_cuda_graph_decode
        configure_cuda_graph_decode(self.model, enabled=False)
        torch.cuda.synchronize()

    def value(self, sign=1, dtype=torch.bfloat16):
        x = torch.randn(1, 1, 64, device='cuda', dtype=dtype)
        x[..., 0] = sign * 2
        return x

    def info(self):
        from models.cuda_graph_decode import cuda_graph_decode_info
        return cuda_graph_decode_info(self.model)

    def capture(self):
        with torch.inference_mode():
            actual = self.model(self.value())
        self.assertEqual(self.info()['captured_layers'], 1, self.info())
        return actual

    def test_both_gpu_routes_change_inputs_without_output_aliasing(self):
        with torch.inference_mode():
            first = self.value(1)
            expected = self.reference(first)
            count = self.expert.fused_lora_projections
            actual = self.model(first)
            self.assertEqual(self.expert.fused_lora_projections, count + 3)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            preserved = actual.clone()
            for sign in (-1, 1, -1):
                x = self.value(sign)
                expected = self.reference(x)
                result = self.model(x)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertNotEqual(result.data_ptr(), actual.data_ptr())
            torch.testing.assert_close(actual, preserved, rtol=0, atol=0)
        self.assertEqual(self.info()['replays'], 4, self.info())
        self.assertEqual(self.info()['captures'], 1, self.info())

    def test_live_adapter_inplace_update_and_parameter_replacement_invalidate(self):
        self.capture()
        with torch.no_grad():
            self.expert.experts[1].down_proj.lora_B['default'].weight.add_(.5)
        x = self.value(-1)
        with torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())
        self.assertEqual(self.info()['fallbacks'].get('changed-state'), 1, self.info())
        from models.cuda_graph_decode import configure_cuda_graph_decode
        configure_cuda_graph_decode(self.model, False)
        configure_cuda_graph_decode(self.model)
        self.capture()
        layer = self.expert.experts[0].gate_proj.lora_A['default']
        layer.weight = nn.Parameter(layer.weight.detach().clone() * .5, requires_grad=False)
        x = self.value(1)
        with torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())

    def test_changed_adapter_scaling_invalidates(self):
        self.capture()
        self.expert.experts[0].gate_proj.scaling['default'] *= 2
        x = self.value(1)
        with torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())

    def test_disabled_adapter_invalidates_and_preserves_output(self):
        self.capture()
        x = self.value()
        with self.model.disable_adapter(), torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.model.requires_grad_(False)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())

    def test_descendant_hook_is_observed_once_after_capture(self):
        self.capture()
        observed = []
        handle = self.expert.experts[0].gate_proj.base_layer.register_forward_hook(
            lambda module, args, value: observed.append(value.clone()))
        with torch.inference_mode():
            self.model(self.value())
        handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())

    def test_hook_before_capture_does_not_run_warmups(self):
        observed = []
        handle = self.expert.experts[0].gate_proj.base_layer.register_forward_hook(
            lambda module, args, value: observed.append(value.clone()))
        with torch.inference_mode():
            self.model(self.value())
        handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertEqual(self.info()['captures'], 0, self.info())

    def test_failure_and_small_memory_budget_fall_back(self):
        from unittest.mock import patch
        from models.cuda_graph_decode import configure_cuda_graph_decode
        x = self.value()
        with torch.inference_mode():
            expected = self.reference(x)
            with patch('torch.cuda.CUDAGraph', side_effect=RuntimeError('injected capture failure')):
                torch.testing.assert_close(self.model(x), expected, rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())
        self.assertIn('injected capture failure', str(self.info()['errors']))
        configure_cuda_graph_decode(self.model, False)
        configure_cuda_graph_decode(self.model, max_memory_bytes=1)
        with torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())
        self.assertEqual(self.info()['memory_bytes'], 0, self.info())

    def test_training_prefill_grad_autocast_and_other_stream_use_reference(self):
        self.capture()
        x = self.value()
        with torch.enable_grad():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        with torch.inference_mode():
            batch = torch.cat([x, x], dim=1)
            torch.testing.assert_close(self.model(batch), self.reference(batch), rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
            stream.synchronize()
        self.assertEqual(self.info()['replays'], 1, self.info())

    def test_router_adapter_toggle_alone_invalidates(self):
        self.capture()
        self.expert.router.enable_adapters(False)
        self.model.requires_grad_(False)
        x = self.value(-1)
        with torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())

    def test_global_hook_and_quantization_state_change_invalidate(self):
        self.capture()
        observed = []
        from torch.nn.modules.module import register_module_forward_hook
        target = self.expert.experts[0].gate_proj.base_layer
        def record(module, inputs, output):
            if module is target:
                observed.append(output.clone())
        handle = register_module_forward_hook(record)
        with torch.inference_mode():
            self.model(self.value())
        handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())
        from models.cuda_graph_decode import configure_cuda_graph_decode
        configure_cuda_graph_decode(self.model, False)
        configure_cuda_graph_decode(self.model)
        self.capture()
        with torch.no_grad():
            target.weight.quant_state.absmax.add_(1)
        x = self.value()
        with torch.inference_mode():
            torch.testing.assert_close(self.model(x), self.reference(x), rtol=0, atol=0)
        self.assertEqual(self.info()['captured_layers'], 0, self.info())

    def test_two_layers_keep_retained_outputs_and_obey_graph_count_budget(self):
        from models.cuda_graph_decode import configure_cuda_graph_decode, cuda_graph_decode_info
        import copy
        # A second independently owned MLP; different weights and buffers.
        configure_cuda_graph_decode(self.model, False)
        second = copy.deepcopy(self.expert)
        container = nn.ModuleList([self.expert, second])
        configure_cuda_graph_decode(container, max_graphs=1)
        with torch.inference_mode():
            x = self.value()
            first = self.expert(x)
            retained = first.clone()
            second(first)
            self.expert(self.value(-1))
            torch.testing.assert_close(first, retained, rtol=0, atol=0)
        info = cuda_graph_decode_info(container)
        self.assertEqual(info['captured_layers'], 1, info)
        self.assertEqual(info['fallbacks'].get('budget'), 1, info)
        configure_cuda_graph_decode(container, False)


if __name__ == '__main__':
    unittest.main()
