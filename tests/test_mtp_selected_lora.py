"""Selected two-row LoRA guards and opt-in, lock-protected CUDA math checks."""
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from models.routed_lora import routed_lora_add


class CudaMetadataTensor(torch.Tensor):
    """CPU storage, CUDA metadata only, for checking dispatch without a device."""
    @property
    def is_cuda(self):
        return True


def cpu_inputs(rows):
    return (torch.ones(rows, 8, dtype=torch.bfloat16).as_subclass(CudaMetadataTensor),
            torch.zeros(rows, 7, dtype=torch.bfloat16), torch.zeros(rows, dtype=torch.long),
            torch.ones(2, 8), torch.ones(7, 2), torch.ones(2, 8), torch.ones(7, 2))


class SelectedLoraGuardTests(unittest.TestCase):
    def test_one_and_two_rows_reach_backend_without_copying_live_parameters(self):
        for rows in (1, 2):
            args = cpu_inputs(rows)
            def backend(*received):
                for actual, original in zip(received, args):
                    self.assertIs(actual, original, 'Dispatch must pass original live tensors')
                raise RuntimeError('selected backend reached')
            module = SimpleNamespace(selected_lora_add=backend)
            with self.subTest(rows=rows), torch.inference_mode(), patch.dict(
                    sys.modules, {'models.routed_lora_kernels': module}):
                with self.assertRaisesRegex(RuntimeError, 'selected backend reached'):
                    routed_lora_add(*args, 2., -.5)

    def test_invalid_rows_or_selection_count_fall_back_before_backend(self):
        for rows, base_rows, selected_rows in ((0, 0, 0), (3, 3, 3), (2, 1, 2),
                                                (2, 2, 1), (1, 1, 2)):
            args = list(cpu_inputs(rows))
            args[1] = torch.zeros(base_rows, 7, dtype=torch.bfloat16)
            args[2] = torch.zeros(selected_rows, dtype=torch.long)
            def backend(*args):
                self.fail('An unsupported shape reached the GPU backend')
            with self.subTest(rows=rows, base_rows=base_rows, selected_rows=selected_rows), \
                    torch.inference_mode(), patch.dict(sys.modules,
                        {'models.routed_lora_kernels': SimpleNamespace(selected_lora_add=backend)}):
                self.assertIsNone(routed_lora_add(*args, 2., -.5))


@unittest.skipUnless(os.environ.get('QWEN_MTP_LORA_GPU_TEST') == '1',
                     'exclusive GPU lease and explicit opt-in required')
class SelectedLoraCUDATests(unittest.TestCase):
    @torch.inference_mode()
    def test_real_model_projection_sizes_routes_dtypes_and_one_row_regression(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        maxima = {}
        for dtype in (torch.bfloat16, torch.float16):
            maximum = 0.
            for hidden, out in ((2560, 9216), (9216, 2560)):
                generator = torch.Generator(device='cuda').manual_seed(803 + hidden)
                def rand(shape, scale, kind=torch.float32):
                    return torch.randn(shape, generator=generator, device='cuda', dtype=kind) * scale
                a0, b0, a1, b1 = (rand((32, hidden), .03), rand((out, 32), .03),
                                   rand((32, hidden), .03), rand((out, 32), .03))
                for routes in ((0,), (1,), (0, 0), (1, 1), (0, 1), (1, 0)):
                    x = rand((len(routes), hidden), 1., dtype)
                    base = rand((len(routes), out), .2, dtype)
                    selected = torch.tensor(routes, device='cuda')
                    untouched = base.clone()
                    expected = torch.cat([
                        base[row:row+1] + (F.linear(F.linear(x[row:row+1].float(), a), b)
                                               * scale).to(dtype)
                        for row, expert in enumerate(routes)
                        for a, b, scale in [(a0, b0, 2.) if expert == 0 else (a1, b1, -.5)]])
                    with self.subTest(dtype=dtype, hidden=hidden, routes=routes):
                        actual = routed_lora_add(x, base, selected, a0, b0, a1, b1, 2., -.5)
                        self.assertIsNotNone(actual)
                        error = (actual.float() - expected.float()).abs().max().item()
                        maximum = max(maximum, error)
                        torch.testing.assert_close(actual, expected, rtol=.01 if dtype == torch.bfloat16 else .003,
                                                   atol=.008 if dtype == torch.bfloat16 else .001)
                        self.assertTrue(torch.equal(base, untouched))
            maxima[str(dtype)] = maximum
        print('selected_lora_max_abs_error=' + str(maxima), flush=True)

    @torch.inference_mode()
    def test_two_row_delta_rounding_and_unselected_nonfinite_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.ones((2, 4), device='cuda', dtype=dtype)
            base = torch.full((2, 7), -1., device='cuda', dtype=dtype)
            a = torch.full((2, 4), .25, device='cuda')
            b = torch.full((7, 2), .502, device='cuda')
            invalid_a, invalid_b = torch.full_like(a, float('nan')), torch.full_like(b, float('nan'))
            selected = torch.zeros(2, device='cuda', dtype=torch.long)
            expected = base + torch.full_like(base, 1.004)
            actual = routed_lora_add(x, base, selected, a, b, invalid_a, invalid_b, 1., 1.)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
