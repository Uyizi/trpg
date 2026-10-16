"""CPU checks for the real NVFP4 format; no CUDA context is created."""
import importlib
import unittest
import os
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch


def probe():
    return importlib.import_module('models.nvfp4_probe')


class NVFP4ProbeTests(unittest.TestCase):
    def test_profiler_refuses_busy_gpu_before_importing_torch(self):
        # -S removes site-packages: a busy lease must exit before any Torch
        # import, CUDA query or allocation, even when --execute was supplied.
        with tempfile.TemporaryDirectory() as folder:
            lock=Path(folder)/'.cache/qwen-debugger/gpu.lock'
            lock.parent.mkdir(parents=True)
            output=Path(folder)/'result.json'
            with lock.open('a+') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run([sys.executable,'-S',str(Path(__file__).resolve().parents[1]/'scripts/profile_nvfp4_projection.py'),
                                       '--execute','--output',str(output)],
                                      env={**os.environ,'HOME':folder},capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['status'],'blocked')
            self.assertFalse(report['cuda_queried'])
            self.assertFalse(report['native_fp4_executed'])

    def test_e2m1_exact_levels_sign_and_low_nibble_order(self):
        levels = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]])
        packed = probe().pack_e2m1(levels)
        assert packed.tolist() == [[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]]
        assert torch.equal(probe().unpack_e2m1(packed), levels)
    
    
    def test_e2m1_round_to_nearest_even_and_saturation(self):
        x = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 100.]])
        assert probe().unpack_e2m1(probe().pack_e2m1(x)).tolist() == [[0., 1., 1., 2., 2., 4., 4., 6.]]
    
    
    def test_swizzle_matches_documented_byte_coordinates_with_ragged_rows(self):
        scales = torch.arange(130*7).reshape(130, 7).remainder(256).to(torch.uint8)
        out = probe().swizzle_scales(scales)
        assert out.numel() == 256*8
        for row in (0, 1, 31, 32, 33, 127, 128, 129):
            for col in range(7):
                offset = ((row//128)*2 + col//4)*512 + (row%32)*16 + ((row%128)//32)*4 + col%4
                assert out[offset] == scales[row, col]
        assert torch.equal(probe().unswizzle_scales(out, 130, 7), scales)
    
    
    def test_quantization_is_finite_for_zero_matrix(self):
        q = probe().quantize_nvfp4(torch.zeros(3, 64))
        assert q.packed.shape == (3, 32)
        assert q.scales.numel() == 128*4
        assert torch.isfinite(probe().dequantize_nvfp4(q)).all()
        assert torch.count_nonzero(probe().dequantize_nvfp4(q)) == 0
    
    
    def test_quantization_has_actual_e4m3_scales_and_bounded_reconstruction_error(self):
        torch.manual_seed(19)
        x = torch.randn(13, 256)
        q = probe().quantize_nvfp4(x)
        assert q.scales.dtype == torch.float8_e4m3fn
        assert q.global_scale.dtype == torch.float32
        assert q.packed.dtype == torch.uint8
        restored = probe().dequantize_nvfp4(q)
        assert (torch.linalg.vector_norm(restored-x)/torch.linalg.vector_norm(x)).item() < .15
    
    
    def test_shape_and_nonfinite_rejected_before_quantization(self):
        for x in (torch.zeros(3, 15), torch.zeros(2, 2, 16), torch.full((1, 16), float('nan'))):
            with self.assertRaises(ValueError):
                probe().quantize_nvfp4(x)
    
    
    def test_error_summary_tracks_reference_and_nonfinite(self):
        result = probe().error_summary(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 4.]))
        assert result['max_abs_error'] == 1.
        self.assertAlmostEqual(result['relative_l2'], (1/14)**.5, places=6)
        assert not probe().error_summary(torch.ones(2), torch.tensor([1., float('nan')]))['finite']

    @unittest.skipUnless(os.environ.get('QWEN_FP4_GPU_TEST') == '1', 'exclusive GPU lease opt-in required')
    def test_triton_pack_bytes_match_reference_on_known_and_ragged_gpu_vectors(self):
        from models.nvfp4_quant_kernels import quantize_nvfp4_triton
        p = probe()
        torch.manual_seed(20261015)
        levels = torch.tensor([0., .25, .5, .75, 1., 1.25, 1.5, 1.75,
                               2., 2.5, 3., 3.5, 4., 5., 6., -0.], device='cuda')
        inputs = [levels.reshape(1,16), -levels.reshape(1,16), torch.zeros(3,64,device='cuda')]
        inputs += [torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                   for m,k in [(1,2560),(16,2560),(128,2560),(130,112),(512,2560)]]
        for x in inputs:
            with self.subTest(shape=tuple(x.shape), dtype=str(x.dtype)):
                expected, actual = p.quantize_nvfp4(x), quantize_nvfp4_triton(x)
                self.assertTrue(torch.equal(actual.packed, expected.packed))
                self.assertTrue(torch.equal(actual.scales.view(torch.uint8), expected.scales.view(torch.uint8)))
                self.assertTrue(torch.equal(actual.global_scale, expected.global_scale))

if __name__ == "__main__":
    unittest.main()
