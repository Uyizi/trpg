#!/usr/bin/env python3
"""Serialized GPU numerical checks and warmed RMSNorm microprofile (no model load).

Run only after the service owner grants a GPU lease. Nothing executes on import.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def benchmark(norm, value, iterations):
    import torch
    for _ in range(10):
        norm(value)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall = time.perf_counter()
    for _ in range(iterations):
        norm(value)
    end.record()
    end.synchronize()
    return dict(cuda_us_per_call=start.elapsed_time(end) * 1000 / iterations,
                wall_us_per_call=(time.perf_counter() - wall) * 1e6 / iterations)


def profile(norm, value):
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile
    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       profile_memory=True) as recorded:
        norm(value)
        torch.cuda.synchronize()
    kernels = Counter(event.name for event in recorded.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA)
    allocations = sum(event.name in ('aten::empty', 'aten::empty_strided')
                      and event.device_memory_usage > 0 for event in recorded.events())
    return dict(cuda_kernel_count=sum(kernels.values()), kernels=dict(kernels),
                tensor_allocation_events=allocations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    from models.text_norm_fusion import configure_text_norm_fusion, text_norm_fusion_info
    # Load the opt-in class directly: unittest's skip marker is enabled here only
    # because executing this script is explicit authorization for GPU checks.
    import os
    os.environ['QWEN35_TEST_TEXT_NORM_CUDA'] = '1'
    from tests.test_text_norm_fusion import TextNormFusionCudaTests
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TextNormFusionCudaTests))
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  numerical_tests=dict(passed=result.wasSuccessful(), count=result.testsRun,
                                       output=stream.getvalue()), cases=[])
    if result.wasSuccessful():
        with torch.inference_mode():
            for name, shape, gap in (
                ('decode_hidden', (1, 1, 2560), 1),
                ('decode_query_gate_view', (1, 1, 16, 256), 2),
                ('prefill_hidden', (1, 128, 2560), 1),
            ):
                norm = Qwen3_5RMSNorm(shape[-1]).to(device='cuda', dtype=torch.bfloat16)
                norm.eval().requires_grad_(False)
                norm.weight.normal_(std=.2)
                value = torch.randn((*shape[:-1], shape[-1] * gap),
                                    device='cuda', dtype=torch.bfloat16)[..., :shape[-1]]
                reference = norm(value)
                baseline = dict(benchmark(norm, value, args.iterations), **profile(norm, value))
                configured = configure_text_norm_fusion(norm)
                actual = norm(value)
                candidate = dict(benchmark(norm, value, args.iterations), **profile(norm, value))
                report['cases'].append(dict(name=name, shape=shape, strides=value.stride(),
                    configured=configured, dispatch=text_norm_fusion_info(norm),
                    max_abs_error=float((actual.float() - reference.float()).abs().max()),
                    baseline=baseline, fused=candidate,
                    cuda_speedup=baseline['cuda_us_per_call'] / candidate['cuda_us_per_call'],
                    wall_speedup=baseline['wall_us_per_call'] / candidate['wall_us_per_call']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
