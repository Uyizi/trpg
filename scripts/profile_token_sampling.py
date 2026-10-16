#!/usr/bin/env python3
"""No-model boundary checks and wall timings; CUDA requires an exclusive lease.

Example (after acquiring the repository's GPU lease):
  .venv-qwen35/bin/python scripts/profile_token_sampling.py --device cuda \
      --vocab-size 248320 --profile --output /tmp/token-sampling.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.token_sampling import TokenSamplingError, sample_token


def reference_sample(logits, temperature, *, check_finite=True, generator=None):
    """The original generator, including its grammar finite-score host check."""
    if check_finite and not logits.isfinite().any().item():
        raise TokenSamplingError('no finite next-token score')
    if temperature > 0:
        values, indices = torch.topk(logits.float() / temperature, min(20, logits.numel()))
        probs = values.softmax(-1)
        probs.masked_fill_(probs.cumsum(-1) - probs > .8, 0)
        return int(indices[torch.multinomial(probs, 1, generator=generator)])
    return int(logits.argmax())


def check_boundaries(device):
    comparisons = 0
    # Different shapes expose argmax ties, topk cutoffs, and masked candidates.
    for width in (1, 5, 20, 31):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            scores = torch.linspace(-1., 1., 2 * width, dtype=dtype, device=device)[::2]
            if width > 1:
                scores[0] = -math.inf
                scores[-2:] = 1.
            for temperature in (0., .2, .7, 2.):
                reference_rng = torch.Generator(device=device).manual_seed(1803)
                actual_rng = torch.Generator(device=device).manual_seed(1803)
                for _ in range(12):
                    expected = reference_sample(scores, temperature, generator=reference_rng)
                    result = sample_token(scores, temperature=temperature, generator=actual_rng)
                    if result.token_id != expected or int(result.device_token) != expected:
                        raise AssertionError((width, dtype, temperature, expected, result.token_id))
                    comparisons += 1
                if not torch.equal(reference_rng.get_state(), actual_rng.get_state()):
                    raise AssertionError('RNG consumption differs from the original sampler')
    invalid = 0
    for scores in ([-math.inf] * 3, [1., math.nan], [1., math.inf], [math.nan], [math.inf]):
        for temperature in (0., .7):
            try:
                sample_token(torch.tensor(scores, device=device), temperature=temperature)
            except TokenSamplingError:
                invalid += 1
            else:
                raise AssertionError('invalid logits returned a token')
    greedy = sample_token(torch.tensor([-math.inf, 4., 4., -3.], device=device), temperature=0.)
    if greedy.token_id != 1:
        raise AssertionError('greedy first-maximum tie changed')
    try:
        sample_token(torch.tensor([torch.finfo(torch.float32).max, 1.], device=device), temperature=.01)
    except TokenSamplingError:
        invalid += 1
    else:
        raise AssertionError('float32 scaling overflow returned a token')
    for temperature in (0., .7, 1e-30):
        result = sample_token(torch.tensor([-math.inf, 0., -math.inf], device=device), temperature=temperature)
        if result.token_id != 1:
            raise AssertionError('single finite candidate was lost')
    return {'passed': True, 'exact_seeded_token_comparisons': comparisons,
            'invalid_rows_rejected': invalid, 'greedy_first_tie_preserved': True}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--vocab-size', type=int, default=248320)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--profile', action='store_true', help='CUDA-only event counts; separate from timing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.vocab_size, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error('sizes/iterations/repeats must be positive; warmup must be nonnegative')
    if args.profile and args.device != 'cuda':
        parser.error('--profile is CUDA-only')
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measure(function):
        for _ in range(args.warmup):
            function()
        samples = []
        for _ in range(args.repeats):
            synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                function()
            synchronize()
            samples.append((time.perf_counter() - started) * 1e6 / args.iterations)
        return statistics.median(samples)

    report = {'device': str(device), 'dtype': args.dtype, 'vocab_size': args.vocab_size,
              'torch': torch.__version__, 'iterations': args.iterations, 'repeats': args.repeats,
              'boundaries': check_boundaries(device), 'timings': [],
              'notes': ['Wall microseconds per selection; synchronization included.',
                        'Timing baseline_checked retains original grammar finite check.',
                        'raw_legacy has no finite check and is unsafe for malformed logits.',
                        'torch.multinomial may synchronize internally in both implementations.',
                        'with_input columns include preparing [1,1] next-token input_ids.',
                        'Invalid new-path stochastic rows may consume a discarded RNG draw.']}
    if device.type == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(device)
    for masked in (False, True):
        scores = torch.randn(args.vocab_size, device=device, dtype=dtype)
        if masked:
            # Legal score candidates at regular positions, all others masked.
            allowed = torch.arange(args.vocab_size, device=device).remainder(37) == 0
            scores.masked_fill_(~allowed, -math.inf)
        for temperature in (0., .7):
            def reference():
                return reference_sample(scores, temperature)

            def optimized():
                return sample_token(scores, temperature=temperature)

            def reference_with_input():
                return torch.tensor([[reference()]], device=device)

            def optimized_with_input():
                return optimized().device_token.view(1, 1)

            row = {'masked': masked, 'temperature': temperature,
                   'raw_legacy_us': measure(lambda: reference_sample(scores, temperature, check_finite=False)),
                   'baseline_checked_us': measure(reference),
                   'optimized_checked_us': measure(optimized),
                   'baseline_checked_with_input_us': measure(reference_with_input),
                   'optimized_checked_with_input_us': measure(optimized_with_input)}
            row['checked_speedup'] = row['baseline_checked_us'] / row['optimized_checked_us']
            row['checked_with_input_speedup'] = row['baseline_checked_with_input_us'] / row['optimized_checked_with_input_us']
            if args.profile:
                from torch.profiler import ProfilerActivity, profile
                events = {}
                for name, function in (('baseline_checked', reference), ('optimized_checked', optimized)):
                    synchronize()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                        for _ in range(10):
                            function()
                        synchronize()
                    selected_events = {}
                    for event in trace.key_averages():
                        if any(term in event.key for term in ('item', 'local_scalar', 'Memcpy', 'Synchronize', 'multinomial')):
                            selected_events[event.key] = event.count
                    events[name] = selected_events
                row['profiler_event_counts_for_10_calls'] = events
            report['timings'].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
