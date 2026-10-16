"""CPU boundaries for validated token selection; no model or CUDA needed."""
import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


def legacy_sample(logits, temperature, generator):
    """Frozen pre-optimization sampling rule, including RNG consumption."""
    values, indices = torch.topk(logits.float() / temperature, min(20, logits.numel()))
    probs = values.softmax(-1)
    probs.masked_fill_(probs.cumsum(-1) - probs > .8, 0)
    return int(indices[torch.multinomial(probs, 1, generator=generator)])


class TokenSamplingTests(unittest.TestCase):
    def test_profile_script_runs_cpu_boundaries_and_writes_inspectable_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'sampling.json'
            result = subprocess.run([
                sys.executable, str(root / 'scripts/profile_token_sampling.py'),
                '--device', 'cpu', '--vocab-size', '29', '--iterations', '2',
                '--warmup', '0', '--repeats', '1', '--output', str(output),
            ], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report['boundaries']['passed'])
            self.assertEqual(report['device'], 'cpu')
            self.assertEqual(len(report['timings']), 4)
            self.assertTrue(all(row['optimized_checked_us'] > 0 for row in report['timings']))

    def test_greedy_returns_first_maximum_and_reusable_device_token(self):
        from models.token_sampling import sample_token
        scores = torch.tensor([-math.inf, 4., 4., -3.])
        original = scores.clone()
        for temperature in (0., -1.):
            result = sample_token(scores, temperature=temperature)
            self.assertEqual(result.token_id, 1)
            self.assertEqual(result.device_token.dtype, torch.long)
            self.assertEqual(result.device_token.device, scores.device)
            self.assertEqual(result.device_token.view(1, 1).tolist(), [[1]])
        self.assertTrue(torch.equal(scores, original))

    def test_sampling_matches_legacy_seeded_draws_and_rng_state(self):
        from models.token_sampling import sample_token
        # Noncontiguous inputs, ties, mixed masked scores, and widths around k=20.
        for width in (1, 5, 20, 29):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                scores = torch.linspace(-1., 1., 2 * width, dtype=dtype)[::2]
                if width > 1:
                    scores[0] = -math.inf
                    scores[-2:] = 1.
                for temperature in (.2, .7, 2.):
                    with self.subTest(width=width, dtype=dtype, temperature=temperature):
                        reference_rng = torch.Generator().manual_seed(1803)
                        actual_rng = torch.Generator().manual_seed(1803)
                        expected = [legacy_sample(scores, temperature, reference_rng) for _ in range(12)]
                        results = [sample_token(scores, temperature=temperature, generator=actual_rng) for _ in expected]
                        self.assertEqual([r.token_id for r in results], expected)
                        self.assertTrue(torch.equal(actual_rng.get_state(), reference_rng.get_state()))
                        self.assertEqual([int(r.device_token) for r in results], expected)

    def test_rejects_all_masked_nan_and_positive_infinity(self):
        from models.token_sampling import TokenSamplingError, sample_token
        for scores in ([-math.inf] * 3, [1., math.nan, -math.inf],
                       [1., math.inf, -math.inf], [math.nan], [math.inf]):
            for temperature in (0., .7):
                with self.subTest(scores=scores, temperature=temperature):
                    with self.assertRaises(TokenSamplingError):
                        sample_token(torch.tensor(scores), temperature=temperature)

    def test_rejects_malformed_vectors_and_nonfinite_temperature(self):
        from models.token_sampling import TokenSamplingError, sample_token
        for scores in (torch.tensor([]), torch.zeros(2, 3), torch.tensor(1.),
                       torch.tensor([1, 2]), torch.tensor([1j])):
            with self.subTest(shape=scores.shape, dtype=scores.dtype):
                with self.assertRaises(TokenSamplingError):
                    sample_token(scores, temperature=0.)
        for temperature in (math.nan, math.inf, -math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaises(TokenSamplingError):
                    sample_token(torch.zeros(3), temperature=temperature)

    def test_scaled_overflow_fails_closed_without_multinomial_runtime_error(self):
        from models.token_sampling import TokenSamplingError, sample_token
        with self.assertRaises(TokenSamplingError):
            sample_token(torch.tensor([torch.finfo(torch.float32).max, 1.]), temperature=.01)

    def test_valid_tiny_temperature_and_one_finite_masked_token(self):
        from models.token_sampling import sample_token
        for temperature in (0., .7, 1e-30):
            result = sample_token(torch.tensor([-math.inf, 0., -math.inf]), temperature=temperature)
            self.assertEqual(result.token_id, 1)


if __name__ == '__main__':
    unittest.main()
