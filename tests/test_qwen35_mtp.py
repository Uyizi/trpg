"""CPU native-MTP math/shift/cache contracts; weights here are synthetic fixtures."""
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest
import subprocess
import json
import sys

import torch
import torch.nn.functional as F
from transformers import Qwen3_5TextConfig
from safetensors.torch import save_file


def fixture():
    torch.manual_seed(1729)
    config = Qwen3_5TextConfig(
        vocab_size=31, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, layer_types=['linear_attention', 'full_attention'],
        mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
        tie_word_embeddings=True, attn_output_gate=True, attention_bias=False,
        hidden_act='silu', rms_norm_eps=1e-6, max_position_embeddings=128,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': .5, 'mrope_section': [1, 1, 0]},
    )
    shapes = {
        'fc': (16, 32), 'pre_fc_norm_embedding': (16,),
        'pre_fc_norm_hidden': (16,), 'norm': (16,),
        'layers.0.input_layernorm': (16,), 'layers.0.post_attention_layernorm': (16,),
        'layers.0.self_attn.q_proj': (32, 16), 'layers.0.self_attn.k_proj': (8, 16),
        'layers.0.self_attn.v_proj': (8, 16), 'layers.0.self_attn.o_proj': (16, 16),
        'layers.0.self_attn.q_norm': (8,), 'layers.0.self_attn.k_norm': (8,),
        'layers.0.mlp.gate_proj': (24, 16), 'layers.0.mlp.up_proj': (24, 16),
        'layers.0.mlp.down_proj': (16, 24),
    }
    weights = {'mtp.' + name + '.weight': (torch.randn(shape) * .13).bfloat16()
               for name, shape in shapes.items()}
    embedding = torch.nn.Embedding(31, 16)
    head = torch.nn.Linear(16, 31, bias=False)
    head.weight = embedding.weight
    return config, weights, embedding, head


def reference(config, weights, embedding, head, ids, hidden, positions):
    """Independent equations, no upstream layer or production helper calls."""
    w = {key.removeprefix('mtp.').removesuffix('.weight'): value.float()
         for key, value in weights.items()}
    def norm(x, key):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + w[key])
    def linear(x, key):
        return F.linear(x, w[key])
    x = linear(torch.cat([norm(embedding(ids), 'pre_fc_norm_embedding'),
                          norm(hidden, 'pre_fc_norm_hidden')], -1), 'fc')
    z = norm(x, 'layers.0.input_layernorm')
    # Official q_proj interleaves [Q_i, gate_i] per head, not [all Q, all gates].
    q, gate = linear(z, 'layers.0.self_attn.q_proj').reshape(1, -1, 2, 16).chunk(2, -1)
    q = norm(q, 'layers.0.self_attn.q_norm').transpose(1, 2)
    k = norm(linear(z, 'layers.0.self_attn.k_proj').reshape(1, -1, 1, 8),
             'layers.0.self_attn.k_norm').transpose(1, 2)
    v = linear(z, 'layers.0.self_attn.v_proj').reshape(1, -1, 1, 8).transpose(1, 2)
    angle = positions.float()[..., None] * torch.tensor([1., .01])
    angle = torch.cat([angle, angle], -1)[:, None]
    def rotate(t):
        a, b = t[..., :4], t[..., 4:]
        rotated = torch.cat([-a[..., 2:], a[..., :2]], -1)
        return torch.cat([a * angle.cos() + rotated * angle.sin(), b], -1)
    q, k = rotate(q), rotate(k).repeat_interleave(2, 1)
    scores = q @ k.transpose(-1, -2) / (8 ** .5)
    scores.masked_fill_(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).triu(1), float('-inf'))
    a = (scores.softmax(-1) @ v.repeat_interleave(2, 1)).transpose(1, 2).reshape(1, -1, 16)
    x = x + linear(a * gate.reshape(1, -1, 16).sigmoid(), 'layers.0.self_attn.o_proj')
    z = norm(x, 'layers.0.post_attention_layernorm')
    x = x + linear(F.silu(linear(z, 'layers.0.mlp.gate_proj')) *
                   linear(z, 'layers.0.mlp.up_proj'), 'layers.0.mlp.down_proj')
    x = norm(x, 'norm')
    return x, head(x)


class Qwen35MTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.qwen35_mtp'),
                             'Native MTP implementation must exist')
        from models import qwen35_mtp
        return qwen35_mtp

    def build(self):
        config, weights, embedding, head = fixture()
        mtp = self.module().Qwen35MTPHead(config, embedding, head, weights,
                                         device='cpu', dtype=torch.float32)
        return mtp, config, weights, embedding, head

    @torch.inference_mode()
    def test_forward_matches_independent_official_equations(self):
        # Catches wrong concat order, non-zero-centered norm, q/gate layout,
        # absent gate/GQA/qk-norm/partial-RoPE, wrong MLP and final norm.
        mtp, config, weights, embedding, head = self.build()
        ids = torch.tensor([[3, 8, 7, 11]])
        hidden = torch.randn(1, 4, 16)
        positions = torch.tensor([[5, 6, 7, 8]])
        expected_hidden, expected_logits = reference(config, weights, embedding, head, ids, hidden, positions)
        result = mtp(ids, hidden, positions)
        torch.testing.assert_close(result.hidden_states, expected_hidden, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(result.logits, expected_logits, atol=5e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_prefill_shifts_tokens_and_preserves_slot_positions(self):
        mtp, *_ = self.build()
        ids = torch.tensor([[3, 8, 7]])
        hidden = torch.randn(1, 3, 16)
        positions = torch.tensor([[9, 10, 11]])
        expected = mtp(torch.tensor([[8, 7, 5]]), hidden, positions)
        actual = mtp.prefill(ids, hidden, torch.tensor([5]), positions, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits)
        self.assertEqual(actual.past_key_values.next_positions.tolist(), [12])
        wrong = mtp(ids, hidden, positions)
        self.assertGreater((actual.logits - wrong.logits).abs().max().item(), .01)

    @torch.inference_mode()
    def test_cached_chunks_equal_full_forward_and_reject_wrong_positions(self):
        mtp, *_ = self.build()
        ids = torch.tensor([[3, 8, 7, 11]])
        hidden = torch.randn(1, 4, 16)
        positions = torch.tensor([[9, 10, 11, 12]])
        full = mtp(ids, hidden, positions)
        first = mtp(ids[:, :2], hidden[:, :2], positions[:, :2], use_cache=True)
        cache = first.past_key_values
        with self.assertRaisesRegex(ValueError, 'position'):
            mtp(ids[:, 2:3], hidden[:, 2:3], torch.tensor([[12]]), past_key_values=cache)
        self.assertEqual(cache.kv.get_seq_length(), 2)
        second = mtp(ids[:, 2:], hidden[:, 2:], positions[:, 2:], past_key_values=cache)
        torch.testing.assert_close(torch.cat([first.logits, second.logits], 1), full.logits, atol=3e-6, rtol=2e-5)
        other, *_ = self.build()
        with self.assertRaisesRegex(ValueError, 'cache'):
            other(ids[:, :1], hidden[:, :1], torch.tensor([[13]]), past_key_values=cache)

    @torch.inference_mode()
    def test_chunk_prefill_uses_next_chunk_tail_token_without_double_shift(self):
        mtp, *_ = self.build()
        ids = torch.tensor([[3, 8, 7, 11]])
        hidden = torch.randn(1, 4, 16)
        positions = torch.tensor([[0, 1, 2, 3]])
        full = mtp.prefill(ids, hidden, torch.tensor([5]), positions)
        first = mtp.prefill(ids[:, :2], hidden[:, :2], ids[:, 2], positions[:, :2], use_cache=True)
        second = mtp.prefill(ids[:, 2:], hidden[:, 2:], torch.tensor([5]), positions[:, 2:],
                             past_key_values=first.past_key_values)
        torch.testing.assert_close(torch.cat([first.logits, second.logits], 1), full.logits,
                                   atol=3e-6, rtol=2e-5)

    def test_shares_target_modules_without_registering_or_mutating_them(self):
        config, weights, embedding, head = fixture()
        before = embedding.weight.detach().clone()
        mtp = self.module().Qwen35MTPHead(config, embedding, head, weights, dtype=torch.float32)
        self.assertIs(mtp.embed_tokens, embedding)
        self.assertIs(mtp.lm_head, head)
        self.assertEqual(len(list(mtp.parameters())), 15)
        mtp.to(dtype=torch.float64)
        self.assertEqual(embedding.weight.dtype, torch.float32)
        self.assertTrue(embedding.weight.requires_grad)
        torch.testing.assert_close(embedding.weight, before, rtol=0, atol=0)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.layer_types, ['linear_attention', 'full_attention'])

    def test_rejects_missing_extra_bad_shape_non_bf16_and_nonfinite_weights(self):
        config, original, embedding, head = fixture()
        for mode in ('missing', 'extra', 'shape', 'dtype', 'nan'):
            with self.subTest(mode=mode):
                weights = copy.deepcopy(original)
                if mode == 'missing': weights.pop('mtp.norm.weight')
                if mode == 'extra': weights['mtp.fake.weight'] = torch.zeros(1).bfloat16()
                if mode == 'shape': weights['mtp.norm.weight'] = torch.zeros(15).bfloat16()
                if mode == 'dtype': weights['mtp.norm.weight'] = weights['mtp.norm.weight'].float()
                if mode == 'nan': weights['mtp.norm.weight'][0] = float('nan')
                with self.assertRaises(ValueError):
                    self.module().Qwen35MTPHead(config, embedding, head, weights)

    def test_safetensors_requires_complete_real_payload_and_tied_target(self):
        config, weights, embedding, head = fixture()
        cls = self.module().Qwen35MTPHead
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.safetensors'
            save_file(weights, path)
            mtp = cls.from_safetensors(path, config, embedding, head, dtype=torch.float32)
            self.assertEqual(len(mtp.state_dict()), 15)
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaises((ValueError, OSError)):
                cls.from_safetensors(path, config, embedding, head)
        untied = torch.nn.Linear(16, 31, bias=False)
        with self.assertRaisesRegex(ValueError, 'tied'):
            cls(config, embedding, untied, weights)

    @torch.inference_mode()
    def test_final_hidden_capture_uses_normalized_target_output_and_removes_hook(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        config, *_ = fixture()
        config.layer_types = ['full_attention', 'full_attention']
        config._attn_implementation = 'eager'
        target = Qwen3_5TextModel(config).eval()
        target.norm.weight.fill_(.75)
        raw = []
        raw_hook = target.norm.register_forward_pre_hook(lambda module, args: raw.append(args[0].clone()))
        hooks_before = len(target.norm._forward_hooks)
        self.assertTrue(hasattr(self.module(), 'FinalHiddenCapture'), 'Normalized target hidden capture must exist')
        with self.module().FinalHiddenCapture(target.norm) as capture:
            result = target(torch.tensor([[3, 8, 7]]), use_cache=False)
            torch.testing.assert_close(capture.hidden_states, result.last_hidden_state, rtol=0, atol=0)
            self.assertGreater((capture.hidden_states - raw[-1]).abs().max().item(), .1)
        raw_hook.remove()
        self.assertEqual(len(target.norm._forward_hooks), hooks_before)

    def test_missing_weights_profile_fails_before_model_or_gpu_load(self):
        # A missing checkpoint must produce a concrete blocked report, not a
        # random head or an attempt to reserve the shared GPU/model.
        self.assertTrue(Path('scripts/profile_qwen35_mtp.py').is_file(), 'MTP profiler must exist')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.json'
            result = subprocess.run([sys.executable, 'scripts/profile_qwen35_mtp.py',
                                     '--mtp', str(Path(folder) / 'missing.safetensors'),
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output.read_text())
            self.assertFalse(report['target_loaded'])
            self.assertFalse(report['cuda_queried'])
            self.assertFalse(report['speculative_decoding_usable'])

    @torch.inference_mode()
    def test_bf16_loading_preserves_rotary_phase_precision(self):
        # Casting inverse frequencies to BF16 before multiplying a long
        # position produces a different phase, even though matmuls use BF16.
        config, weights, embedding, head = fixture()
        mtp = self.module().Qwen35MTPHead(config, embedding, head, weights)
        x = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
        positions = torch.tensor([[[10000]]]).expand(3, 1, 1)
        cos, sin = mtp.rotary_emb(x, positions)
        expected_phase = torch.tensor([10000., 100., 10000., 100.])
        torch.testing.assert_close(cos[0, 0], expected_phase.cos().bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(sin[0, 0], expected_phase.sin().bfloat16(), rtol=0, atol=0)
        mtp.to(dtype=torch.bfloat16)
        cos, sin = mtp.rotary_emb(x, positions)
        torch.testing.assert_close(cos[0, 0], expected_phase.cos().bfloat16(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
