"""CPU-only checkpoint boundary tests; no target model or CUDA is imported."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


def probe_module():
    spec = importlib.util.find_spec('models.mtp_probe')
    if spec is None:
        return None
    from models import mtp_probe
    return mtp_probe


CONFIG = {'text_config': {
    'model_type': 'qwen3_5_text', 'hidden_size': 8, 'intermediate_size': 16,
    'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4,
    'num_hidden_layers': 4, 'mtp_num_hidden_layers': 1,
    'mtp_use_dedicated_embeddings': False, 'attn_output_gate': True,
    'attention_bias': False, 'tie_word_embeddings': True, 'hidden_act': 'silu',
    'rms_norm_eps': 1e-6, 'vocab_size': 12,
    'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000000,
                        'partial_rotary_factor': 0.5},
}}


def literal_shapes():
    return {
        'mtp.fc.weight': [8, 16],
        'mtp.pre_fc_norm_embedding.weight': [8],
        'mtp.pre_fc_norm_hidden.weight': [8],
        'mtp.norm.weight': [8],
        'mtp.layers.0.input_layernorm.weight': [8],
        'mtp.layers.0.post_attention_layernorm.weight': [8],
        'mtp.layers.0.self_attn.q_proj.weight': [16, 8],
        'mtp.layers.0.self_attn.k_proj.weight': [4, 8],
        'mtp.layers.0.self_attn.v_proj.weight': [4, 8],
        'mtp.layers.0.self_attn.o_proj.weight': [8, 8],
        'mtp.layers.0.self_attn.q_norm.weight': [4],
        'mtp.layers.0.self_attn.k_norm.weight': [4],
        'mtp.layers.0.mlp.gate_proj.weight': [16, 8],
        'mtp.layers.0.mlp.up_proj.weight': [16, 8],
        'mtp.layers.0.mlp.down_proj.weight': [8, 16],
        'model.language_model.embed_tokens.weight': [12, 8],
    }


def make_header(shapes):
    offset = 0
    header = {}
    for name, shape in shapes.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {'dtype': 'BF16', 'shape': shape,
                        'data_offsets': [offset, offset + size]}
        offset += size
    return header, offset


def write_safetensors(path, header, payload_size):
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    with path.open('wb') as file:
        file.write(struct.pack('<Q', len(raw)))
        file.write(raw)
        file.write(bytes(payload_size))


class MTPProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe = probe_module()
        self.assertIsNotNone(self.probe, 'The offline MTP compatibility probe is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkpoint(self, shapes=None):
        shapes = literal_shapes() if shapes is None else shapes
        header, size = make_header(shapes)
        write_safetensors(self.root / 'model.safetensors', header, size)
        (self.root / 'config.json').write_text(json.dumps(CONFIG))
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {name: 'model.safetensors' for name in shapes}}))
        return header

    def test_complete_head_shapes_are_available_but_integration_is_not_usable(self):
        self.checkpoint()
        report = self.probe.inspect_checkpoint(self.root)
        self.assertEqual(report['native_head_status'], 'available_headers_only')
        self.assertEqual(report['mtp_tensor_count_verified'], 15)
        self.assertEqual(report['errors'], [])
        self.assertFalse(report['speculative_decoding_usable'])

    def test_indexed_tensors_without_payload_are_blocked(self):
        self.checkpoint()
        (self.root / 'model.safetensors').unlink()
        report = self.probe.inspect_checkpoint(self.root)
        self.assertEqual(report['native_head_status'], 'blocked')
        self.assertEqual(report['mtp_tensor_count_indexed'], 15)
        self.assertEqual(report['mtp_tensor_count_verified'], 0)
        self.assertTrue(any('Missing shard' in error for error in report['errors']))

    def test_ungated_q_projection_is_rejected(self):
        shapes = literal_shapes()
        shapes['mtp.layers.0.self_attn.q_proj.weight'] = [8, 8]
        self.checkpoint(shapes)
        report = self.probe.inspect_checkpoint(self.root)
        self.assertEqual(report['native_head_status'], 'blocked')
        self.assertTrue(any('q_proj.weight' in error and 'shape' in error for error in report['errors']))

    def test_missing_head_tensor_is_rejected(self):
        shapes = literal_shapes()
        del shapes['mtp.pre_fc_norm_hidden.weight']
        self.checkpoint(shapes)
        report = self.probe.inspect_checkpoint(self.root)
        self.assertTrue(any('pre_fc_norm_hidden' in error for error in report['errors']))
        self.assertEqual(report['native_head_status'], 'blocked')

    def test_unknown_quantization_scale_layout_is_not_silently_skipped(self):
        shapes = literal_shapes()
        shapes['mtp.fc.weight_scale'] = [1]
        self.checkpoint(shapes)
        report = self.probe.inspect_checkpoint(self.root)
        self.assertTrue(any('Unsupported MTP tensor' in error for error in report['errors']))
        self.assertEqual(report['native_head_status'], 'blocked')

    def test_shared_embedding_is_required(self):
        shapes = literal_shapes()
        del shapes['model.language_model.embed_tokens.weight']
        self.checkpoint(shapes)
        report = self.probe.inspect_checkpoint(self.root)
        self.assertTrue(any('embed_tokens' in error for error in report['errors']))

    def test_truncated_header_is_rejected_before_json_read(self):
        path = self.root / 'truncated.safetensors'
        path.write_bytes(struct.pack('<Q', 1000000) + b'{}')
        with self.assertRaisesRegex(ValueError, 'header'):
            self.probe.read_safetensors_header(path)

    def test_truncated_payload_is_rejected(self):
        path = self.root / 'truncated.safetensors'
        header, size = make_header({'a': [4]})
        write_safetensors(path, header, size - 1)
        with self.assertRaisesRegex(ValueError, 'payload|offset'):
            self.probe.read_safetensors_header(path)

    def test_overlapping_tensor_offsets_are_rejected(self):
        path = self.root / 'overlap.safetensors'
        header, size = make_header({'a': [4], 'b': [4]})
        header['b']['data_offsets'] = [0, 8]
        write_safetensors(path, header, size)
        with self.assertRaisesRegex(ValueError, 'overlap|contiguous'):
            self.probe.read_safetensors_header(path)

    def test_wrong_byte_count_is_rejected(self):
        path = self.root / 'wrong.safetensors'
        header, size = make_header({'a': [4]})
        header['a']['shape'] = [8]
        write_safetensors(path, header, size)
        with self.assertRaisesRegex(ValueError, 'byte'):
            self.probe.read_safetensors_header(path)

    def test_duplicate_json_tensor_names_are_rejected(self):
        path = self.root / 'duplicate.safetensors'
        raw = b'{"a":{},"a":{}}  '
        path.write_bytes(struct.pack('<Q', len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.probe.read_safetensors_header(path)

    def test_index_cannot_escape_checkpoint_directory(self):
        self.checkpoint()
        index_path = self.root / 'model.safetensors.index.json'
        index = json.loads(index_path.read_text())
        index['weight_map']['mtp.fc.weight'] = '../outside.safetensors'
        index_path.write_text(json.dumps(index))
        report = self.probe.inspect_checkpoint(self.root)
        self.assertTrue(any('outside checkpoint' in error for error in report['errors']))

    def test_index_and_header_must_agree(self):
        header = self.checkpoint()
        header['not_in_index'] = {'dtype': 'BF16', 'shape': [1],
                                  'data_offsets': [sum(v['data_offsets'][1] - v['data_offsets'][0] for v in header.values()), 0]}
        size = header['not_in_index']['data_offsets'][0] + 2
        header['not_in_index']['data_offsets'][1] = size
        write_safetensors(self.root / 'model.safetensors', header, size)
        report = self.probe.inspect_checkpoint(self.root)
        self.assertTrue(any('index' in error for error in report['errors']))

    def test_multi_layer_mtp_is_explicitly_unsupported(self):
        config = copy.deepcopy(CONFIG)
        config['text_config']['mtp_num_hidden_layers'] = 2
        self.checkpoint()
        (self.root / 'config.json').write_text(json.dumps(config))
        report = self.probe.inspect_checkpoint(self.root)
        self.assertEqual(report['native_head_status'], 'blocked')
        self.assertTrue(any('one' in error.lower() for error in report['errors']))

    def test_unimplemented_rotary_scaling_is_not_marked_compatible(self):
        self.checkpoint()
        config = copy.deepcopy(CONFIG)
        config['text_config']['rope_parameters']['rope_type'] = 'yarn'
        (self.root / 'config.json').write_text(json.dumps(config))
        report = self.probe.inspect_checkpoint(self.root)
        self.assertEqual(report['native_head_status'], 'blocked')
        self.assertTrue(any('rotary' in error.lower() for error in report['errors']))

    def test_nonfinite_normalization_epsilon_is_rejected(self):
        self.checkpoint()
        config = copy.deepcopy(CONFIG)
        config['text_config']['rms_norm_eps'] = float('nan')
        (self.root / 'config.json').write_text(json.dumps(config))
        report = self.probe.inspect_checkpoint(self.root)
        self.assertEqual(report['native_head_status'], 'blocked')
        self.assertTrue(any('epsilon' in error.lower() for error in report['errors']))


if __name__ == '__main__':
    unittest.main()
