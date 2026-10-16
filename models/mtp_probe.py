"""Offline, header-only Qwen3.5 dense MTP compatibility inspection.

Run ``python -m models.mtp_probe``. Uses only the standard library: no torch,
CUDA, model construction, package downloads, tensor reads, or service changes.
Header validity proves storage layout, never tensor values or draft quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import struct


ROOT = Path('/mnt/d/ai_project_data/qwen35_cloud')
MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2,
    'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8,
    'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object)


def read_safetensors_header(path):
    """Validate a complete file's header and byte ranges without reading data."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header length')
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > MAX_HEADER_BYTES or length > size - 8:
            raise ValueError('Invalid, oversized, or truncated safetensors header')
        header = _json(file.read(length))
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be an object')
    payload_size = size - 8 - length
    ranges = []
    for name, tensor in header.items():
        if name == '__metadata__':
            if not isinstance(tensor, dict) or not all(isinstance(v, str) for v in tensor.values()):
                raise ValueError('Invalid safetensors metadata')
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f'Invalid tensor descriptor: {name}')
        shape, dtype, offsets = (tensor.get(key) for key in ('shape', 'dtype', 'data_offsets'))
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'Invalid tensor shape: {name}')
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f'Unsupported tensor dtype: {name}: {dtype}')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError(f'Invalid tensor offsets: {name}')
        begin, end = offsets
        if not 0 <= begin <= end <= payload_size:
            raise ValueError(f'Tensor offsets outside payload: {name}')
        if end - begin != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f'Tensor byte count disagrees with shape/dtype: {name}')
        ranges.append((begin, end, name))
    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            raise ValueError(f'Tensor payload is not contiguous or contains overlap: {name}')
        cursor = end
    if cursor != payload_size:
        raise ValueError('Unindexed bytes remain in tensor payload')
    return {name: value for name, value in header.items() if name != '__metadata__'}


def expected_mtp_shapes(config):
    """The one-layer, dense, unquantized, shared-embedding official layout."""
    c = config.get('text_config', config)
    if c.get('model_type') != 'qwen3_5_text':
        raise ValueError('Only dense qwen3_5_text MTP is supported by this inspection')
    if c.get('mtp_num_hidden_layers') != 1:
        raise ValueError('Exactly one MTP layer is required')
    if c.get('mtp_use_dedicated_embeddings', False):
        raise ValueError('Dedicated MTP embeddings are unsupported')
    if not c.get('tie_word_embeddings', config.get('tie_word_embeddings', False)):
        raise ValueError('This inspection requires shared embedding/lm_head weights')
    if c.get('attention_bias', False) or c.get('attn_output_gate') is not True:
        raise ValueError('Expected bias-free attention with per-head output gate')
    if c.get('hidden_act') != 'silu':
        raise ValueError('Expected SiLU gated dense MLP')
    dims = []
    for key in ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size'):
        value = c.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f'Invalid model dimension: {key}')
        dims.append(value)
    h, intermediate, heads, kv_heads, dim, _ = dims
    if heads % kv_heads:
        raise ValueError('Query heads must be divisible by KV heads')
    epsilon = c.get('rms_norm_eps')
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid RMS normalization epsilon')
    rope = c.get('rope_parameters', {})
    if not isinstance(rope, dict) or rope.get('rope_type') != 'default':
        raise ValueError('Only default partial rotary embeddings are supported')
    fraction = rope.get('partial_rotary_factor', 1.0)
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Invalid partial rotary fraction')
    rotary_dim = dim * fraction
    if int(rotary_dim) != rotary_dim:
        raise ValueError('Partial rotary dimension must be integral')
    if int(rotary_dim) <= 0 or int(rotary_dim) % 2:
        raise ValueError('Partial rotary dimension must be positive and even')
    theta = rope.get('rope_theta')
    if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
        raise ValueError('Invalid rotary theta')
    shapes = {'mtp.fc.weight': [h, 2 * h]}
    for norm in ('pre_fc_norm_embedding', 'pre_fc_norm_hidden', 'norm',
                 'layers.0.input_layernorm', 'layers.0.post_attention_layernorm'):
        shapes[f'mtp.{norm}.weight'] = [h]
    for name, shape in {
        'self_attn.q_proj': [2 * heads * dim, h],
        'self_attn.k_proj': [kv_heads * dim, h],
        'self_attn.v_proj': [kv_heads * dim, h],
        'self_attn.o_proj': [h, heads * dim],
        'self_attn.q_norm': [dim], 'self_attn.k_norm': [dim],
        'mlp.gate_proj': [intermediate, h], 'mlp.up_proj': [intermediate, h],
        'mlp.down_proj': [h, intermediate],
    }.items():
        shapes[f'mtp.layers.0.{name}.weight'] = shape
    return shapes


def inspect_checkpoint(folder):
    """Return fail-closed availability/layout evidence for one local checkpoint."""
    folder = Path(folder).resolve()
    report = {
        'path': str(folder), 'native_head_status': 'blocked',
        'speculative_decoding_usable': False, 'mtp_tensor_count_indexed': 0,
        'mtp_tensor_count_verified': 0, 'errors': [], 'tensors': {},
        'payload_values_verified': False, 'headers_valid': False,
        'notes': ['Header inspection only; this probe does not execute the separate opt-in MTP runtime.'],
    }
    errors = report['errors']
    try:
        config = _json((folder / 'config.json').read_bytes())
        if not isinstance(config, dict):
            raise ValueError('Configuration must be an object')
        c = config.get('text_config', config)
        index = _json((folder / 'model.safetensors.index.json').read_bytes())
        weights = index['weight_map']
        if not isinstance(weights, dict) or not weights or not all(isinstance(v, str) for v in weights.values()):
            raise ValueError('Invalid checkpoint weight_map')
        report['mtp_tensor_count_indexed'] = sum(name.startswith('mtp.') for name in weights)
        report['quantization_config'] = config.get('quantization_config')
        tensors = {}
        for shard in sorted(set(weights.values())):
            path = (folder / shard).resolve()
            if not path.is_relative_to(folder):
                errors.append(f'Shard path is outside checkpoint: {shard}')
                continue
            if not path.is_file():
                errors.append(f'Missing shard: {shard}')
                continue
            try:
                header = read_safetensors_header(path)
            except (ValueError, OSError, UnicodeError) as error:
                errors.append(f'Invalid shard {shard}: {error}')
                continue
            expected_names = {name for name, value in weights.items() if value == shard}
            for name in sorted(expected_names - header.keys()):
                errors.append(f'Indexed tensor absent from shard {shard}: {name}')
            for name in sorted(header.keys() - expected_names):
                errors.append(f'Header tensor absent/mismatched in index {shard}: {name}')
            for name, value in header.items():
                if name in tensors:
                    errors.append(f'Duplicate tensor across shards: {name}')
                tensors[name] = dict(value, shard=shard)
        report['headers_valid'] = not errors
        report['tensor_count_verified'] = len(tensors)
        report['dtype_counts'] = dict(Counter(value['dtype'] for value in tensors.values()))
        report['payload_bytes'] = sum(v['data_offsets'][1] - v['data_offsets'][0] for v in tensors.values())
        report['mtp_tensor_count_verified'] = sum(name.startswith('mtp.') for name in tensors)
        shared_name = 'model.language_model.embed_tokens.weight'
        if shared_name not in tensors and 'model.embed_tokens.weight' in tensors:
            shared_name = 'model.embed_tokens.weight'
        report['tensors'] = {name: value for name, value in tensors.items()
                             if name.startswith('mtp.') or name in (shared_name, 'lm_head.weight')}
        if not report['mtp_tensor_count_indexed'] and not report['mtp_tensor_count_verified']:
            errors.append('No MTP tensors in checkpoint; config/index availability is not weight availability')
        expected = expected_mtp_shapes(config)
        report['expected_mtp_shapes'] = expected
        report['expected_mtp_bf16_bytes'] = sum(math.prod(shape) * 2 for shape in expected.values())
        for name in sorted(tensors.keys() | weights.keys()):
            if name.startswith('mtp.') and name not in expected:
                errors.append(f'Unsupported MTP tensor/quantization scale layout: {name}')
        for name, shape in expected.items():
            value = tensors.get(name)
            if value is None:
                errors.append(f'Missing MTP tensor payload: {name}')
                continue
            if value['shape'] != shape:
                errors.append(f'MTP tensor shape mismatch: {name}: {value["shape"]} != {shape}')
            if value['dtype'] not in ('BF16', 'F16', 'F32'):
                errors.append(f'Unsupported packed/quantized MTP dtype: {name}: {value["dtype"]}')
        embed = tensors.get(shared_name)
        if embed is None or embed['shape'] != [c['vocab_size'], c['hidden_size']]:
            errors.append(f'Missing or incompatible shared embedding: {shared_name}')
        elif embed['dtype'] not in ('BF16', 'F16', 'F32'):
            errors.append('Shared embedding must have floating-point storage')
        head = tensors.get('lm_head.weight')
        if head is not None:
            # Header shape equality cannot prove two independently stored tensors are tied.
            errors.append('Separately stored lm_head requires value/tie verification beyond this header probe')
        if not errors:
            report['native_head_status'] = 'available_headers_only'
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors.append(f'Checkpoint inspection failed: {error}')
    return report


def fp4_environment():
    """Read distribution metadata only; do not import GPU packages or query CUDA."""
    names = ('torch', 'transformers', 'bitsandbytes', 'triton', 'kernels',
             'flashinfer-python', 'flashinfer-cubin', 'flashinfer-jit-cache',
             'nvidia-modelopt', 'tensorrt-llm', 'vllm', 'sglang', 'torchao',
             'compressed-tensors', 'cuda-python', 'nvidia-cutlass-dsl')
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        'packages': packages, 'cuda_queried': False,
        'sm120_hardware_verified': False, 'sm120_native_fp4_execution_verified': False,
        'native_fp4_usable': False,
        'blockers': [
            'NF4 packed bytes and double-quant scales are incompatible with NVFP4 E2M1/block-16 scales.',
            'Use the BF16 dense export for fresh packing; preserve both trained LoRA branches, router, and sharing.',
            'No native FP4 runtime/kernel path has been executed or validated for this custom target.',
            'SM120 hardware capability alone does not establish an installed compatible kernel.',
        ],
    }


def build_report(base, dense_source=None, nf4_source=None):
    result = {
        'probe': 'qwen35-offline-mtp-fp4-v1',
        'status': 'blocked', 'speculative_decoding_usable': False,
        'base': inspect_checkpoint(base), 'fp4': fp4_environment(),
        'integration_blockers': [
            'The main target skips mtp.*; the opt-in MTP runtime requires its separate official tensor file.',
            'Modified NF4 plus trained Top-1 LoRA experts require actual target hidden states and acceptance measurements.',
            'Hybrid DeltaNet convolution/recurrent state cannot be rolled back by cropping full-attention KV alone.',
            'JSON grammar, EOS, sampling and streaming must commit only verified accepted tokens.',
        ],
    }
    for key, path in (('dense_source', dense_source), ('nf4_source', nf4_source)):
        if path is not None:
            result[key] = inspect_checkpoint(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'base/Qwen3.5-4B')
    parser.add_argument('--dense-source', type=Path, default=ROOT / 'exports/Qwen3.5-4B-Dense-style-v1')
    parser.add_argument('--nf4-source', type=Path, default=ROOT / 'runtime-cache/Qwen3.5-4B-Dense-style-v1-text-nf4-v1')
    args = parser.parse_args(argv)
    report = build_report(args.base, args.dense_source, args.nf4_source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Header-only availability is deliberately insufficient for a usability gate.
    return 0 if report['speculative_decoding_usable'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
