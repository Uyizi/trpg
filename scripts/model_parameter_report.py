"""Count actual safetensors shapes without loading weights; size MoE candidates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import struct


def report(folder):
    config = json.loads((folder / 'config.json').read_text())
    text_config = config.get('text_config', config)
    counts, tensors, ffn_layers = defaultdict(int), {}, defaultdict(int)
    for path in sorted(folder.glob('*.safetensors')):
        with path.open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            if size > 20_000_000:
                raise ValueError('Unexpectedly large safetensors header')
            header = json.loads(source.read(size))
        for name, metadata in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError('Duplicate tensor in shards: ' + name)
            n = math.prod(metadata['shape'])
            tensors[name] = n
            if '.visual.' in name or name.startswith('visual.'):
                group = 'vision'
            elif '.mtp.' in name or name.startswith('mtp.'):
                group = 'mtp'
            elif 'embed_tokens' in name:
                group = 'embedding'
            elif name.endswith('lm_head.weight'):
                group = 'head'
            elif '.mlp.' in name:
                group = 'ffn'
                ffn_layers[name.split('.mlp.')[0]] += n
            elif '.self_attn.' in name or '.linear_attn.' in name:
                group = 'attention_and_delta'
            elif 'norm' in name:
                group = 'norm'
            else:
                group = 'other'
            counts[group] += n
    if not tensors:
        raise ValueError('No safetensors found')
    total = sum(counts.values())
    language = total - counts['vision'] - counts['mtp']
    hidden = text_config['hidden_size']
    candidates = []
    # For hybrid Qwen3.5 these are arithmetic only; do not convert its layers.
    for experts in (2, 4, 8):
        router = len(ffn_layers) * hidden * experts
        for top_k in (1, 2):
            if top_k <= experts:
                candidate = language + (experts - 1) * counts['ffn'] + router
                active = language + (top_k - 1) * counts['ffn'] + router
                candidates.append({'experts_per_layer': experts, 'top_k': top_k,
                    'total_language_parameters': candidate, 'active_language_parameters': active,
                    'frozen_bf16_weight_gib': candidate * 2 / 1024 ** 3})
    return {'model_directory': str(folder), 'model_type': config.get('model_type'),
        'stored_parameters': total, 'language_parameters': language, 'by_component': dict(counts),
        'ffn_layer_count': len(ffn_layers), 'ffn_per_layer': dict(ffn_layers),
        'tie_word_embeddings': text_config.get('tie_word_embeddings', config.get('tie_word_embeddings', False)),
        'moe_candidates': candidates,
        'notes': ['Counts follow actual stored tensors; an omitted tied lm_head is not counted twice.',
                  'Active parameter count is not measured FLOPs or throughput.',
                  'BF16 weights exclude gradients, LoRA, optimizer states, activations and KV cache.',
                  'Qwen3.5 may store vision/MTP weights; language-only counts exclude them.',
                  'Only Qwen3 dense is the main upcycling experiment; Qwen3.5 estimates are arithmetic.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = report(args.model)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
