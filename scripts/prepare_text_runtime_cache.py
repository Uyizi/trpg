"""Explicitly serialize the frozen text NF4 base once; retain all source files."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from models.runtime_paths import DEFAULT_BUNDLE, read_bundle
    from models.runtime_weight_cache import cache_manifest, cache_path, source_identity, validated_cache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    manifest = read_bundle(args.bundle)
    if (manifest.get('state') != 'completed' or manifest.get('format') != 'qwen35-top1-nf4-v2'
        or manifest.get('quantization') != 'nf4' or not manifest.get('merged_base')):
        raise ValueError('Cache preparation requires a published merged Dense NF4 bundle')
    source = Path(manifest['merged_base']).resolve()
    destination = cache_path(source)
    if destination.exists():
        raise FileExistsError(f'Cache destination already exists; no files overwritten: {destination}')
    lease_path = Path.home() / '.cache/qwen-debugger/gpu.lock'
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from models.qwen35_moe import load_dense
        before = source_identity(source)
        model = load_dense(source, quantized=True).eval().requires_grad_(False)
        if not getattr(model, 'is_loaded_in_4bit', False):
            raise ValueError('Loader did not produce a serialized NF4 base')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=destination.name + '.staging-', dir=destination.parent) as staging:
            model.save_pretrained(staging, safe_serialization=True, max_shard_size='2GB')
            receipt = cache_manifest(source, staging)
            if receipt['source'] != before:
                raise ValueError('Source checkpoint changed during cache preparation')
            (Path(staging) / 'runtime_cache.json').write_text(json.dumps(receipt, indent=2) + '\n')
            if validated_cache(source, staging) is None:
                raise ValueError('Prepared cache failed validation')
            if destination.exists():
                raise FileExistsError(destination)
            Path(staging).rename(destination)
        print(json.dumps({'prepared': str(destination), 'source': str(source)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
