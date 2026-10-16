"""Optional derivative NF4 cache. Published source weights are never modified.

Small JSON files use SHA256; large shard provenance uses size/mtime/ctime plus
safetensors header validation. Stat fingerprints are not cryptographic proof of
tensor contents. Cache creation is explicit; ordinary inference only reads it.
"""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from safetensors import SafetensorError, safe_open
from .runtime_paths import ROOT

FORMAT = 'qwen35-text-nf4-cache-v1'
QUANTIZATION = dict(quant_method='bitsandbytes', load_in_4bit=True, load_in_8bit=False,
                    bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype='bfloat16', bnb_4bit_quant_storage='uint8',
                    llm_int8_skip_modules=['lm_head'], llm_int8_enable_fp32_cpu_offload=False)


def cache_path(source):
    return ROOT / 'runtime-cache' / (Path(source).resolve().name + '-text-nf4-v1')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_files(folder):
    index = folder / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    names = sorted(set(mapping.values())) if mapping is not None else ['model.safetensors']
    if not names or any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
        raise ValueError('Invalid safetensors shard index')
    return names, mapping


def source_identity(source):
    source = Path(source).resolve()
    names, _ = _checkpoint_files(source)
    names += [name for name in ('config.json', 'model.safetensors.index.json', 'generation_config.json')
              if (source / name).exists()]
    if 'config.json' not in names:
        raise ValueError('Checkpoint config is missing')
    files = {}
    for name in names:
        path, stamp = source / name, (source / name).stat()
        files[name] = dict(size=stamp.st_size, mtime_ns=stamp.st_mtime_ns, ctime_ns=stamp.st_ctime_ns)
        if name.endswith('.json'):
            files[name]['sha256'] = _sha(path)
    return dict(path=str(source), files=files)


def _semantics():
    return dict(versions={name: version(name) for name in
                         ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'safetensors')},
                quantization=QUANTIZATION,
                text_loader_sha256=_sha(Path(__file__).with_name('text_only_qwen35.py')))


def _validate_text_nf4(folder):
    config = json.loads((folder / 'config.json').read_text())
    quantization = config.get('quantization_config', {})
    if config.get('model_type') != 'qwen3_5' or any(quantization.get(k) != v for k, v in QUANTIZATION.items()):
        raise ValueError('Cache must contain the configured Qwen3.5 text NF4 base')
    names, mapping = _checkpoint_files(folder)
    actual = {}
    for filename in names:
        with safe_open(folder / filename, framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                if (not key.startswith(('model.language_model.', 'lm_head.'))
                    or any(part in key for part in ('.visual.', '.experts.', '.router.', 'lora_', 'adapter'))
                    or key in actual):
                    raise ValueError('Cache contains a visual, expert, adapter, or duplicate tensor')
                actual[key] = filename
    if not any(key.endswith('.quant_state.bitsandbytes__nf4') for key in actual):
        raise ValueError('Cache does not contain serialized NF4 quantization state')
    if mapping is not None and mapping != actual:
        raise ValueError('Cache shard index does not match tensor headers')


def cache_manifest(source, folder):
    folder = Path(folder)
    _validate_text_nf4(folder)
    cached = source_identity(folder)
    cached.pop('path')  # Staging directories are renamed atomically.
    return dict(format=FORMAT, state='completed', source=source_identity(source),
                semantics=_semantics(), cache=cached,
                verification='JSON SHA256 and shard size/mtime_ns/ctime_ns with safetensors headers; '
                             'stat fingerprints are not cryptographic proof of tensor contents')


def validated_cache(source, candidate=None):
    folder = Path(candidate) if candidate is not None else cache_path(source)
    try:
        manifest = json.loads((folder / 'runtime_cache.json').read_text())
        if manifest.get('format') != FORMAT or manifest.get('state') != 'completed':
            return None
        expected = cache_manifest(source, folder)
        return folder if manifest == expected else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, SafetensorError, ImportError):
        return None
