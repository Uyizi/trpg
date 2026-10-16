"""Resolve a downloaded cloud bundle without rewriting its source files."""
import json
import os
from pathlib import Path

CLOUD_ROOT = Path('/mnt/workspace/qwen35_update')
LOCAL_ROOT = Path('/mnt/d/ai_project_data/qwen35_cloud')
ROOT = Path(os.environ.get('QWEN35_ROOT', str(CLOUD_ROOT if Path('/etc/dsw').exists() else LOCAL_ROOT)))
DEFAULT_BUNDLE = ROOT / 'exports/Qwen3.5-MoE-NF4-DPO-long-v3'


def read_bundle(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'moe_manifest.json').read_text())
    # The export lives in ROOT/exports/name; keep all dependency paths inside
    # the downloaded tree, including tokenizer files and both adapters.
    root = folder.parent.parent
    for key in ('base', 'dense_adapter', 'moe_adapter', 'merged_base', 'routing_state'):
        if key not in manifest:
            continue
        path = Path(manifest[key])
        if path.is_relative_to(CLOUD_ROOT):
            path = root / path.relative_to(CLOUD_ROOT)
        elif not path.is_absolute():
            path = folder / path
        manifest[key] = str(path)
        if not path.exists():
            raise ValueError(f'模型依赖缺失：{path}')
    return manifest
