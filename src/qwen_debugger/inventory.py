"""Read-only runtime identity; file metadata is not a weights checksum."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT = Path(os.environ.get('QDEBUG_PROJECT_ROOT', next(
    (str(p) for p in Path(__file__).resolve().parents if (p/'models/pretrained.py').is_file()),
    str(Path.cwd())))).resolve()
DEFAULT_MODEL = Path(os.environ.get('QWEN35_ROOT','/mnt/d/ai_project_data/qwen35_cloud'))/'exports/Qwen3.5-MoE-NF4-DPO-long-v3'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_local(model_path=None):
    folder=Path(model_path or DEFAULT_MODEL).resolve()
    manifest=folder/'moe_manifest.json'
    if not manifest.is_file(): raise ValueError('找不到当前 MoE bundle：'+str(manifest))
    data=json.loads(manifest.read_text()); paths={}
    for key in ('base','merged_base','dense_adapter','moe_adapter','routing_state'):
        if key not in data: continue
        p=Path(data[key]); cloud=Path('/mnt/workspace/qwen35_update')
        if p.is_relative_to(cloud): p=folder.parent.parent/p.relative_to(cloud)
        elif not p.is_absolute(): p=folder/p
        if not p.exists(): raise ValueError('模型依赖缺失：'+str(p))
        paths[key]=str(p)
    configs={}; files=[]
    for kind in ('base','merged_base','dense_adapter','moe_adapter'):
        if kind not in paths: continue
        root=Path(paths[kind])
        for name in ('config.json','tokenizer_config.json','generation_config.json','chat_template.jinja','adapter_config.json'):
            p=root/name
            configs[kind+'/'+name]={'path':str(p),'sha256':file_hash(p)} if p.is_file() else None
        for p in sorted(root.glob('*.safetensors')):
            st=p.stat(); files.append({'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':None})
    config=json.loads((Path(paths['base'])/'config.json').read_text())
    text=config.get('text_config',config)
    code={str(p.relative_to(PROJECT)):file_hash(p) for foldername in ('models','src/qwen_debugger')
          for p in sorted((PROJECT/foldername).glob('*.py'))}
    return {'model_path':str(folder),'paths':paths,'bundle_sha256':file_hash(manifest),
            'bundle_declared_metadata':data,'configs':configs,'weight_files':files,'code_sha256':code,
            'text_config':text,'weight_identity':'partial: file metadata; manifest checksums are declarations',
            'template_policy':'existing PretrainedChatModel non-thinking prefix patch; actual template saved per request',
            'seed_capability':{'state':'declared','source':'torch.manual_seed + fork_rng; not a determinism guarantee'},
            'thinking_capability':{'state':'declared','source':'existing runtime template'},
            'retrieval':'disabled; exact case messages only','cross_observation_cache':'cleared'}


def doctor():
    deps={}
    for name in ('pydantic','torch','transformers','peft','bitsandbytes'):
        try: deps[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: deps[name]=None
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],
                           capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired): gpu=None
    try: local=inspect_local()
    except (OSError,ValueError) as exc: local={'error':str(exc)}
    return {'python':sys.executable,'python_version':platform.python_version(),'os':platform.platform(),
            'core_dependencies':deps,'gpu_device_totals_not_model_usage':gpu,
            'model_worker_python':str(PROJECT/'.venv-qwen35/bin/python'), 'local':local,
            'probe':'NOT_RUN: doctor does not load weights or generate'}
