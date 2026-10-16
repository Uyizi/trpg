"""Persistent DSW queue: finish CPT -> style SFT -> two-expert MoE -> chat bundle.

No model/data downloads. A failed or manually stopped stage halts the queue.
Restart explicitly with --resume; completed stages are never trained twice.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/mnt/workspace/qwen35_update')
STATE=ROOT/'logs/qwen35_pipeline.json'
STOP=ROOT/'logs/qwen35_pipeline.stop'


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return {}


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    tmp.replace(path)


def status(stage,state,**kwargs):
    write(STATE,dict(stage=stage,state=state,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),**kwargs))
    print(time.strftime('%H:%M:%S'),stage,state,flush=True)


def stopped():
    return STOP.exists()


def is_locked(path):
    if not path.exists(): return False
    with path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(lock,fcntl.LOCK_UN)
    return False


def require_complete(stage):
    folder=ROOT/'runs'/f'qwen35_{stage}_v1'
    completed=read(folder/'completed.json')
    if (completed.get('state')!='completed'
        or not (folder/'adapter/adapter_model.safetensors').is_file()):
        raise RuntimeError(f'{stage} has not completed; refusing to advance')
    # The atomic completion marker is canonical after a process has released its lock.
    if read(folder/'status.json').get('state')!='completed':
        write(folder/'status.json',completed)


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def publish():
    require_complete('style');require_complete('moe')
    destination=ROOT/'exports/Qwen3.5-4B-MoE-style-v1'
    destination.mkdir(parents=True,exist_ok=True)
    base=ROOT/'base/Qwen3.5-4B'
    dense=ROOT/'runs/qwen35_style_v1/adapter'
    moe=ROOT/'runs/qwen35_moe_v1/adapter'
    manifest={'format':'qwen35-top1-v1','state':'completed','base':str(base),
        'dense_adapter':str(dense),'moe_adapter':str(moe),'experts':2,'top_k':1,
        'dense_adapter_sha256':digest(dense/'adapter_model.safetensors'),
        'moe_adapter_sha256':digest(moe/'adapter_model.safetensors'),
        'base_receipt_sha256':digest(base/'download-complete.json'),
        'note':'Reconstruct cloned frozen Dense FFNs, then load expert LoRAs and trained routers; custom project loader required.'}
    write(destination/'moe_manifest.json',manifest)
    # An atomic pointer switch affects the NEXT cloud chat, never an existing session.
    pointer=ROOT/'active-model'
    pending=ROOT/'active-model.next'
    if pending.is_symlink(): pending.unlink()
    pending.symlink_to(destination,target_is_directory=True)
    pending.replace(pointer)
    return destination


def run_stage(stage,resume):
    folder=ROOT/'runs'/f'qwen35_{stage}_v1'
    if is_locked(folder/'.lock'):
        raise RuntimeError(f'{stage} is already running outside this queue')
    if read(folder/'completed.json').get('state')=='completed':
        require_complete(stage);return
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    cmd=[str(ROOT/'.venv35/bin/python'),'-u',str(ROOT/'scripts/cloud_qwen35_followup.py'),'--stage',stage]
    if resume: cmd.append('--resume')
    with (ROOT/'logs'/f'qwen35_{stage}.log').open('ab') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage,'running',child_pid=child.pid)
        signaled=False
        while child.poll() is None:
            if stopped() and not signaled:
                child.send_signal(signal.SIGTERM);signaled=True
                status(stage,'stopping',child_pid=child.pid)
            time.sleep(5)
        if stopped():
            raise InterruptedError('Stop requested; current stage saved when possible')
        if child.returncode!=0:
            raise RuntimeError(f'{stage} exited {child.returncode}; see logs/qwen35_{stage}.log')
    require_complete(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with (ROOT/'logs/qwen35_pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A stop request is sticky; restarting requires explicitly removing its file.
        def request_stop(*_): STOP.touch()
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        stage='cpt'
        try:
            status(stage,'waiting_for_current_training')
            while is_locked(ROOT/'runs/qwen35_cpt_v1/.lock'):
                if stopped(): raise InterruptedError('Queue stopped; independent CPT left running')
                time.sleep(5)
            if stopped(): raise InterruptedError('Queue stopped')
            require_complete('cpt')
            for stage in ('style','moe'):
                if stopped(): raise InterruptedError('Queue stopped')
                run_stage(stage,args.resume)
            stage='publish'
            if stopped(): raise InterruptedError('Stopped before publishing')
            destination=publish()
            status('completed','completed',model=str(destination))
        except InterruptedError as exc:
            status(stage,'stopped',reason=str(exc));return
        except BaseException as exc:
            status(stage,'failed',reason=f'{type(exc).__name__}: {str(exc)[:500]}')
            raise


if __name__=='__main__': main()
