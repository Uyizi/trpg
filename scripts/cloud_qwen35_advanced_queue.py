"""Detached sequential cloud pipeline: QLoRA -> DPO -> 4K SFT -> RAG -> publish."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/workspace/qwen35_update')
STOP=ROOT/'logs/qwen35_advanced.stop'
CHILD=None
STOP_REQUESTED=False

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def state(stage,status,**extra):
    atomic(ROOT/'logs/qwen35_advanced_queue.json',dict(stage=stage,state=status,pid=os.getpid(),
        updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))
    print(time.strftime('%Y-%m-%d %H:%M:%S'),stage,status,flush=True)

def stopping(*_):
    global STOP_REQUESTED
    STOP_REQUESTED=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)

def main():
    global CHILD
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/qwen35_advanced.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
        stage='wait_qlora'
        try:
            state(stage,'waiting')
            while True:
                if STOP_REQUESTED or STOP.exists():state(stage,'stopped');return
                parent=ROOT/'logs/qwen35_v2_queue.json'
                if not parent.exists():raise RuntimeError('QLoRA queue has not started')
                status=json.loads(parent.read_text())
                if status['state']=='completed':break
                if status['state'] in ('failed','stopped_saved','stopped') or not Path('/proc',str(status['pid'])).exists():
                    raise RuntimeError('QLoRA queue stopped; inspect its log before resuming')
                time.sleep(15)
            env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
            env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
                       TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'app'),
                       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            python=str(ROOT/'.venv35/bin/python')
            stages=[('dpo',[python,'-u',str(ROOT/'scripts/cloud_qwen35_dpo.py')],ROOT/'runs/qwen35_dpo_v1/completed.json'),
                    ('long_context',[python,'-u',str(ROOT/'scripts/cloud_qwen35_long.py')],ROOT/'runs/qwen35_long_v1/completed.json'),
                    ('semantic_index',[python,'-u','-m','models.semantic_retrieval','--build'],ROOT/'data/semantic_arxiv/manifest.json'),
                    ('runtime_publish',[python,'-u',str(ROOT/'scripts/cloud_qwen35_finish.py')],ROOT/'exports/Qwen3.5-MoE-NF4-DPO-long-v3/completed.json')]
            for stage,command,marker in stages:
                if STOP_REQUESTED or STOP.exists():state(stage,'stopped');return
                if marker.exists():continue
                state(stage,'running')
                with (ROOT/'logs'/f'qwen35_advanced_{stage}.log').open('ab') as log:
                    CHILD=subprocess.Popen(command,cwd=ROOT/'app',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    state(stage,'running',child_pid=CHILD.pid)
                    signalled=False
                    while CHILD.poll() is None:
                        if (STOP_REQUESTED or STOP.exists()) and not signalled:
                            CHILD.send_signal(signal.SIGTERM);signalled=True
                        time.sleep(3)
                if STOP_REQUESTED or STOP.exists():state(stage,'stopped');return
                if CHILD.returncode or not marker.exists():raise RuntimeError(f'{stage}: exit={CHILD.returncode}, completion marker={marker.exists()}')
                CHILD=None
            state('finished','completed',model=str((ROOT/'active-model').resolve()))
        except BaseException as exc:
            state(stage,'failed',error=type(exc).__name__,detail=str(exc)[:500]);raise
if __name__=='__main__':main()
