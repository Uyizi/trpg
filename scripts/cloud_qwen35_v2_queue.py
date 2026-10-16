"""Cloud-only NF4 continuation queue; stop on any stage failure."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/workspace/qwen35_update')
def state(stage,status,**extra):
 path=ROOT/'logs/qwen35_v2_queue.json';temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(dict(stage=stage,state=status,pid=os.getpid(),updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**extra),indent=2))
 temp.replace(path)
 print(stage,status,flush=True)
def main():
 with (ROOT/'logs/qwen35_v2_queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
  env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
  for stage,script in [('merge_dense','cloud_qwen35_export_dense.py'),('qlora','cloud_qwen35_qlora.py')]:
   state(stage,'running')
   with (ROOT/'logs'/f'qwen35_v2_{stage}.log').open('ab') as log:
    code=subprocess.call([str(ROOT/'.venv35/bin/python'),'-u',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   if code:
    state(stage,'failed',exit_code=code);return
  completed=json.loads((ROOT/'runs/qwen35_qlora_v2/status.json').read_text())
  state('finished',completed['state'])
if __name__=='__main__':main()
