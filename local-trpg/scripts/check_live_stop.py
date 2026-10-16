"""Explicit destructive fault injection for this app's own idle model worker only."""
import json,time,signal,os,uuid
from pathlib import Path
import httpx
root=Path(__file__).resolve().parents[1]
with httpx.Client(base_url='http://127.0.0.1:8768',trust_env=False,timeout=15) as c:
 h=c.get('/api/health').json();assert h['model_available'] and not h['model_busy'] and not h['mock']
 n=c.post('/api/campaigns',json={'player_setup':{'name':'模型停止恢复检查'}}).json();cid=n['campaign_id'];before=n['view']
 parent=int((root/'data/server.pid').read_text());children=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes();status=(p/'status').read_text();ppid=int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
   if ppid==parent and b'local_trpg/model/worker.py' in cmd and b'--tokenizer-only' not in cmd:children.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert len(children)==1,children
 os.kill(children[0],signal.SIGTERM);time.sleep(1)
 body={'client_request_id':uuid.uuid4().hex,'expected_revision':0,'text':'我等待一格。'}
 r=c.post(f'/api/campaigns/{cid}/turns',json=body);tid=r.json()['turn_id']
 deadline=time.monotonic()+15
 while True:
  result=c.get(f'/api/campaigns/{cid}/turns/{tid}').json()
  if result['is_terminal']:break
  assert time.monotonic()<deadline;time.sleep(.2)
 after=c.get(f'/api/campaigns/{cid}').json();health=c.get('/api/health').json()
 assert result['status']=='failed' and before==after and not health['model_available']
 report={'model_pid':children[0],'campaign_id':cid,'status':result['status'],'reply':result['public_reply'],'snapshot_unchanged':True,'health':health,'real_model':True,'scope':'only this application idle worker'}
 (root/'artifacts/live-model-stop.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
