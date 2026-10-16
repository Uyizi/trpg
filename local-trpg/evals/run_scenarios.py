"""Real HTTP playthrough; no fake backend, direct networking and durable evidence.

Run against the started app. Each step uses the same player endpoint as the UI.
Failures remain in JSONL. A failed step stops so a human can inspect/rephrase it.
"""
import argparse,json,time,uuid,sqlite3,hashlib
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
def source_version():
 files=sorted((ROOT/'src/local_trpg').rglob('*.py'))+sorted((ROOT/'src/local_trpg/agents/prompts').glob('*.txt'))
 return hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+b'\0'+p.read_bytes() for p in files)).hexdigest()
ROUTES={
 'repair_short':[
  ('前往动力舱',[]),
  ('检查流量面板和阀组操作卡，记录它们显示的状态。',[]),
  ('顾成，请执行手动隔离支路',['mechanic']),
  ('执行恢复主循环',[]),
  ('选择前往最近维修站',[]),
 ],
 'evacuate_finish':[
  ('方榆，请执行转存细胞',['doctor']),
  ('韩松，请执行准备救生艇',['guard']),
  ('执行组织集合',[]),
  ('前往气闸与救生艇',[]),
  ('执行登艇',[]),
  ('执行启动救生艇',[]),
 ],
 'evacuate_explicit':[
  ('林舟，请执行发送标准求救',['comms']),
  ('前往气闸与救生艇',[]),
  ('执行取应急保冷箱',[]),
  ('前往货舱',[]),
  ('方榆，我向你出示补给票E07。',['doctor']),
  ('方榆，请执行转存细胞',['doctor']),
  ('韩松，请执行准备救生艇',['guard']),
  ('执行组织集合',[]),
  ('执行登艇',[]),
  ('执行启动救生艇',[]),
 ],
 'self_transfer_finish':[
  ('执行转存细胞',[]),
  ('执行转存细胞',[]),
  ('执行组织集合',[]),
  ('前往气闸与救生艇',[]),
  ('执行登艇',[]),
  ('执行启动救生艇',[]),
 ],
 'repair':[
  ('前往动力舱',[]),
  ('检查流量面板和阀组操作卡，记录它们显示的状态。',[]),
  ('顾成，临时支路为什么接在这里？请告诉我改接工单的内容和安全处置步骤。',['mechanic']),
  ('前往驾驶舱',[]),
  ('阅读审计页，查看保持功能的启用时间和刷卡记录。',[]),
  ('林舟，我向你出示审计页E02。我保证保留完整记录与出处，请核对事故前12分钟的值班记录。',['comms']),
  ('方榆，四份细胞需要什么保冷条件？请把本次货物说明告诉我。',['doctor']),
  ('前往气闸与救生艇',[]),
  ('取走应急柜里的两只保冷箱和两块储冷芯，并阅读补给票。',[]),
  ('前往货舱',[]),
  ('方榆，这是我拿到的补给票E07，两箱两芯都已在货舱。请按说明把四份细胞转入保冷箱。',['doctor']),
  ('顾成，请现在手动隔离货舱支路。',['mechanic']),
  ('前往动力舱',[]),
  ('我执行恢复主循环操作：解除保持、排气、启动。',[]),
  ('主循环已经稳定，我选择驾驶同路号前往最近维修站。',[]),
 ],
 'evacuate':[
  ('林舟，请发送标准求救，报告预计救援到达时间。',['comms']),
  ('方榆，请说明细胞转存所需条件和保冷期限。',['doctor']),
  ('前往气闸与救生艇',[]),
  ('取走两只保冷箱、两块储冷芯，并阅读补给票。',[]),
  ('前往货舱',[]),
  ('方榆，我把E07补给票给你看，两箱两芯都在此处，请把四份细胞转入保冷箱。',['doctor']),
  ('韩松，请准备好救生艇，先别起飞。',['guard']),
  ('许岚，已经求救并保护好细胞，救生艇也已准备。请同意全员集合到气闸。',['captain']),
  ('我组织所有能够行动的船员在气闸集合撤离。',[]),
  ('前往气闸与救生艇',[]),
  ('让实际在场且同意的船员登艇，我带着保冷箱一起登上救生艇。',[]),
  ('我启动救生艇离开同路号，等待救援交接。',[]),
 ],
 'interaction':[
  ('我对韩松说：你守货舱时看到过哪些操作？只说你亲眼看见的。',['guard']),
  ('许岚，你为什么要求先别动货舱支路？你优先想保住什么？',['captain']),
  ('我对顾成谎称：船长已经批准我接手检查。',['mechanic']),
  ('林舟，我听顾成说船长要求暂时保持支路，你亲眼见过吗？',['comms']),
  ('我想暗中套话，同时表示我没有怀疑任何人。',['captain']),
  ('我不按原路线调查，先去货舱确认手动阀能不能安全卸压。',[]),
 ],
 'isolation_probe':[
  ('韩松，事故前12分钟是谁动过保持？请只说你亲眼所见。',['guard']),
  ('林舟，你知道我刚才私下和韩松说了什么吗？只说你真正听到的。',['comms']),
  ('方榆，请区分你上周亲眼见过的储冷芯情况与这次航程已确认的情况。',['doctor']),
 ]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--route',choices=ROUTES,default='repair');ap.add_argument('--url',default='http://127.0.0.1:8768');ap.add_argument('--campaign');ap.add_argument('--start',type=int,default=0);ap.add_argument('--continue-errors',action='store_true');ap.add_argument('--text');ap.add_argument('--recipients',default='');ap.add_argument('--private-aside',default='');args=ap.parse_args()
 if args.route in ('evacuate_finish','self_transfer_finish') and not args.campaign:ap.error('This route resumes an existing prepared campaign; pass --campaign')
 with httpx.Client(base_url=args.url,timeout=15,trust_env=False) as c:
  until=time.monotonic()+370
  while True:
   try:
    if c.get('/api/health').json()['model_available']:break
   except httpx.ConnectError:pass
   if time.monotonic()>until:raise RuntimeError('real model not ready; no fallback')
   time.sleep(2)
  health=c.get('/api/health').json();assert health['mock'] is False and health['backend']=='runner',health
  if args.campaign:cid=args.campaign
  else:
   r=c.post('/api/campaigns',json={'player_setup':{'name':'实机试玩-'+args.route}});r.raise_for_status();cid=r.json()['campaign_id']
  out=ROOT/'artifacts'/f'live-{cid}.jsonl'
  print(json.dumps({'campaign_id':cid,'evidence':str(out),'route':args.route},ensure_ascii=False),flush=True)
  steps=[(args.text,args.recipients.split(',') if args.recipients else [])] if args.text else ROUTES[args.route]
  for i,(text,recipients) in enumerate(steps[args.start:],args.start):
   view=c.get(f'/api/campaigns/{cid}').json();body={'client_request_id':uuid.uuid4().hex,'expected_revision':view['revision'],'text':text,'speech_channel':'private' if recipients else 'local','recipient_ids':recipients,'private_aside':args.private_aside}
   begin=time.monotonic();r=c.post(f'/api/campaigns/{cid}/turns',json=body);r.raise_for_status();tid=r.json()['turn_id']
   until=begin+150
   while True:
    status=c.get(f'/api/campaigns/{cid}/turns/{tid}').json()
    if status['is_terminal']:break
    if time.monotonic()>until:raise RuntimeError('turn exceeded 150s observation budget')
    time.sleep(.6)
   row={'step':i,'campaign_id':cid,'turn_id':tid,'request':body,'source_version':source_version(),'elapsed_ms':round((time.monotonic()-begin)*1000),'status':status}
   with sqlite3.connect(ROOT/'data/game.sqlite') as db:
    db.row_factory=sqlite3.Row
    stored=db.execute('SELECT elapsed FROM turns WHERE id=?',(tid,)).fetchone()
    row['server_elapsed_ms']=round(stored['elapsed']*1000) if stored and stored['elapsed'] is not None else None
    row['model_runs']=[dict(x) for x in db.execute('SELECT role,usage,elapsed,error FROM model_runs WHERE turn_id=?',(tid,))]
   with out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   result=status.get('result') or {};v=result.get('view') or view
   print(json.dumps({'step':i,'turn':tid,'status':status['status'],'tick':v['tick'],'calls':len(row['model_runs']),'seconds':round(row['elapsed_ms']/1000,1),'degraded':result.get('degraded'),'reply':status['public_reply'],'dialogue':result.get('dialogue'),'outcome':v.get('outcome')},ensure_ascii=False),flush=True)
   if status['status']!='committed' and not args.continue_errors:break
  r=c.get(f'/api/campaigns/{cid}/transcript');(ROOT/'artifacts'/f'live-{cid}.md').write_text(r.text)
if __name__=='__main__':main()
