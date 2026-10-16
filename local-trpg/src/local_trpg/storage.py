"""Short SQLite transactions; unpublished work never enters the canonical snapshot."""
import hashlib,json,secrets,sqlite3,time,uuid
from contextlib import contextmanager
from pathlib import Path
from .schemas import GameSnapshot,PlayerCommand,GameEvent

def wire(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(wire(x).encode()).hexdigest()
def uid():return uuid.uuid4().hex
def turn_transcript(turn_id,tick,player_name,command,result):
 """One definition for persisted and returned logs; DM asides stay private."""
 entries=[{'turn_id':turn_id,'speaker':'DM','speaker_id':'dm','text':result['narration'],'tick':tick}]
 if command.get('text','').strip():entries.insert(0,{'turn_id':turn_id,'speaker':player_name,'speaker_id':'player','text':command['text'],'tick':tick})
 entries.extend({'turn_id':turn_id,'speaker':d['speaker'],'speaker_id':d.get('speaker_id',d['speaker']),'text':d['text'],'tick':tick} for d in result['dialogue'])
 return entries
class Conflict(ValueError):pass
class Store:
 def __init__(self,path):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
  with self.tx() as db:
   version=db.execute('PRAGMA user_version').fetchone()[0]
   if version not in (0,1):raise ValueError('数据库版本高于当前程序；未加载')
   db.executescript('''
CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, state TEXT NOT NULL, initial_state TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS turns(id TEXT PRIMARY KEY,campaign TEXT NOT NULL REFERENCES campaigns(id), request_id TEXT NOT NULL,body_hash TEXT NOT NULL,command TEXT NOT NULL,base_revision INTEGER NOT NULL,status TEXT NOT NULL,stage TEXT NOT NULL,epoch INTEGER NOT NULL DEFAULT 0,cancel INTEGER NOT NULL DEFAULT 0,draft TEXT,result TEXT,reply TEXT NOT NULL DEFAULT '',created REAL NOT NULL,elapsed REAL, UNIQUE(campaign,request_id));
CREATE TABLE IF NOT EXISTS events(campaign TEXT NOT NULL REFERENCES campaigns(id), id TEXT NOT NULL,turn_id TEXT NOT NULL,body TEXT NOT NULL,seq INTEGER NOT NULL,PRIMARY KEY(campaign,id));
CREATE TABLE IF NOT EXISTS rolls(turn_id TEXT NOT NULL REFERENCES turns(id),roll_key TEXT NOT NULL,dice TEXT NOT NULL,PRIMARY KEY(turn_id,roll_key));
CREATE TABLE IF NOT EXISTS archived_rolls(campaign TEXT NOT NULL REFERENCES campaigns(id),turn_id TEXT NOT NULL,roll_key TEXT NOT NULL,dice TEXT NOT NULL,PRIMARY KEY(campaign,turn_id,roll_key));
CREATE TABLE IF NOT EXISTS saves(id TEXT PRIMARY KEY,campaign TEXT NOT NULL,name TEXT NOT NULL,revision INTEGER NOT NULL,tick INTEGER NOT NULL,bundle TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_runs(id TEXT PRIMARY KEY,turn_id TEXT,role TEXT,usage TEXT,elapsed REAL,error TEXT);
CREATE TABLE IF NOT EXISTS calls(turn_id TEXT,call_key TEXT,view_hash TEXT,output TEXT,PRIMARY KEY(turn_id,call_key));
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied REAL);
INSERT OR IGNORE INTO schema_migrations VALUES(1,strftime('%s','now'));
PRAGMA user_version=1;
''')
 @contextmanager
 def tx(self):
  db=sqlite3.connect(self.path,timeout=10);db.row_factory=sqlite3.Row
  db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA journal_mode=WAL')
  try:
   db.execute('BEGIN IMMEDIATE');yield db;db.commit()
  except BaseException:db.rollback();raise
  finally:db.close()
 def snapshot(self,campaign):
  with self.tx() as db:r=db.execute('SELECT state FROM campaigns WHERE id=?',(campaign,)).fetchone()
  if not r:raise KeyError('战役不存在')
  return GameSnapshot.model_validate_json(r['state'])
 def create(self,state):
  with self.tx() as db:db.execute('INSERT INTO campaigns VALUES(?,?,?,?,?)',(state.campaign_id,state.revision,state.model_dump_json(),state.model_dump_json(),time.time()))
  return state.campaign_id
 def list_campaigns(self):
  with self.tx() as db:rows=db.execute('SELECT * FROM campaigns ORDER BY updated DESC').fetchall()
  return [{'campaign_id':r['id'],'module_id':json.loads(r['state']).get('module_id','reserve_cooling'),'player_name':json.loads(r['state'])['actors']['player']['name'],'revision':r['revision'],'tick':json.loads(r['state'])['tick'],'updated_at':r['updated']} for r in rows]
 def submit(self,campaign,command):
  body=command.model_dump();h=digest(body)
  with self.tx() as db:
   prior=db.execute('SELECT * FROM turns WHERE campaign=? AND request_id=?',(campaign,command.client_request_id)).fetchone()
   if prior:
    if prior['body_hash']!=h:raise Conflict('同一请求编号已用于不同内容')
    return prior['id']
   c=db.execute('SELECT revision FROM campaigns WHERE id=?',(campaign,)).fetchone()
   if not c:raise KeyError('战役不存在')
   if c['revision']!=command.expected_revision:raise Conflict('战役已变化，请刷新后重新提交')
   if db.execute("SELECT 1 FROM turns WHERE campaign=? AND status IN ('queued','running','recoverable')",(campaign,)).fetchone():raise Conflict('该战役有待处理回合，请等待、恢复或取消')
   id=uid();db.execute('INSERT INTO turns(id,campaign,request_id,body_hash,command,base_revision,status,stage,created) VALUES(?,?,?,?,?,?,?,?,?)',(id,campaign,command.client_request_id,h,wire(body),command.expected_revision,'queued','等待处理',time.time()))
   return id
 def turn(self,id):
  with self.tx() as db:r=db.execute('SELECT * FROM turns WHERE id=?',(id,)).fetchone()
  if not r:raise KeyError('回合不存在')
  return dict(r)
 def active(self,campaign):
  with self.tx() as db:r=db.execute("SELECT id FROM turns WHERE campaign=? AND status IN ('queued','running','recoverable')",(campaign,)).fetchone()
  return r['id'] if r else None
 def claim(self,id):
  with self.tx() as db:
   n=db.execute("UPDATE turns SET status='running',stage='解释行动',epoch=epoch+1 WHERE id=? AND status='queued' AND cancel=0",(id,)).rowcount
   if n!=1:raise Conflict('回合已被领取或取消')
  return self.turn(id)
 def progress(self,id,epoch,stage,draft=None):
  with self.tx() as db:
   n=db.execute("UPDATE turns SET stage=?,draft=COALESCE(?,draft) WHERE id=? AND epoch=? AND status='running' AND cancel=0",(stage,wire(draft) if draft else None,id,epoch)).rowcount
   if not n:raise Conflict('回合已取消或旧执行失效')
 def finish(self,id,epoch,status,reply):
  with self.tx() as db:db.execute("UPDATE turns SET status=?,stage=?,reply=?,elapsed=?-created WHERE id=? AND epoch=? AND status='running' AND cancel=0",(status,'需要说明' if status=='needs_input' else '已结束',reply,time.time(),id,epoch))
 def cancel(self,campaign,id):
  with self.tx() as db:
   row=db.execute('SELECT * FROM turns WHERE campaign=? AND id=?',(campaign,id)).fetchone()
   if not row:raise KeyError('回合不存在')
   db.execute("UPDATE turns SET cancel=1,epoch=epoch+1,status='cancelled',stage='已取消',reply='本回合已取消，未提交任何游戏变化。' WHERE campaign=? AND id=? AND status IN ('running','queued','recoverable')",(campaign,id))
  return self.status(campaign,id)
 def recover(self):
  with self.tx() as db:db.execute("UPDATE turns SET status='recoverable',epoch=epoch+1,stage='运行被中断，可恢复或取消' WHERE status IN ('running','queued')")
 def resume(self,campaign,id):
  with self.tx() as db:
   t=db.execute('SELECT * FROM turns WHERE id=? AND campaign=?',(id,campaign)).fetchone()
   c=db.execute('SELECT revision FROM campaigns WHERE id=?',(campaign,)).fetchone()
   if not t or t['status']!='recoverable' or t['base_revision']!=c['revision']:raise Conflict('只能恢复原状态尚未变化的中断回合')
   db.execute("UPDATE turns SET status='queued',stage='等待恢复',cancel=0 WHERE id=?",(id,))
  return id
 def roll(self,turn_id,key,fixed=None):
  with self.tx() as db:
   row=db.execute('SELECT dice FROM rolls WHERE turn_id=? AND roll_key=?',(turn_id,key)).fetchone()
   if row:return json.loads(row['dice'])
   dice=list(fixed) if fixed else [secrets.randbelow(6)+1,secrets.randbelow(6)+1]
   if len(dice)!=2 or any(type(x)!=int or not 1<=x<=6 for x in dice):raise ValueError('非法骰点')
   db.execute('INSERT INTO rolls VALUES(?,?,?)',(turn_id,key,wire(dice)));return dice
 def roll_dice(self,turn_id,key,*,sides,count=1,fixed=None):
  if sides not in (6,20) or type(count)!=int or not 1<=count<=3:raise ValueError('非法骰子规格')
  key=f'dice:{sides}:{count}:'+key
  with self.tx() as db:
   row=db.execute('SELECT dice FROM rolls WHERE turn_id=? AND roll_key=?',(turn_id,key)).fetchone()
   if row:return json.loads(row['dice'])
   dice=list(fixed) if fixed is not None else [secrets.randbelow(sides)+1 for _ in range(count)]
   if len(dice)!=count or any(type(x)!=int or not 1<=x<=sides for x in dice):raise ValueError('非法骰点')
   db.execute('INSERT INTO rolls VALUES(?,?,?)',(turn_id,key,wire(dice)))
   return dice
 def commit(self,id,epoch,state,events,result):
  with self.tx() as db:
   t=db.execute('SELECT * FROM turns WHERE id=?',(id,)).fetchone()
   if not t or t['epoch']!=epoch or t['status']!='running' or t['cancel']:raise Conflict('回合不能提交')
   if state.revision!=t['base_revision'] or state.campaign_id!=t['campaign']:raise Conflict('draft 来源不符')
   new=state.model_copy(deep=True);new.revision+=1
   new.transcript.extend(turn_transcript(id,new.tick,new.actors['player'].name,json.loads(t['command']),result))
   n=db.execute('UPDATE campaigns SET state=?,revision=?,updated=? WHERE id=? AND revision=?',(new.model_dump_json(),new.revision,time.time(),new.campaign_id,t['base_revision'])).rowcount
   if n!=1:raise Conflict('旧回合不能覆盖新状态')
   seq=db.execute('SELECT COUNT(*) FROM events WHERE campaign=?',(new.campaign_id,)).fetchone()[0]
   for i,event in enumerate(events):db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(new.campaign_id,event.event_id,id,event.model_dump_json(),seq+i))
   db.execute("UPDATE turns SET status='committed',stage='已完成',result=?,reply=?,draft=NULL,elapsed=?-created WHERE id=?",(wire(result),result['narration'],time.time(),id))
  return new.revision
 def status(self,campaign,id):
  t=self.turn(id)
  if t['campaign']!=campaign:raise KeyError('回合不存在')
  result=json.loads(t['result']) if t['result'] else None
  with self.tx() as db:call=db.execute("SELECT output FROM calls WHERE turn_id=? AND call_key='guide:'",(id,)).fetchone()
  origin='model' if t['status']=='completed_ooc' and call and json.loads(call['output']).get('text')==t['reply'] else 'legacy' if t['status']=='completed_ooc' else 'system'
  return dict(campaign_id=campaign,turn_id=id,mode=json.loads(t['command']).get('mode','in_character'),epoch=t['epoch'],base_revision=t['base_revision'],status=t['status'],stage=t['stage'],is_terminal=t['status'] not in ('running','queued'),public_reply=t['reply'],reply_source=origin,revision=self.snapshot(campaign).revision,result=result,cancel_requested=bool(t['cancel']))
 def table_talk(self,campaign,limit=40):
  # Separate from in-world transcript/observations: NPCs never receive this.
  with self.tx() as db:
   rows=db.execute("SELECT t.id,t.command,t.reply,t.created,c.output FROM turns t LEFT JOIN calls c ON c.turn_id=t.id AND c.call_key='guide:' WHERE t.campaign=? AND t.status='completed_ooc' ORDER BY t.created DESC,t.id DESC LIMIT ?",(campaign,limit)).fetchall()
  return [{'turn_id':r['id'],'question':json.loads(r['command'])['text'],'answer':r['reply'],'created':r['created'],
    'reply_source':'model' if r['output'] and json.loads(r['output']).get('text')==r['reply'] else 'legacy'} for r in reversed(rows)]
 def record_call(self,turn,epoch,key,h,output,metrics):
  with self.tx() as db:
   row=db.execute('SELECT epoch,status FROM turns WHERE id=?',(turn,)).fetchone()
   if not row or row['epoch']!=epoch or row['status']!='running':raise Conflict('旧生成结果已丢弃')
   db.execute('INSERT OR REPLACE INTO calls VALUES(?,?,?,?)',(turn,key,h,wire(output)))
 def record_metric(self,turn,metrics):
  with self.tx() as db:
   db.execute('INSERT INTO model_runs VALUES(?,?,?,?,?,?)',(uid(),turn,metrics['role'],wire(metrics.get('usage',{})),metrics.get('elapsed_ms',0),metrics.get('error'),))
 def cached_call(self,turn,key,h):
  with self.tx() as db:r=db.execute('SELECT * FROM calls WHERE turn_id=? AND call_key=?',(turn,key)).fetchone()
  return json.loads(r['output']) if r and r['view_hash']==h else None
 def export(self,campaign,db=None):
  if db is None:
   with self.tx() as db:return self.export(campaign,db)
  row=db.execute('SELECT * FROM campaigns WHERE id=?',(campaign,)).fetchone()
  if not row:raise KeyError('战役不存在')
  events=[json.loads(r['body']) for r in db.execute('SELECT body FROM events WHERE campaign=? ORDER BY seq',(campaign,))]
  rolls=[dict(r) for r in db.execute('SELECT rolls.* FROM rolls JOIN turns ON turns.id=rolls.turn_id WHERE turns.campaign=? AND turns.status=\'committed\'',(campaign,))]
  rolls += [dict(r) for r in db.execute('SELECT turn_id,roll_key,dice FROM archived_rolls WHERE campaign=? ORDER BY turn_id,roll_key',(campaign,))]
  return {'schema_version':1,'module_hash':json.loads(row['state'])['module_hash'],'state':json.loads(row['state']),'initial_state':json.loads(row['initial_state']),'events':events,'rolls':rolls,'warning':'完整备份包含剧情秘密；分享请导出玩家记录。'}
 def save(self,campaign,name):
  with self.tx() as db:
   bundle=self.export(campaign,db);s=bundle['state'];id=uid()
   db.execute('INSERT INTO saves VALUES(?,?,?,?,?,?)',(id,campaign,name[:80],s['revision'],s['tick'],wire(bundle)))
   active=db.execute("SELECT 1 FROM turns WHERE campaign=? AND status IN ('running','queued','recoverable')",(campaign,)).fetchone()
  return {'save_id':id,'revision':s['revision'],'during_turn':bool(active)}
 def saves(self,campaign):
  with self.tx() as db:rows=db.execute('SELECT id,name,revision,tick FROM saves WHERE campaign=?',(campaign,)).fetchall()
  return [{'save_id':r['id'],'name':r['name'],'revision':r['revision'],'tick':r['tick']} for r in rows]
 def fork(self,id,module_hash):
  with self.tx() as db:r=db.execute('SELECT bundle FROM saves WHERE id=?',(id,)).fetchone()
  if not r:raise KeyError('存档不存在')
  return self.restore(json.loads(r['bundle']),module_hash)
 def restore(self,bundle,module_hash):
  if len(wire(bundle))>16*1024*1024:raise ValueError('存档超过16MiB')
  if bundle.get('schema_version')!=1 or bundle.get('module_hash')!=module_hash:raise ValueError('存档版本或冒险内容不匹配')
  s=GameSnapshot.model_validate(bundle['state']);initial=GameSnapshot.model_validate(bundle['initial_state'])
  if s.module_hash!=module_hash or initial.module_hash!=module_hash:raise ValueError('快照身份不匹配')
  evs=[GameEvent.model_validate(e) for e in bundle['events']]
  if len({e.event_id for e in evs})!=len(evs):raise ValueError('重复事件')
  from .engine.reducer import apply_event
  from .modules import module_by_id
  module=module_by_id(s.module_id);world=module['world']
  if module['module_hash']!=module_hash or initial.module_id!=s.module_id:raise ValueError('冒险身份不匹配')
  adventure=module.get('engine')=='scene_d20'
  if adventure:
   from .adventure.engine import initial_adventure
   template=initial_adventure(module,initial.actors['player'].name,initial.flags.get('sheets',{}).get('player',''))
   # Snapshot identity and public name vary; all actual initial resources and
   # private state must match a genuine initialized module, not imported claims.
   exclude_initial={'campaign_id','revision','transcript','mock'}
   if initial.model_dump(exclude=exclude_initial)!=template.model_dump(exclude=exclude_initial):raise ValueError('初始冒险状态不匹配')
   allowed_actors=set(template.actors);locations=set(world['scenes'])
   registered_items=set(template.items);registered_facts={}
  else:
   allowed_actors={'player',*world['actors']};locations=set(world['locations'])
   registered_items=set(world['items']);registered_facts={key:True for key in world['facts']}
  if set(s.actors)!=allowed_actors or set(initial.actors)!=allowed_actors:raise ValueError('存档角色集合错误')
  if s.world_facts!=registered_facts or initial.world_facts!=s.world_facts:raise ValueError('固定世界真相不匹配')
  for snapshot in (initial,s):
   if set(snapshot.items)!=registered_items:raise ValueError('物品集合与冒险不一致')
   for item,owner in snapshot.items.items():
    if owner not in allowed_actors|locations|{'consumed','destroyed','pod'}:raise ValueError('物品持有者无效')
   for actor in snapshot.actors.values():
    if actor.location_id not in locations or not set(actor.inventory)<=registered_items:raise ValueError('角色位置或背包无效')
    if set(actor.inventory)!={i for i,owner in snapshot.items.items() if owner==actor.id}:raise ValueError('背包与物品所有权不一致')
   if not set(snapshot.evidence)<=allowed_actors or any(not set(refs)<=set(world['evidence']) for refs in snapshot.evidence.values()):raise ValueError('线索引用无效')
  replay=initial
  for e in evs:
   if e.actor_id not in allowed_actors or e.location_id not in locations:raise ValueError('事件引用未知角色或地点')
   if e.kind=='ActorMoved' and (e.payload['actor_id'] not in allowed_actors or e.payload['destination'] not in locations):raise ValueError('非法移动引用')
   replay=apply_event(replay,e)
  exclude={'revision','transcript'}
  if replay.model_dump(exclude=exclude)!=s.model_dump(exclude=exclude):raise ValueError('存档快照与事件重放不一致')
  saved_rolls=bundle.get('rolls',[])
  keys=set();event_turns={e.turn_id for e in evs}
  for r in saved_rolls:
   if set(r)!={'turn_id','roll_key','dice'} or (r['turn_id'],r['roll_key']) in keys:raise ValueError('重复或无效骰点记录')
   if r['turn_id'] not in event_turns:raise ValueError('骰点引用不存在的回合')
   dice=json.loads(r['dice'])
   if r['roll_key'].startswith('dice:'):
    try:_,sides,count,_=r['roll_key'].split(':',3);sides=int(sides);count=int(count)
    except (ValueError,TypeError):raise ValueError('非法骰子规格')
    if not adventure or sides not in (6,20) or not 1<=count<=3:raise ValueError('非法骰子规格')
   else:sides,count=6,2
   if not isinstance(dice,list) or len(dice)!=count or any(type(n)!=int or not 1<=n<=sides for n in dice):raise ValueError('非法骰点')
   keys.add((r['turn_id'],r['roll_key']))
  id=uid();s.campaign_id=id;initial.campaign_id=id
  with self.tx() as db:
   db.execute('INSERT INTO campaigns VALUES(?,?,?,?,?)',(id,s.revision,s.model_dump_json(),initial.model_dump_json(),time.time()))
   for i,e in enumerate(evs):db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(id,e.event_id,e.turn_id,e.model_dump_json(),i))
   for r in saved_rolls:db.execute('INSERT INTO archived_rolls VALUES(?,?,?,?)',(id,r['turn_id'],r['roll_key'],r['dice']))
  return id
