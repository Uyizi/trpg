"""Finite rule authority for the original reserve-cooling adventure."""
import hashlib
from ..schemas import Actor,Observation,BeliefRecord,GameSnapshot,GameEvent,RuleResult
from .reducer import apply_event
from .scenario import ScenarioEffects

HEALTH=['healthy','strained','injured','incapacitated']
def classify_check(total):return 'success' if total>=10 else 'success_with_cost' if total>=7 else 'failure'
def initial_state(module,name,mock=False,campaign_id=None):
 import uuid
 w=module['world'];actors={}
 for id,p in {'player':w['player'],**w['actors']}.items():
  actors[id]=Actor(id=id,name=name if id=='player' else p['name'],location_id=p['location_id'],health=p.get('health','healthy'),skills=p.get('skills',{}),inventory=p.get('inventory',[]),goals=p.get('goals',[]))
 state=GameSnapshot(campaign_id=campaign_id or uuid.uuid4().hex,module_hash=module['module_hash'],actors=actors,
  items={id:(p.get('location_id','cargo') if p.get('owner')=='ship' else p.get('owner',p.get('location_id','cargo'))) for id,p in w['items'].items()},world_facts={id:True for id in w['facts']},
  flags=w['initial_flags'],clocks={k:v for k,v in w['clocks'].items() if k not in ('phase',)},relations={id:0 for id in w['actors']},mock=mock,
  cells=[{'id':f'cell_{i}','container':'original_cargo_box','expiry':None,'valid':True} for i in range(1,5)])
 for id,a in actors.items():
  for item in a.inventory:state.items[item]=id
 for o in w['initial_observations']:
  text=' '.join(w['facts'][f]['text'] for f in o['proposition_ids'])
  state.observations.append(Observation(observation_id=o['observation_id'],event_id='initial:'+o['observation_id'],observer_id=o['actor_id'],modality='direct',content=text,proposition_ids=o['proposition_ids'],observed_at=0))
 for id,p in w['actors'].items():
  for b in p.get('initial_beliefs',[]):state.beliefs.append(BeliefRecord(actor_id=id,proposition_id=b['proposition_id'],stance=b['stance'],source_observation_ids=b['source_observation_ids'],updated_at=0,basis=b['basis']))
 state.transcript=[{'turn_id':'opening','speaker':'DM','text':module['public']['opening'],'tick':0}]
 return state

class RuleError(ValueError):pass
class Engine(ScenarioEffects):
 def __init__(self,state,module,turn_id,rolls):
  self.state=state.model_copy(deep=True);self.module=module;self.turn_id=turn_id;self.rolls=rolls;self.events=[];self.checks=[];self.authorization=None
 def emit(self,kind,payload,actor='player',location=None):
  e=GameEvent(event_id=f'{self.turn_id}:{len(self.events)}',turn_id=self.turn_id,beat_index=len(self.events),actor_id=actor,kind=kind,payload=payload,
   location_id=location or self.state.actors[actor].location_id,game_tick=self.state.tick)
  self.state=apply_event(self.state,e);self.events.append(e);return e
 def notice(self,text,recipients=None,actor='player'):self.emit('PublicNotice',{'text':text,'recipient_ids':list(dict.fromkeys((recipients or ['player'])+[actor]))},actor)
 def flag(self,key,value):self.emit('FlagSet',{'key':key,'value':value})
 def move(self,actor,destination):
  if destination not in self.module['world']['locations']:raise RuleError('没有这个地点')
  if destination=='engine' and self.state.flags.get('engine_destroyed'):raise RuleError('动力舱已热损坏并封闭')
  if self.state.actors[actor].health=='incapacitated':raise RuleError('该人物无法自行移动，需要协助')
  if self.state.actors[actor].location_id!=destination:
   if self.state.actors[actor].onboard:self.flag('onboard:'+actor,False)
   self.emit('ActorMoved',{'actor_id':actor,'destination':destination},actor)
 def give_evidence(self,id,recipients,source='player'):
  if id not in self.module['world']['evidence']:raise RuleError('没有这份已注册证据')
  p=self.module['world']['evidence'][id]
  self.emit('EvidenceObserved',{'observer_ids':recipients,'evidence_id':id,'content':p['text'],'proposition_ids':p['proposition_ids'],'source':source},source)
 def speech(self,actor,text,channel,recipients,evidence=None):
  if not text:return
  if any(id not in self.state.actors for id in recipients):raise RuleError('收件人不存在')
  # All six participants have the established ship working radio/private handset.
  if channel=='private' and not recipients:raise RuleError('请指定私聊对象')
  if channel=='broadcast':recipients=[id for id,a in self.state.actors.items() if a.outcome=='present']
  self.emit('SpeechOccurred',{'text':text,'channel':channel,'recipient_ids':recipients,'evidence_refs':evidence or []},actor)
 def advance(self,cost):
  start=self.state.tick
  for tick in range(start+1,start+cost+1):
   self.emit('ClockAdvanced',{'tick':tick})
   self.expire_cells()
   if tick<start+cost:self.finish_clock()
 def expire_cells(self):
  cells=[dict(c) for c in self.state.cells];changed=False
  for c in cells:
   if c['valid'] and c['expiry'] is not None and self.state.tick>=c['expiry']:c['valid']=False;changed=True
  if changed:
   self.emit('CellsChanged',{'cells':cells});self.notice('部分细胞超过有效保冷期限，已经失效。')
 def check(self,actor,skill,description):
  modifier=-1 if self.state.actors[actor].health=='strained' or (skill=='technical' and self.state.tick>=6) else 0
  dice=self.rolls(f'{actor}:0:{skill}');bonus=self.state.actors[actor].skills.get(skill,0);total=sum(dice)+bonus+modifier
  payload={'dice':dice,'skill':bonus,'modifier':modifier,'total':total,'outcome':classify_check(total),'description':description}
  self.emit('CheckResolved',payload,actor);self.checks.append(payload);return payload['outcome']
 def resolve_player(self,intent):
  a=intent;s=self.state;actor='player'
  if s.actors[actor].health=='incapacitated':raise RuleError('你已无法主动行动；可以查看状态或等待救援。')
  if any(t not in s.actors and t not in self.module['world']['locations'] and t not in self.module['world']['evidence'] and t not in s.items for t in a.target_ids):raise RuleError('行动引用未知对象')
  if a.kind=='MOVE':
   if s.phase=='epilogue':raise RuleError('本次航程已经结束；可以继续交接已知记录，不能重新进入事故现场。')
   if len(a.target_ids)!=1:raise RuleError('请说明要前往哪一个地点')
   self.move(actor,a.target_ids[0]);self.describe_room();return
  if a.kind=='INSPECT' and not a.module_action_id:
   refs=self.inspection_refs(a)
   if not refs and any(word in a.description for word in ('简图','地图')):
    from .guidance import map_text
    self.notice(map_text(self.module));return
   if refs and set(refs)<=set(s.evidence.get(actor,[])):
    self.inspect(a);return
   if not refs and any(x in a.description for x in ['这里有什么','周围有什么','查看房间','环顾']):
    p=self.module['world']['locations'][s.actors[actor].location_id]
    people=[a.name for id,a in s.actors.items() if id!=actor and a.location_id==s.actors[actor].location_id and a.outcome=='present']
    self.notice('这里可见：'+'、'.join(p['visible_objects'])+'。'+('同舱的还有：'+'、'.join(people)+'。' if people else '这里只有你。'));return
  if s.phase=='epilogue':
   self.speech('player',a.spoken_text or a.description,a.channel,a.recipient_ids)
   self.notice('你可以交接或说明已经获得的记录；本次冒险的救援结果已经确定。');return
  if a.module_action_id and a.module_action_id not in self.module['world'].get('actions',{}) and a.module_action_id not in ('choose_station','assist_evacuation'):
   raise RuleError('动作尚未注册，请说明具体操作')
  if a.kind in ('TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE') and not set(a.evidence_refs)<=set(s.evidence.get('player',[])):raise RuleError('你还没有取得这份证据')
  # A registered end-of-voyage choice belongs to the player. The model may
  # phrase it as ASK, but it never authorizes an NPC task or bypasses repair.
  if a.module_action_id=='choose_station':self.module_action('choose_station','player',[]);return
  if a.kind in ('TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE'):
   # Check actual listeners before advancing time or granting work permission.
   # Merely mentioning a remote NPC must not make local speech audible to them.
   reachable={id for id,p in s.actors.items() if id!='player' and p.outcome=='present' and p.health!='incapacitated'}
   if a.channel=='local':reachable={id for id in reachable if s.actors[id].location_id==s.actors['player'].location_id}
   elif a.channel=='private':reachable&=set(a.recipient_ids or a.target_ids)
   if not reachable:raise RuleError('当前发言范围内没有能听见你的船员。请切换“工作频道 · 广播”或选择私人通话对象；本次未消耗时间。')
   # Action targets may also include a patient. Explicit message recipients
   # take precedence; for a work request, identify the capable worker only.
   addressed={id for id in (a.recipient_ids or a.target_ids) if id in self.module['world']['actors']}
   if a.module_action_id and not a.recipient_ids:addressed={id for id in addressed if a.module_action_id in self.module['world']['actors'][id]['actions']}
   if addressed-reachable:raise RuleError('指定的船员不在当前发言范围内。请切换工作频道或私人通话；本次未消耗时间。')
  if a.kind=='GIVE':
   if len(a.item_ids)!=1 or len(a.target_ids)!=1 or s.items.get(a.item_ids[0])!='player' or a.target_ids[0] not in s.actors:raise RuleError('请指定你持有的物品及在场接收人')
   if s.actors[a.target_ids[0]].location_id!=s.actors['player'].location_id:raise RuleError('接收人不在场')
  if a.kind=='ASK' and a.module_action_id:
   targets=[id for id in (a.recipient_ids or a.target_ids) if id in self.module['world']['actors'] and a.module_action_id in self.module['world']['actors'][id]['actions']]
   if len(targets)!=1:raise RuleError('请明确请求哪一名NPC执行工作')
   npc=targets[0]
   if a.module_action_id not in self.module['world']['actors'][npc]['actions']:raise RuleError('这个人物不能执行所请求的工作')
   self.authorization={'actor_id':npc,'module_action_id':a.module_action_id,'item_ids':a.item_ids,'target_ids':[x for x in a.target_ids if x!=npc],'tick':s.tick+1}
  if a.kind in ('PERSUADE','DECEIVE'):
   key='social:'+hashlib.sha256((a.spoken_text+'|'.join(a.target_ids)+'|'.join(sorted(s.evidence.get('player',[])))).encode()).hexdigest()
   if s.flags.get(key):raise RuleError('相同说法在情境未变化时不重复检定，请补充依据或改变做法')
   self.flag(key,True)
  # A request to perform work still takes time. Asking for already-held
  # information or ordinary conversation is not a five-minute work action.
  if a.kind not in ('ASK','TALK') or a.module_action_id:self.advance(1)
  if self.authorization:self.authorization['tick']=self.state.tick
  if a.module_action_id and not self.authorization:
   self.module_action(a.module_action_id,'player',a.target_ids);return
  if a.kind=='INSPECT':
   self.inspect(a);return
  if a.kind in ('TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE'):
   self.speech(actor,a.spoken_text or a.description,a.channel,a.recipient_ids or [x for x in a.target_ids if x in s.actors],a.evidence_refs)
   recipients=a.recipient_ids or [x for x in a.target_ids if x in s.actors]
   if a.channel=='broadcast':recipients=list(self.state.actors)
   elif a.channel=='local':recipients=[x.id for x in self.state.actors.values() if x.location_id==self.state.actors['player'].location_id]
   for e in a.evidence_refs:self.give_evidence(e,recipients)
   if a.kind in ('DECEIVE','PERSUADE'):
    result=self.check(actor,'social','交涉检定：说法受到质疑。' if a.kind=='DECEIVE' else '交涉检定')
    if result=='failure':
     target=next((x for x in a.target_ids if x in self.state.relations),'mechanic')
     self.emit('RelationChanged',{'actor_id':target,'value':-2});self.notice('对方质疑你的依据；仍可查阅公开记录、补充证据或改变方案。')
    elif result=='success_with_cost':self.flag('promise_preserve_records',True);self.notice('对方愿意配合，条件是完整保留记录与出处。')
   return
  if a.kind=='GIVE':self.emit('ItemTransferred',{'item_id':a.item_ids[0],'from_owner':'player','to_owner':a.target_ids[0]});return
  if a.kind=='TAKE':
   for item in a.item_ids:
    if self.state.items.get(item)!=self.state.actors['player'].location_id:raise RuleError('物品不在可取用范围')
    self.emit('ItemTransferred',{'item_id':item,'from_owner':self.state.items[item],'to_owner':'player'})
   if not a.item_ids:raise RuleError('请说明取走哪一件已发现物品')
   return
  if a.kind=='WAIT':self.notice('你等待一个行动格，留意工作频道。');return
  if a.kind=='CONFLICT':
   r=self.check(actor,'action','短暂冲突检定')
   if r=='failure':self.emit('HealthChanged',{'actor_id':actor,'health':'strained'});self.notice('你未能达成阻拦目标，受到轻伤，可以改用协商或寻求协助。')
   else:self.notice('冲突暂时平息；没有替你作出严重伤害或处决决定。')
   return
  raise RuleError('请说明操作的对象和方式；此行动尚不能确定性结算')
 def describe_room(self):
  location=self.state.actors['player'].location_id
  place=self.module['public']['locations'][location]
  people=[a.name for id,a in self.state.actors.items() if id!='player' and a.location_id==location and a.outcome=='present']
  self.notice('你来到'+place['name']+'。这里能看见：'+'、'.join(place['visible_objects'])+'。'+('在场的人有：'+'、'.join(people)+'。' if people else '目前没有其他人在这里。')+'你准备查看什么，或和谁交谈？')
 def inspection_refs(self,a):
  evidence=self.module['world']['evidence']
  refs=list(dict.fromkeys(a.evidence_refs+[x for x in a.target_ids if x in evidence]))
  if any(id not in evidence for id in refs):raise RuleError('没有这份已注册证据')
  return refs
 def inspect(self,a):
  loc=self.state.actors['player'].location_id;text=a.description;refs=self.inspection_refs(a)
  if not refs:
   if loc=='engine':refs=['E01','E04'] if not any(w in text for w in ['工单','改接']) else ['E03']
   elif loc=='bridge':refs=['E06'] if '值班' in text or '完整记录' in text else ['E08'] if '航' in text else ['E02']
   elif loc=='clinic' or loc=='cargo':refs=['E05']
   # Reading the cabinet list reveals its registered evidence, but never
   # authorizes taking its contents. Taking remains an explicit player action.
   elif loc=='lifeboat':refs=['E07']
   else:raise RuleError('生活舱只有公开简图，请前往对应地点调查')
  for id in refs:
   if id in self.state.evidence.get('player',[]):
    self.notice(self.module['world']['evidence'][id]['text']);continue
   allowed={'E01':['engine','bridge'],'E02':['bridge'],'E03':['engine','bridge'],'E04':['engine'],'E05':['clinic','cargo'],'E06':['bridge'],'E07':['lifeboat'],'E08':['bridge','lifeboat']}
   if loc not in allowed.get(id,[]):raise RuleError('这份记录不在当前位置')
   self.give_evidence(id,['player'])
