import re
from ..schemas import AgentView,NarrationView,ContextBudget
from ..memory import retrieve
from .guidance import ending_summary

def scene(state,actor_id,module=None):
 actor=state.actors[actor_id];world=(module or {}).get('world',{})
 locations=world.get('locations',{})
 place=locations.get(actor.location_id,{})
 return {'location_id':actor.location_id,'location':place.get('name',actor.location_id),'tick':state.tick,
  'people':[{'id':a.id,'name':a.name,'health':a.health} for a in state.actors.values() if a.location_id==actor.location_id and a.outcome=='present'],
  'inventory':[{'id':i,'name':world.get('items',{}).get(i,{}).get('name',i)} for i,owner in state.items.items() if owner==actor_id],
  'public_conditions':{'phase':state.phase,'main_loop_restored':bool(state.flags.get('main_loop_restored')),'engine_destroyed':bool(state.flags.get('engine_destroyed')),'pod_prepared':bool(state.flags.get('pod_prepared'))}}

def disclosure_options(state,actor_id,module):
 """Compute authority from this actor's received observations, never player inventory."""
 world=module['world'];profile=world['actors'].get(actor_id,{})
 observations=[o for o in state.observations if o.observer_id==actor_id]
 messages=[o for o in observations if o.source_actor_id=='player' and o.modality=='heard']
 text=' '.join(o.content for o in messages)[-5000:]
 own=set(state.evidence.get(actor_id,[]))
 def mentions(words):return any(word in text for word in words)
 def full_record_promise(o):
  return (o.channel=='private' and not re.search(r'不(?:会|愿|能)?(?:承诺|保证|保留)|不保留',o.content)
   and bool(re.search(r'我(?:会|愿意|承诺|保证)',o.content))
   and any(x in o.content for x in ('完整','全部'))
   and any(x in o.content for x in ('保留','保存','记录')) and '记录' in o.content)
 timed=mentions(('12','十二'))
 gates={
  'reasonable_connection_question':mentions(('改接','接入','临时管路','工单')),
  'captain_hold_corroborated':'E02' in own and ('E06' in own or state.flags.get('guard_witness:'+actor_id,False)),
  'private_record_promise_and_cooperation':any(full_record_promise(o) for o in messages) and (state.relations.get(actor_id,0)>=1 or state.flags.get('cooperation_with:'+actor_id,False)),
  'cargo_cooling_or_transfer_question':mentions(('保冷','货物','细胞','内盒','转存','移动','搬运','补给')),
  'reasonable_repair_question':mentions(('改接','工单','维修','修复','阀','循环','保护','故障')),
  'private_timed_query_with_source_promise_or_audit':('E02' in own and mentions(('互证','核对','印证','对照','记录'))) or any(o.channel=='private' and any(t in o.content for t in ('12','十二')) and any(t in o.content for t in ('出处','来源')) and any(t in o.content for t in ('保留','保存','承诺','保证')) for o in messages),
  'reasonable_route_question':mentions(('航路','维修站','求救','救援','信标','多久','到达')),
  'panel_and_work_order_or_hold_time_question':{'E01','E03'}<=own or (timed and mentions(('操作','刷卡','保持','动过','发生'))),
  'mechanic_only_access_accusation':'只有' in text and mentions(('机械师','顾成')) and mentions(('碰','操作','系统','动过')),
 }
 rules=profile.get('disclosure_rules',[])
 allowed_facts=set();allowed_evidence=set(own)
 restricted={p for rule in rules for p in rule.get('proposition_ids',[])}
 for rule in rules:
  if gates.get(rule['guard_id'],False):
   allowed_facts.update(rule.get('proposition_ids',[]));allowed_evidence.update(rule.get('evidence_ids',[]))
 # A permitted card also permits its individually sourced propositions. The
 # card supplies scope, never personal knowledge or unrelated private facts.
 observed_props={p for o in observations for p in o.proposition_ids}
 card_props={p for eid in allowed_evidence if eid in world['evidence']
  for p in world['evidence'][eid]['proposition_ids']}
 allowed_facts.update(card_props & observed_props)
 # Received, sourced statements may be relayed; initial private memories still
 # require their own disclosure rule. A relay never becomes a new eyewitness.
 for o in observations:
  if o.modality in ('read','reported'):
   allowed_facts.update(p for p in o.proposition_ids if p not in restricted)
 result={}
 for pid in sorted(allowed_facts):
  sources=[o for o in observations if pid in o.proposition_ids]
  if pid not in world['facts'] or not sources:continue
  direct=next((o for o in sources if o.modality=='direct'),None)
  source=direct or sources[-1]
  result[pid]={'id':pid,'kind':'proposition','text':world['facts'][pid]['text'],
   'source_kind':'firsthand' if direct else 'testimony','source_observation_ids':[source.observation_id]}
 for eid in sorted(allowed_evidence):
  if eid not in world['evidence']:continue
  evidence=world['evidence'][eid]
  received=[o for o in observations if eid in o.evidence_refs and o.modality in ('read','reported')]
  source=next((s for s in evidence['sources'] if s.get('actor_id')==actor_id),None)
  kind=(received[-1].source_kind or ('record' if received[-1].modality=='read' else 'testimony')) if received else ('testimony' if source and source['kind']=='testimony' else 'record')
  result[eid]={'id':eid,'kind':'evidence','text':evidence['text'],'source_kind':kind,
   'source_observation_ids':[received[-1].observation_id] if received else [o.observation_id for o in observations if set(o.proposition_ids)&set(evidence['proposition_ids'])]}
 return result

def disclosure(state,actor_id,module):
 return sorted(disclosure_options(state,actor_id,module))

def build_agent_view(state,actor_id,budget=None,module=None,authorization=None,status_report=False,reply_to_player=False,current_event_ids=None):
 if actor_id not in state.actors:raise ValueError('未知角色')
 if module is None:
  from ..modules import load_module
  module=load_module()
 profile=module['world']['actors'].get(actor_id,{})
 choices=disclosure_options(state,actor_id,module);available=set(choices)
 actions=[{'kind':x} for x in ['SPEAK','WITHHOLD','WAIT','REQUEST_HELP']]
 if authorization and authorization.get('actor_id')==actor_id:
  actions.append({'kind':'MODULE_ACTION','module_action_id':authorization['module_action_id'],'item_ids':authorization.get('item_ids',[]),'target_ids':authorization.get('target_ids',[])})
 observations=retrieve(state,actor_id,limit=16)
 ownids={o['observation_id'] for o in observations}
 beliefs=[b.model_dump() for b in state.beliefs if b.actor_id==actor_id and set(b.source_observation_ids)<=ownids][-12:]
 goal_choices=[{'id':f'{actor_id}:goal:{i}','text':goal} for i,goal in enumerate(state.actors[actor_id].goals)]
 current_goal=state.actors[actor_id].current_goal_id or (goal_choices[0]['id'] if goal_choices else None)
 view=AgentView(actor_id=actor_id,name=state.actors[actor_id].name,role=profile.get('role','人物'),style=profile.get('speaking_style','简短具体'),goals=state.actors[actor_id].goals,
  scene=scene(state,actor_id,module),observations=observations,beliefs=beliefs,private_background=profile.get('private_background',[]),
  disclosure_rules=[{'evidence_id':e,'allowed':e in available} for e in module['world']['evidence']],allowed_actions=actions,
  available_evidence=[{'id':e,'title':module['world']['evidence'][e]['title'],'text':module['world']['evidence'][e]['text']} for e in sorted(available) if e in module['world']['evidence']],
  available_disclosures=[{k:v for k,v in row.items() if k in ('id','kind','source_kind') or (k=='text' and row['kind']=='proposition')} for row in choices.values()],
  available_goals=goal_choices,current_goal_id=current_goal)
 if current_event_ids is not None:
  view.scene['current_player_observation_ids']=[o['observation_id'] for o in view.observations if o['event_id'] in current_event_ids and o.get('source_actor_id')=='player' and o.get('modality')=='heard']
  reply_to_player=reply_to_player and bool(view.scene['current_player_observation_ids'])
  status_report=status_report and bool(view.scene['current_player_observation_ids'])
 if reply_to_player:view.scene['reply_to_player']=True
 if status_report:
  # A routine radio check needs current surroundings, not every private
  # historical memory. Filter whole sourced rows; never rewrite their content
  # or erase character memory in the persisted state. Specific conversations
  # still use the ordinary full actor-local view and disclosure gates.
  view.scene['reply_focus']='status_report'
  view.observations=[o for o in view.observations if set(o.get('proposition_ids',[]))<=available
   and not (o.get('modality')=='heard' and o.get('source_actor_id') not in ('player',actor_id))]
  view.private_background=[];view.beliefs=[];view.goals=[]
  view.available_goals=[];view.current_goal_id=None
 return view

def build_narration_view(state,events,player_id='player',module=None):
 ids={e.event_id for e in events}
 public=[{'event_id':o.event_id,'speaker_id':o.source_actor_id or 'system','speaker':state.actors.get(o.source_actor_id).name if o.source_actor_id in state.actors else '船上系统','text':o.content,'kind':o.modality} for o in state.observations if o.observer_id==player_id and o.event_id in ids]
 return NarrationView(scene=scene(state,player_id,module),events=public)

def player_view(state,module,active_turn_id=None):
 world=module['world'];player=state.actors['player'];sc=scene(state,'player',module)
 known=set(state.evidence.get('player',[]))
 # UI affordances are basic communication/perception, never a solution path
 # inferred from unobserved flags, cargo expiry or author-only prerequisites.
 suggestions=['查看这里有什么可以调查的','使用工作频道询问当前情况'] if not state.outcome else []
 # A suggestion is a complete editable command, including its communication
 # scope. Keep labels for old saves/clients; never infer routing from arbitrary
 # player prose in the browser (which could expose a private conversation).
 suggestion_commands={}
 for text in suggestions:
  channel='local';recipients=[]
  if text in ('使用工作频道询问当前情况','向大家说明已掌握的证据'):channel='broadcast'
  elif text in ('询问医生如何保冷和转存','向方榆出示补给票E07'):channel='private';recipients=['doctor']
  else:
   for actor,profile in world['actors'].items():
    if text.startswith(profile['name']+'，请执行'):channel='private';recipients=[actor];break
  suggestion_commands[text]={'text':text,'mode':'in_character','speech_channel':channel,'recipient_ids':recipients}
 return {'campaign_id':state.campaign_id,'revision':state.revision,'module_title':module['public']['title'],
  'player':{'id':'player','name':player.name,'role':module['public']['player']['role'],'skills':player.skills,'location_id':player.location_id,'location':sc['location'],'health':player.health,'inventory':sc['inventory']},
  'tick':state.tick,'phase':state.phase,'clocks':([{'label':'主回路热负荷','value':state.tick,'limit':12}] if state.phase=='crisis' else [{'label':'撤离期限','value':state.tick,'limit':15}] if state.phase=='evacuation' else []),
  'locations':[{'id':k,'name':v['name']} for k,v in world['locations'].items()],
  'npcs':[{'id':id,'name':a.name,'role':world['actors'].get(id,{}).get('role',''), 'location_id':a.location_id if a.location_id==player.location_id and a.outcome=='present' else None,'present':a.location_id==player.location_id and a.outcome=='present'} for id,a in state.actors.items() if id!='player'],
  'evidence':[{'id':e,'title':world['evidence'][e]['title'],'text':world['evidence'][e]['text'],'source':next((o.source_actor_id or '记录' for o in reversed(state.observations) if o.observer_id=='player' and e in o.evidence_refs),'记录')} for e in sorted(known) if e in world['evidence']],
  'suggestions':suggestions,'suggestion_commands':suggestion_commands,'ending_summary':ending_summary(state),'transcript':state.transcript,'outcome':state.outcome,'active_turn_id':active_turn_id,'mock':state.mock}
