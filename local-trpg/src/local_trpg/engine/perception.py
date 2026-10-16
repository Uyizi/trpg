"""Whitelist observations at the instant the event occurs, never forward payloads."""
from ..schemas import Observation

# Public vocabulary only: perception does not load author truths or NPC profiles.
PLACES={'bridge':'驾驶舱','common':'生活舱','clinic':'医务间','engine':'动力舱','cargo':'货舱','lifeboat':'气闸与救生艇','pod':'救生艇内'}
HEALTH_NAMES={'healthy':'正常','strained':'轻伤','injured':'受伤','incapacitated':'失能'}
ITEM_NAMES={'multitool':'多用工具','recorder':'记录终端','face_mask':'面罩','work_card':'检修工作卡','ordinary_tools':'普通工具','original_cargo_box':'原货箱'}

def item_label(identifier):
 if identifier in ITEM_NAMES:return ITEM_NAMES[identifier]
 for prefix,label in (('medical_supply_','普通处置物资'),('cold_case_','保冷箱'),('cold_core_','储冷芯'),('cell_','细胞内盒')):
  if identifier.startswith(prefix):return label+identifier[len(prefix):]
 return '物品'

def owner_label(state,identifier):
 if identifier in state.actors:return state.actors[identifier].name
 return PLACES.get(identifier,{'consumed':'已使用物资区','ship':'船上'}.get(identifier,'存放处'))

def perceive(state,event):
 p=event.payload;kind=event.kind;recipients=[];content='';modality='direct';refs=[];props=[];source_kind=None;source_ids=[];channel=None
 local=[a.id for a in state.actors.values() if a.location_id==event.location_id and a.outcome=='present' and a.health!='incapacitated']
 if kind=='SpeechOccurred':
  modality='heard';content=p['text'];refs=p.get('evidence_refs',[]);channel=p['channel']
  recipients=local if p['channel']=='local' else list(dict.fromkeys([event.actor_id]+p['recipient_ids']))
 elif kind=='ActorMoved':
  arrivals=[a.id for a in state.actors.values() if a.location_id==p['destination'] and a.outcome=='present' and a.health!='incapacitated']
  recipients=list(set(local+arrivals+[p['actor_id']]));content=f"{state.actors[p['actor_id']].name}前往{p.get('destination_name') or PLACES.get(p['destination'],'目的地')}。"
 elif kind=='ItemTransferred':
  recipients=local;content=f"{item_label(p['item_id'])}交给{owner_label(state,p['to_owner'])}。"
 elif kind=='EvidenceObserved':
  recipients=p['observer_ids'];source_kind=p.get('source_kind','record');source_ids=p.get('source_observation_ids',[])
  modality='reported' if source_kind=='testimony' else 'read';content=p['content'];refs=[p['evidence_id']];props=p['proposition_ids']
 elif kind=='FactDisclosed':
  recipients=p['observer_ids'];modality='reported';content=p['content'];props=p['proposition_ids']
  source_kind=p['source_kind'];source_ids=p['source_observation_ids']
 elif kind=='PublicNotice':recipients=p['recipient_ids'];content=p['text']
 elif kind=='HealthChanged':
  recipients=list(set(local+[p['actor_id']]));content=f"{state.actors[p['actor_id']].name}的伤势变为{HEALTH_NAMES[p['health']]}。"
 elif kind=='CheckResolved':recipients=['player'];content=p['description']
 else:return []
 return [Observation(observation_id=f'{event.event_id}:{a}',event_id=event.event_id,observer_id=a,modality=modality,content=content,
   source_actor_id=event.actor_id,channel=channel,source_kind=source_kind,source_observation_ids=source_ids,
   evidence_refs=refs,proposition_ids=props,observed_at=event.game_tick) for a in sorted(set(recipients)) if a in state.actors]
