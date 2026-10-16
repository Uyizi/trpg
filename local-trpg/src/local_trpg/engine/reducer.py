from ..schemas import BeliefRecord
from .perception import perceive

def apply_event(state,event):
 new=state.model_copy(deep=True);p=event.payload;k=event.kind
 if k=='FactDisclosed':
  own={o.observation_id:o for o in state.observations if o.observer_id==event.actor_id}
  if not p['source_observation_ids'] or not set(p['source_observation_ids'])<=set(own):raise ValueError('事实披露引用越权')
  sources=[own[oid] for oid in p['source_observation_ids']]
  supported={prop for o in sources for prop in o.proposition_ids}
  if not set(p['proposition_ids'])<=supported or not set(p['proposition_ids'])<=set(state.world_facts):raise ValueError('事实披露缺少命题来源')
  if p['source_kind']=='firsthand' and any(o.modality!='direct' for o in sources):raise ValueError('转述不能伪装为亲历')
 observations=perceive(state,event)
 if k=='ActorMoved':new.actors[p['actor_id']].location_id=p['destination']
 elif k=='ItemTransferred':
  if new.items.get(p['item_id'])!=p['from_owner']:raise ValueError('物品所有权不一致')
  new.items[p['item_id']]=p['to_owner']
  if event.actor_id=='player' and p['from_owner']=='player' and p['to_owner'] in new.relations:
   new.flags['cooperation_with:'+p['to_owner']]=True
 elif k=='HealthChanged':new.actors[p['actor_id']].health=p['health']
 elif k=='ClockAdvanced':
  if p['tick']<new.tick:raise ValueError('时钟不能倒退')
  new.tick=p['tick']
  if p.get('phase'):new.phase=p['phase']
 elif k=='FlagSet':
  new.flags[p['key']]=p['value']
  if p['key'].startswith('onboard:'):new.actors[p['key'].split(':',1)[1]].onboard=bool(p['value'])
  if p['key'].startswith('lost:'):new.actors[p['key'].split(':',1)[1]].outcome='lost'
 elif k=='RelationChanged':new.relations[p['actor_id']]=p['value']
 elif k=='CellsChanged':new.cells=p['cells']
 elif k=='TriggerFired':
  if p['trigger_id'] in new.fired_triggers:raise ValueError('触发器重复')
  new.fired_triggers.append(p['trigger_id'])
 elif k=='ScenarioResolved':new.outcome=p['outcome'];new.phase='epilogue'
 elif k=='PlansUpdated':new.pending_npcs=p['pending_npcs']
 elif k=='NpcGoalChanged':
  aid=p['actor_id']
  if aid!=event.actor_id or aid not in new.actors or p['goal_id'] not in {f'{aid}:goal:{i}' for i in range(len(new.actors[aid].goals))}:raise ValueError('非法人物目标')
  new.actors[aid].current_goal_id=p['goal_id']
 elif k=='BeliefRevised':
  b=BeliefRecord.model_validate(p)
  own={o.observation_id for o in new.observations if o.observer_id==b.actor_id}
  if not set(b.source_observation_ids)<=own:raise ValueError('信念引用越权')
  new.beliefs=[x for x in new.beliefs if (x.actor_id,x.proposition_id)!=(b.actor_id,b.proposition_id)]+[b]
 new.observations.extend(observations)
 for o in observations:
  if k=='FactDisclosed' and event.actor_id=='guard' and p['source_kind']=='firsthand' and {'captain_enabled_hold','guard_witnessed_hold'}&set(o.proposition_ids):
   new.flags['guard_witness:'+o.observer_id]=True
   if o.observer_id=='player':new.flags['guard_witness']=True
  for ref in o.evidence_refs:
   if ref not in new.evidence.setdefault(o.observer_id,[]):new.evidence[o.observer_id].append(ref)
  for prop in o.proposition_ids:
   b=BeliefRecord(actor_id=o.observer_id,proposition_id=prop,stance='accepted',source_observation_ids=[o.observation_id],updated_at=new.tick,basis='testimony' if o.modality=='reported' else 'observation')
   new.beliefs=[x for x in new.beliefs if (x.actor_id,x.proposition_id)!=(b.actor_id,prop)]+[b]
 for a in new.actors.values():a.inventory=sorted(i for i,owner in new.items.items() if owner==a.id)
 return new
