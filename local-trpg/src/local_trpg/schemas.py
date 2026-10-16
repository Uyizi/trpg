from __future__ import annotations
from typing import Literal,Any
from pydantic import BaseModel,ConfigDict,Field,model_validator

class Strict(BaseModel):
 model_config=ConfigDict(extra='forbid',validate_assignment=True)

Health=Literal['healthy','strained','injured','incapacitated']
Channel=Literal['local','private','broadcast']
ActionKind=Literal['TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE','INSPECT','MOVE','TAKE','GIVE','USE_ITEM','REPAIR','WAIT','CONFLICT','IMPROVISE']
class Actor(Strict):
 id:str;name:str;location_id:str
 health:Health='healthy'
 skills:dict[str,int]=Field(default_factory=dict)
 inventory:list[str]=Field(default_factory=list)
 goals:list[str]=Field(default_factory=list)
 current_goal_id:str|None=None
 busy:bool=False
 onboard:bool=False
 outcome:Literal['present','saved','lost']='present'
class Observation(Strict):
 observation_id:str;event_id:str;observer_id:str
 modality:Literal['direct','heard','reported','read']
 content:str
 source_actor_id:str|None=None
 channel:Channel|None=None
 source_kind:Literal['record','firsthand','testimony']|None=None
 source_observation_ids:list[str]=Field(default_factory=list)
 evidence_refs:list[str]=Field(default_factory=list)
 proposition_ids:list[str]=Field(default_factory=list)
 observed_at:int
class BeliefRecord(Strict):
 actor_id:str;proposition_id:str
 stance:Literal['unknown','accepted','doubted','rejected']
 source_observation_ids:list[str]
 updated_at:int
 basis:Literal['observation','testimony','inference']
class GameSnapshot(Strict):
 schema_version:Literal[1]=1
 campaign_id:str;revision:int=0
 module_id:str='reserve_cooling';module_version:str='1.0.0';module_hash:str
 tick:int=0
 phase:Literal['crisis','stable','evacuation','epilogue','ended']='crisis'
 actors:dict[str,Actor]
 items:dict[str,str]=Field(default_factory=dict)
 world_facts:dict[str,bool]=Field(default_factory=dict)
 observations:list[Observation]=Field(default_factory=list)
 beliefs:list[BeliefRecord]=Field(default_factory=list)
 relations:dict[str,int]=Field(default_factory=dict)
 flags:dict[str,Any]=Field(default_factory=dict)
 clocks:dict[str,int|None]=Field(default_factory=dict)
 cells:list[dict]=Field(default_factory=list)
 fired_triggers:list[str]=Field(default_factory=list)
 evidence:dict[str,list[str]]=Field(default_factory=dict)
 transcript:list[dict]=Field(default_factory=list)
 pending_npcs:dict[str,int]=Field(default_factory=dict)
 outcome:dict|None=None
 mock:bool=False
class PlayerCommand(Strict):
 client_request_id:str=Field(min_length=1,max_length=100)
 expected_revision:int=Field(ge=0)
 mode:Literal['in_character','ooc']='in_character'
 text:str=Field(default='',max_length=6000)
 speech_channel:Channel='local'
 recipient_ids:list[str]=Field(default_factory=list,max_length=5)
 private_aside:str=Field(default='',max_length=3000)
 reply_to_turn_id:str|None=None
class PlayerIntent(Strict):
 actor_id:Literal['player']='player'
 kind:ActionKind|None=None
 target_ids:list[str]=Field(default_factory=list)
 attempt_description:str=''
 spoken_text:str=''
 speech_channel:Channel='local'
 recipient_ids:list[str]=Field(default_factory=list)
 private_intent:str=''
 evidence_refs:list[str]=Field(default_factory=list)
 needs_clarification:bool=False
 clarification_question:str=''
 module_action_id:str|None=None
 item_ids:list[str]=Field(default_factory=list)
class ActionIntent(Strict):
 actor_id:str
 kind:ActionKind
 target_ids:list[str]=Field(default_factory=list)
 description:str=''
 spoken_text:str=''
 channel:Channel='local'
 recipient_ids:list[str]=Field(default_factory=list)
 evidence_refs:list[str]=Field(default_factory=list)
 module_action_id:str|None=None
 item_ids:list[str]=Field(default_factory=list)
class NpcAction(Strict):
 kind:Literal['SPEAK','MOVE','OFFER_ITEM','REQUEST_HELP','WITHHOLD','WAIT','MODULE_ACTION']='SPEAK'
 target_ids:list[str]=Field(default_factory=list)
 description:str=''
 module_action_id:str|None=None
 item_ids:list[str]=Field(default_factory=list)
class Utterance(Strict):
 channel:Channel='local'
 recipient_ids:list[str]=Field(default_factory=lambda:['player'])
 text:str=Field(max_length=800)
 claim_refs:list[str]=Field(default_factory=list)
 disclose_fact_ids:list[str]=Field(default_factory=list)
class BeliefUpdate(Strict):
 proposition_id:str;stance:Literal['unknown','accepted','doubted','rejected']
 source_observation_ids:list[str]
 basis:Literal['observation','testimony','inference']='inference'
class NpcDecision(Strict):
 actor_id:str
 action:NpcAction=Field(default_factory=NpcAction)
 utterance:Utterance|None=None
 belief_updates:list[BeliefUpdate]=Field(default_factory=list,max_length=4)
 next_goal_id:str|None=None
class ContextBudget(Strict):
 input_limit:int=4096;output_limit:int=384;reserve_tokens:int=256
class InterpreterView(Strict):
 text:str;channel:Channel;recipient_ids:list[str]
 scene:dict;known_evidence:list[dict];legal_actions:list[dict]
 clarification:str|None=None
class AgentView(Strict):
 actor_id:str;name:str;role:str;style:str;goals:list[str]
 scene:dict
 observations:list[dict];beliefs:list[dict]
 private_background:list[dict];disclosure_rules:list[dict]
 allowed_actions:list[dict];available_evidence:list[dict]
 available_disclosures:list[dict]=Field(default_factory=list)
 available_goals:list[dict]=Field(default_factory=list)
 current_goal_id:str|None=None
class NarrationView(Strict):
 scene:dict;events:list[dict]
class GuideView(Strict):
 question:str;location:str;tick:int;sources:list[dict]
 history:list[dict]=Field(default_factory=list)
class GuideReviewView(Strict):
 question:str;sources:list[dict];answer:str
 speaker_name:str|None=None
class GuideReviewResult(Strict):
 supported:bool;unsupported_claims:list[str]=Field(default_factory=list,max_length=3)
class AdventureIntent(Strict):
 kind:Literal['look','move','inspect','talk','act','attack','ability','retreat','wait','clarify']
 target_id:str=''
 target_ids:list[str]=Field(default_factory=list,max_length=3)
 method:str=''
 skill:str=''
 item_id:str=''
 attack_id:str=''
 ability_id:str=''
 helper_id:str=''
 words:str=''
 clarification:str=''
class AdventureView(Strict):
 text:str;scene:dict;player:dict;party:list[dict];legal_actions:list[dict]
 npcs:list[dict];known_evidence:list[dict]
 clarification:str|None=None
class DirectorView(Strict):
 action:dict;scene:dict;available_actions:list[dict];private_aside:str=''
class SceneProposal(Strict):
 module_action_id:str|None=None
 target_ids:list[str]=Field(default_factory=list)
 clarification:str=''
class NarrationResult(Strict):
 text:str=Field(max_length=1800);source_event_ids:list[str]
class SpeechPayload(Strict):
 text:str;channel:Channel;recipient_ids:list[str]
 evidence_refs:list[str]=Field(default_factory=list)
class MovePayload(Strict):
 actor_id:str;destination:str
 destination_name:str|None=None
class ItemPayload(Strict):
 item_id:str;from_owner:str;to_owner:str
class EvidencePayload(Strict):
 observer_ids:list[str];evidence_id:str;content:str;proposition_ids:list[str];source:str
 source_kind:Literal['record','testimony']='record'
 source_observation_ids:list[str]=Field(default_factory=list)
class FactDisclosurePayload(Strict):
 observer_ids:list[str]
 proposition_ids:list[str]=Field(min_length=1)
 content:str
 source_kind:Literal['firsthand','testimony']
 source_observation_ids:list[str]=Field(min_length=1)
class NpcGoalPayload(Strict):
 actor_id:str;goal_id:str
class HealthPayload(Strict):
 actor_id:str;health:Health
class ClockPayload(Strict):
 tick:int;phase:str|None=None
class FlagPayload(Strict):
 key:str;value:Any
class RelationPayload(Strict):
 actor_id:str;value:int=Field(ge=-2,le=2)
class CheckPayload(Strict):
 dice:list[int];skill:int;modifier:int;total:int;outcome:Literal['success','success_with_cost','failure'];description:str
 die_sides:int=6
 difficulty:int|None=None
class PublicPayload(Strict):
 text:str;recipient_ids:list[str]
class CellsPayload(Strict):
 cells:list[dict]
class TriggerPayload(Strict):
 trigger_id:str
class OutcomePayload(Strict):
 outcome:dict
class PlanPayload(Strict):
 pending_npcs:dict[str,int]
PAYLOADS={'SpeechOccurred':SpeechPayload,'ActorMoved':MovePayload,'ItemTransferred':ItemPayload,'EvidenceObserved':EvidencePayload,
 'FactDisclosed':FactDisclosurePayload,'NpcGoalChanged':NpcGoalPayload,
 'HealthChanged':HealthPayload,'ClockAdvanced':ClockPayload,'FlagSet':FlagPayload,'RelationChanged':RelationPayload,
 'CheckResolved':CheckPayload,'PublicNotice':PublicPayload,'CellsChanged':CellsPayload,'TriggerFired':TriggerPayload,
 'ScenarioResolved':OutcomePayload,'BeliefRevised':BeliefRecord,'PlansUpdated':PlanPayload}
class GameEvent(Strict):
 event_id:str;turn_id:str;beat_index:int;actor_id:str
 kind:str;payload:dict;location_id:str;audience:list[str]=Field(default_factory=list)
 caused_by:str|None=None;game_tick:int
 @model_validator(mode='after')
 def valid_payload(self):
  if self.kind not in PAYLOADS:raise ValueError('未知事件类型')
  PAYLOADS[self.kind].model_validate(self.payload)
  return self
class RuleResult(Strict):
 accepted:bool;events:list[GameEvent]=Field(default_factory=list)
 check_result:dict|None=None;reason_code:str='ok'
class TurnDraft(Strict):
 turn_id:str;base_revision:int;epoch:int;state:GameSnapshot
 events:list[GameEvent]=Field(default_factory=list)
class TurnReceipt(Strict):
 turn_id:str;status:str
class PlayerTurnResult(Strict):
 narration:str;dialogue:list[dict];checks:list[dict];degraded:bool;view:dict
 rule_log:list[str]=Field(default_factory=list)
