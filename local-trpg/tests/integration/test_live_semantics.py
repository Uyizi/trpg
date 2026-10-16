"""Regressions observed with the actual local Qwen runner."""
import pytest
from local_trpg.modules import load_module
from local_trpg.storage import Store
from local_trpg.engine.rules import initial_state, Engine, RuleError
from local_trpg.engine.turns import TurnService
from local_trpg.schemas import ActionIntent, PlayerCommand, PlayerIntent, NpcDecision

def test_station_choice_does_not_require_a_fictitious_npc_authorization():
 module=load_module();state=initial_state(module,'航线')
 engine=Engine(state,module,'route',lambda k:[3,4])
 action=ActionIntent(actor_id='player',kind='ASK',module_action_id='choose_station',spoken_text='我选择前往最近维修站。')
 with pytest.raises(RuleError,match='主循环'):
  engine.resolve_player(action)
 engine.flag('main_loop_restored',True)
 engine.resolve_player(action)
 assert engine.state.outcome['ship_saved'] is True
 assert engine.state.outcome['arrival_tick']==5

@pytest.mark.asyncio
async def test_dialogue_is_shown_verbatim_without_a_second_narrator_paraphrase(tmp_path):
 module=load_module();state=initial_state(module,'通话')
 store=Store(tmp_path/'s.db');store.create(state);narrator_views=[]
 class Agent:
  calls=repairs=0;metrics=[]
  async def interpret(self,v):return PlayerIntent(kind='ASK',target_ids=['doctor'],spoken_text='你收到我的消息了吗？')
  async def decide_npc(self,v):return NpcDecision(actor_id='doctor',action={'kind':'SPEAK'},utterance={'text':'我收到你的消息了。','channel':'private','recipient_ids':['player']})
  async def narrate(self,v):narrator_views.append(v);raise AssertionError('Only dialogue occurred; do not paraphrase it.')
 service=TurnService(store,module,None,agent_factory=lambda *a:Agent())
 receipt=await service.submit_turn(state.campaign_id,PlayerCommand(client_request_id='p',expected_revision=0,text='你收到我的消息了吗？',speech_channel='private',recipient_ids=['doctor']))
 await service.tasks[receipt.turn_id]
 result=service.status(state.campaign_id,receipt.turn_id)['result']
 assert result['dialogue']==[{'speaker':'方榆','speaker_id':'doctor','text':'我收到你的消息了。'}]
 assert not narrator_views
 assert result['degraded'] is False

def test_public_station_suggestion_uses_rules_not_model_navigation_guess():
 module=load_module();service=TurnService(None,module,None)
 intent=service.shortcut(PlayerCommand(client_request_id='station',expected_revision=0,text='选择前往最近维修站'))
 assert intent.kind=='USE_ITEM' and intent.module_action_id=='choose_station'
 engine=Engine(initial_state(module,'未修复'),module,'route',lambda k:[3,4])
 with pytest.raises(RuleError,match='主循环'):
  engine.resolve_player(ActionIntent(actor_id='player',kind=intent.kind,module_action_id=intent.module_action_id))
 assert engine.state.tick==0 and engine.state.outcome is None

def test_exact_public_work_request_still_requires_npc_acceptance_and_authority():
 module=load_module();service=TurnService(None,module,None)
 intent=service.shortcut(PlayerCommand(client_request_id='work',expected_revision=0,text='方榆，请执行转存细胞。'))
 assert intent.kind=='ASK' and intent.module_action_id=='transfer_cells' and intent.target_ids==['doctor']
 engine=Engine(initial_state(module,'请求'),module,'request',lambda k:[3,4])
 # The player starts alone in the common room; this public request uses radio.
 engine.resolve_player(ActionIntent(actor_id='player',kind=intent.kind,module_action_id=intent.module_action_id,target_ids=intent.target_ids,spoken_text=intent.spoken_text,channel='broadcast'))
 assert engine.authorization['actor_id']=='doctor'
 assert not engine.state.flags.get('cells_transferred')
 for text in ('方榆，请执行恢复主循环','我没有请方榆执行转存细胞','方榆，请执行转存细胞然后销毁记录'):
  assert service.shortcut(PlayerCommand(client_request_id='no',expected_revision=0,text=text)) is None
