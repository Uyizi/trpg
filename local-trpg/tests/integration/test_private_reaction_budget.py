import pytest
from local_trpg.modules import load_module
from local_trpg.storage import Store
from local_trpg.engine.rules import initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.schemas import PlayerCommand,PlayerIntent,NpcDecision,NarrationResult

@pytest.mark.asyncio
async def test_private_single_recipient_defers_unrelated_pending_dialogue(tmp_path):
 module=load_module();state=initial_state(module,'预算');state.pending_npcs={'captain':10,'guard':8}
 store=Store(tmp_path/'s.db');store.create(state);seen=[]
 class Agent:
  calls=repairs=0;metrics=[]
  async def interpret(self,v):return PlayerIntent(kind='ASK',target_ids=['doctor'],spoken_text='请说明细胞保冷条件。')
  async def decide_npc(self,v):seen.append(v.actor_id);return NpcDecision(actor_id=v.actor_id)
  async def narrate(self,v):return NarrationResult(text='本轮通话已记录。',source_event_ids=[v.events[0]['event_id']])
 service=TurnService(store,module,None,agent_factory=lambda *a:Agent())
 receipt=await service.submit_turn(state.campaign_id,PlayerCommand(client_request_id='p',expected_revision=0,text='请说明细胞保冷条件。',speech_channel='private',recipient_ids=['doctor']))
 await service.tasks[receipt.turn_id]
 assert seen==['doctor']
 assert set(store.snapshot(state.campaign_id).pending_npcs)>={'captain','guard'}
