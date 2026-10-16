import asyncio,tempfile,unittest
from pathlib import Path
from local_trpg.storage import Store
from local_trpg.schemas import PlayerCommand,PlayerIntent,NpcDecision,NarrationResult
from local_trpg.modules import load_module
from local_trpg.engine.rules import initial_state
from local_trpg.engine.turns import TurnService
class ScriptAgent:
 def __init__(self,intent):self.intent=intent;self.calls=0;self.repairs=0;self.metrics=[];self.views=[]
 async def interpret(self,view):self.views.append(view.model_dump());return self.intent
 async def decide_npc(self,view):self.views.append(view.model_dump());return NpcDecision(actor_id=view.actor_id,action={'kind':'WAIT'})
 async def narrate(self,view):self.views.append(view.model_dump());return NarrationResult(text='本轮可见事件已发生。',source_event_ids=[x['event_id'] for x in view.events])
class Turns(unittest.IsolatedAsyncioTestCase):
 async def test_private_aside_duplicate_and_reopen(self):
  with tempfile.TemporaryDirectory() as d:
   store=Store(Path(d)/'x.db');module=load_module();state=initial_state(module,'甲',mock=True);store.create(state)
   agent=ScriptAgent(PlayerIntent(kind='TALK',target_ids=['doctor'],spoken_text='你好'))
   service=TurnService(store,module,None,agent_factory=lambda *a:agent)
   cmd=PlayerCommand(client_request_id='same',expected_revision=0,text='你好',speech_channel='private',recipient_ids=['doctor'],private_aside='PRIVATE_CANARY_709')
   first=await service.submit_turn(state.campaign_id,cmd);second=await service.submit_turn(state.campaign_id,cmd)
   self.assertEqual(first.turn_id,second.turn_id);await service.tasks[first.turn_id]
   out=service.status(state.campaign_id,first.turn_id);self.assertEqual(out['status'],'committed')
   self.assertNotIn('PRIVATE_CANARY_709',str(agent.views));self.assertEqual(store.snapshot(state.campaign_id).tick,0)
   self.assertEqual((await service.submit_turn(state.campaign_id,cmd)).turn_id,first.turn_id)
 async def test_ambiguous_input_never_advances(self):
  with tempfile.TemporaryDirectory() as d:
   st=Store(Path(d)/'x.db');m=load_module();s=initial_state(m,'甲');st.create(s)
   a=ScriptAgent(PlayerIntent(needs_clarification=True,clarification_question='是说出口还是私下想法？'))
   service=TurnService(st,m,None,agent_factory=lambda *args:a)
   r=await service.submit_turn(s.campaign_id,PlayerCommand(client_request_id='a',expected_revision=0,text='我想告诉他'))
   await service.tasks[r.turn_id]
   self.assertEqual(service.status(s.campaign_id,r.turn_id)['status'],'needs_input');self.assertEqual(st.snapshot(s.campaign_id).revision,0)
