import unittest,tempfile
from pathlib import Path
from local_trpg.schemas import GameSnapshot,Actor,GameEvent,PlayerCommand
from local_trpg.engine.reducer import apply_event
from local_trpg.engine.perception import perceive
from local_trpg.engine.projection import build_agent_view
from local_trpg.storage import Store,Conflict


def minimal():
 return GameSnapshot(campaign_id='c',module_hash='test',actors={x:Actor(id=x,name=x,location_id='common') for x in ['player','a','b']},items={'key':'player'})

def event(kind,payload,actor='a',beat=0):
 return GameEvent(event_id=f'e{beat}',turn_id='t',beat_index=beat,actor_id=actor,kind=kind,payload=payload,location_id='common',game_tick=0)

class Core(unittest.TestCase):
 def test_lie_does_not_move_key_and_movement_changes_audience(self):
  s=minimal();e=event('SpeechOccurred',{'text':'钥匙在别处','channel':'local','recipient_ids':[]})
  obs=perceive(s,e);self.assertEqual({o.observer_id for o in obs},{'player','a','b'})
  moved=apply_event(s,event('ActorMoved',{'actor_id':'b','destination':'bridge'},beat=1))
  self.assertNotIn('b',{o.observer_id for o in perceive(moved,e)})
  self.assertEqual(moved.items['key'],'player');self.assertEqual(s.actors['b'].location_id,'common')
 def test_unknown_event_payload_rejected(self):
  with self.assertRaises(ValueError):event('SpeechOccurred',{'text':'x','channel':'local','recipient_ids':[],'private_intent':'secret'})
 def test_storage_duplicate_cancel_and_save_isolation(self):
  with tempfile.TemporaryDirectory() as d:
   st=Store(Path(d)/'game.db');s=minimal();st.create(s)
   cmd=PlayerCommand(client_request_id='request1',expected_revision=0,text='等待')
   first=st.submit('c',cmd);self.assertEqual(first,st.submit('c',cmd))
   changed=cmd.model_copy(update={'text':'走'})
   with self.assertRaises(Conflict):st.submit('c',changed)
   turn=st.claim(first);self.assertEqual(st.save('c','before')['revision'],0)
   st.cancel('c',first)
   with self.assertRaises(Conflict):st.commit(first,turn['epoch'],s,[],{'narration':'x','dialogue':[],'checks':[],'degraded':False})
   self.assertEqual(st.snapshot('c').revision,0)
 def test_roll_persists_and_commit_increments_once(self):
  with tempfile.TemporaryDirectory() as d:
   st=Store(Path(d)/'game.db');s=minimal();st.create(s)
   cmd=PlayerCommand(client_request_id='request1',expected_revision=0,text='等')
   tid=st.submit('c',cmd);t=st.claim(tid);r=st.roll(tid,'player:0:social')
   self.assertEqual(r,st.roll(tid,'player:0:social'))
   st.commit(tid,t['epoch'],s,[],{'narration':'完成','dialogue':[],'checks':[],'degraded':False})
   self.assertEqual(st.snapshot('c').revision,1);self.assertEqual(st.submit('c',cmd),tid)
