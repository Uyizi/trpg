import tempfile,json,time
from pathlib import Path
from fastapi.testclient import TestClient
from local_trpg.api import create_app
from local_trpg.config import Settings
from local_trpg.schemas import PlayerIntent,NpcDecision,NarrationResult
class Queue:
 active=0
 async def warmup(self):pass
 async def close(self):pass
class Agent:
 calls=0;repairs=0;metrics=[]
 async def interpret(self,view):return PlayerIntent(kind='INSPECT',attempt_description='查看房间这里有什么')
 async def decide_npc(self,view):return NpcDecision(actor_id=view.actor_id,action={'kind':'WAIT'})
 async def narrate(self,view):return NarrationResult(text='观察到房间里的公开物品。',source_event_ids=[x['event_id'] for x in view.events])
def test_public_api_and_save_export(tmp_path):
 app=create_app(Settings(data_dir=tmp_path,backend='mock'),queue=Queue(),agent_factory=lambda *args:Agent())
 with TestClient(app) as c:
  modules=c.get('/api/modules').json();assert {m['module_id'] for m in modules}=={'pinebrook','reserve_cooling'};assert 'facts' not in str(modules)
  new=c.post('/api/campaigns',json={'player_setup':{'name':'试玩'}}).json();id=new['campaign_id']
  assert new['view']['mock'];assert 'private_background' not in json.dumps(new)
  command={'client_request_id':'a','expected_revision':0,'text':'查看这里有什么','private_aside':'PRIVATE_CANARY_TEST'}
  receipt=c.post(f'/api/campaigns/{id}/turns',json=command).json()
  for _ in range(60):
   status=c.get(f"/api/campaigns/{id}/turns/{receipt['turn_id']}").json()
   if status['is_terminal']:break
   time.sleep(.01)
  assert status['status']=='committed',status
  assert 'PRIVATE_CANARY_TEST' not in json.dumps(status)
  saved=c.post(f'/api/campaigns/{id}/saves',json={'name':'checkpoint'}).json()
  fork=c.post(f"/api/saves/{saved['save_id']}/fork").json()
  assert fork['campaign_id']!=id;assert fork['view']['tick']==0
  share=c.get(f'/api/campaigns/{id}/transcript').text;assert 'PRIVATE_CANARY_TEST' not in share;assert 'captain_enabled_hold' not in share
  backup=c.get(f'/api/campaigns/{id}/export').json();assert 'world_facts' in backup['state'];assert '/home/' not in json.dumps(backup)
  assert c.get('/modules/reserve_cooling/world.json').status_code==404
  assert c.post('/api/campaigns',json={},headers={'Origin':'https://evil.example'}).status_code==403
  assert c.get('/api/campaigns',headers={'Host':'evil.example'}).status_code==403
