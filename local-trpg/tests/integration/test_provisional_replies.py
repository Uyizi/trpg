"""Visible, validated speech can precede commit; previews never become canon."""
import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from local_trpg.api import create_app
from local_trpg.config import Settings
from local_trpg.schemas import NpcDecision, PlayerIntent


class Queue:
 active = 0
 async def warmup(self): pass
 async def close(self): pass


@pytest.mark.parametrize('ending', ['commit', 'cancel', 'failure', 'recover'])
def test_reply_available_while_later_npc_waits_without_changing_saves(tmp_path, ending):
 waiting, release = threading.Event(), threading.Event()
 class Agent:
  calls = repairs = 0
  metrics = []
  async def interpret(self, view):
   assert 'DM_PRIVATE_CANARY' not in str(view)
   return PlayerIntent(kind='ASK', target_ids=['guard', 'doctor'], spoken_text='收到消息了吗？')
  async def decide_npc(self, view):
   self.calls += 1
   if self.calls == 2:
    waiting.set()
    while not release.is_set(): await asyncio.sleep(.002)
    if ending == 'failure': raise RuntimeError('later call failed')
   return NpcDecision(actor_id=view.actor_id, action={'kind':'SPEAK'}, utterance={'text':'VISIBLE_REPLY_'+view.actor_id, 'channel':'private', 'recipient_ids':['player']})
 app = create_app(Settings(data_dir=tmp_path, backend='mock', narration_mode='brief'), queue=Queue(), agent_factory=lambda *args: Agent())
 with TestClient(app) as client:
  campaign = client.post('/api/campaigns', json={}).json()['campaign_id']
  other = client.post('/api/campaigns', json={}).json()['campaign_id']
  command = {'client_request_id':'once', 'expected_revision':0, 'text':'收到消息了吗？', 'speech_channel':'broadcast', 'private_aside':'DM_PRIVATE_CANARY'}
  receipt = client.post(f'/api/campaigns/{campaign}/turns', json=command).json()
  turn = receipt['turn_id']; path = f'/api/campaigns/{campaign}/turns/{turn}'
  try:
   assert waiting.wait(3), 'second NPC did not start'
   status = client.get(path).json()
   assert status['status'] == 'running' and status['result'] is None
   preview = status['provisional']
   assert preview['campaign_id'] == campaign and preview['turn_id'] == turn
   assert preview['epoch'] == status['epoch'] and preview['base_revision'] == 0
   assert preview['dialogue'] == [{'event_id':preview['dialogue'][0]['event_id'], 'speaker':'韩松', 'speaker_id':'guard', 'text':'VISIBLE_REPLY_guard'}]
   assert 'DM_PRIVATE_CANARY' not in json.dumps(status)
   assert 'private_background' not in json.dumps(status) and 'world_facts' not in json.dumps(status)
   assert client.get(f'/api/campaigns/{other}/turns/{turn}').status_code == 404
   assert client.post(f'/api/campaigns/{campaign}/turns', json=command).json()['turn_id'] == turn
   assert client.get(f'/api/campaigns/{campaign}').json()['revision'] == 0
   saved = client.post(f'/api/campaigns/{campaign}/saves', json={'name':'during'}).json()
   fork = client.post(f"/api/saves/{saved['save_id']}/fork").json()['view']
   assert fork['revision'] == 0 and 'VISIBLE_REPLY' not in str(fork)
   assert 'VISIBLE_REPLY' not in client.get(f'/api/campaigns/{campaign}/export').text
   assert 'VISIBLE_REPLY' not in client.get(f'/api/campaigns/{campaign}/transcript').text
   if ending == 'cancel':
    assert client.post(path+'/cancel').json()['provisional'] is None
    assert client.get(path).json()['provisional'] is None
   if ending == 'recover':
    app.state.store.recover()
    assert client.get(path).json()['provisional'] is None
   release.set()
   client.portal.call(lambda: app.state.service.tasks[turn])
   final = client.get(path).json()
   assert final['provisional'] is None
   if ending == 'commit':
    assert final['status'] == 'committed' and final['revision'] == 1
    assert [d['text'] for d in final['result']['dialogue']] == ['VISIBLE_REPLY_guard', 'VISIBLE_REPLY_doctor']
    assert sum(e['text'] == 'VISIBLE_REPLY_guard' for e in final['result']['view']['transcript']) == 1
   else:
    assert final['status'] == {'cancel':'cancelled', 'failure':'failed', 'recover':'recoverable'}[ending]
    assert final['revision'] == 0
  finally:
   release.set()


@pytest.mark.parametrize('speech', ['private_other', 'local_unheard', 'invalid_disclosure'])
def test_no_preview_for_unheard_or_rejected_npc_proposal(tmp_path, speech):
 waiting, release = threading.Event(), threading.Event()
 class Agent:
  calls = repairs = 0
  metrics = []
  async def interpret(self, view): return PlayerIntent(kind='ASK', target_ids=['guard', 'doctor'], spoken_text='收到消息了吗？')
  async def decide_npc(self, view):
   self.calls += 1
   if self.calls == 2:
    waiting.set()
    while not release.is_set(): await asyncio.sleep(.002)
    return NpcDecision(actor_id=view.actor_id)
   utterance = {'text':'HIDDEN_CANARY', 'channel':'private', 'recipient_ids':['mechanic']}
   if speech == 'local_unheard': utterance.update(channel='local', recipient_ids=[])
   if speech == 'invalid_disclosure': utterance.update(recipient_ids=['player'], disclose_fact_ids=['invented_secret'])
   return NpcDecision(actor_id=view.actor_id, action={'kind':'SPEAK'}, utterance=utterance)
 app = create_app(Settings(data_dir=tmp_path, backend='mock', narration_mode='brief'), queue=Queue(), agent_factory=lambda *args:Agent())
 with TestClient(app) as client:
  campaign = client.post('/api/campaigns', json={}).json()['campaign_id']
  receipt = client.post(f'/api/campaigns/{campaign}/turns', json={'client_request_id':'privacy','expected_revision':0,'text':'询问','speech_channel':'broadcast'}).json()
  try:
   assert waiting.wait(3)
   response = client.get(f"/api/campaigns/{campaign}/turns/{receipt['turn_id']}").json()
   assert response['status'] == 'running'
   assert response['provisional'] is None
   assert 'HIDDEN_CANARY' not in json.dumps(response)
  finally: release.set()


@pytest.mark.asyncio
async def test_old_execution_cannot_republish_or_remove_resumed_preview(tmp_path):
 from local_trpg.modules import load_module
 from local_trpg.storage import Store
 from local_trpg.engine.rules import initial_state
 from local_trpg.engine.turns import TurnService
 from local_trpg.schemas import PlayerCommand
 module = load_module(); state = initial_state(module, '恢复')
 store = Store(tmp_path/'epochs.db'); store.create(state)
 waiting = {1: asyncio.Event(), 3: asyncio.Event()}
 release = {1: asyncio.Event(), 3: asyncio.Event()}
 class Agent:
  calls = repairs = 0
  metrics = []
  def __init__(self, epoch): self.epoch = epoch
  async def interpret(self, view): return PlayerIntent(kind='ASK', target_ids=['guard', 'doctor'], spoken_text='收到消息了吗？')
  async def decide_npc(self, view):
   self.calls += 1
   if self.calls == 2:
    waiting[self.epoch].set(); await release[self.epoch].wait()
   return NpcDecision(actor_id=view.actor_id, action={'kind':'SPEAK'}, utterance={'text':f'epoch-{self.epoch}-{view.actor_id}', 'channel':'private', 'recipient_ids':['player']})
 service = TurnService(store, module, None, agent_factory=lambda id, epoch:Agent(epoch), narration_mode='brief')
 receipt = await service.submit_turn(state.campaign_id, PlayerCommand(client_request_id='resume', expected_revision=0, text='询问', speech_channel='broadcast'))
 id = receipt.turn_id; old_task = service.tasks[id]
 try:
  await asyncio.wait_for(waiting[1].wait(), 3)
  assert service.status(state.campaign_id,id)['provisional']['dialogue'][0]['text'] == 'epoch-1-guard'
  store.recover()
  assert service.status(state.campaign_id,id)['provisional'] is None
  restarted = TurnService(store, module, None)
  assert restarted.status(state.campaign_id,id)['provisional'] is None
  await service.resume(state.campaign_id,id)
  await asyncio.wait_for(waiting[3].wait(), 3)
  release[1].set(); await old_task
  preview = service.status(state.campaign_id,id)['provisional']
  assert preview['epoch'] == 3
  assert [d['text'] for d in preview['dialogue']] == ['epoch-3-guard']
  release[3].set(); await service.tasks[id]
  final = service.status(state.campaign_id,id)
  assert final['provisional'] is None and final['revision'] == 1
  assert [d['text'] for d in final['result']['dialogue']] == ['epoch-3-guard', 'epoch-3-doctor']
 finally:
  for event in release.values(): event.set()
  await asyncio.gather(old_task, service.tasks[id], return_exceptions=True)
