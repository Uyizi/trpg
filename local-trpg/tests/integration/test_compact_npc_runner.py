"""Actual runner, projection, parser and engine boundaries with offline transport."""
import json

import pytest

from local_trpg.agents.runner import AgentRunner
from local_trpg.engine.projection import build_agent_view
from local_trpg.engine.rules import initial_state
from local_trpg.model.base import GenerationResult
from local_trpg.modules import load_module
from local_trpg.schemas import PlayerCommand
from local_trpg.storage import Store


class CompactQueue:
    def __init__(self):
        self.requests = []

    def count_tokens(self, messages):
        return 100

    async def generate(self, request):
        self.requests.append(request)
        return GenerationResult(final_text='{"a":"SPEAK","t":"目前尚未核实。","c":"private","r":["player"]}',
                                finish_reason='stop')


@pytest.mark.asyncio
async def test_runner_compact_output_keeps_complete_view_and_isolates_probe_prefix():
    module = load_module()
    view = build_agent_view(initial_state(module, '玩家'), 'doctor', module=module)
    queue = CompactQueue()
    runner = AgentRunner(queue)
    decision = await runner.decide_npc(view)
    assert decision.actor_id == 'doctor'
    assert decision.utterance.channel == 'private'
    request = queue.requests[-1]
    data, _ = json.JSONDecoder().raw_decode(request.messages[1].content)
    assert data == view.model_dump()
    assert 'actor_id' not in request.response_schema['properties']
    assert request.prefix_message_count == 1
    await runner.repair_npc(view, '请保留观察依据')
    repair = queue.requests[-1]
    assert request.messages[0] == repair.messages[0]
    assert request.cache_namespace == repair.cache_namespace
    assert '请保留观察依据' in repair.messages[1].content
    await AgentRunner(queue).decide_npc(view)
    assert queue.requests[-1].cache_namespace != request.cache_namespace


@pytest.mark.asyncio
async def test_runner_prefix_namespace_is_campaign_and_actor_scoped(tmp_path):
    module = load_module()
    state = initial_state(module, '玩家')
    store = Store(tmp_path / 'compact.db')
    store.create(state)
    turn_id = store.submit(state.campaign_id, PlayerCommand(client_request_id='compact', expected_revision=0))
    claimed = store.claim(turn_id)
    queue = CompactQueue()
    runner = AgentRunner(queue, store=store, turn_id=turn_id, epoch=claimed['epoch'])
    for actor_id in ('doctor', 'guard'):
        await runner.decide_npc(build_agent_view(state, actor_id, module=module))
    doctor, guard = queue.requests
    assert doctor.cache_namespace != guard.cache_namespace
    assert state.campaign_id in doctor.cache_namespace
