"""Regressions for current-question loss, verbal promises, and brief resolution.

Only the slow model transport is scripted. Permissions, effects and persistence
use the real implementation; these fixtures do not assert model intelligence.
"""
import json
import pytest

from local_trpg.agents.runner import AgentRunner
from local_trpg.engine.projection import build_agent_view
from local_trpg.engine.rules import Engine, initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.model.base import GenerationResult
from local_trpg.modules import load_module
from local_trpg.schemas import PlayerCommand
from local_trpg.storage import Store


def payload(request):
    return json.JSONDecoder().raw_decode(request.messages[1].content)[0]


class Queue:
    def __init__(self, replies=(), pressure=False):
        self.replies = iter(replies)
        self.requests = []
        self.pressure = pressure

    def count_tokens(self, messages):
        # Force trimming at four observations independently of wording.
        if self.pressure:
            data = json.JSONDecoder().raw_decode(messages[1].content)[0]
            return 4500 if len(data['observations']) > 3 else 2000
        return 128

    async def generate(self, request):
        self.requests.append(request)
        output = next(self.replies, {'actor_id': payload(request).get('actor_id', 'doctor'), 'action': {'kind': 'WAIT'}})
        return GenerationResult(final_text=json.dumps(output, ensure_ascii=False),
            finish_reason='stop', usage={'input_tokens': 128, 'output_tokens': 30})


@pytest.mark.asyncio
async def test_budget_trimming_preserves_current_question_and_explicit_focus():
    module = load_module()
    engine = Engine(initial_state(module, '玩家'), module, 'focus', lambda _: [3, 4])
    engine.advance(1)
    engine.speech('player', '请只回答本次航程是否确认储冷芯充好。', 'private', ['doctor'])
    view = build_agent_view(engine.state, 'doctor', module=module)
    current = view.observations[-1]
    history = [{**current, 'observation_id': f'old-{i}', 'event_id': f'old-event-{i}',
                'observed_at': 0, 'content': '这是已经结束的旧对话。'} for i in range(6)]
    # Unordered retrieval must not turn the latest question into a disposable row.
    view.observations = [current, *history]
    before = view.model_dump_json()
    queue = Queue(pressure=True)
    await AgentRunner(queue).decide_npc(view)
    request = queue.requests[0]
    data, end = json.JSONDecoder().raw_decode(request.messages[1].content)
    assert current['observation_id'] in {o['observation_id'] for o in data['observations']}
    assert current['content'] in request.messages[1].content[end:]
    assert view.model_dump_json() == before


@pytest.mark.asyncio
async def test_old_player_speech_is_not_promoted_to_a_new_question():
    module = load_module()
    engine = Engine(initial_state(module, '玩家'), module, 'old', lambda _: [3, 4])
    engine.speech('player', '旧问题标记', 'private', ['doctor'])
    engine.advance(1)
    queue = Queue()
    await AgentRunner(queue).decide_npc(build_agent_view(engine.state, 'doctor', module=module))
    content = queue.requests[0].messages[1].content
    _, end = json.JSONDecoder().raw_decode(content)
    assert '旧问题标记' not in content[end:]


async def work_turn(tmp_path, replies):
    module = load_module()
    state = initial_state(module, '请求者')
    store = Store(tmp_path/'game.db');store.create(state)
    queue = Queue(replies)
    service = TurnService(store, module, queue, narration_mode='brief')
    receipt = await service.submit_turn(state.campaign_id, PlayerCommand(
        client_request_id='prepare', expected_revision=0,
        text='韩松，请执行准备救生艇', speech_channel='private', recipient_ids=['guard']))
    await service.tasks[receipt.turn_id]
    return store, state, queue, service.status(state.campaign_id, receipt.turn_id)


def guard(action, text):
    return {'actor_id': 'guard', 'action': action,
        'utterance': {'text': text, 'channel': 'private', 'recipient_ids': ['player']}}


@pytest.mark.asyncio
async def test_promising_work_without_action_is_repaired_before_commit(tmp_path):
    store, state, queue, status = await work_turn(tmp_path, [
        guard({'kind': 'SPEAK'}, '我现在就准备救生艇。'),
        guard({'kind': 'MODULE_ACTION', 'module_action_id': 'prepare_pod'}, '我来备艇。')])
    assert status['status'] == 'committed', status
    assert store.snapshot(state.campaign_id).flags['pod_prepared'] is True
    assert len(queue.requests) == 2
    assert all(r.role_kind == 'npc' for r in queue.requests)
    assert '我现在就准备救生艇。' not in json.dumps(status['result'], ensure_ascii=False)
    assert status['result']['degraded'] is False
    assert '已执行' in status['public_reply']


@pytest.mark.asyncio
@pytest.mark.parametrize('reply', ['我不执行这项工作。', '如果先确认人员名单，我会准备救生艇。'])
async def test_refusal_or_conditional_offer_does_not_force_execution(tmp_path, reply):
    store, state, queue, status = await work_turn(tmp_path, [guard({'kind': 'SPEAK'}, reply)])
    assert status['status'] == 'committed', status
    assert not store.snapshot(state.campaign_id).flags.get('pod_prepared')
    assert status['result']['dialogue'][0]['text'] == reply
    assert '本轮未执行' in status['public_reply']
    assert len(queue.requests) == 1


@pytest.mark.asyncio
async def test_failed_repair_never_turns_a_promise_into_a_world_effect(tmp_path):
    promise = guard({'kind': 'SPEAK'}, '我现在就准备救生艇。')
    store, state, queue, status = await work_turn(tmp_path, [promise, promise])
    assert status['result']['degraded'] is True
    assert not store.snapshot(state.campaign_id).flags.get('pod_prepared')
    assert len(queue.requests) == 2
    assert not status['result']['dialogue']
    assert '本轮未执行' in status['public_reply']


@pytest.mark.asyncio
async def test_brief_resolution_keeps_actual_evidence_without_narration_call(tmp_path):
    module = load_module();state = initial_state(module, '调查者')
    store = Store(tmp_path/'game.db');store.create(state)
    queue = Queue([{'kind': 'INSPECT', 'attempt_description': '检查动力舱阀组面板', 'evidence_refs': ['E01']}])
    # Move is an existing deterministic zero-time action, not a fabricated fixture.
    service = TurnService(store, module, queue, narration_mode='brief')
    for i, text in enumerate(['前往动力舱', '检查动力舱阀组面板']):
        receipt = await service.submit_turn(state.campaign_id, PlayerCommand(
            client_request_id=f'inspect-{i}', expected_revision=i, text=text))
        await service.tasks[receipt.turn_id]
    result = service.status(state.campaign_id, receipt.turn_id)
    assert result['status'] == 'committed', result
    assert 'E01' in {e['id'] for e in result['result']['view']['evidence']}
    # Reading a panel needs interpretation, not two unsolicited NPC replies.
    assert [r.role_kind for r in queue.requests] == ['interpreter']
    assert result['result']['degraded'] is False
