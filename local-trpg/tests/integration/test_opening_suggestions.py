"""Opening UI choices must reach listeners and must not spend time on bad routing."""
import pytest

from local_trpg.engine.projection import player_view, build_agent_view
from local_trpg.engine.rules import initial_state, Engine, RuleError
from local_trpg.engine.turns import TurnService
from local_trpg.modules import load_module
from local_trpg.schemas import NpcDecision, PlayerCommand, PlayerIntent
from local_trpg.storage import Store


class ListenerAgent:
    calls = 0
    repairs = 0
    metrics = []

    def __init__(self, silent=False):
        self.views = []
        self.silent = silent

    async def interpret(self, view):
        # Deliberately wrong for the shipped look-around choice: it must bypass
        # probabilistic interpretation, just like the other exact UI choices.
        return PlayerIntent(kind='ASK', spoken_text=view.text, attempt_description=view.text)

    async def decide_npc(self, view):
        self.views.append(view)
        if self.silent:
            return NpcDecision(actor_id=view.actor_id, action={'kind': 'WAIT'})
        assert any(o['source_actor_id'] == 'player' and o['modality'] == 'heard'
                   and o['observed_at'] == view.scene['tick'] for o in view.observations)
        return NpcDecision(actor_id=view.actor_id, action={'kind': 'SPEAK'},
            utterance={'text': '收到你的询问，我在'+view.scene['location']+'。', 'channel': 'broadcast'})


async def run(tmp_path, text, channel='local', silent=False):
    module = load_module()
    state = initial_state(module, '入口回归')
    store = Store(tmp_path/'game.db'); store.create(state)
    agent = ListenerAgent(silent)
    service = TurnService(store, module, None, agent_factory=lambda *args: agent, narration_mode='brief')
    receipt = await service.submit_turn(state.campaign_id, PlayerCommand(
        client_request_id='opening', expected_revision=0, text=text, speech_channel=channel))
    await service.tasks[receipt.turn_id]
    return service.status(state.campaign_id, receipt.turn_id), store.snapshot(state.campaign_id), agent


def test_work_channel_suggestion_carries_explicit_public_routing():
    module = load_module()
    view = player_view(initial_state(module, '入口'), module)
    command = view['suggestion_commands']['使用工作频道询问当前情况']
    assert command['speech_channel'] == 'broadcast'
    assert command['recipient_ids'] == []
    assert command['mode'] == 'in_character'


@pytest.mark.asyncio
async def test_local_question_in_empty_room_does_not_spend_a_turn(tmp_path):
    result, state, agent = await run(tmp_path, '使用工作频道询问当前情况')
    assert result['status'] == 'needs_input'
    assert '工作频道' in result['public_reply']
    assert '没有' in result['public_reply']
    assert state.tick == state.revision == 0
    assert not agent.views


def test_status_report_excludes_unrelated_private_memories_without_erasing_them():
    module = load_module()
    engine = Engine(initial_state(module, '入口'), module, 'status', lambda _: [3, 4])
    engine.advance(1)
    engine.speech('player', '各位，请汇报你所在岗位的当前情况。', 'broadcast', [])
    before = engine.state.model_dump_json()
    view = build_agent_view(engine.state, 'guard', module=module, status_report=True)
    assert view.scene['reply_focus'] == 'status_report'
    assert not view.private_background
    assert not view.beliefs
    assert 'captain_enabled_hold' not in view.model_dump_json()
    assert any(o['source_actor_id'] == 'player' for o in view.observations)
    assert engine.state.model_dump_json() == before
    assert build_agent_view(engine.state, 'guard', module=module).private_background
    decision = NpcDecision(actor_id='guard', action={'kind': 'SPEAK'},
        utterance={'text': '各位，请汇报你所在岗位的当前情况。', 'channel': 'broadcast'})
    with pytest.raises(RuleError, match='复述'):
        TurnService(None, module, None).validate_npc(engine, 'guard', decision, view)


@pytest.mark.asyncio
async def test_broadcast_reaches_npcs_and_reports_reply(tmp_path):
    result, state, agent = await run(tmp_path, '使用工作频道询问当前情况', 'broadcast')
    assert result['status'] == 'committed'
    assert result['result']['dialogue']
    assert '回应' in result['public_reply']
    assert '没有执行新的操作' not in result['public_reply']
    assert len(agent.views) == 2
    assert state.tick == 0


@pytest.mark.asyncio
async def test_no_reply_is_reported_as_no_reply(tmp_path):
    result, _, _ = await run(tmp_path, '使用工作频道询问当前情况', 'broadcast', silent=True)
    assert not result['result']['dialogue']
    assert '没有收到' in result['public_reply']


@pytest.mark.asyncio
async def test_look_around_suggestion_shows_public_room_without_time_cost(tmp_path):
    result, state, agent = await run(tmp_path, '查看这里有什么可以调查的')
    assert result['status'] == 'committed'
    assert '这里可见' in result['public_reply']
    assert state.tick == 0
    assert not agent.views
