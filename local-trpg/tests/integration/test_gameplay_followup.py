"""Reproduce player-visible interaction bugs with the real rules and storage."""
import pytest
from local_trpg.engine.projection import player_view, build_agent_view
from local_trpg.engine.rules import Engine, RuleError, initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.modules import load_module
from local_trpg.schemas import NpcDecision, PlayerCommand, PlayerIntent
from local_trpg.storage import Store


class Agent:
    calls = repairs = 0
    metrics = []
    async def interpret(self, view):
        return PlayerIntent(kind='ASK', target_ids=['doctor'], spoken_text='你在哪里？')
    async def decide_npc(self, view):
        return NpcDecision(actor_id=view.actor_id, action={'kind': 'SPEAK'},
            utterance={'text': '我在医务间。', 'channel': 'private', 'recipient_ids': ['player']})


@pytest.mark.asyncio
async def test_own_action_and_same_name_npc_both_survive_save_and_replay(tmp_path):
    module = load_module();state = initial_state(module, '方榆')
    store = Store(tmp_path/'game.db');store.create(state)
    service = TurnService(store, module, None, agent_factory=lambda *args: Agent(), narration_mode='brief')
    command = PlayerCommand(client_request_id='same-name', expected_revision=0,
        text='你在哪里？', speech_channel='private', recipient_ids=['doctor'], private_aside='DM_ONLY_CANARY')
    receipt = await service.submit_turn(state.campaign_id, command)
    await service.tasks[receipt.turn_id]
    result = service.status(state.campaign_id, receipt.turn_id)
    assert result['status'] == 'committed'
    assert result['result']['dialogue'][0]['speaker_id'] == 'doctor'
    snapshot = store.snapshot(state.campaign_id)
    assert snapshot.transcript[-3]['speaker_id'] == 'player'
    assert snapshot.transcript[-3]['text'] == command.text
    assert snapshot.transcript[-1]['speaker_id'] == 'doctor'
    assert snapshot.transcript == result['result']['view']['transcript']
    assert 'DM_ONLY_CANARY' not in str(snapshot.transcript)
    fork = store.restore(store.export(state.campaign_id), module['module_hash'])
    assert store.snapshot(fork).transcript == snapshot.transcript


def test_remote_private_question_cannot_be_answered_with_inaudible_local_speech():
    module = load_module();engine = Engine(initial_state(module, '入口'), module, 'reply', lambda _: [3, 4])
    engine.advance(1);engine.speech('player', '你在哪里？', 'private', ['doctor'])
    view = build_agent_view(engine.state, 'doctor', module=module, reply_to_player=True)
    decision = NpcDecision(actor_id='doctor', utterance={'text': '我在医务间。', 'channel': 'local'})
    with pytest.raises(RuleError, match='听|频道|现场'):
        TurnService(None, module, None).apply_npc(engine, 'doctor', decision, view)
    assert not any(e.actor_id == 'doctor' for e in engine.events)


def test_opening_suggestions_do_not_offer_remote_inspection_from_common_room():
    module = load_module();state = initial_state(module, '入口')
    view = player_view(state, module)
    assert view['suggestions'] == ['查看这里有什么可以调查的', '使用工作频道询问当前情况']
    assert '检查动力舱的阀组与操作卡' not in view['suggestions']
    state.actors['player'].location_id = 'engine'
    service = TurnService(None, module, None)
    assert service.shortcut(PlayerCommand(client_request_id='inspect', expected_revision=0,
        text='检查动力舱的阀组与操作卡')).kind == 'INSPECT'


def test_explicit_evidence_suggestion_preserves_recipient_and_owned_card():
    module = load_module();state = initial_state(module, '入口')
    state.actors['player'].location_id = 'cargo';state.evidence['player'] = ['E07']
    command = {'text':'向方榆出示补给票E07','mode':'in_character','speech_channel':'private','recipient_ids':['doctor']}
    intent = TurnService(None, module, None).shortcut(PlayerCommand(
        **command, client_request_id='show', expected_revision=0), state)
    assert intent.kind == 'SHOW_EVIDENCE'
    assert intent.target_ids == command['recipient_ids'] == ['doctor']
    assert intent.evidence_refs == ['E07']
    state.flags['main_loop_restored'] = True
    shared = TurnService(None, module, None).shortcut(PlayerCommand(
        client_request_id='share', expected_revision=0, text='向大家说明已掌握的证据'), state)
    assert shared.evidence_refs == ['E07']
    state.evidence.clear()
    assert '向大家说明已掌握的证据' not in player_view(state, module)['suggestions']


@pytest.mark.asyncio
async def test_asking_who_is_here_reports_visible_people_without_spending_time(tmp_path):
    module = load_module();state = initial_state(module, '入口')
    store = Store(tmp_path/'look.db');store.create(state)
    service = TurnService(store, module, None, agent_factory=lambda *args: Agent(), narration_mode='brief')
    receipt = await service.submit_turn(state.campaign_id, PlayerCommand(
        client_request_id='who', expected_revision=0, text='都有谁在'))
    await service.tasks[receipt.turn_id]
    result = service.status(state.campaign_id, receipt.turn_id)
    assert result['status'] == 'committed'
    assert '这里只有你' in result['public_reply']
    assert store.snapshot(state.campaign_id).tick == 0


def test_stable_ship_and_absent_actors_are_not_presented_as_active_crisis():
    module = load_module();state = initial_state(module, '入口')
    state.phase = 'stable';state.tick = 9;state.flags['main_loop_restored'] = True
    state.actors['guard'].location_id = state.actors['player'].location_id
    state.actors['guard'].outcome = 'lost'
    view = player_view(state, module)
    assert not view['clocks']
    assert not next(n for n in view['npcs'] if n['id'] == 'guard')['present']
    assert not any(p['id'] == 'guard' for p in build_agent_view(state, 'player', module=module).scene['people'])


def test_status_cannot_infer_room_safety_from_own_health():
    module = load_module();engine = Engine(initial_state(module, '入口'), module, 'status', lambda _: [3, 4])
    engine.advance(1);engine.speech('player', '各位，请汇报你所在岗位的当前情况。', 'broadcast', [])
    view = build_agent_view(engine.state, 'doctor', module=module, status_report=True)
    bad = NpcDecision(actor_id='doctor', utterance={'text': '我身体健康，医务间一切正常，没有发现异常情况。', 'channel': 'broadcast'})
    with pytest.raises(RuleError, match='检查|依据'):
        TurnService(None, module, None).validate_npc(engine, 'doctor', bad, view)


def test_status_report_cannot_copy_another_actors_location():
    module = load_module();engine = Engine(initial_state(module, '入口'), module, 'status', lambda _: [3, 4])
    engine.advance(1);engine.speech('player', '各位，请汇报你所在岗位的当前情况。', 'broadcast', [])
    engine.speech('guard', '我在货舱，身体状态良好。', 'broadcast', [])
    before = engine.state.model_dump_json()
    view = build_agent_view(engine.state, 'doctor', module=module, status_report=True)
    assert not any(o.get('source_actor_id') == 'guard' and o.get('modality') == 'heard' for o in view.observations)
    assert engine.state.model_dump_json() == before
    bad = NpcDecision(actor_id='doctor', utterance={'text': '我在货舱，身体状态良好。', 'channel': 'broadcast'})
    with pytest.raises(RuleError, match='位置|医务间'):
        TurnService(None, module, None).validate_npc(engine, 'doctor', bad, view)
