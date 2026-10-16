"""Information boundaries and a genuinely model-backed table conversation."""
import json

import pytest

from local_trpg.engine.guidance import guide_view, map_text
from local_trpg.engine.projection import player_view
from local_trpg.engine.projection import build_agent_view
from local_trpg.memory import current_player_messages
from local_trpg.engine.rules import Engine, initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.modules import load_module
from local_trpg.schemas import ActionIntent, NarrationResult, PlayerCommand
from local_trpg.storage import Store


class ModelGuide:
    calls = 0
    repairs = 0
    metrics = []

    def __init__(self):
        self.views = []

    async def guide(self, view):
        self.views.append(view.model_dump())
        return NarrationResult(text='模型回答：'+view.question, source_event_ids=['rules'])


@pytest.mark.asyncio
async def test_rules_map_and_next_step_all_reach_the_model(tmp_path):
    module = load_module(); state = initial_state(module, '桌外玩家')
    store = Store(tmp_path/'game.db'); store.create(state)
    agent = ModelGuide(); service = TurnService(store, module, None, agent_factory=lambda *_: agent)
    before = store.snapshot(state.campaign_id).model_dump_json()
    questions = ['简图是什么？', '格数怎么算？', '下一步做什么？']
    for question in questions:
        r = await service.submit_turn(state.campaign_id, PlayerCommand(
            client_request_id=question, expected_revision=0, mode='ooc', text=question))
        await service.tasks[r.turn_id]
        assert service.status(state.campaign_id, r.turn_id)['public_reply'] == '模型回答：'+question
    assert [v['question'] for v in agent.views] == questions
    assert store.snapshot(state.campaign_id).model_dump_json() == before


def test_hidden_cargo_state_cannot_change_player_hints_or_dm_sources():
    module = load_module(); state = initial_state(module, '保密边界')
    other = state.model_copy(deep=True)
    other.flags['cells_transferred'] = True
    other.flags['cargo_branch_isolated'] = True
    other.flags['solo_transfer_prepared'] = True
    for cell in other.cells:
        cell['expiry'] = 999; cell['valid'] = False
    assert player_view(state, module) == player_view(other, module)
    assert guide_view('货物怎么样', state, module).model_dump() == guide_view('货物怎么样', other, module).model_dump()


def test_map_names_rooms_without_revealing_unvisited_room_contents():
    text = map_text(load_module())
    assert '动力舱' in text and '医务间' in text
    assert '储冷芯' not in text and '补给票' not in text and '改接工单' not in text


def test_current_clock_is_available_in_sources_used_by_answer_review():
    module = load_module(); state = initial_state(module, '时钟')
    state.tick = 3
    view = guide_view('现在第几格？', state, module)
    scene = json.loads(next(s['text'] for s in view.sources if s['event_id'] == 'scene'))
    assert scene['tick'] == 3


def test_inspecting_cabinet_reads_local_list_without_taking_items():
    module = load_module(); state = initial_state(module, '查看')
    state.actors['player'].location_id = 'lifeboat'
    engine = Engine(state, module, 'cabinet', lambda _: [3, 4])
    engine.resolve_player(ActionIntent(actor_id='player', kind='INSPECT', description='打开应急柜查看清单'))
    assert engine.state.items == state.items
    assert 'E07' in engine.state.evidence['player']
    assert engine.state.tick == 1
    assert not any(e.kind == 'ItemTransferred' for e in engine.events)


def test_entering_room_describes_obvious_objects_without_revealing_documents():
    module = load_module(); engine = Engine(initial_state(module, '场景'), module, 'move', lambda _: [3, 4])
    engine.resolve_player(ActionIntent(actor_id='player', kind='MOVE', target_ids=['engine']))
    perceived = '\n'.join(o.content for o in engine.state.observations if o.observer_id == 'player' and o.event_id.startswith('move:'))
    assert '流量面板' in perceived and '操作卡' in perceived
    assert '18%' not in perceived and not engine.state.evidence.get('player')
    assert engine.state.tick == 0


def test_simple_question_does_not_consume_five_minutes():
    module = load_module(); engine = Engine(initial_state(module, '发问'), module, 'ask', lambda _: [3, 4])
    engine.resolve_player(ActionIntent(actor_id='player', kind='ASK', spoken_text='方榆，你现在在哪里？',
        target_ids=['doctor'], channel='private', recipient_ids=['doctor']))
    assert engine.state.tick == 0
    assert any(o.observer_id == 'doctor' and o.content == '方榆，你现在在哪里？' for o in engine.state.observations)


def test_previous_dm_questions_are_projected_separately_without_private_asides(tmp_path):
    module = load_module(); state = initial_state(module, '历史')
    store = Store(tmp_path/'game.db'); store.create(state)
    cmd = PlayerCommand(client_request_id='history', expected_revision=0, mode='ooc', text='格数是什么？', private_aside='PRIVATE_HISTORY_CANARY')
    turn = store.submit(state.campaign_id, cmd); lease = store.claim(turn)
    store.record_call(turn, lease['epoch'], 'guide:', 'h', {'text':'一格约五分钟。','source_event_ids':['rules']}, {})
    store.finish(turn, lease['epoch'], 'completed_ooc', '一格约五分钟。')
    service = TurnService(store, module, None)
    history = service.view(state.campaign_id)['table_talk']
    assert history[0]['question'] == '格数是什么？' and history[0]['reply_source'] == 'model'
    projected = guide_view('你上一句是什么意思？', state, module, history)
    assert projected.history == [{'question': '格数是什么？', 'answer': '一格约五分钟。'}]
    serialized = json.dumps(history, ensure_ascii=False)
    assert 'PRIVATE_HISTORY_CANARY' not in serialized
    assert not store.snapshot(state.campaign_id).transcript[-1].get('speaker_id') == 'table_dm'
    restored = TurnService(Store(tmp_path/'game.db'), module, None).view(state.campaign_id)
    assert restored['table_talk'] == history


def test_requesting_actual_work_still_costs_time():
    module = load_module(); engine = Engine(initial_state(module, '工作'), module, 'work', lambda _: [3, 4])
    engine.resolve_player(ActionIntent(actor_id='player', kind='ASK', module_action_id='transfer_cells',
        spoken_text='方榆，请转存细胞。', target_ids=['doctor'], channel='private', recipient_ids=['doctor']))
    assert engine.state.tick == 1 and engine.authorization['actor_id'] == 'doctor'
    assert not engine.state.flags.get('cells_transferred')


def test_same_tick_old_radio_message_is_not_a_new_question_to_remote_npc():
    module = load_module(); engine = Engine(initial_state(module, '分轮'), module, 'old', lambda _: [3, 4])
    engine.speech('player', '各位在哪？', 'broadcast', [])
    newer = Engine(engine.state, module, 'new', lambda _: [3, 4])
    newer.move('player', 'engine')
    newer.speech('player', '顾成，这个面板是什么？', 'local', [])
    view = build_agent_view(newer.state, 'guard', module=module, reply_to_player=True,
                            current_event_ids={event.event_id for event in newer.events})
    assert newer.state.tick == 0
    assert any(o['content'] == '各位在哪？' for o in view.observations)
    assert current_player_messages(view.observations, view.scene) == []
    assert not view.scene.get('reply_to_player')
