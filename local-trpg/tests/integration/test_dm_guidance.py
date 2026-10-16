"""Regressions from the first human playthrough: DM questions and cargo loss."""
import json

import pytest

from local_trpg.engine.projection import player_view
from local_trpg.engine.rules import Engine, initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.model.parsing import InvalidOutput
from local_trpg.modules import load_module
from local_trpg.schemas import ActionIntent, NarrationResult, PlayerCommand
from local_trpg.storage import Store


class GuideAgent:
    calls = 0
    repairs = 0
    metrics = []

    def __init__(self, fail=False):
        self.views = []
        self.fail = fail

    async def guide(self, view):
        self.views.append(view.model_dump())
        if self.fail:
            raise InvalidOutput('unavailable')
        return NarrationResult(text='本轮问题：'+view.question+'。还不能判断船长的动机。你可以去驾驶舱核对公开记录。',
                               source_event_ids=['briefing'])


async def ask(service, store, campaign, text):
    state = store.snapshot(campaign)
    receipt = await service.submit_turn(campaign, PlayerCommand(
        client_request_id=text, expected_revision=state.revision, mode='ooc', text=text,
        private_aside='PRIVATE_ASIDE_CANARY'))
    await service.tasks[receipt.turn_id]
    return service.status(campaign, receipt.turn_id)


@pytest.mark.asyncio
async def test_ooc_answers_map_and_clock_separately_without_touching_world(tmp_path):
    module = load_module(); state = initial_state(module, '提问者')
    store = Store(tmp_path/'game.db'); store.create(state)
    agent = GuideAgent(); service = TurnService(store, module, None, agent_factory=lambda *_: agent)
    before = store.snapshot(state.campaign_id).model_dump_json()
    map_reply = await ask(service, store, state.campaign_id, '简图是什么？')
    clock_reply = await ask(service, store, state.campaign_id, '格数的切换依据什么？')
    assert map_reply['status'] == clock_reply['status'] == 'completed_ooc'
    assert len(agent.views) == 2
    assert '简图是什么' in map_reply['public_reply']
    assert '格数的切换依据什么' in clock_reply['public_reply']
    assert map_reply['public_reply'] != clock_reply['public_reply']
    assert store.snapshot(state.campaign_id).model_dump_json() == before


@pytest.mark.asyncio
async def test_open_ooc_uses_real_agent_interface_with_only_player_sources(tmp_path):
    module = load_module(); state = initial_state(module, '提问者')
    state.actors['captain'].goals.append('NPC_PRIVATE_GOAL_CANARY')
    store = Store(tmp_path/'game.db'); store.create(state)
    agent = GuideAgent(); service = TurnService(store, module, None, agent_factory=lambda *_: agent)
    answer = await ask(service, store, state.campaign_id, '船长为什么这样坚持？')
    assert agent.views and agent.views[0]['question'] == '船长为什么这样坚持？'
    serialized = json.dumps(agent.views, ensure_ascii=False)
    assert 'NPC_PRIVATE_GOAL_CANARY' not in serialized
    assert 'PRIVATE_ASIDE_CANARY' not in serialized
    assert 'captain_enabled_hold' not in serialized
    assert '还不能判断' in answer['public_reply']
    assert store.snapshot(state.campaign_id).tick == 0


@pytest.mark.asyncio
async def test_unavailable_ooc_is_explicit_without_an_invented_walkthrough(tmp_path):
    module = load_module(); state = initial_state(module, '提问者')
    store = Store(tmp_path/'game.db'); store.create(state)
    service = TurnService(store, module, None, agent_factory=lambda *_: GuideAgent(True))
    answer = await ask(service, store, state.campaign_id, '船长为什么这样坚持？')
    assert answer['status'] == 'failed'
    assert '未能生成' in answer['public_reply']
    assert '动力舱' not in answer['public_reply']


def repaired_without_cargo():
    module = load_module(); engine = Engine(initial_state(module, '试玩'), module, 'repair', lambda _: [6, 6])
    engine.move('player', 'engine')
    engine.resolve_player(ActionIntent(
        actor_id='player', kind='INSPECT', evidence_refs=['E01', 'E04']))
    engine.advance(1); engine.module_action('isolate_branch', 'mechanic', [])
    engine.advance(1); engine.module_action('restore_main_loop', 'player', [])
    return module, engine


@pytest.mark.asyncio
@pytest.mark.parametrize('confirmation', ['确认前往最近维修站', '确认'])
async def test_irreversible_departure_is_confirmed_without_a_hidden_deadline_spoiler(tmp_path, confirmation):
    module, engine = repaired_without_cargo()
    store = Store(tmp_path/'game.db'); store.create(engine.state)
    service = TurnService(store, module, None, agent_factory=lambda *_: GuideAgent(), narration_mode='brief')
    before = store.snapshot(engine.state.campaign_id).model_dump_json()
    r = await service.submit_turn(engine.state.campaign_id, PlayerCommand(client_request_id='depart', expected_revision=0, text='选择前往最近维修站'))
    await service.tasks[r.turn_id]
    preview = service.status(engine.state.campaign_id, r.turn_id)
    assert preview['status'] == 'needs_input' and '结束本章' in preview['public_reply']
    assert store.snapshot(engine.state.campaign_id).model_dump_json() == before
    r = await service.submit_turn(engine.state.campaign_id, PlayerCommand(client_request_id='confirmed', expected_revision=0, text=confirmation, reply_to_turn_id=r.turn_id))
    await service.tasks[r.turn_id]
    assert service.status(engine.state.campaign_id, r.turn_id)['status'] == 'committed'
    assert store.snapshot(engine.state.campaign_id).outcome is not None


def test_ending_reports_observed_result_and_leaves_detailed_recap_to_dm():
    module, engine = repaired_without_cargo()
    engine.module_action('choose_station', 'player', [])
    view = player_view(engine.state, module)
    assert view['outcome']['crew_saved'] == 6 and view['outcome']['cells_saved'] == 0
    explanation = ' '.join(view['ending_summary'])
    assert '0/4' in explanation
    assert '结束' in explanation and '后续章节' in explanation
    assert view['suggestions'] == []


def test_map_is_a_real_zero_cost_inspection():
    module = load_module(); engine = Engine(initial_state(module, '地图'), module, 'map', lambda _: [3, 4])
    engine.resolve_player(ActionIntent(actor_id='player', kind='INSPECT', description='查看简图'))
    assert engine.state.tick == 0
    assert any('动力舱' in o.content and '医务间' in o.content for o in engine.state.observations if o.observer_id == 'player')


def test_sticker_shortcut_inspects_instead_of_spending_time_on_npc_chatter():
    module = load_module(); state = initial_state(module, '贴纸')
    state.actors['player'].location_id = 'cargo'
    intent = TurnService(None, module, None).shortcut(PlayerCommand(
        client_request_id='sticker', expected_revision=0, text='故障贴纸'), state)
    assert intent is not None and intent.kind == 'INSPECT'
    assert intent.evidence_refs == ['E05']


def test_boarded_player_is_not_given_an_automatic_solution_menu():
    module = load_module(); state = initial_state(module, '登艇')
    state.phase = 'evacuation'; state.tick = 13
    state.flags['pod_prepared'] = True; state.flags['evacuation_agreed'] = True
    state.actors['player'].location_id = 'lifeboat'; state.actors['player'].onboard = True
    assert player_view(state, module)['suggestions'] == ['查看这里有什么可以调查的', '使用工作频道询问当前情况']


@pytest.mark.parametrize('known', [['E04', 'E05'], ['E05']])
def test_cold_case_solution_is_not_automatically_listed_as_a_ui_task(known):
    module = load_module(); state = initial_state(module, '应急柜')
    state.actors['player'].location_id = 'lifeboat'
    state.evidence['player'] = known
    assert '执行取应急保冷箱' not in player_view(state, module)['suggestions']
