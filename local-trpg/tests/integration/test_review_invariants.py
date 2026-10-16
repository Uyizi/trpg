"""Independent P1–P5 review regressions. Test doubles never enter production.

Run with: .venv/bin/python -m pytest tests/integration/test_review_invariants.py -q
These tests use the real module, rules, reducer, store and role runner; no model
weights, network service, GPU or wall-clock timeout sleeps are required.
"""
import asyncio
import copy
import json
import sqlite3

import pytest

from local_trpg.agents.runner import AgentRunner
from local_trpg.engine.projection import build_agent_view, build_narration_view, player_view
from local_trpg.engine.reducer import apply_event
from local_trpg.engine.rules import Engine, RuleError, initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.memory import retrieve
from local_trpg.model.base import GenerationResult
from local_trpg.model.parsing import InvalidOutput
from local_trpg.modules import load_module
from local_trpg.schemas import ActionIntent, GameEvent, NarrationResult, NpcDecision, PlayerCommand, PlayerIntent
from local_trpg.storage import Conflict, Store


@pytest.fixture(scope="module")
def module():
    return load_module()


@pytest.fixture
def state(module):
    return initial_state(module, "审阅玩家")


@pytest.fixture
def store(tmp_path, state):
    result = Store(tmp_path / "review.db")
    result.create(state)
    return result


def engine_for(state, module, turn="review"):
    return Engine(state, module, turn, lambda key: [6, 6])


def npc_view(engine, module, npc):
    return build_agent_view(engine.state, npc, module=module, authorization=engine.authorization)


class ScriptAgent:
    """Explicit test-only agent for deterministic fault injection."""

    def __init__(self, intent, decisions=None, pause_narration=False):
        self.intent = intent
        self.decisions = decisions or {}
        self.views = []
        self.calls = self.repairs = 0
        self.metrics = []
        self.narration_entered = asyncio.Event()
        self.release_narration = asyncio.Event()
        self.pause_narration = pause_narration

    async def interpret(self, view):
        self.views.append(("interpreter", view.model_dump()))
        return self.intent

    async def decide_npc(self, view):
        self.views.append((view.actor_id, view.model_dump()))
        return self.decisions.get(view.actor_id, NpcDecision(actor_id=view.actor_id, action={"kind": "WAIT"}))

    async def narrate(self, view):
        self.views.append(("narrator", view.model_dump()))
        self.narration_entered.set()
        if self.pause_narration:
            await self.release_narration.wait()
        return NarrationResult(text="本轮可见言行已经记录。", source_event_ids=[e["event_id"] for e in view.events])


async def run_command(store, module, state, agent, **fields):
    service = TurnService(store, module, None, agent_factory=lambda turn, epoch: agent)
    command = PlayerCommand(client_request_id="review", expected_revision=state.revision, **fields)
    receipt = await service.submit_turn(state.campaign_id, command)
    await service.tasks[receipt.turn_id]
    return service, receipt, command


@pytest.mark.asyncio
async def test_explicit_private_aside_and_internal_deception_never_enter_role_views(store, module, state):
    canary = "PRIVATE_ASIDE_CANARY_31f894"
    agent = ScriptAgent(PlayerIntent(kind="DECEIVE", target_ids=["doctor"], spoken_text="我没有看过门禁记录。", private_intent=canary))
    service, receipt, _ = await run_command(store, module, state, agent, text="告诉医生我没看过记录", speech_channel="private", recipient_ids=["doctor"], private_aside=canary)
    result = service.status(state.campaign_id, receipt.turn_id)
    assert result["status"] == "committed"
    visible = json.dumps([agent.views, result, service.view(state.campaign_id)], ensure_ascii=False)
    assert canary not in visible
    assert '"private_intent"' not in visible
    assert "DECEIVE" not in visible


@pytest.mark.asyncio
async def test_empty_spoken_text_does_not_publish_raw_private_thought_fallback(store, module, state):
    canary = "UNSPOKEN_THOUGHT_CANARY_c472"
    agent = ScriptAgent(PlayerIntent(kind="ASK", target_ids=["doctor"], private_intent=canary))
    service, receipt, _ = await run_command(store, module, state, agent, text="我心想" + canary, recipient_ids=["doctor"], speech_channel="private")
    assert service.status(state.campaign_id, receipt.turn_id)["status"] == "needs_input"
    assert store.snapshot(state.campaign_id) == state
    assert not [v for role, v in agent.views if role != "interpreter"]


@pytest.mark.parametrize("kind", ["TALK", "ASK", "PERSUADE", "DECEIVE", "SHOW_EVIDENCE"])
def test_unowned_evidence_reference_cannot_grant_canonical_clue(kind, state, module):
    engine = engine_for(state, module)
    try:
        engine.resolve_player(ActionIntent(actor_id="player", kind=kind, spoken_text="我听说有值班记录。", channel="broadcast", evidence_refs=["E06"], target_ids=["comms"]))
    except RuleError:
        pass
    assert all("E06" not in refs for refs in engine.state.evidence.values()), "An unowned reference must not materialize a real clue card"
    assert module["world"]["evidence"]["E06"]["text"] not in json.dumps(player_view(engine.state, module), ensure_ascii=False)


def test_discovering_new_evidence_by_legal_inspection_does_not_require_prior_ownership(state, module):
    state.actors["player"].location_id = "engine"
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="INSPECT", evidence_refs=["E01", "E04"]))
    assert {"E01", "E04"} <= set(engine.state.evidence["player"])
    assert engine.state.tick == 1


def test_npc_authorization_cannot_change_patient(state, module):
    state.actors["player"].health = "strained"
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="ASK", spoken_text="医生，请用第一份物资治疗我。", target_ids=["doctor", "player"], recipient_ids=["doctor"], channel="private", module_action_id="treat_actor", item_ids=["medical_supply_1"]))
    decision = NpcDecision(actor_id="doctor", action={"kind": "MODULE_ACTION", "module_action_id": "treat_actor", "target_ids": ["mechanic"], "item_ids": ["medical_supply_1"]})
    before = engine.state.model_copy(deep=True)
    service = TurnService(None, module, None)
    with pytest.raises(RuleError):
        service.apply_npc(engine, "doctor", decision, npc_view(engine, module, "doctor"))
    assert engine.state == before


def test_npc_uses_only_the_exact_authorized_medical_supply(state, module):
    state.actors["player"].health = "strained"
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="ASK", spoken_text="医生，请用第二份物资治疗我。", target_ids=["doctor", "player"], recipient_ids=["doctor"], channel="private", module_action_id="treat_actor", item_ids=["medical_supply_2"]))
    decision = NpcDecision(actor_id="doctor", action={"kind": "MODULE_ACTION", "module_action_id": "treat_actor", "target_ids": ["player"], "item_ids": ["medical_supply_2"]})
    TurnService(None, module, None).apply_npc(engine, "doctor", decision, npc_view(engine, module, "doctor"))
    assert engine.state.items["medical_supply_2"] == "consumed"
    assert engine.state.items["medical_supply_1"] == "doctor"
    assert engine.state.actors["player"].health == "healthy"
    assert engine.state.tick == 1


def test_invalid_npc_utterance_cannot_leave_partially_applied_medical_action(state, module):
    state.actors["player"].health = "strained"
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="ASK", spoken_text="医生，请用第一份物资治疗我。", target_ids=["doctor", "player"], recipient_ids=["doctor"], channel="private", module_action_id="treat_actor", item_ids=["medical_supply_1"]))
    before = engine.state.model_copy(deep=True)
    before_events = list(engine.events)
    decision = NpcDecision(actor_id="doctor", action={"kind": "MODULE_ACTION", "module_action_id": "treat_actor", "target_ids": ["player"], "item_ids": ["medical_supply_1"]}, utterance={"text": "处置完毕。", "channel": "private", "recipient_ids": []})
    with pytest.raises(RuleError):
        TurnService(None, module, None).apply_npc(engine, "doctor", decision, npc_view(engine, module, "doctor"))
    assert engine.state == before
    assert engine.events == before_events


def test_explicit_request_can_distinguish_doctor_from_npc_patient(state, module):
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="ASK", spoken_text="医生，请用第一份物资治疗机械师。", target_ids=["doctor", "mechanic"], recipient_ids=["doctor"], channel="private", module_action_id="treat_actor", item_ids=["medical_supply_1"]))
    assert engine.authorization["actor_id"] == "doctor"
    assert engine.authorization["target_ids"] == ["mechanic"]


def test_registered_npc_job_is_one_tick_single_use_and_not_authorized_by_move(state, module):
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="MOVE", target_ids=["bridge"], module_action_id="send_distress"))
    decision = NpcDecision(actor_id="comms", action={"kind": "MODULE_ACTION", "module_action_id": "send_distress"})
    service = TurnService(None, module, None)
    with pytest.raises(RuleError):
        service.apply_npc(engine, "comms", decision, npc_view(engine, module, "comms"))
    assert engine.state.tick == 0
    engine.resolve_player(ActionIntent(actor_id="player", kind="ASK", spoken_text="请发送标准求救，注明是我提出的。", target_ids=["comms"], module_action_id="send_distress"))
    service.apply_npc(engine, "comms", decision, npc_view(engine, module, "comms"))
    assert engine.state.tick == engine.state.flags["first_distress_tick"] == 1
    with pytest.raises(RuleError):
        service.apply_npc(engine, "comms", decision, npc_view(engine, module, "comms"))


def test_arriving_observer_sees_movement_and_subsequent_local_speech(state, module):
    state.actors["doctor"].location_id = "clinic"
    engine = engine_for(state, module)
    engine.move("player", "clinic")
    movement_id = engine.events[-1].event_id
    assert any(o.observer_id == "doctor" and o.event_id == movement_id for o in engine.state.observations), "Destination witnesses must see someone arrive"
    engine.speech("player", "到场后的话", "local", [])
    assert any(o.observer_id == "doctor" and o.content == "到场后的话" for o in engine.state.observations)


def test_departure_speech_order_and_replay_are_equivalent(state, module):
    state.actors["doctor"].location_id = "common"
    first = engine_for(state, module, "leave-first")
    first.move("doctor", "clinic")
    first.speech("player", "离开后才说的话", "local", [])
    assert not any(o.observer_id == "doctor" and o.content == "离开后才说的话" for o in first.state.observations)
    second = engine_for(state, module, "speak-first")
    second.speech("player", "离开前说的话", "local", [])
    second.move("doctor", "clinic")
    assert any(o.observer_id == "doctor" and o.content == "离开前说的话" for o in second.state.observations)
    replayed = state
    for event in second.events:
        replayed = apply_event(replayed, event)
    assert replayed == second.state


@pytest.mark.asyncio
async def test_second_npc_gets_fresh_perception_and_broadcast_does_not_wake_all(store, module, state):
    first = module["world"]["initiative_order"][0]
    utterance = "FIRST_NPC_BROADCAST_61c2"
    agent = ScriptAgent(PlayerIntent(kind="TALK", spoken_text="大家报告情况。"), {first: NpcDecision(actor_id=first, utterance={"text": utterance, "channel": "broadcast"})})
    service, receipt, _ = await run_command(store, module, state, agent, text="大家报告情况。", speech_channel="broadcast")
    assert service.status(state.campaign_id, receipt.turn_id)["status"] == "committed"
    npc_views = [(role, v) for role, v in agent.views if role not in ("interpreter", "narrator")]
    assert len(npc_views) == 2
    assert npc_views[0][0] == first
    assert utterance in json.dumps(npc_views[1][1], ensure_ascii=False)
    final = store.snapshot(state.campaign_id)
    for actor in final.actors:
        assert any(o.observer_id == actor and o.content == "大家报告情况。" for o in final.observations)
    assert set(final.pending_npcs) == set(module["world"]["actors"]) - {role for role, _ in npc_views}


def test_private_retrieval_and_blocked_secret_do_not_become_player_evidence(state, module):
    engine = engine_for(state, module)
    canary = "CAPTAIN_ONLY_CANARY_378"
    engine.speech("player", canary, "private", ["captain"])
    assert canary not in json.dumps(retrieve(engine.state, "doctor", query=canary), ensure_ascii=False)
    decision = NpcDecision(actor_id="captain", utterance={"text": "这是记录", "disclose_fact_ids": ["E06"]})
    with pytest.raises(RuleError):
        TurnService(None, module, None).apply_npc(engine, "captain", decision, npc_view(engine, module, "captain"))
    assert "E06" not in engine.state.evidence.get("player", [])


@pytest.mark.asyncio
async def test_duplicate_after_roll_cancel_before_commit_and_save_do_not_leave_ghost_memory(store, module, state):
    agent = ScriptAgent(PlayerIntent(kind="DECEIVE", target_ids=["doctor"], spoken_text="我从没见过记录。"), pause_narration=True)
    service = TurnService(store, module, None, agent_factory=lambda turn, epoch: agent)
    command = PlayerCommand(client_request_id="concurrent", expected_revision=0, text="我从没见过记录。", speech_channel="private", recipient_ids=["doctor"])
    first, second = await asyncio.gather(service.submit_turn(state.campaign_id, command), service.submit_turn(state.campaign_id, command))
    assert first.turn_id == second.turn_id
    await asyncio.wait_for(agent.narration_entered.wait(), 2)
    dice = store.roll(first.turn_id, "player:0:social")
    assert dice == store.roll(first.turn_id, "player:0:social", fixed=[1, 1])
    save = store.save(state.campaign_id, "during draft")
    assert save["during_turn"] and save["revision"] == 0
    assert store.snapshot(state.campaign_id) == state
    store.cancel(state.campaign_id, first.turn_id)
    agent.release_narration.set()
    await service.tasks[first.turn_id]
    assert store.snapshot(state.campaign_id) == state
    assert service.status(state.campaign_id, first.turn_id)["status"] == "cancelled"
    assert store.export(state.campaign_id)["events"] == []


def test_old_epoch_cannot_publish_cached_call_or_commit_after_resume(store, state):
    command = PlayerCommand(client_request_id="restart", expected_revision=0, text="等待")
    turn = store.submit(state.campaign_id, command)
    old = store.claim(turn)
    store.roll(turn, "player:0:technical", [2, 5])
    store.recover()
    store.resume(state.campaign_id, turn)
    current = store.claim(turn)
    assert current["epoch"] > old["epoch"]
    with pytest.raises(Conflict):
        store.record_call(turn, old["epoch"], "npc:doctor", "view", {}, {"role": "npc"})
    with pytest.raises(Conflict):
        store.commit(turn, old["epoch"], state, [], {"narration": "旧结果", "dialogue": []})
    assert store.roll(turn, "player:0:technical", [6, 6]) == [2, 5]
    assert store.snapshot(state.campaign_id) == state


def test_event_insert_failure_rolls_back_snapshot_and_turn_status(store, state, module):
    command = PlayerCommand(client_request_id="atomic", expected_revision=0, text="等待")
    turn = store.submit(state.campaign_id, command)
    claim = store.claim(turn)
    engine = engine_for(state, module, turn)
    engine.resolve_player(ActionIntent(actor_id="player", kind="WAIT"))
    duplicate_events = [engine.events[0], engine.events[0]]
    with pytest.raises((sqlite3.IntegrityError, ValueError)):
        store.commit(turn, claim["epoch"], engine.state, duplicate_events, {"narration": "不应提交", "dialogue": []})
    assert store.snapshot(state.campaign_id) == state
    assert store.turn(turn)["status"] == "running"
    assert store.export(state.campaign_id)["events"] == []


@pytest.mark.parametrize("corruption", ["future_version", "foreign_actor", "contradictory_snapshot", "invalid_dice", "unknown_roll_turn", "unknown_item"])
def test_invalid_backup_is_rejected_before_any_campaign_is_created(store, state, module, corruption):
    bundle = store.export(state.campaign_id)
    if corruption == "future_version":
        bundle["schema_version"] = 999
    elif corruption == "foreign_actor":
        event = GameEvent(event_id="foreign-event", turn_id="foreign-turn", beat_index=0, actor_id="unknown_actor", kind="SpeechOccurred", payload={"text": "bad", "channel": "private", "recipient_ids": ["player"]}, location_id="common", game_tick=0)
        bundle["events"] = [event.model_dump()]
    elif corruption == "contradictory_snapshot":
        key = next(iter(bundle["state"]["world_facts"]))
        bundle["state"]["world_facts"][key] = False
    elif corruption == "invalid_dice":
        bundle["rolls"] = [{"turn_id": "missing-turn", "roll_key": "player:0:social", "dice": "[0,7]"}]
    elif corruption == "unknown_roll_turn":
        bundle["rolls"] = [{"turn_id": "missing-turn", "roll_key": "player:0:social", "dice": "[3,4]"}]
    else:
        for key in ("state", "initial_state"):
            bundle[key]["items"]["unregistered_magic_key"] = "player"
            bundle[key]["actors"]["player"]["inventory"].append("unregistered_magic_key")
    before = store.list_campaigns()
    with pytest.raises((ValueError, KeyError)):
        store.restore(bundle, module["module_hash"])
    assert store.list_campaigns() == before


def test_export_restore_preserves_recorded_random_results(store, state, module):
    command = PlayerCommand(client_request_id="rolled", expected_revision=0, text="劝说")
    turn = store.submit(state.campaign_id, command)
    claim = store.claim(turn)
    engine = Engine(state, module, turn, lambda key: store.roll(turn, key, [4, 5]))
    engine.resolve_player(ActionIntent(actor_id="player", kind="PERSUADE", spoken_text="请听我说明", target_ids=["doctor"], channel="private", recipient_ids=["doctor"]))
    engine.finish_clock()
    store.commit(turn, claim["epoch"], engine.state, engine.events, {"narration": "已记录", "dialogue": []})
    bundle = store.export(state.campaign_id)
    assert bundle["rolls"]
    restored = store.restore(copy.deepcopy(bundle), module["module_hash"])
    assert store.export(restored)["rolls"] == bundle["rolls"]
    assert store.submit(state.campaign_id, command) == turn


def test_arrival_before_cargo_expiry_counts_saved_cells_even_if_ship_deadline_is_later(state, module):
    state.tick = 8
    state.flags.update(pod_prepared=True, first_distress_tick=1, rescue_eta=7, evacuation_agreed=True, cells_transferred=True)
    for actor in state.actors.values():
        actor.location_id = "lifeboat"
        actor.onboard = True
    for i, cell in enumerate(state.cells):
        cell.update(container=f"cold_case_{1 + i // 2}", expiry=12, valid=True)
    for item in ("cold_case_1", "cold_case_2", "cold_core_1", "cold_core_2"):
        state.items[item] = "pod"
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="USE_ITEM", module_action_id="launch_pod"))
    assert engine.state.outcome["arrival_tick"] == 9
    assert engine.state.outcome["cells_saved"] == 4
    assert engine.state.flags["engine_destroyed"]


def test_completion_at_tick12_beats_damage_but_crossing_does_not_revive_engine(state, module):
    state.tick = 11
    state.flags["cargo_branch_isolated"] = True
    state.actors["player"].location_id = "engine"
    exact = engine_for(state, module, "exact")
    exact.resolve_player(ActionIntent(actor_id="player", kind="REPAIR", module_action_id="restore_main_loop"))
    exact.finish_clock()
    assert exact.state.flags["main_loop_restored"] and not exact.state.flags["engine_destroyed"]
    crossed = engine_for(state, module, "crossed")
    crossed.advance(2)
    with pytest.raises(RuleError):
        crossed.module_action("restore_main_loop", "player", [])
    assert crossed.state.flags["engine_destroyed"] and not crossed.state.flags["main_loop_restored"]
    fired = list(crossed.state.fired_triggers)
    crossed.finish_clock()
    assert crossed.state.fired_triggers == fired


def test_expired_cells_cannot_be_transferred_or_have_their_expiry_refreshed(state, module):
    state.tick = 4
    state.actors["player"].location_id = "cargo"
    state.evidence["player"] = ["E05"]
    state.flags["solo_transfer_prepared"] = True
    for item in ("cold_case_1", "cold_case_2", "cold_core_1", "cold_core_2"):
        state.items[item] = "cargo"
    for cell in state.cells:
        cell["expiry"] = 5
    engine = engine_for(state, module)
    with pytest.raises(RuleError):
        engine.resolve_player(ActionIntent(actor_id="player", kind="USE_ITEM", module_action_id="transfer_cells"))
    assert all(not c["valid"] and c["expiry"] == 5 for c in engine.state.cells)


@pytest.mark.parametrize("unavailable", ["doctor_supplies", "guard_incapacitated", "comms_incapacitated"])
def test_unavailable_scheduled_worker_does_not_block_all_clock_progress(state, module, unavailable):
    if unavailable == "doctor_supplies":
        state.tick = 2
        state.items["medical_supply_1"] = state.items["medical_supply_2"] = "consumed"
    elif unavailable == "guard_incapacitated":
        state.tick = 7
        state.actors["guard"].health = "incapacitated"
    else:
        state.tick = 8
        state.actors["comms"].health = "incapacitated"
    engine = engine_for(state, module)
    engine.resolve_player(ActionIntent(actor_id="player", kind="WAIT"))
    engine.finish_clock()
    assert engine.state.tick == state.tick + 1
    assert str(engine.state.tick) in engine.state.fired_triggers
    if unavailable == "doctor_supplies":
        assert engine.state.actors["mechanic"].health == "strained"
    elif unavailable == "guard_incapacitated":
        assert not engine.state.flags["pod_prepared"]
    else:
        assert engine.state.flags.get("first_distress_tick") is None


class OfflineQueue:
    """Scripted model transport only for AgentRunner budget tests."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def count_tokens(self, messages):
        return 100

    async def generate(self, request):
        self.requests.append(request)
        return next(self.responses)


@pytest.mark.asyncio
async def test_runner_stops_all_further_generation_after_backend_timeout(module, state):
    queue = OfflineQueue([GenerationResult(error="timeout: backend_busy_unknown")])
    runner = AgentRunner(queue, store=None, turn_id="test-timeout", epoch=1)
    view = build_narration_view(state, [], module=module)
    with pytest.raises(InvalidOutput):
        await runner.narrate(view)
    with pytest.raises(InvalidOutput):
        await runner.narrate(view)
    assert len(queue.requests) == 1 and runner.calls == 1


@pytest.mark.asyncio
async def test_runner_has_only_one_format_repair_across_roles(module, state):
    queue = OfflineQueue([GenerationResult(final_text="invalid", finish_reason="stop"), GenerationResult(final_text='{"actor_id":"doctor","action":{"kind":"WAIT"}}', finish_reason="stop"), GenerationResult(final_text="invalid again", finish_reason="stop")])
    runner = AgentRunner(queue, store=None, turn_id="test-repair", epoch=1)
    await runner.decide_npc(build_agent_view(state, "doctor", module=module))
    with pytest.raises(InvalidOutput):
        await runner.narrate(build_narration_view(state, [], module=module))
    assert runner.repairs == 1 and len(queue.requests) == 3
