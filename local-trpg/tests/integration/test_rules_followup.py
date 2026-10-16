"""Finite rules regressions; real engine and replay, with no model or server."""

import pytest

from local_trpg.engine.reducer import apply_event
from local_trpg.engine.rules import Engine, RuleError, initial_state
from local_trpg.modules import load_module
from local_trpg.schemas import ActionIntent


@pytest.fixture(scope="module")
def module():
    return load_module()


def engine_for(module, location="lifeboat", known=()):
    state = initial_state(module, "规则回归")
    state.actors["player"].location_id = location
    state.flags["pod_prepared"] = True
    engine = Engine(state, module, "rules-followup", lambda key: [6, 6])
    for reference in known:
        engine.give_evidence(reference, ["player"])
    return state, engine


def assert_replays(start, engine):
    replay = start
    for event in engine.events:
        replay = apply_event(replay, event)
    assert replay == engine.state


def test_leaving_the_pod_clears_boarding_and_blocks_remote_launch(module):
    start, engine = engine_for(module)
    engine.module_action("board_pod", "player", [])
    engine.move("player", "bridge")

    assert not engine.state.actors["player"].onboard
    assert engine.state.flags["onboard:player"] is False
    before = engine.state.model_copy(deep=True)
    with pytest.raises(RuleError):
        engine.module_action("launch_pod", "player", [])
    assert engine.state == before
    assert_replays(start, engine)


def test_returning_to_airlock_requires_boarding_again(module):
    _, engine = engine_for(module)
    engine.module_action("board_pod", "player", [])
    engine.move("player", "bridge")
    engine.move("player", "lifeboat")

    with pytest.raises(RuleError):
        engine.module_action("launch_pod", "player", [])
    engine.module_action("board_pod", "player", [])
    engine.module_action("launch_pod", "player", [])
    assert engine.state.phase == "epilogue"
    assert engine.state.outcome["crew_saved"] == 1


def test_stale_boarding_flag_cannot_launch_from_another_room(module):
    state, _ = engine_for(module, location="bridge")
    state.actors["player"].onboard = True
    state.flags["onboard:player"] = True
    engine = Engine(state, module, "stale-boarding", lambda key: [6, 6])

    with pytest.raises(RuleError):
        engine.module_action("launch_pod", "player", [])
    assert engine.state == state
    assert engine.events == []


@pytest.mark.parametrize("departure", [False, True])
def test_launch_saves_only_crew_who_remain_boarded_and_replays(module, departure):
    state, _ = engine_for(module)
    state.flags["evacuation_agreed"] = True
    for actor in state.actors.values():
        actor.location_id = "lifeboat"
    engine = Engine(state, module, "crew-boarding", lambda key: [6, 6])
    engine.module_action("board_pod", "player", [])
    engine.move("player", "lifeboat")
    assert engine.state.actors["player"].onboard
    if departure:
        engine.move("guard", "bridge")
    engine.module_action("launch_pod", "player", [])

    assert engine.state.outcome["crew_saved"] == (5 if departure else 6)
    assert engine.state.actors["guard"].onboard is not departure
    assert engine.state.phase == "epilogue"
    assert sum(event.kind == "ScenarioResolved" for event in engine.events) == 1
    assert_replays(state, engine)


@pytest.mark.parametrize("location", ["cargo", "bridge"])
@pytest.mark.parametrize("reference_field", ["evidence_refs", "target_ids"])
def test_explicit_known_record_reread_is_free_in_any_room(module, location, reference_field):
    start, engine = engine_for(module, location=location, known=["E05"])
    before = engine.state.model_copy(deep=True)
    first_new_event = len(engine.events)
    engine.resolve_player(ActionIntent(
        actor_id="player", kind="INSPECT", description="重读已得到的货物说明E05",
        **{reference_field: ["E05"]},
    ))

    assert engine.state.tick == 0
    assert engine.state.evidence == before.evidence
    assert engine.state.beliefs == before.beliefs
    assert not any(event.kind == "EvidenceObserved" for event in engine.events[first_new_event:])
    assert any(module["world"]["evidence"]["E05"]["text"] in event.payload.get("text", "")
               for event in engine.events[first_new_event:])
    assert_replays(start, engine)


def test_mixed_known_and_new_records_cost_one_tick_and_discover_only_new_record(module):
    _, engine = engine_for(module, location="bridge", known=["E05"])
    first_new_event = len(engine.events)
    engine.resolve_player(ActionIntent(
        actor_id="player", kind="INSPECT", evidence_refs=["E05", "E02"],
    ))

    assert engine.state.tick == 1
    assert set(engine.state.evidence["player"]) == {"E05", "E02"}
    assert [event.payload["evidence_id"] for event in engine.events[first_new_event:]
            if event.kind == "EvidenceObserved"] == ["E02"]


def test_evidence_target_cannot_be_ignored_beside_a_known_reference(module):
    _, engine = engine_for(module, location="bridge", known=["E05"])
    engine.resolve_player(ActionIntent(
        actor_id="player", kind="INSPECT", evidence_refs=["E05"], target_ids=["E02"],
    ))

    assert engine.state.tick == 1
    assert set(engine.state.evidence["player"]) == {"E05", "E02"}


@pytest.mark.parametrize("references", [["E02"], ["E05", "E02"]])
def test_unknown_record_still_requires_its_actual_location(module, references):
    _, engine = engine_for(module, location="cargo", known=["E05"])
    with pytest.raises(RuleError):
        engine.resolve_player(ActionIntent(
            actor_id="player", kind="INSPECT", evidence_refs=references,
        ))
    assert engine.state.evidence["player"] == ["E05"]


@pytest.mark.parametrize("description", ["重读货物说明", "环顾并重读记录"])
def test_invalid_record_reference_cannot_be_silently_treated_as_a_free_look(module, description):
    _, engine = engine_for(module, location="cargo", known=["E05"])
    with pytest.raises(RuleError):
        engine.resolve_player(ActionIntent(
            actor_id="player", kind="INSPECT", description=description,
            evidence_refs=["E05", "E99"],
        ))
    assert engine.state.evidence["player"] == ["E05"]


def test_new_record_with_room_look_wording_still_costs_time(module):
    _, engine = engine_for(module, location="cargo")
    engine.resolve_player(ActionIntent(
        actor_id="player", kind="INSPECT", description="环顾并调查货物说明",
        evidence_refs=["E05"],
    ))

    assert engine.state.tick == 1
    assert engine.state.evidence["player"] == ["E05"]


def test_known_record_does_not_make_registered_operation_free(module):
    _, engine = engine_for(module, location="bridge", known=["E05"])
    engine.resolve_player(ActionIntent(
        actor_id="player", kind="INSPECT", evidence_refs=["E05"],
        module_action_id="send_distress",
    ))

    assert engine.state.tick == engine.state.flags["first_distress_tick"] == 1


def test_unspecified_inspection_remains_a_paid_investigation(module):
    _, engine = engine_for(module, location="cargo")
    engine.resolve_player(ActionIntent(actor_id="player", kind="INSPECT", description="调查货箱"))

    assert engine.state.tick == 1
    assert engine.state.evidence["player"] == ["E05"]
