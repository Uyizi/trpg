"""Exercise provenance, disclosure gates and replay through the real engine."""
import json

import pytest

from local_trpg.engine.projection import build_agent_view
from local_trpg.engine.reducer import apply_event
from local_trpg.engine.rules import Engine, RuleError, initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.modules import load_module
from local_trpg.model.parsing import InvalidOutput
from local_trpg.schemas import ActionIntent, NpcDecision, PlayerCommand, PlayerIntent, NarrationResult
from local_trpg.storage import Store


@pytest.fixture
def setup():
    module = load_module()
    state = initial_state(module, "核验员")
    engine = Engine(state, module, "disclosure", lambda key: [4, 4])
    return module, state, engine, TurnService(None, module, None)


def view(engine, module, actor):
    return build_agent_view(engine.state, actor, module=module, authorization=engine.authorization)


def options(engine, module, actor):
    return {row["id"] for row in view(engine, module, actor).model_dump().get("available_disclosures", [])}


def say(engine, actor, text, channel="private"):
    engine.speech("player", text, channel, [actor])


def disclose(engine, service, module, actor, refs, recipients=None, text="这是我的亲历说明。", channel="private"):
    decision = NpcDecision(actor_id=actor, utterance={"text": text, "channel": channel,
        "recipient_ids": recipients or ["player"], "disclose_fact_ids": refs})
    service.apply_npc(engine, actor, decision, view(engine, module, actor))


def test_guard_firsthand_testimony_is_scoped_and_keeps_canonical_world_unchanged(setup):
    module, initial, engine, service = setup
    assert "captain_enabled_hold" not in options(engine, module, "guard")
    say(engine, "guard", "事故前12分钟，你看见谁操作保持？")
    assert {"captain_enabled_hold", "guard_witnessed_hold"} <= options(engine, module, "guard")
    before = dict(engine.state.world_facts)
    disclose(engine, service, module, "guard", ["captain_enabled_hold", "guard_witnessed_hold"])
    assert engine.state.flags.get("guard_witness") is True
    assert engine.state.flags.get("guard_witness:player") is True
    assert not engine.state.flags.get("guard_witness:doctor")
    heard = [o for o in engine.state.observations if o.observer_id == "player" and "captain_enabled_hold" in o.proposition_ids]
    assert len(heard) == 1 and heard[0].modality == "reported"
    assert heard[0].source_actor_id == "guard"
    assert heard[0].source_kind == "firsthand"
    assert heard[0].source_observation_ids
    assert "亲历证词" in heard[0].content
    assert any(b.actor_id == "player" and b.proposition_id == "captain_enabled_hold" and b.basis == "testimony" for b in engine.state.beliefs)
    assert "E06" not in engine.state.evidence.get("player", [])
    assert engine.state.world_facts == before
    replayed = initial
    for event in engine.events:
        replayed = apply_event(replayed, event)
    assert replayed == engine.state


def test_guard_requires_actual_evidence_receipt_or_specific_question(setup):
    module, _, engine, service = setup
    for eid in ("E01", "E03"):
        engine.give_evidence(eid, ["player"])
    assert "captain_enabled_hold" not in options(engine, module, "guard")
    for eid in ("E01", "E03"):
        engine.give_evidence(eid, ["guard"])
    assert "captain_enabled_hold" in options(engine, module, "guard")


def test_guard_corrects_exclusive_mechanic_accusation(setup):
    module, _, engine, _ = setup
    say(engine, "guard", "只有机械师碰过系统，肯定是他。", "broadcast")
    assert "captain_enabled_hold" in options(engine, module, "guard")


def test_captain_disclosure_uses_own_received_evidence_not_player_inventory(setup):
    module, _, engine, service = setup
    for eid in ("E02", "E06"):
        engine.give_evidence(eid, ["player"])
    assert "captain_enabled_hold" not in options(engine, module, "captain")
    before = engine.state.model_copy(deep=True)
    with pytest.raises(RuleError, match="披露"):
        disclose(engine, service, module, "captain", ["captain_enabled_hold"])
    assert engine.state == before
    engine.give_evidence("E02", ["captain"])
    assert "captain_enabled_hold" not in options(engine, module, "captain")
    engine.give_evidence("E06", ["captain"])
    assert "captain_enabled_hold" in options(engine, module, "captain")
    disclose(engine, service, module, "captain", ["captain_enabled_hold"], text="我亲自启用了保持，当时低估了危险。")
    assert any(o.observer_id == "player" and o.source_actor_id == "captain" and "captain_enabled_hold" in o.proposition_ids for o in engine.state.observations)


def test_witness_to_captain_does_not_verify_player_case_until_player_hears_it(setup):
    module, _, engine, service = setup
    say(engine, "guard", "12分钟前是谁操作保持？")
    disclose(engine, service, module, "guard", ["captain_enabled_hold"], ["captain"])
    engine.give_evidence("E02", ["captain", "player"])
    assert "captain_enabled_hold" in options(engine, module, "captain")
    assert not engine.state.flags.get("guard_witness")
    engine.final_outcome("station")
    assert not engine.state.outcome["captain_action_verified"]
    disclose(engine, service, module, "guard", ["captain_enabled_hold"])
    engine.final_outcome("station")
    assert engine.state.outcome["captain_action_verified"]


@pytest.mark.parametrize("channel,trust,text,allowed", [
    ("private", 1, "我承诺完整保留事故记录和出处。", True),
    ("private", 0, "我承诺完整保留事故记录和出处。", False),
    ("broadcast", 1, "我承诺完整保留事故记录和出处。", False),
    ("private", 1, "我不承诺完整保留事故记录。", False),
    ("private", 1, "我想看看记录。", False),
])
def test_captain_private_promise_requires_explicit_heard_commitment_and_cooperation(setup, channel, trust, text, allowed):
    module, _, engine, _ = setup
    engine.state.relations["captain"] = trust
    engine.state.flags["promise_preserve_records"] = True  # A global promise is insufficient.
    say(engine, "captain", text, channel)
    assert ("captain_enabled_hold" in options(engine, module, "captain")) is allowed


def test_concrete_resource_cooperation_can_replace_trust_bonus(setup):
    module, _, engine, _ = setup
    say(engine, "captain", "我保证完整保留全部事故记录。")
    assert "captain_enabled_hold" not in options(engine, module, "captain")
    engine.move("player", "bridge")
    engine.emit("ItemTransferred", {"item_id": "recorder", "from_owner": "player", "to_owner": "captain"})
    assert "captain_enabled_hold" in options(engine, module, "captain")


def test_registered_fact_relay_retains_testimony_source_without_new_eyewitness(setup):
    module, _, engine, service = setup
    say(engine, "guard", "12分钟前是谁操作保持？")
    disclose(engine, service, module, "guard", ["captain_enabled_hold"], ["doctor"])
    assert "captain_enabled_hold" in options(engine, module, "doctor")
    disclose(engine, service, module, "doctor", ["captain_enabled_hold"], text="韩松刚才向我说明了他的目击。")
    observations = [o for o in engine.state.observations if o.observer_id == "player" and "captain_enabled_hold" in o.proposition_ids]
    assert observations[-1].source_kind == "testimony"
    assert observations[-1].source_observation_ids
    assert not engine.state.flags.get("guard_witness")


def test_reported_observation_cannot_be_relabelled_as_firsthand(setup):
    module, _, engine, service = setup
    say(engine, "guard", "12分钟前是谁操作保持？")
    disclose(engine, service, module, "guard", ["captain_enabled_hold"], ["doctor"])
    source = next(o for o in reversed(engine.state.observations) if o.observer_id == "doctor" and "captain_enabled_hold" in o.proposition_ids)
    before = engine.state.model_copy(deep=True)
    with pytest.raises(ValueError, match="亲历"):
        engine.emit("FactDisclosed", {"observer_ids": ["player"], "proposition_ids": ["captain_enabled_hold"],
            "content": "伪装为亲历的转述", "source_kind": "firsthand", "source_observation_ids": [source.observation_id]}, "doctor")
    assert engine.state == before


def test_local_testimony_requires_listener_to_be_present(setup):
    module, _, engine, service = setup
    say(engine, "guard", "12分钟前是谁操作保持？")
    disclose(engine, service, module, "guard", ["captain_enabled_hold"], channel="local")
    assert not engine.state.flags.get("guard_witness")
    assert not any(o.observer_id == "player" and "captain_enabled_hold" in o.proposition_ids for o in engine.state.observations)


def test_doctor_explanation_and_reading_original_label_keep_different_sources(setup):
    module, _, engine, service = setup
    say(engine, "doctor", "如何保冷和转存细胞？")
    disclose(engine, service, module, "doctor", ["E05"], text="我来说明货物用途和转存步骤。")
    spoken = [o for o in engine.state.observations if o.observer_id == "player" and "E05" in o.evidence_refs][-1]
    assert spoken.modality == "reported" and spoken.source_kind == "testimony"
    engine.give_evidence("E05", ["player"])
    record = [o for o in engine.state.observations if o.observer_id == "player" and "E05" in o.evidence_refs][-1]
    assert record.modality == "read" and record.source_kind == "record"


def test_authorized_cargo_card_allows_sourced_atoms_without_granting_whole_card(setup):
    module, _, engine, service = setup
    say(engine, "doctor", "请说明细胞的转存条件。")
    atoms = {"cell_transfer_standard_cases", "cell_expiry_4_ticks"}
    assert atoms <= options(engine, module, "doctor")
    before = dict(engine.state.world_facts)
    disclose(engine, service, module, "doctor", sorted(atoms), text="内盒可转入标准保冷箱，原箱断冷后四格失效。")
    received = {p for o in engine.state.observations if o.observer_id == "player" for p in o.proposition_ids}
    assert atoms <= received
    assert "E05" not in engine.state.evidence.get("player", [])
    assert engine.state.world_facts == before


def test_authorized_card_cannot_supply_an_atom_missing_from_npc_observations(setup):
    module, _, engine, _ = setup
    engine.state.observations = [o for o in engine.state.observations
        if not (o.observer_id == "doctor" and "cell_expiry_4_ticks" in o.proposition_ids)]
    say(engine, "doctor", "请说明细胞的转存条件。")
    available = options(engine, module, "doctor")
    assert {"E05", "cell_transfer_standard_cases"} <= available
    assert "cell_expiry_4_ticks" not in available
    assert "cell_expiry_4_ticks" in engine.state.world_facts


def test_authorized_cards_do_not_expand_into_uncovered_private_atoms(setup):
    module, _, engine, _ = setup
    engine.give_evidence("E02", ["captain"])
    assert "captain_card_used" in options(engine, module, "captain")
    assert "captain_enabled_hold" not in options(engine, module, "captain")
    engine.give_evidence("E07", ["doctor"])
    assert "cold_cores_charged" in options(engine, module, "doctor")
    assert "doctor_saw_old_uncharged_cores" not in options(engine, module, "doctor")


def test_authorized_card_can_cover_a_restricted_atom_but_not_other_secrets(setup):
    module, _, engine, _ = setup
    assert "captain_ordered_connection" not in options(engine, module, "captain")
    engine.give_evidence("E03", ["captain"])
    assert "captain_ordered_connection" in options(engine, module, "captain")
    assert "captain_enabled_hold" not in options(engine, module, "captain")


def test_next_goal_is_validated_persisted_and_replayed_atomically(setup):
    module, initial, engine, service = setup
    choice = "guard:goal:2"
    decision = NpcDecision(actor_id="guard", action={"kind": "WAIT"}, next_goal_id=choice)
    service.apply_npc(engine, "guard", decision, view(engine, module, "guard"))
    assert engine.state.actors["guard"].current_goal_id == choice
    assert view(engine, module, "guard").current_goal_id == choice
    replayed = initial
    for event in engine.events:
        replayed = apply_event(replayed, event)
    assert replayed == engine.state
    before = engine.state.model_copy(deep=True)
    bad = NpcDecision(actor_id="guard", next_goal_id="guard:goal:0", utterance={"text": "秘密", "disclose_fact_ids": ["made_up_fact"]})
    with pytest.raises(RuleError):
        service.apply_npc(engine, "guard", bad, view(engine, module, "guard"))
    assert engine.state == before


def test_limited_critical_action_language_rejects_wrong_role_and_premature_success(setup):
    module, _, engine, service = setup
    with pytest.raises(RuleError, match="转存"):
        service.apply_npc(engine, "mechanic", NpcDecision(actor_id="mechanic", utterance={"text": "我按说明开始转存。"}), view(engine, module, "mechanic"))
    engine.resolve_player(ActionIntent(actor_id="player", kind="ASK", spoken_text="医生，请协助转存。", channel="private", recipient_ids=["doctor"], target_ids=["doctor"], module_action_id="transfer_cells"))
    decision = NpcDecision(actor_id="doctor", action={"kind": "MODULE_ACTION", "module_action_id": "transfer_cells"}, utterance={"text": "我已经完成转存，细胞安全了。"})
    before = engine.state.model_copy(deep=True)
    with pytest.raises(RuleError, match="裁决|完成"):
        service.apply_npc(engine, "doctor", decision, view(engine, module, "doctor"))
    assert engine.state == before


def test_public_event_text_uses_chinese_names(setup):
    _, _, engine, _ = setup
    engine.move("player", "engine")
    engine.emit("HealthChanged", {"actor_id": "player", "health": "strained"})
    engine.emit("ItemTransferred", {"item_id": "medical_supply_1", "from_owner": "doctor", "to_owner": "player"}, "doctor")
    text = " ".join(o.content for o in engine.state.observations if o.event_id.startswith("disclosure:"))
    assert "动力舱" in text and "轻伤" in text and "处置物资" in text
    assert "engine" not in text and "strained" not in text and "medical_supply" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["semantic", "parse", "budget_exhausted"])
async def test_semantic_npc_failure_uses_one_same_view_repair(tmp_path, failure_mode):
    module = load_module()
    state = initial_state(module, "核验员")
    store = Store(tmp_path / "repair.db")
    store.create(state)

    class RepairingAgent:
        calls = 0
        repairs = 0
        metrics = []
        repaired_views = []

        async def interpret(self, _):
            return PlayerIntent(kind="ASK", target_ids=["guard"], spoken_text="12分钟前谁操作了保持？")

        async def decide_npc(self, v):
            if failure_mode == "parse":
                raise InvalidOutput("格式解析失败，原调用已处理格式修复预算")
            return NpcDecision(actor_id=v.actor_id, utterance={"text": "说明", "disclose_fact_ids": ["made_up_fact"]})

        async def repair_npc(self, v, reason):
            self.repairs += 1
            self.repaired_views.append((v.model_dump(), reason))
            return NpcDecision(actor_id=v.actor_id, utterance={"text": "我在场看到了操作。", "channel": "private", "recipient_ids": ["player"], "disclose_fact_ids": ["captain_enabled_hold"]})

        async def narrate(self, v):
            return NarrationResult(text="船员提供了证词。", source_event_ids=[e["event_id"] for e in v.events])

    agent = RepairingAgent()
    if failure_mode == "budget_exhausted":
        agent.repairs = 1
    service = TurnService(store, module, None, agent_factory=lambda *args: agent)
    receipt = await service.submit_turn(state.campaign_id, PlayerCommand(client_request_id="repair", expected_revision=0, text="问保安事故前12分钟操作", speech_channel="private", recipient_ids=["guard"]))
    await service.tasks[receipt.turn_id]
    assert service.status(state.campaign_id, receipt.turn_id)["status"] == "committed"
    assert agent.repairs == (0 if failure_mode == "parse" else 1)
    assert bool(store.snapshot(state.campaign_id).flags.get("guard_witness")) is (failure_mode == "semantic")
    assert "made_up_fact" not in json.dumps(store.snapshot(state.campaign_id).world_facts)
