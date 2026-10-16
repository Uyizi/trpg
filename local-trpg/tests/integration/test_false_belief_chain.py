"""Scripted model proposals exercise the real turn/engine belief boundary.

This does not test whether a live model spontaneously believes a lie.
"""
import pytest

from local_trpg.engine.projection import build_agent_view
from local_trpg.engine.rules import initial_state
from local_trpg.engine.turns import TurnService
from local_trpg.modules import load_module
from local_trpg.schemas import NarrationResult, NpcDecision, PlayerCommand, PlayerIntent
from local_trpg.storage import Store


@pytest.mark.asyncio
async def test_heard_lie_can_revise_sourced_belief_without_changing_canon(tmp_path):
    module = load_module()
    initial = initial_state(module, "信念边界核验员")
    proposition = "cold_cores_charged"
    lie = "我检查过了，这次补给的两块储冷芯都没有充好。"
    assert initial.world_facts[proposition] is True
    assert next(b for b in initial.beliefs if b.actor_id == "doctor"
                and b.proposition_id == proposition).stance == "doubted"
    store = Store(tmp_path / "false-belief.db")
    store.create(initial)

    class FakeModelProposals:
        """Only model outputs are scripted; speech and its source ID are real."""
        calls = 0
        repairs = 0
        metrics = []
        received_source = None

        async def interpret(self, view):
            self.calls += 1
            assert view.text == lie
            return PlayerIntent(kind="TALK", target_ids=["doctor"], spoken_text=lie)

        async def decide_npc(self, view):
            self.calls += 1
            assert view.actor_id == "doctor"
            self.received_source = next(o for o in view.observations
                if o["observer_id"] == "doctor" and o["source_actor_id"] == "player"
                and o["modality"] == "heard" and o["content"] == lie)
            return NpcDecision(actor_id="doctor", action={"kind": "WAIT"},
                belief_updates=[{"proposition_id": proposition, "stance": "rejected",
                    "basis": "testimony",
                    "source_observation_ids": [self.received_source["observation_id"]]}])

        async def narrate(self, view):
            self.calls += 1
            return NarrationResult(text="本轮可见情况已记录。",
                source_event_ids=[event["event_id"] for event in view.events])

    fake = FakeModelProposals()
    service = TurnService(store, module, None, agent_factory=lambda *_: fake)
    receipt = await service.submit_turn(initial.campaign_id, PlayerCommand(
        client_request_id="heard-lie", expected_revision=0, text=lie,
        speech_channel="private", recipient_ids=["doctor"]))
    await service.tasks[receipt.turn_id]
    assert service.status(initial.campaign_id, receipt.turn_id)["status"] == "committed"
    assert service.last_metrics[receipt.turn_id]["degraded"] is False

    after = store.snapshot(initial.campaign_id)
    events = store.export(initial.campaign_id)["events"]
    speech, = [e for e in events if e["kind"] == "SpeechOccurred"]
    revised, = [e for e in events if e["kind"] == "BeliefRevised"]
    source = fake.received_source
    assert speech["actor_id"] == "player" and speech["payload"]["text"] == lie
    assert speech["beat_index"] < revised["beat_index"]
    assert source["event_id"] == speech["event_id"]
    assert source["proposition_ids"] == [] and source["evidence_refs"] == []
    assert revised["actor_id"] == "doctor"
    belief, = [b for b in after.beliefs if b.actor_id == "doctor"
               and b.proposition_id == proposition]
    assert belief.stance == "rejected" and belief.basis == "testimony"
    assert belief.source_observation_ids == [source["observation_id"]]
    assert revised["payload"] == belief.model_dump()
    assert after.world_facts == initial.world_facts
    assert after.items == initial.items and after.cells == initial.cells
    assert after.evidence == initial.evidence

    next_view = build_agent_view(after, "doctor", module=module)
    assert belief.model_dump() in next_view.beliefs
    assert source in next_view.observations
    assert all(o["observer_id"] == "doctor" for o in next_view.observations)
