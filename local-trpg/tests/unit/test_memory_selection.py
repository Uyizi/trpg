"""Bounded retrieval must retain what the NPC just heard and actually did."""
import pytest

from local_trpg.engine.projection import build_agent_view
from local_trpg.engine.rules import initial_state
from local_trpg.memory import retrieve
from local_trpg.modules import load_module
from local_trpg.schemas import Observation


def observation(identifier, tick, content, *, observer="doctor", source="player", modality="heard", refs=(), source_ids=()):
    return Observation(observation_id=identifier, event_id="event:" + identifier,
        observer_id=observer, source_actor_id=source, modality=modality,
        content=content, evidence_refs=list(refs), observed_at=tick,
        source_kind="record" if refs else None,
        source_observation_ids=list(source_ids))


@pytest.fixture
def crowded_history():
    module = load_module()
    state = initial_state(module, "记忆回归")
    state.tick = 10
    state.observations = [
        observation("supply-receipt", 1, "本次补给票：两块储冷芯已经充好。", source="guard", modality="read", refs=["E07"], source_ids=["guard-receipt"]),
        *[observation(f"old-record-{i}", 2, f"旧货物说明第{i}次核对。", modality="read", refs=["E05"]) for i in range(32)],
        observation("own-transfer-completed", 7, "我已将4份内盒转入保冷箱，首次储冷期限为第25格。", source="doctor", modality="direct"),
        *[observation(f"recent-chatter-{i}", 9, f"已结束的普通话题{i}。") for i in range(24)],
        observation("current-player-question", 10, "刚才转存完成了吗，接下来怎样撤离？"),
        observation("other-private-secret", 10, "OTHER_ACTOR_SECRET_CANARY", observer="captain", source="captain", modality="direct"),
    ]
    return module, state


def test_old_evidence_cannot_evict_current_question_and_recent_own_result(crowded_history):
    module, state = crowded_history
    before = state.model_dump_json()
    selected = build_agent_view(state, "doctor", module=module).observations
    ids = {row["observation_id"] for row in selected}
    assert {"current-player-question", "own-transfer-completed", "supply-receipt"} <= ids
    assert len(selected) <= 16
    assert all(row["observer_id"] == "doctor" for row in selected)
    assert "OTHER_ACTOR_SECRET_CANARY" not in str(selected)
    originals = {row.observation_id: row.model_dump() for row in state.observations}
    assert all(row == originals[row["observation_id"]] for row in selected)
    assert [row["observed_at"] for row in selected] == sorted(row["observed_at"] for row in selected)
    assert state.model_dump_json() == before


def test_repeated_same_source_evidence_does_not_fill_all_memory_slots(crowded_history):
    _, state = crowded_history
    state.observations = [observation(f"repeat-{i}", i, "储冷芯已充好。", source="guard", modality="read", refs=["E07"], source_ids=["original-receipt"]) for i in range(20)]
    state.observations.extend([
        observation("independent-report", 20, "储冷芯已充好。", source="comms", modality="reported", refs=["E07"], source_ids=["comms-inspection"]),
        observation("contradicting-report", 21, "储冷芯可能没有充好。", source="guard", modality="reported", refs=["E07"], source_ids=["original-receipt"]),
    ])
    selected = retrieve(state, "doctor", limit=16)
    ids = {row["observation_id"] for row in selected}
    assert {identifier for identifier in ids if identifier.startswith("repeat-")} == {"repeat-19"}
    assert {"independent-report", "contradicting-report"} <= ids
    assert next(row for row in selected if row["observation_id"] == "repeat-19")["source_observation_ids"] == ["original-receipt"]


@pytest.mark.parametrize("limit", [0, 1, 3, 8])
def test_small_limits_keep_current_context_without_exceeding_budget(crowded_history, limit):
    _, state = crowded_history
    selected = retrieve(state, "doctor", limit=limit)
    assert len(selected) <= limit
    if limit:
        assert "current-player-question" in {row["observation_id"] for row in selected}
    if limit >= 3:
        assert "own-transfer-completed" in {row["observation_id"] for row in selected}
        assert any(row["evidence_refs"] and row["event_id"] for row in selected)


def test_retrieval_does_not_reconstruct_missing_results_from_world_state(crowded_history):
    _, state = crowded_history
    state.observations = [row for row in state.observations if row.source_actor_id != "doctor"]
    state.flags["cells_transferred"] = True
    state.flags["private_operation_note"] = "UNOBSERVED_WORLD_RESULT_CANARY"
    selected = retrieve(state, "doctor", query="UNOBSERVED_WORLD_RESULT_CANARY", limit=16)
    assert "UNOBSERVED_WORLD_RESULT_CANARY" not in str(selected)
    assert "own-transfer-completed" not in {row["observation_id"] for row in selected}
    assert all(row["observation_id"] in {o.observation_id for o in state.observations if o.observer_id == "doctor"} for row in selected)
