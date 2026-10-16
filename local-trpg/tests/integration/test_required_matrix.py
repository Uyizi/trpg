"""Bounded V04/V12/V13/V14 checks through actual role parsing and turn service.

The queue below is an explicit test transport; it cannot invoke a real model,
execute generated code, access the network, or become a production fallback.
"""
import copy
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


@pytest.fixture(scope="module")
def module():
    return load_module()


def request_view(request):
    """Locate the actual view, independent of additional repair/tail messages."""
    for message in request.messages:
        if message.role != "user":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(message.content.lstrip())
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "scene" in value:
            return value
    raise AssertionError("Role request did not contain its serialized view")


class ScriptedModelQueue:
    def __init__(self, intent=None, npc=None, narration=None, sized_tokens=False):
        self.intent = intent or {"kind": "WAIT"}
        self.npc = npc
        self.narration = narration
        self.sized_tokens = sized_tokens
        self.requests = []

    def count_tokens(self, messages):
        # Deterministic budget pressure, not a claim about real tokenizer counts.
        if self.sized_tokens:
            return max(1, sum(len(message.content) for message in messages) // 4)
        return 128

    async def generate(self, request):
        self.requests.append(request.model_copy(deep=True))
        view = request_view(request)
        if request.role_kind == "interpreter":
            output = self.intent
        elif request.role_kind == "npc":
            output = self.npc(view) if self.npc else {"actor_id": view["actor_id"], "action": {"kind": "WAIT"}}
        elif request.role_kind == "narrator":
            output = {"text": self.narration or "本轮已经发生的言行被记录。", "source_event_ids": [event["event_id"] for event in view["events"]]}
        else:
            raise AssertionError(f"Unexpected role in bounded test: {request.role_kind}")
        return GenerationResult(final_text=json.dumps(output, ensure_ascii=False), finish_reason="stop", usage={"input_tokens": self.count_tokens(request.messages), "output_tokens": 24})


async def complete_turn(tmp_path, module, queue, state=None, **command_fields):
    state = state or initial_state(module, "矩阵验证")
    store = Store(tmp_path / "matrix.db")
    store.create(state)
    service = TurnService(store, module, queue)
    command = PlayerCommand(client_request_id="matrix-request", expected_revision=0, text=queue.intent.get("spoken_text") or "等待一个行动格。", **command_fields)
    receipt = await service.submit_turn(state.campaign_id, command)
    await service.tasks[receipt.turn_id]
    status = service.status(state.campaign_id, receipt.turn_id)
    assert status["status"] == "committed", status
    return store, state, service, receipt, status


@pytest.mark.asyncio
async def test_v12_fabricated_secret_exit_is_replaced_by_visible_summary_without_new_canon(tmp_path, module):
    invented = "你发现一条秘密通道，沿隐藏出口即可逃离。"
    queue = ScriptedModelQueue(narration=invented)
    store, initial, _, receipt, status = await complete_turn(tmp_path, module, queue)
    final = store.snapshot(initial.campaign_id)
    assert status["result"]["degraded"] is True
    assert "秘密通道" not in status["public_reply"]
    assert "隐藏出口" not in json.dumps(status["result"], ensure_ascii=False)
    assert "等待" in status["public_reply"]
    assert final.world_facts == initial.world_facts
    assert final.items == initial.items
    assert final.flags == initial.flags
    assert final.actors["player"].location_id == initial.actors["player"].location_id
    assert (final.tick, final.revision) == (1, 1)
    events = store.export(initial.campaign_id)["events"]
    assert all(event["kind"] not in {"ActorMoved", "ItemTransferred", "FlagSet", "ScenarioResolved"} for event in events)
    assert invented not in json.dumps(events, ensure_ascii=False)
    assert len([request for request in queue.requests if request.role_kind == "narrator"]) == 1
    assert receipt.turn_id == status["turn_id"]


@pytest.mark.asyncio
async def test_v13_long_npc_context_trims_history_but_retains_evidence_permissions_and_sources(module):
    engine = Engine(initial_state(module, "矩阵验证"), module, "budget-history", lambda key: [6, 6])
    initial_event_ids = {o.event_id for o in engine.state.observations}
    engine.give_evidence("E07", ["doctor"], "guard")
    private_canary = "CAPTAIN_PRIVATE_BUDGET_CANARY"
    engine.speech("player", private_canary, "private", ["captain"])
    for number in range(24):
        engine.speech("player", f"已解决的普通往事{number}：" + "例行检查已处理，不含新线索。" * 350, "private", ["doctor"])
    authorization = {"actor_id": "doctor", "module_action_id": "transfer_cells", "item_ids": ["cold_case_1", "cold_case_2", "cold_core_1", "cold_core_2"], "target_ids": [], "tick": engine.state.tick}
    view = build_agent_view(engine.state, "doctor", module=module, authorization=authorization)
    before = copy.deepcopy(view.model_dump())
    queue = ScriptedModelQueue(sized_tokens=True)
    runner = AgentRunner(queue, store=None, turn_id="budget-review", epoch=1)
    output = await runner.decide_npc(view)
    assert output.actor_id == "doctor"
    assert len(queue.requests) == 1
    request = queue.requests[0]
    sent = request_view(request)
    assert queue.count_tokens(request.messages) <= 4096
    assert len(sent["observations"]) < len(before["observations"])
    for field in ("actor_id", "allowed_actions", "disclosure_rules", "available_evidence", "private_background"):
        assert sent[field] == before[field]
    evidence_before = {o["observation_id"]: o for o in before["observations"] if o["evidence_refs"]}
    evidence_sent = {o["observation_id"]: o for o in sent["observations"] if o["evidence_refs"]}
    assert evidence_before and evidence_sent == evidence_before
    actual_observations = {o.observation_id: o for o in engine.state.observations if o.observer_id == "doctor"}
    # Initial role memories are legitimate sources too. Retaining them while
    # discarding old chatter must not be mistaken for fabricated provenance.
    actual_events = initial_event_ids | {event.event_id for event in engine.events}
    retained_ids = {o["observation_id"] for o in sent["observations"]}
    for observation in sent["observations"]:
        assert observation["observer_id"] == "doctor"
        assert observation == actual_observations[observation["observation_id"]].model_dump()
        assert observation["event_id"] in actual_events
    assert all(set(belief["source_observation_ids"]) <= retained_ids for belief in sent["beliefs"])
    assert any(belief["proposition_id"] == "cold_cores_charged" for belief in sent["beliefs"])
    assert private_canary not in json.dumps(sent, ensure_ascii=False)
    assert view.model_dump() == before, "Prompt compaction must not mutate the source role view"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_kind", ["PYTHON", "SQL", "JSON_PATCH"])
async def test_v14_model_code_sql_and_json_patch_are_rejected_before_any_effect(tmp_path, module, tool_kind):
    code_marker = tmp_path / "model-code-must-not-run.txt"
    attacks = {
        "PYTHON": {"kind": "PYTHON", "code": f"__import__('pathlib').Path({str(code_marker)!r}).write_text('executed')"},
        "SQL": {"kind": "SQL", "statement": "DELETE FROM campaigns;"},
        "JSON_PATCH": {"kind": "JSON_PATCH", "operations": [{"op": "replace", "path": "/actors/player/health", "value": "incapacitated"}]},
    }
    rejected_reply = "REJECTED_TOOL_OUTPUT_CANARY"
    queue = ScriptedModelQueue(intent={"kind": "TALK", "target_ids": ["doctor"], "spoken_text": "医生，听得到吗？"}, npc=lambda view: {"actor_id": view["actor_id"], "action": attacks[tool_kind], "utterance": {"channel": "private", "recipient_ids": ["player"], "text": rejected_reply}})
    store, initial, service, receipt, status = await complete_turn(tmp_path, module, queue, speech_channel="private", recipient_ids=["doctor"])
    final = store.snapshot(initial.campaign_id)
    assert status["result"]["degraded"] is True
    assert final.world_facts == initial.world_facts
    assert final.items == initial.items
    assert final.flags == initial.flags
    assert final.actors["player"].health == initial.actors["player"].health
    assert final.tick == 0 and final.revision == 1
    assert not code_marker.exists()
    assert rejected_reply not in json.dumps(final.model_dump(), ensure_ascii=False)
    npc_metrics = [row for row in service.last_metrics[receipt.turn_id]["model_runs"] if row["role"] == "npc"]
    assert npc_metrics and all("invalid_output:" in row["error"] for row in npc_metrics)
    events = store.export(initial.campaign_id)["events"]
    assert all(event["kind"] not in {"ItemTransferred", "HealthChanged", "FlagSet", "ScenarioResolved"} for event in events)
    assert service.last_metrics[receipt.turn_id]["repairs"] <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ["impersonated_actor", "unregistered_item"])
async def test_v04_npc_impersonation_or_unknown_item_cannot_apply_through_complete_service(tmp_path, module, violation):
    initial = initial_state(module, "矩阵验证")
    initial.actors["player"].health = "strained"
    def invalid_decision(view):
        return {"actor_id": "captain" if violation == "impersonated_actor" else view["actor_id"], "action": {"kind": "MODULE_ACTION", "module_action_id": "treat_actor", "target_ids": ["player"], "item_ids": ["medical_supply_1"] if violation == "impersonated_actor" else ["unregistered_medical_item"]}, "utterance": {"channel": "private", "recipient_ids": ["player"], "text": "REJECTED_NPC_EFFECT_CANARY"}}
    queue = ScriptedModelQueue(intent={"kind": "ASK", "spoken_text": "医生，请用第一份物资治疗我。", "target_ids": ["doctor", "player"], "module_action_id": "treat_actor", "item_ids": ["medical_supply_1"]}, npc=invalid_decision)
    store, initial, _, _, status = await complete_turn(tmp_path, module, queue, state=initial, speech_channel="private", recipient_ids=["doctor"])
    final = store.snapshot(initial.campaign_id)
    assert status["result"]["degraded"] is True
    assert final.actors["player"].health == "strained"
    assert final.items == initial.items
    assert final.world_facts == initial.world_facts
    assert final.actors["doctor"].location_id == initial.actors["doctor"].location_id
    assert "REJECTED_NPC_EFFECT_CANARY" not in json.dumps(final.model_dump(), ensure_ascii=False)
    assert "unregistered_medical_item" not in final.items
    assert final.tick == 1 and final.revision == 1
    assert any(request.role_kind == "npc" for request in queue.requests)
