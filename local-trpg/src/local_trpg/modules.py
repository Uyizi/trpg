"""Load server-owned adventure data without evaluating configuration strings.

Only ``module['public']`` is suitable for an unauthenticated presentation DTO.
The world dictionary contains author truths, private observations and test routes;
it must never be serialized wholesale into an NPC or player-facing context.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_MODULE_PATH = Path(__file__).resolve().parents[2] / "modules" / "reserve_cooling"
MODULE_ACTION_IDS = frozenset({
    "take_cold_cases", "transfer_cells", "begin_safe_isolation",
    "complete_safe_isolation", "isolate_branch", "restore_main_loop",
    "treat_actor", "send_distress", "prepare_pod", "assemble_crew",
    "board_pod", "launch_pod", "assist_evacuation",
})
NPC_ACTION_ALLOWLIST = {
    "captain": frozenset({"assemble_crew"}),
    "doctor": frozenset({"treat_actor", "transfer_cells"}),
    "mechanic": frozenset({"isolate_branch", "restore_main_loop"}),
    "comms": frozenset({"send_distress"}),
    "guard": frozenset({"prepare_pod", "assist_evacuation"}),
}
# These are declarative names for engine dispatch, not Python expressions.
GUARD_IDS = frozenset("""
    actor_capable at_emergency_cabinet at_identified_manual_valve
    audit_and_witness_or_watch_log boarding_consent_and_help
    capable_unblocked_normal_corridor captain_hold_corroborated
    cargo_branch_isolated cargo_cooling_or_transfer_question
    cargo_instructions_and_supply_seen cargo_instructions_known
    cases_available_at_cargo cause_evidence_or_equivalent_validation
    cells_valid_at_completion communication_equipment_usable
    cooling_purpose_explained_or_cargo_instructions correct_operation_card_sequence
    crisis_captain_available crisis_comms_available crisis_comms_available_no_distress
    crisis_guard_available_pod_unprepared crisis_unresolved
    distress_cells_and_pod_confirmed doctor_and_mechanic_available
    doctor_available_or_acting engine_not_destroyed evacuation_deadline_reached
    evacuation_phase evacuation_plan_accepted explicit_standard_distress_with_requester
    injured_target_accessible main_loop_not_restored mechanic_only_access_accusation
    mechanic_or_valid_guidance medical_supply_available nearby_injury
    normal_corridor_access normal_maintenance_access ordinary_tools_available
    panel_and_work_order_or_hold_time_question participants_and_items_at_airlock
    player_onboard pod_launched pod_operable pod_prepared
    private_record_promise_and_cooperation private_timed_query_with_source_promise_or_audit
    reasonable_connection_question reasonable_inspection reasonable_repair_question
    reasonable_route_question restored_and_station_selected restored_with_partial_cargo_loss
    safe_isolation_preparation_valid specific_bridge_watch_query supply_receipt_seen
    tick_at_least_9 unsupported_public_accusation usable_cold_cores
""".split())
EFFECT_IDS = frozenset("""
    accept_and_announce_evacuation accept_evacuation announce_valve_heat_risk
    ask_about_distress assemble_available_crew assist_cold_cases assist_named_actor_to_airlock board_present_participants
    broadcast_flow_warning destroy_engine_and_expose_cargo form_captain_operator_belief
    improve_health_one_step isolate_cargo_branch isolation_failed_strain launch_actual_onboard
    mark_remaining_ship_occupants_lost move_available_crew_to_airlock perform_requested_action
    prepare_pod prepare_safe_isolation prepare_solo_transfer prioritize_treatment restore_main_loop
    send_first_distress support_cargo_diversion take_cases_and_cores transfer_valid_cells
    update_cold_core_belief withdraw_personal_judgment
""".split())
_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_PUBLIC_KEYS = {
    "schema_version", "module_id", "module_version", "title", "opening", "premise",
    "locations", "characters", "player", "rules",
}


class ModuleValidationError(ValueError):
    """Malformed module or unknown registry reference."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModuleValidationError(message)


def _object(value: Any, path: str) -> dict:
    _require(isinstance(value, dict), f"{path}: expected object")
    return value


def _list(value: Any, path: str) -> list:
    _require(isinstance(value, list), f"{path}: expected list")
    return value


def _text(value: Any, path: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{path}: expected integer >= {minimum}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ModuleValidationError(f"non-finite JSON number: {value}")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModuleValidationError(f"cannot read {path.name}: {exc}") from exc
    return _object(data, path.name)


def _table(world: dict, key: str) -> dict:
    table = _object(world.get(key), f"world.{key}")
    _require(bool(table), f"world.{key}: empty table")
    for identifier, value in table.items():
        _require(bool(_ID.fullmatch(identifier)), f"world.{key}: invalid ID {identifier!r}")
        entry = _object(value, f"world.{key}.{identifier}")
        _require(entry.get("id") == identifier, f"world.{key}.{identifier}: id differs from key")
    return table


def _validate(public: dict, world: dict) -> None:
    for label, section in (("public", public), ("world", world)):
        _require(type(section.get("schema_version")) is int and section["schema_version"] == 1,
                 f"{label}.schema_version: only integer 1 is supported")
        for key in ("module_id", "module_version", "title"):
            _text(section.get(key), f"{label}.{key}")
    for key in ("module_id", "module_version"):
        _require(public[key] == world[key], f"public/world {key} mismatch")
    _require(set(public) == _PUBLIC_KEYS, "public: unexpected or missing fields; private data is not public")
    for key in ("opening", "premise"):
        _text(public.get(key), f"public.{key}")
    _list(public.get("rules"), "public.rules")
    tables = {key: _table(world, key) for key in
              ("locations", "actors", "facts", "evidence", "items", "actions", "routes")}
    _require(set(tables["actions"]) == MODULE_ACTION_IDS, "world.actions: unknown or missing registered action")
    _require(set(tables["actors"]) == set(NPC_ACTION_ALLOWLIST), "world.actors: expected the five registered NPCs")
    player = _object(world.get("player"), "world.player")
    _require(player.get("id") == "player", "world.player.id must be player")
    actor_ids = set(tables["actors"]) | {"player"}
    participants = _list(world.get("participants"), "world.participants")
    _require(len(participants) == len(actor_ids) and set(participants) == actor_ids,
             "world.participants: expected player and five distinct NPCs")
    initiative = _list(world.get("initiative_order"), "world.initiative_order")
    _require(len(initiative) == 5 and set(initiative) == set(tables["actors"]),
             "world.initiative_order: expected each NPC once")

    triggers = _list(world.get("triggers"), "world.triggers")
    trigger_ids = set()
    for i, raw in enumerate(triggers):
        trigger = _object(raw, f"world.triggers[{i}]")
        tid = _text(trigger.get("trigger_id"), f"world.triggers[{i}].trigger_id")
        _require(tid not in trigger_ids, f"duplicate trigger ID: {tid}")
        trigger_ids.add(tid)
        _integer(trigger.get("due_tick"), f"trigger.{tid}.due_tick")
        _require(trigger.get("once") is True, f"trigger.{tid}: must be once")
        _require(("module_action_id" in trigger) != ("effect_id" in trigger),
                 f"trigger.{tid}: expected exactly one action or effect")

    observations = _list(world.get("initial_observations"), "world.initial_observations")
    observation_ids = set()
    observation_actors = {}
    for i, raw in enumerate(observations):
        observation = _object(raw, f"world.initial_observations[{i}]")
        oid = _text(observation.get("observation_id"), f"observation[{i}].observation_id")
        _require(oid not in observation_ids, f"duplicate observation ID: {oid}")
        observation_ids.add(oid)
        observation_actors[oid] = observation.get("actor_id")
        _text(observation.get("source"), f"observation.{oid}.source")

    references = {}
    def refs(keys: str, ids: set | frozenset, plural: bool = False) -> None:
        for key in keys.split():
            references[key] = (ids, plural)
    refs("location_id destination_id", set(tables["locations"]))
    refs("location_ids connected_to", set(tables["locations"]), True)
    refs("actor_id target_id npc_actor_id", actor_ids)
    refs("actor_ids participants initiative_order expected_new_evidence_recipient_ids", actor_ids, True)
    refs("proposition_id", set(tables["facts"]))
    refs("proposition_ids expected_proposition_ids corroboration_proposition_ids causal_chain", set(tables["facts"]), True)
    refs("evidence_id", set(tables["evidence"]))
    refs("evidence_ids expected_evidence_ids presented_evidence_ids corroboration_evidence_ids", set(tables["evidence"]), True)
    refs("core_id container_id", set(tables["items"]))
    refs("item_ids inventory", set(tables["items"]), True)
    refs("module_action_id automatic_on_action_id obtain_action shared_effect_with", MODULE_ACTION_IDS)
    refs("guard_id access_guard_id", GUARD_IDS)
    refs("guard_ids", GUARD_IDS, True)
    refs("effect_id automatic_effect_id prepare_effect_id failure_effect_id", EFFECT_IDS)
    refs("expected_trigger_ids scheduled_plans", trigger_ids, True)
    refs("source_observation_ids", observation_ids, True)

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key in references:
                    known, plural = references[key]
                    values = _list(child, child_path) if plural else [child]
                    for identifier in values:
                        _require(isinstance(identifier, str) and identifier in known,
                                 f"{child_path}: unknown reference {identifier!r}")
                walk(child, child_path)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{path}[{i}]")
    walk(world, "world")
    walk(public, "public")

    for aid, actor in {**tables["actors"], "player": player}.items():
        path = f"world.actors.{aid}"
        for key in ("name", "role", "location_id"):
            _text(actor.get(key), f"{path}.{key}")
        _require(actor.get("health") in {"healthy", "strained", "injured", "incapacitated"},
                 f"{path}.health: unknown health")
        skills = _object(actor.get("skills"), f"{path}.skills")
        _require(set(skills) == {"observe", "technical", "social", "action"}, f"{path}.skills: invalid skill names")
        for skill, modifier in skills.items():
            _integer(modifier, f"{path}.skills.{skill}", -10)
        inventory = _list(actor.get("inventory"), f"{path}.inventory")
        for iid in inventory:
            _require(tables["items"][iid].get("owner") == aid, f"{path}.inventory: item {iid} has another owner")
        allowed = _list(actor.get("actions"), f"{path}.actions")
        for action_id in allowed:
            _require(action_id in MODULE_ACTION_IDS, f"{path}.actions: unknown action {action_id}")
            if aid != "player":
                _require(action_id in NPC_ACTION_ALLOWLIST[aid], f"{path}.actions: action violates NPC allowlist")
            _require(aid in tables["actions"][action_id]["actor_ids"], f"{path}.actions: actor missing from action allowlist")
        if aid == "player":
            continue
        for key in ("goals", "private_background", "initial_beliefs", "disclosure_rules", "cooperation_rules"):
            _list(actor.get(key), f"{path}.{key}")
        _text(actor.get("speaking_style"), f"{path}.speaking_style")
        for background in actor["private_background"]:
            _require(background.get("text") == tables["facts"][background["proposition_id"]]["text"],
                     f"{path}.private_background: proposition text differs from canon")
            _text(background.get("source"), f"{path}.private_background.source")
        for belief in actor["initial_beliefs"]:
            _require(belief.get("stance") in {"accepted", "believed", "doubted", "rejected", "uncertain"},
                     f"{path}.initial_beliefs: invalid stance")
            for oid in belief["source_observation_ids"]:
                _require(observation_actors[oid] == aid, f"{path}: belief uses another actor's private observation")

    for iid, item in tables["items"].items():
        _text(item.get("name"), f"item.{iid}.name")
        _text(item.get("kind"), f"item.{iid}.kind")
        _require(item.get("owner") in actor_ids | {"ship"}, f"item.{iid}.owner: unknown owner")
    for pid, fact in tables["facts"].items():
        _text(fact.get("text"), f"fact.{pid}.text")
        _require(fact.get("fact_id") in {f"F{i:02d}" for i in range(1, 10)}, f"fact.{pid}: unknown fact group")
        _require(fact.get("truth") is True, f"fact.{pid}: initial canon truth must be true")
    grouped_facts = _object(world.get("fact_groups"), "world.fact_groups")
    _require(set(grouped_facts) == {f"F{i:02d}" for i in range(1, 10)}, "world.fact_groups: missing fixed fact group")
    grouped_ids = []
    for group, ids in grouped_facts.items():
        for pid in _list(ids, f"fact_groups.{group}"):
            _require(pid in tables["facts"] and tables["facts"][pid]["fact_id"] == group,
                     f"fact_groups.{group}: invalid proposition reference {pid}")
            grouped_ids.append(pid)
    _require(len(grouped_ids) == len(set(grouped_ids)) and set(grouped_ids) == set(tables["facts"]),
             "world.fact_groups: duplicate or ungrouped proposition")

    source_ids = set()
    for eid, evidence in tables["evidence"].items():
        for key in ("title", "text", "support_scope"):
            _text(evidence.get(key), f"evidence.{eid}.{key}")
        _require(evidence.get("initially_visible") is False, f"evidence.{eid}: initial evidence must be observed before granted")
        _require(bool(_list(evidence.get("proposition_ids"), f"evidence.{eid}.proposition_ids")),
                 f"evidence.{eid}: empty proposition list")
        sources = _list(evidence.get("sources"), f"evidence.{eid}.sources")
        _require(bool(sources), f"evidence.{eid}: missing sources")
        for source in sources:
            sid = _text(source.get("source_id"), f"evidence.{eid}.source_id")
            _require(sid not in source_ids, f"duplicate source ID: {sid}")
            source_ids.add(sid)
            _require("actor_id" in source or "location_id" in source, f"source.{sid}: missing source owner/location")
            _require(set(source.get("proposition_ids", [])) <= set(evidence["proposition_ids"]),
                     f"source.{sid}: proposition subset exceeds evidence support")

    for aid, action in tables["actions"].items():
        _integer(action.get("tick_cost"), f"action.{aid}.tick_cost")
        for actor_id in action["actor_ids"]:
            if actor_id != "player":
                _require(aid in NPC_ACTION_ALLOWLIST[actor_id], f"action.{aid}: NPC action allowlist violation")
    for trigger in triggers:
        if "module_action_id" in trigger:
            action = tables["actions"][trigger["module_action_id"]]
            _require(trigger.get("actor_id") in action["actor_ids"],
                     f"trigger.{trigger['trigger_id']}: actor action allowlist violation")
    for rid, route in tables["routes"].items():
        previous_tick = -1
        for step in _list(route.get("steps"), f"route.{rid}.steps"):
            tick = _integer(step.get("completion_tick"), f"route.{rid}.completion_tick")
            _require(tick >= previous_tick, f"route.{rid}: completion ticks out of order")
            previous_tick = tick
            if "npc_actor_id" in step:
                _require(step.get("module_action_id") in NPC_ACTION_ALLOWLIST[step["npc_actor_id"]],
                         f"route.{rid}: NPC action allowlist violation")
        result = _object(route.get("expected_outcome"), f"route.{rid}.expected_outcome")
        _require(0 <= _integer(result.get("crew_saved"), f"route.{rid}.crew_saved") <= 6,
                 f"route.{rid}: crew count exceeds six")
        _require(0 <= _integer(result.get("cells_saved"), f"route.{rid}.cells_saved") <= 4,
                 f"route.{rid}: cell count exceeds four")

    public_locations = _object(public.get("locations"), "public.locations")
    public_characters = _object(public.get("characters"), "public.characters")
    _require(set(public_locations) == set(tables["locations"]), "public.locations: unknown or missing location")
    _require(set(public_characters) == set(tables["actors"]), "public.characters: unknown or missing actor")
    for lid, loc in public_locations.items():
        _require(set(loc) == {"id", "name", "visible_objects"}, "public.location: unexpected private fields")
        _require(loc["id"] == lid, "public.location: id differs from key")
        _text(loc["name"], f"public.location.{lid}.name")
        for obj in _list(loc["visible_objects"], f"public.location.{lid}.visible_objects"):
            _text(obj, f"public.location.{lid}.visible_objects")
    for aid, actor in public_characters.items():
        _require(set(actor) == {"id", "name", "role"}, "public.character: unexpected private fields")
        _require(actor["id"] == aid, "public.character: id differs from key")
        for key in ("name", "role"):
            _require(actor[key] == tables["actors"][aid][key], f"public.character.{aid}: identity mismatch")
    public_player = _object(public.get("player"), "public.player")
    _require(set(public_player) == {"id", "role", "location_id", "health", "skills", "inventory"},
             "public.player: unexpected or missing fields")
    for key, value in public_player.items():
        _require(value == player[key], f"public.player.{key}: differs from player initial state")
    for rule in public["rules"]:
        _text(rule, "public.rules")


def load_module(path: Path | None = None) -> dict:
    """Read and validate ``public.json`` / ``world.json`` from a module directory.

    The SHA-256 identifies canonical JSON of both files, sorted by object keys,
    encoded as UTF-8 without insignificant whitespace. List order remains material.
    Every invocation returns fresh dictionaries; there is no mutable shared cache.
    """
    module_path = DEFAULT_MODULE_PATH if path is None else Path(path)
    if (module_path / 'adventure.json').is_file():
        from .adventure.registry import validate_adventure
        try:
            return validate_adventure(_read_json(module_path / 'adventure.json'))
        except (KeyError, TypeError, ValueError) as exc:
            raise ModuleValidationError(f'invalid adventure: {exc}') from exc
    public = _read_json(module_path / "public.json")
    world = _read_json(module_path / "world.json")
    try:
        _validate(public, world)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModuleValidationError(f"invalid module structure: {exc}") from exc
    canonical = json.dumps({"public": public, "world": world}, sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "public": public,
        "world": world,
        "module_id": world["module_id"],
        "module_version": world["module_version"],
        "module_hash": hashlib.sha256(canonical).hexdigest(),
    }


def module_by_id(module_id):
    # A fixed registry prevents user-supplied paths or executable modules.
    paths = {'reserve_cooling': DEFAULT_MODULE_PATH,
             'pinebrook': DEFAULT_MODULE_PATH.parent / 'pinebrook'}
    if module_id not in paths:
        raise ModuleValidationError('冒险不存在')
    return load_module(paths[module_id])


def available_modules():
    return [module_by_id(id) for id in ('pinebrook', 'reserve_cooling')]
