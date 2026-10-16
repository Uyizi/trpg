"""Module contract tests: data identity, private knowledge, and invalid input."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ModuleTests(unittest.TestCase):
    def loader(self):
        spec = importlib.util.find_spec("local_trpg.modules")
        self.assertIsNotNone(spec, "the module loader has not been implemented")
        from local_trpg.modules import load_module
        return load_module

    def module(self):
        return self.loader()()

    def write_module(self, path, module):
        for section in ("public", "world"):
            (path / f"{section}.json").write_text(
                json.dumps(module[section], ensure_ascii=False), encoding="utf-8"
            )

    def assert_invalid(self, mutate, message):
        module = copy.deepcopy(self.module())
        mutate(module)
        with tempfile.TemporaryDirectory() as tmp:
            self.write_module(Path(tmp), module)
            with self.assertRaisesRegex(ValueError, message):
                self.loader()(Path(tmp))

    def test_loads_complete_adventure_with_exact_resources(self):
        module = self.module()
        self.assertEqual(set(module), {"public", "world", "module_id", "module_version", "module_hash"})
        self.assertEqual(module["module_id"], "reserve_cooling")
        self.assertEqual(module["module_version"], "1.0.0")
        world = module["world"]
        self.assertEqual(set(world["actors"]), {"captain", "doctor", "mechanic", "comms", "guard"})
        self.assertEqual(set(world["locations"]), {"bridge", "common", "clinic", "engine", "cargo", "lifeboat"})
        self.assertEqual(world["participants"], ["player", "captain", "doctor", "mechanic", "comms", "guard"])
        self.assertEqual(world["player"]["skills"], {"observe": 2, "technical": 1, "social": 1, "action": 0})
        self.assertEqual(world["player"]["inventory"], ["multitool", "recorder", "face_mask", "work_card"])
        for kind, count in (("cell", 4), ("cold_case", 2), ("cold_core", 2), ("medical_supply", 2)):
            self.assertEqual(sum(item["kind"] == kind for item in world["items"].values()), count)
        self.assertEqual({f["fact_id"] for f in world["facts"].values()}, {f"F{i:02d}" for i in range(1, 10)})
        self.assertEqual(set(world["evidence"]), {f"E{i:02d}" for i in range(1, 9)})

    def test_public_opening_does_not_include_private_truths_or_evidence(self):
        public = self.module()["public"]
        self.assertEqual(set(public), {"schema_version", "module_id", "module_version", "title", "opening", "premise", "locations", "characters", "player", "rules"})
        serialized = json.dumps(public, ensure_ascii=False)
        for private in ("captain_enabled_hold", "doctor_saw_old_uncharged_cores", "E06", "F04", "已充好", "亲自刷卡"):
            self.assertNotIn(private, serialized)
        self.assertIn("0/12", public["opening"])

    def test_evidence_grants_only_its_atomic_supported_propositions(self):
        world = self.module()["world"]
        e02 = world["evidence"]["E02"]
        self.assertEqual(set(e02["proposition_ids"]), {"captain_card_used", "captain_signed_work_order"})
        self.assertNotIn("captain_enabled_hold", e02["proposition_ids"])
        e07 = world["evidence"]["E07"]
        self.assertEqual(set(e07["proposition_ids"]), {"cases_fit_four_cells", "cold_cores_charged", "emergency_cold_18_ticks", "guard_signed_supply_receipt"})
        self.assertNotIn("doctor_saw_old_uncharged_cores", e07["proposition_ids"])
        for evidence in world["evidence"].values():
            self.assertFalse(evidence["initially_visible"])
            self.assertTrue(evidence["sources"])

    def test_initial_actor_knowledge_keeps_guard_and_doctor_distinct(self):
        actors = self.module()["world"]["actors"]
        doctor_facts = {f["proposition_id"] for f in actors["doctor"]["private_background"]}
        guard_facts = {f["proposition_id"] for f in actors["guard"]["private_background"]}
        mechanic_facts = {f["proposition_id"] for f in actors["mechanic"]["private_background"]}
        self.assertIn("doctor_saw_old_uncharged_cores", doctor_facts)
        self.assertNotIn("cold_cores_charged", doctor_facts)
        self.assertNotIn("captain_enabled_hold", doctor_facts | mechanic_facts)
        self.assertIn("cold_cores_charged", guard_facts)
        self.assertNotIn("cases_fit_four_cells", guard_facts)
        belief = next(b for b in actors["doctor"]["initial_beliefs"] if b["proposition_id"] == "cold_cores_charged")
        self.assertEqual(belief["stance"], "doubted")
        self.assertEqual(belief["source_observation_ids"], ["doctor_saw_old_uncharged_cores"])
        self.assertEqual(actors["mechanic"]["health"], "strained")

    def test_routes_and_deadline_rules_match_three_acceptance_outcomes(self):
        world = self.module()["world"]
        for route, arrival, cells, ship in (("A", 14, 4, True), ("B", 11, 0, True), ("C", 9, 4, False)):
            result = world["routes"][route]["expected_outcome"]
            self.assertEqual(result["arrival_tick"], arrival)
            self.assertEqual(result["crew_saved"], 6)
            self.assertEqual(result["cells_saved"], cells)
            self.assertEqual(result["ship_saved"], ship)
        self.assertEqual(world["timing"], {"original_cargo_hold_ticks": 4, "emergency_cold_ticks": 18, "repair_station_travel_ticks": 5, "rescue_response_ticks": 6})
        self.assertEqual({t["due_tick"] for t in world["triggers"]}, {3, 6, 8, 9, 10, 12, 15})
        self.assertEqual(world["resolution_rules"]["cargo_validity"], "arrival_tick < expiry_tick")
        self.assertTrue(world["resolution_rules"]["clocks_continue_after_departure"])

    def test_guard_evacuation_assistance_is_registered_with_specific_target(self):
        world = self.module()["world"]
        self.assertIn("assist_evacuation", world["actors"]["guard"]["actions"])
        action = world["actions"]["assist_evacuation"]
        self.assertEqual(action["tick_cost"], 1)
        self.assertTrue(action["target_required"])
        self.assertEqual(action["target_contract"]["field"], "target_ids")
        self.assertEqual(action["target_contract"]["minimum"], 1)
        self.assertEqual(action["target_contract"]["maximum"], 1)
        self.assertEqual(action["actor_ids"], ["player", "guard"])

    def test_hash_ignores_json_order_and_formatting_but_tracks_both_files(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.write_module(path, module)
            self.assertEqual(self.loader()(path)["module_hash"], module["module_hash"])
            for section in ("public", "world"):
                file = path / f"{section}.json"
                content = dict(reversed(list(module[section].items())))
                file.write_text(json.dumps(content, ensure_ascii=True, indent=4), encoding="utf-8")
            self.assertEqual(self.loader()(path)["module_hash"], module["module_hash"])
            for section in ("public", "world"):
                edited = copy.deepcopy(module)
                edited[section]["title"] += "（修订）"
                self.write_module(path, edited)
                self.assertNotEqual(self.loader()(path)["module_hash"], module["module_hash"])

    def test_rejects_unsupported_schema_and_mismatched_identity(self):
        for version in (2, True, "1"):
            self.assert_invalid(lambda m, v=version: m["world"].update(schema_version=v), "schema_version")
        self.assert_invalid(lambda m: m["public"].update(module_version="2.0.0"), "module_version")

    def test_rejects_unknown_proposition_location_and_item_references(self):
        self.assert_invalid(lambda m: m["world"]["evidence"]["E07"]["proposition_ids"].append("made_up_fact"), "proposition")
        self.assert_invalid(lambda m: m["world"]["actors"]["doctor"].update(location_id="nowhere"), "location")
        self.assert_invalid(lambda m: m["world"]["player"]["inventory"].append("magic_tool"), "inventory")

    def test_rejects_unknown_action_unpermitted_actor_and_trigger_code(self):
        self.assert_invalid(lambda m: m["world"]["actors"]["doctor"]["actions"].append("destroy_ship"), "action")
        self.assert_invalid(lambda m: m["world"]["actors"]["doctor"]["actions"].append("launch_pod"), "allowlist")
        self.assert_invalid(lambda m: m["world"]["triggers"][0].update(guard_id="__import__('os').system('false')"), "guard")

    def test_rejects_duplicate_ids_broken_observation_and_owner(self):
        self.assert_invalid(lambda m: m["world"]["triggers"].append(m["world"]["triggers"][0]), "duplicate")
        self.assert_invalid(lambda m: m["world"]["actors"]["doctor"]["initial_beliefs"][0].update(source_observation_ids=["not_seen"]), "observation")
        self.assert_invalid(lambda m: m["world"]["items"]["multitool"].update(owner="nobody"), "owner")

    def test_rejects_private_public_fields_and_malformed_shapes(self):
        self.assert_invalid(lambda m: m["public"]["player"].update(private_background=["secret"]), "public.player")
        self.assert_invalid(lambda m: m["public"]["characters"]["captain"].update(id="doctor"), "public.character")
        self.assert_invalid(lambda m: m["world"]["participants"].append({"id": "intruder"}), "structure|participants")
        self.assert_invalid(lambda m: m["world"]["actors"]["doctor"]["initial_beliefs"][0].pop("source_observation_ids"), "structure|observation")

    def test_rejects_duplicate_json_keys_and_nonfinite_numbers(self):
        module = self.module()
        for malformed, expected in (('{"schema_version":1,"schema_version":1}', "duplicate JSON key"),
                                    ('{"schema_version":NaN}', "non-finite")):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                self.write_module(path, module)
                (path / "public.json").write_text(malformed, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    self.loader()(path)

    def test_missing_files_and_bad_json_produce_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cannot read public.json"):
                self.loader()(Path(tmp))
            (Path(tmp) / "public.json").write_text("{unfinished", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot read public.json"):
                self.loader()(Path(tmp))

    def test_beliefs_cannot_use_another_actors_private_observation(self):
        self.assert_invalid(lambda m: m["world"]["actors"]["doctor"]["initial_beliefs"][0].update(
            source_observation_ids=["guard_cold_cores_charged"]), "another actor")

    def test_validation_cli_exits_nonzero_for_invalid_module(self):
        self.module()
        script = ROOT / "scripts" / "validate_module.py"
        success = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertIn("reserve_cooling", success.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            bad = copy.deepcopy(self.module())
            bad["world"]["evidence"]["E02"]["proposition_ids"].append("invented")
            self.write_module(Path(tmp), bad)
            failure = subprocess.run([sys.executable, str(script), tmp], capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("proposition", failure.stderr)


if __name__ == "__main__":
    unittest.main()
