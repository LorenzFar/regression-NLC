"""Small authored fixtures test import safety, not corpus quality."""
import copy
import contextlib
import io
import json
import os
import tempfile
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_craigslist_candidates.py"


def importer():
    spec = importlib.util.spec_from_file_location("craigslist_candidates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dialogue(uuid="d1", reversed_roles=False):
    buyer, seller = (1, 0) if reversed_roles else (0, 1)
    def message(agent, text):
        return {"agent": agent, "action": "message", "data": text}
    return {
        "uuid": uuid,
        "scenario": {"kbs": [
            {"personal": {"Role": "seller" if reversed_roles else "buyer", "Target": 999}},
            {"personal": {"Role": "buyer" if reversed_roles else "seller", "Target": 888}},
        ]},
        "outcome": {"private": "SECRET"},
        "events": [message(buyer, "Hello"), message(seller, "Welcome"),
                   message(seller, "I ask 90"), message(buyer, "I can offer 70"),
                   message(seller, "FUTURE")],
    }


class CandidateTests(unittest.TestCase):
    def test_role_reversal_and_pre_turn_history(self):
        module = importer()
        rows, ids, counts = module.make_candidates([dialogue(reversed_roles=True)])
        self.assertEqual(ids, ["d1"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["speaker"], "BUYER")
        self.assertEqual(row["response"], "I can offer 70")
        self.assertEqual(row["message"], "I ask 90")
        self.assertEqual(row["history"], [
            {"speaker": "BUYER", "text": "Hello"},
            {"speaker": "SELLER", "text": "Welcome"},
        ])
        self.assertEqual(row["source_event_index"], 3)
        self.assertEqual(row["seller_event_index"], 2)
        import json
        encoded = json.dumps(row)
        for private in ("SECRET", "FUTURE", "Target", "999", "888"):
            self.assertNotIn(private, encoded)

    def test_train_allowlist_is_intersected_and_labeled(self):
        module = importer()
        rows, ids, counts = module.make_candidates(
            [dialogue(), dialogue("d2")], train_ids=["d2", "not-in-official-train"])
        self.assertEqual(ids, ["d2"])
        self.assertEqual([r["dialogue_id"] for r in rows], ["d2"])
        self.assertEqual(rows[0]["split_status"], "team_allowlist_intersection")
        self.assertEqual(counts["excluded_by_team_allowlist"], 1)
        self.assertEqual(counts["team_ids_outside_official_train"], 1)
        self.assertEqual(module.make_candidates([dialogue()], train_ids=[])[0], [])

    def test_duplicate_dialogue_ids_fail_closed(self):
        module = importer()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.make_candidates([dialogue(), dialogue()])

    def test_parser_join_checks_every_event_and_uuid(self):
        module = importer()
        raw = dialogue()
        parsed = copy.deepcopy(raw)
        parsed["events"][3]["metadata"] = {"intent": "counter-price", "price": 70}
        rows, _, _ = module.make_candidates([raw], parsed=[parsed])
        self.assertEqual(rows[0]["proposed_action"], "counter")
        self.assertEqual(rows[0]["action_candidates"], ["counter"])
        self.assertEqual(rows[0]["parser_intent"], "counter-price")
        for field, value in (("data", "different"), ("agent", 1), ("action", "offer")):
            wrong = copy.deepcopy(parsed)
            wrong["events"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "alignment"):
                module.make_candidates([raw], parsed=[wrong])
        short = copy.deepcopy(parsed)
        short["events"].pop()
        with self.assertRaisesRegex(ValueError, "alignment"):
            module.make_candidates([raw], parsed=[short])
        wrong = copy.deepcopy(parsed)
        wrong["uuid"] = "other"
        with self.assertRaisesRegex(ValueError, "UUID"):
            module.make_candidates([raw], parsed=[wrong])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.make_candidates([raw], parsed=[parsed, parsed])

    def test_unapproved_proposals_cannot_prepare_until_human_edits(self):
        module = importer()
        from retrieval_pipeline.bank import prepare_candidate
        from retrieval_pipeline.contracts import ContractError
        rows, ids, _ = module.make_candidates([dialogue()])
        row = rows[0]
        self.assertIsNone(row["action"])
        self.assertIsNone(row["mixed_action"])
        self.assertIsNone(row["unresolved_reference"])
        self.assertFalse(row["approved"])
        self.assertFalse(row["price_spans"][0]["validated"])
        self.assertIn("provisional", row["split_status"])
        with self.assertRaises(ContractError):
            prepare_candidate(row, ids)
        edited = copy.deepcopy(row)
        edited.update(action="counter", mixed_action=False, unresolved_reference=False)
        with self.assertRaises(ContractError):
            prepare_candidate(edited, ids)
        edited["price_spans"][0]["validated"] = True
        prepared = prepare_candidate(edited, ids)
        self.assertEqual(prepared["template"], "I can offer {price}")
        self.assertFalse(prepared["approved"])

    def test_agree_is_never_mapped_to_accept_or_confirm(self):
        module = importer()
        raw = dialogue()
        parsed = copy.deepcopy(raw)
        parsed["events"][3]["metadata"] = {"intent": "agree", "price": 70}
        rows, _, counts = module.make_candidates([raw], parsed=[parsed])
        self.assertEqual(rows, [])
        self.assertEqual(counts["skipped_agreement_intent"], 1)

    def test_private_intervening_event_blocks_pair_and_never_enters_history(self):
        module = importer()
        raw = dialogue()
        raw["events"].insert(3, {"action": "offer", "agent": 1, "data": {"price": 1234}})
        self.assertEqual(module.make_candidates([raw])[0], [])
        raw["events"].insert(0, {"action": "eval", "agent": 1, "data": "PRIVATE"})
        raw["events"].pop(4)
        rows, _, _ = module.make_candidates([raw])
        self.assertEqual(rows[0]["source_event_index"], 4)
        self.assertEqual(len(rows[0]["history"]), 2)
        self.assertNotIn("PRIVATE", str(rows[0]))

    def test_bank_restrictions_not_relaxed_for_currency_or_multiple_numbers(self):
        module = importer()
        for text in ("I offer $70", "I offer 70 or 80", "70 including delivery", "70.123"):
            raw = dialogue()
            raw["events"][3]["data"] = text
            with self.subTest(text=text):
                self.assertEqual(module.make_candidates([raw])[0], [])
    def test_cli_writes_queue_ids_and_hashed_source_metadata(self):
        import hashlib
        module = importer()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            raw_path, parsed_path, team_path = (root / p for p in ("raw.json", "parsed.json", "team.json"))
            raw = [dialogue(), dialogue("d2")]
            parsed = copy.deepcopy(raw)
            parsed.reverse()  # Joining must use UUID, not list position.
            for d in parsed:
                d["events"][3]["metadata"] = {"intent": "init-price", "price": 12345}
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            parsed_path.write_text(json.dumps(parsed), encoding="utf-8")
            team_path.write_text('["d2"]', encoding="utf-8")
            output = root / "review"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = module.main([str(raw_path), "--parsed", str(parsed_path),
                                      "--train-ids", str(team_path), "--output-dir", str(output)])
            self.assertEqual(status, 0)
            rows = [json.loads(line) for line in (output / "sources.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(json.loads((output / "train-ids.json").read_text()), ["d2"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["proposed_action"], "propose")
            self.assertEqual(rows[0]["price_spans"][0]["value"], "70")  # Raw span, not parser price.
            self.assertNotIn("12345", json.dumps(rows))
            self.assertNotIn('"validated": true', json.dumps(rows))
            self.assertNotIn('"approved": true', json.dumps(rows))
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["counts"]["queued"], len(rows))
            self.assertEqual(summary["sources"]["raw"]["sha256"], hashlib.sha256(raw_path.read_bytes()).hexdigest())
            self.assertEqual(summary["sources"]["parsed"]["sha256"], hashlib.sha256(parsed_path.read_bytes()).hexdigest())
            self.assertEqual(summary["split_status"], "team_allowlist_intersection")

    def test_cli_rejects_invalid_whole_json_before_creating_output(self):
        module = importer()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            raw = root / "bad.json"
            raw.write_text(json.dumps([dialogue()]) + " trailing data", encoding="utf-8")
            output = root / "review"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main([str(raw), "--output-dir", str(output)]), 2)
            self.assertFalse(output.exists())
    def test_cli_rejects_null_optional_files(self):
        module = importer()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            raw, optional = root / "raw.json", root / "null.json"
            raw.write_text(json.dumps([dialogue()]), encoding="utf-8")
            optional.write_text("null", encoding="utf-8")
            for flag in ("--parsed", "--train-ids"):
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(module.main([str(raw), flag, str(optional), "--output-dir", str(root / "out")]), 2)
            self.assertFalse((root / "out").exists())

    def test_invalid_roles_or_team_id_lists_rejected(self):
        module = importer()
        raw = dialogue()
        raw["scenario"]["kbs"][1]["personal"]["Role"] = "buyer"
        with self.assertRaisesRegex(ValueError, "buyer and seller"):
            module.make_candidates([raw])
        for ids in (["d1", "d1"], "d1", [""], [False]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "team train IDs"):
                module.make_candidates([dialogue()], train_ids=ids)


if __name__ == "__main__":
    unittest.main()
