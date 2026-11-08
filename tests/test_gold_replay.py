"""Replay's public seam uses real collaborators except bounded fake NLU inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.gate_metrics import gate_report, sweep_threshold

from retrieval_pipeline.gold_replay import replay_gold

BANK = Path(__file__).resolve().parents[1] / "examples/approved_fixture_bank.jsonl"
ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")


class FakeNLU:
    def __init__(self, confidence=0.55):
        self.confidence = confidence
        self.calls = []

    def predict(self, history, message):
        self.calls.append((history, message))
        return {act: self.confidence if act == "accept" else (1 - self.confidence) / 5
                for act in ACTS}


def acceptance_case():
    return {"case_id": "accept-live", "origin": "authored_test", "listing_price": "100",
            "history": [{"speaker": "seller", "text": "I can do 100", "act": "propose", "price": "100"},
                        {"speaker": "buyer", "text": "I offer 80", "act": "counter", "price": "80"}],
            "message": "I accept your offer of 80", "act": "accept", "price": "80",
            "needs_clarification": False, "deal_agreed": True}


def test_gold_history_restores_live_buyer_offer_and_gate_changes_only_current_turn():
    row = acceptance_case()
    row["message"] = "Okay, deal."  # Exercise NLU confidence, not explicit-acceptance correction.
    records = replay_gold([row], FakeNLU(), bank=BANK)
    assert all(record["error"] is None for record in records)
    off, reference, full = records
    assert off["reconstruction_mode"] == "gold_labels"
    assert off["state_before"]["state"]["offers"][-1] == {
        "id": "buyer-2", "by": "buyer", "terms": {"price": "80.00"}, "status": "live"}
    assert off["policy"]["action"] == "confirm"
    assert off["policy"]["price"] == "80.00"
    assert off["state_after"]["state"]["agreement"] == "confirmed"
    assert reference["policy"]["action"] == "confirm"
    assert full["policy"]["action"] == "clarify"
    assert "low_confidence" in full["gate"]["reasons"]
    after = dict(full["state_after"]["state"])
    assert after.pop("pending_clarification")["act"] == "accept"
    assert full["state_before"]["state"] == after
    assert all(record["state_before"] == off["state_before"] for record in records)


def test_records_score_with_existing_evaluator_without_translation():
    row = acceptance_case()
    row["message"] = "Okay, deal."  # Keep the low-confidence clarification case.
    original = deepcopy(row)
    records = replay_gold([row], FakeNLU(), bank=BANK)
    report = gate_report(records, {row["case_id"]: row})
    assert report["off"]["correct_confirmations"] == 1
    assert report["full"]["unnecessary_clarifications"] == 1
    assert report["full"]["extra_clarification_turns"] == 1
    assert all(entry["failed_cases"] == 0 for entry in report.values())
    assert sweep_threshold(records, {row["case_id"]: row}, [0.7])["cases"] == 1
    assert row == original
    json.dumps(records, allow_nan=False)


def test_partial_labels_predict_entire_history_with_seller_own_prefix():
    row = acceptance_case()
    del row["history"][0]["act"]
    row["history"].append({"speaker": "seller", "text": "Still interested?"})
    nlu = FakeNLU()
    records = replay_gold([row], nlu, bank=BANK)
    assert all(r["error"] is None for r in records)
    assert records[0]["reconstruction_mode"] == "predicted"
    assert nlu.calls[0] == ([], "I can do 100")
    assert nlu.calls[1] == ([("SELLER", "I can do 100"), ("BUYER", "I offer 80")],
                            "Still interested?")
    assert records[0]["state_before"]["state"]["offers"][-1]["terms"]["price"] == "80.00"


def test_listing_and_limit_defaults_use_text_not_current_gold_price():
    row = acceptance_case()
    del row["listing_price"]
    row["price"] = "99999"  # withheld current-turn target cannot affect inference
    record = replay_gold([row], FakeNLU(), modes=["off"], bank=BANK)[0]
    assert record["error"] is None
    state = record["state_before"]["state"]
    assert state["listing_price"] == "100.00"
    assert state["limits"] == {"target": "70.00", "walk_away": "90.00"}
    assert record["listing_price_source"] == "max_extracted_absolute"
    row.update(target="60", walk_away="85")
    record = replay_gold([row], FakeNLU(), modes=["off"], bank=BANK)[0]
    assert record["state_before"]["state"]["limits"] == {"target": "60.00", "walk_away": "85.00"}


def test_errors_are_per_case_mode_and_do_not_stop_valid_rows():
    bad = {"case_id": "no-price", "history": [], "message": "ok"}
    row = acceptance_case()
    records = replay_gold([bad, row, row, None], FakeNLU(), bank=BANK)
    assert len(records) == 12
    assert all(r["error"] and "cannot derive" in r["error"]["message"] for r in records[:3])
    assert all(r["error"] is None for r in records[3:6])
    assert all(r["error"]["message"] == "duplicate case_id" for r in records[6:9])
    assert all(r["error"] is not None for r in records[9:])
    report = gate_report(records, {row["case_id"]: row})
    assert all(r["failed_cases"] == 3 for r in report.values())


@pytest.mark.parametrize("kwargs", [{"modes": []}, {"modes": ["off", "off"]},
                                    {"threshold": float("nan")}, {"target_ratio": -1}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        replay_gold([], FakeNLU(), **kwargs)


def test_cli_reports_malformed_rows_and_scores_valid_rows(tmp_path, monkeypatch):
    from retrieval_pipeline.gold_replay import main

    source, output = tmp_path / "cases.jsonl", tmp_path / "replay.jsonl"
    source.write_text("{broken\n" + json.dumps(acceptance_case()) + "\n", encoding="utf-8")
    monkeypatch.setattr("retrieval_pipeline.gold_replay.load_nlu", lambda _: FakeNLU())
    assert main([str(source), "--output", str(output), "--bank", str(BANK)]) == 1
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert all(r["error"] is not None for r in records[:3])
    assert all(r["error"] is None for r in records[3:])


def test_cli_resource_failure_is_error_records_not_fabricated_predictions(tmp_path, monkeypatch):
    from retrieval_pipeline.gold_replay import main

    source, output = tmp_path / "cases.jsonl", tmp_path / "replay.jsonl"
    source.write_text(json.dumps(acceptance_case()) + "\n", encoding="utf-8")
    def unavailable(_):
        raise FileNotFoundError("missing checkpoint")
    monkeypatch.setattr("retrieval_pipeline.gold_replay.load_nlu", unavailable)
    assert main([str(source), "--output", str(output), "--bank", str(BANK)]) == 1
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(r["error"]["message"] == "missing checkpoint" for r in records)
    assert all("probabilities" not in r and "generation" not in r for r in records)
