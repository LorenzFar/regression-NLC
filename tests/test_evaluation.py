"""Gold-set extraction and scorers on small synthetic data; no model or dataset download."""
import csv
import json

import pytest

from evaluation import evaluate_nlu, evaluate_prices, metrics
from evaluation.extract_test import SHEET_COLUMNS, dialogue_from_record, seller_rows

RECORD = {
    "agent_info": {"Role": ["buyer", "seller"], "Target": [49.0, 65.0]},
    "agent_turn": [0, 1, 0, 1, 0],
    "utterance": ["can you do 25", "too low. how about 60?", "40?", "ok deal", ""],
    "items": {"Category": ["electronics"] * 2, "Title": ["Router"] * 2,
              "Price": [65.0, 65.0], "Description": ["As new"] * 2},
}


def test_extraction_keeps_turn_numbers_and_drops_empty_turns():
    dialogue = dialogue_from_record(7, RECORD)
    assert [t["turn"] for t in dialogue["turns"]] == [0, 1, 2, 3]
    assert dialogue["buyer_target"] == 49.0 and dialogue["seller_target"] == 65.0
    rows = seller_rows(dialogue)
    assert [r["case_id"] for r in rows] == ["cb-test-0007-t1", "cb-test-0007-t3"]
    assert rows[1]["context"] == "BUYER: can you do 25\nSELLER: too low. how about 60?\nBUYER: 40?"
    assert set(rows[0]) == set(SHEET_COLUMNS) and rows[0]["gold_act"] == ""


def test_classification_metrics_match_hand_computed_values():
    gold = ["accept", "accept", "counter", "reject"]
    pred = ["accept", "counter", "counter", "counter"]
    scores = metrics.per_class(gold, pred, ["accept", "counter", "reject"])
    assert scores["accept"] == pytest.approx({"precision": 1.0, "recall": 0.5, "f1": 2 / 3,
                                              "support": 2, "predicted": 1})
    assert scores["counter"]["precision"] == pytest.approx(1 / 3)
    assert metrics.macro_f1(gold, pred, ["accept", "counter", "reject", "vague"]) == pytest.approx((2 / 3 + 0.5 + 0) / 3)
    assert metrics.accuracy(gold, pred) == 0.5
    assert metrics.confusion_matrix(gold, pred, ["accept", "counter", "reject"]) == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]


def test_kappa_calibration_coverage_and_bootstrap():
    assert metrics.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert metrics.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    calibration = metrics.reliability([0.95, 0.95, 0.55, 0.55], [True, True, True, False])
    assert calibration["ece"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.05)
    coverage = metrics.coverage_curve([0.9, 0.8, 0.5], [True, False, True], [0.7])
    assert coverage == [{"threshold": 0.7, "coverage": pytest.approx(2 / 3), "kept": 2, "accuracy_on_kept": 0.5}]
    ci = metrics.bootstrap_ci([[1], [1], [1]], lambda s: sum(s) / len(s), n_resamples=50)
    assert ci["low"] == ci["high"] == 1.0


def _write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _probabilities(top, confidence):
    rest = (1 - confidence) / 5
    return {f"p_{act}": confidence if act == top else rest for act in evaluate_nlu.ACTS}


def test_nlu_scorer_rescores_saved_predictions(tmp_path):
    labels = [
        {"case_id": "c1", "dialogue_index": 0, "turn": 1, "message": "how about 60?", "gold_act": "counter"},
        {"case_id": "c2", "dialogue_index": 0, "turn": 3, "message": "ok deal", "gold_act": "Accept"},
        {"case_id": "c3", "dialogue_index": 1, "turn": 1, "message": "is it new?", "gold_act": "inquire"},
        {"case_id": "c4", "dialogue_index": 1, "turn": 3, "message": "hmm", "gold_act": "vague"},
        {"case_id": "c5", "dialogue_index": 2, "turn": 1, "message": "not labelled yet", "gold_act": ""},
    ]
    predictions = [
        {"case_id": "c1", **_probabilities("counter", 0.9)},
        {"case_id": "c2", **_probabilities("accept", 0.8)},
        {"case_id": "c3", **_probabilities("inquire", 0.95)},
        {"case_id": "c4", **_probabilities("accept", 0.75)},
    ]
    _write(tmp_path / "labels.csv", labels)
    _write(tmp_path / "predictions.csv", predictions)
    second = [dict(row, gold_act=act) for row, act in zip(labels, ["counter", "accept", "inquire", "reject", ""])]
    _write(tmp_path / "second.csv", second)
    out = tmp_path / "out"

    assert evaluate_nlu.main([str(tmp_path / "labels.csv"), "--predictions", str(tmp_path / "predictions.csv"),
                              "--second-annotator", str(tmp_path / "second.csv"),
                              "--bootstrap", "50", "--output", str(out)]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    scores = report["scores"]
    assert scores["n_turns"] == 4 and scores["n_dialogues"] == 2
    assert scores["accuracy"] == 0.75
    assert scores["accept_errors"] == {"precision": 0.5, "recall": 1.0, "false_accepts": 1,
                                       "confident_false_accepts": 1, "threshold": 0.7}
    assert report["agreement"]["n_overlap"] == 4 and report["agreement"]["raw_agreement"] == 0.75
    assert "Most confident mistakes" in (out / "report.md").read_text(encoding="utf-8")
    # Saved predictions round-trip into the same scores.
    assert evaluate_nlu.main([str(tmp_path / "labels.csv"), "--predictions", str(out / "predictions.csv"),
                              "--bootstrap", "50", "--output", str(tmp_path / "again")]) == 0


def test_nlu_scorer_rejects_unknown_acts(tmp_path):
    _write(tmp_path / "labels.csv", [{"case_id": "c1", "dialogue_index": 0, "turn": 1,
                                      "message": "x", "gold_act": "agree"}])
    _write(tmp_path / "predictions.csv", [{"case_id": "c1", **_probabilities("accept", 0.9)}])
    with pytest.raises(SystemExit) as exit_info:
        evaluate_nlu.main([str(tmp_path / "labels.csv"), "--predictions", str(tmp_path / "predictions.csv")])
    assert exit_info.value.code == 2


def test_price_parsing_normalizes_values_and_validates_roles():
    assert evaluate_prices.parse_prices("$1.5k:new_offer; 60.00 ; 2,600:reference") == [
        ("1500", "new_offer"), ("60", None), ("2600", "reference")]
    assert evaluate_prices.parse_prices("") == []
    with pytest.raises(evaluate_prices.PriceLabelError):
        evaluate_prices.parse_prices("60:offer")
    assert evaluate_prices.digits_baseline("I paid $200 in 2015, can do 1.5k") == [
        ("200", "new_offer"), ("2015", "new_offer"), ("1500", "new_offer")]


def test_price_scorer_counts_values_roles_and_offers(tmp_path):
    labels = [
        {"case_id": "c1", "dialogue_index": 0, "message": "too low, how about 60?",
         "gold_act": "counter", "gold_prices": "60:new_offer"},
        {"case_id": "c2", "dialogue_index": 0, "message": "I paid 200, so 150 is my limit",
         "gold_act": "counter", "gold_prices": "200:not_offer; 150:new_offer"},
        {"case_id": "c3", "dialogue_index": 1, "message": "it is a 2015 model",
         "gold_act": "vague", "gold_prices": ""},
    ]
    predictions = [{"case_id": "c1", "pred_prices": "60:new_offer"},
                   {"case_id": "c2", "pred_prices": "200:new_offer; 150:new_offer"},
                   {"case_id": "c3", "pred_prices": "2015:new_offer"}]
    _write(tmp_path / "labels.csv", labels)
    _write(tmp_path / "predictions.csv", predictions)
    rows = evaluate_prices.load_rows(tmp_path / "labels.csv", tmp_path / "predictions.csv")
    scores = evaluate_prices.score(rows, n_resamples=50)
    assert scores["value"] == pytest.approx({"precision": 0.75, "recall": 1.0, "f1": 6 / 7,
                                             "tp": 3, "predicted": 4, "gold": 3})
    assert scores["value_and_role"]["tp"] == 2
    assert scores["new_offer"]["precision"] == 0.5 and scores["new_offer"]["recall"] == 1.0
    assert scores["role_confusion"] == {"not_offer->new_offer": 1}
    assert scores["message_exact_match"] == pytest.approx(2 / 3)
    assert evaluate_prices.main([str(tmp_path / "labels.csv"), "--baseline", "digits", "--bootstrap", "50",
                                 "--output", str(tmp_path / "out")]) == 0
