"""Price extractor evaluation: set scoring and the held-out guard."""
import csv

import pytest

from evaluation.price_extractor import run
from evaluation.price_extractor.metrics import score, to_set

FIELDS = ["split", "dialogue", "turn", "role", "category", "listing_price", "text",
          "parser_intent", "parser_price", "gold_price", "gold_note"]


@pytest.mark.parametrize("cell, expected", [
    ("none", set()), ("", set()), (None, set()), ("45.00", {"45.00"}),
    ("2025.00 | 2000.00", {"2025.00", "2000.00"}), ("45", {"45.00"}), ("45.0", {"45.00"}),
])
def test_to_set(cell, expected):
    assert to_set(cell) == expected


def test_score_counts_amounts_and_presence():
    rows = [{"gold": to_set(g), "role": "buyer", "category": "bike", "pred": to_set(p)} for g, p in [
        ("100.00", "100.00"),             # exact
        ("100.00 | 125.00", "100.00"),    # one amount missed
        ("none", "40.00"),                # price where none belongs
        ("50.00", "none"),                # missed price
        ("none", "none"),                 # correct none
    ]]
    result = score(rows, lambda r: r["pred"])
    o = result["overall"]
    assert (o["n"], o["exact_match"]) == (5, 0.4)
    assert o["confusion"] == {"price_found": 2, "false_price": 1, "missed_price": 1, "correct_none": 1}
    assert (o["precision"], o["recall"]) == (2 / 3, 0.5)   # tp 2, fp 1, fn 2
    assert result["role"]["buyer"]["n"] == 5 and result["category"]["bike"]["n"] == 5


def test_evaluate_scores_labelled_rows_and_counts_unread_false_alarms(tmp_path):
    rows = [
        ["train", 0, 0, "buyer", "bike", "100.0", "I can do $80", "offer", "80.0", "80.00", ""],
        ["train", 0, 1, "seller", "bike", "100.0", "no, 90 is my best", "counter", "", "90.00", ""],
        ["train", 0, 2, "buyer", "bike", "100.0", "thanks, bye", "bye", "", "", ""],
    ]
    with (tmp_path / "train_turns.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    report = run.evaluate(["train"], str(tmp_path))
    assert report["rows_scored"] == 2 and report["unread_rows"] == 1
    assert report["systems"]["extractor"]["overall"]["exact_match"] == 1.0
    assert report["systems"]["parser"]["overall"]["exact_match"] == 0.5


def test_test_split_needs_final(capsys):
    with pytest.raises(SystemExit):
        run.main(["--split", "test"])
    assert "--final" in capsys.readouterr().err
