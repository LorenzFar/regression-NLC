"""Gold-labelling tools: the filter and the safe applier (temporary CSVs only)."""
import csv

import pytest

from scripts import gold_apply
from scripts.gold_csv import read
from scripts.number_filter import needs_reading

FIELDS = ["split", "dialogue", "turn", "role", "category", "listing_price", "text",
          "parser_intent", "parser_price"]


@pytest.mark.parametrize("text, expected", [
    ("I can do $5", True), ("how about 1850", True), ("thirteen hundred", True), ("two grand", True),
    ("fifty five", True), ("a few bucks off", True), ("I could do a dozen", True), ("$%)", True),
    ("hello, is it available?", False), ("there are then right done might", False), ("", False),
    ("   ", False),
])
def test_filter(text, expected):
    assert needs_reading(text) is expected


@pytest.fixture
def sheet(tmp_path):
    rows = [
        ["train", "0", "0", "buyer", "phone", "10.0", "Can you sell it to me for $5?", "init-price", "5.0"],
        ["train", "0", "1", "seller", "phone", "10.0", "Sure, it is 2 years old", "", ""],
        ["train", "0", "2", "buyer", "phone", "10.0", "hello there", "", ""],
        ["train", "0", "3", "buyer", "phone", "10.0", "", "offer", "7.0"],
        ["train", "0", "4", "seller", "phone", "10.0", "7 or maybe 13 000, ok", "", ""],
    ]
    path = tmp_path / "turns.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path


def labels(tmp_path, text):
    path = tmp_path / "labels.tsv"
    path.write_text(text, encoding="utf-8")
    return path


def test_apply_adds_only_two_columns_and_normalises(sheet, tmp_path, capsys):
    before_fields, before = read(sheet)
    lab = labels(tmp_path, "0:0\t$5\n0:1\tnone\n0:4\t7 | 13 000\tuncertain: bare 7\n")
    gold_apply.main([str(sheet), str(lab)])
    fields, rows = read(sheet)
    assert fields == before_fields + ["gold_price", "gold_note"]
    assert [r["gold_price"] for r in rows] == ["5.00", "none", "", "", "7.00 | 13000.00"]
    assert rows[4]["gold_note"] == "uncertain: bare 7"
    for old, new in zip(before, rows):
        assert all(old[c] == new[c] for c in before_fields)


def test_invalid_labels_are_rejected_and_the_rest_applied(sheet, tmp_path, capsys):
    lab = labels(tmp_path, "0:0\t$6\n0:1\t2 years\n0:2\t5\n0:3\t7\n0:9\t5\n0:4\tnone\n")
    gold_apply.main([str(sheet), str(lab)])
    out = capsys.readouterr().out
    assert "applied 1 of 6" in out
    for fragment in ("not an exact piece", "needs no label", "no such row", "cannot read"):
        assert fragment in out
    assert [r["gold_price"] for r in read(sheet)[1]] == ["", "", "", "", "none"]


def test_existing_labels_are_not_overwritten_without_the_flag(sheet, tmp_path):
    gold_apply.main([str(sheet), str(labels(tmp_path, "0:0\t$5\n"))])
    gold_apply.main([str(sheet), str(labels(tmp_path, "0:0\tnone\n"))])
    assert read(sheet)[1][0]["gold_price"] == "5.00"
    gold_apply.main([str(sheet), str(labels(tmp_path, "0:0\tnone\n")), "--overwrite"])
    assert read(sheet)[1][0]["gold_price"] == "none"


def test_uncertain_rows_stay_empty_but_are_marked_and_not_reshown(sheet, tmp_path, capsys):
    gold_apply.main([str(sheet), str(labels(tmp_path, "0:4\t?\tbare 7, maybe a count\n"))])
    row = read(sheet)[1][4]
    assert row["gold_price"] == "" and row["gold_note"] == "uncertain: bare 7, maybe a count"
    assert "uncertain (gold_price empty, exclude from scoring): 1" in capsys.readouterr().out
    from scripts import gold_batch
    gold_batch.main([str(sheet), "--n", "10"])
    assert "[0:4]" not in capsys.readouterr().out


@pytest.mark.parametrize("text", ["I can give you mine for free.", "it comes with no cost to you"])
def test_free_is_read_as_a_zero_price(text):
    assert needs_reading(text)


def test_free_label_becomes_zero_and_add_on_free_is_none(tmp_path):
    path = tmp_path / "turns.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerow(["train", "0", "0", "seller", "phone", "10.0", "I can give it for free", "", ""])
        writer.writerow(["train", "0", "1", "seller", "phone", "10.0", "a case is thrown in for free", "", ""])
    gold_apply.main([str(path), str(labels(tmp_path, "0:0\tfree\n0:1\tnone\n"))])
    assert [r["gold_price"] for r in read(path)[1]] == ["0.00", "none"]


def test_dry_run_writes_nothing(sheet, tmp_path):
    original = sheet.read_bytes()
    gold_apply.main([str(sheet), str(labels(tmp_path, "0:0\t$5\n")), "--dry-run"])
    assert sheet.read_bytes() == original
