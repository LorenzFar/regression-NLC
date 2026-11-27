import csv
import json

from scripts.build_gold_jsonl import build, prices, seller_price


def test_prices_by_role():
    assert prices("200:not_offer; 150:new_offer", "new_offer") == ["150"]
    assert prices("7:reference; 10:new_offer", "reference") == ["7"]
    assert prices("", "new_offer") == []


def test_seller_price_rules():
    assert seller_price("counter", "1250:reference; 2000:new_offer") == "2000"
    assert seller_price("counter", "2750:reference") == "2750"
    assert seller_price("accept", "60:reference; 54:reference") == "54"
    assert seller_price("accept", "") is None
    assert seller_price("reject", "2900:not_offer") is None


def test_build_labels_history_and_limits(tmp_path):
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "dialogue_index", "turn", "message", "gold_act", "gold_prices"])
        w.writeheader()
        w.writerow(dict(case_id="cb-test-0001-t1", dialogue_index=1, turn=1, message="80",
                        gold_act="propose", gold_prices="80:new_offer"))
        w.writerow(dict(case_id="cb-test-0001-t3", dialogue_index=1, turn=3, message="ok",
                        gold_act="accept", gold_prices=""))
    dialogues = tmp_path / "dialogues.jsonl"
    dialogues.write_text(json.dumps({
        "dialogue_index": 1, "list_price": 80.0, "buyer_target": 75.0,
        "turns": [{"turn": 0, "speaker": "buyer", "text": "how much?"},
                  {"turn": 1, "speaker": "seller", "text": "80"},
                  {"turn": 2, "speaker": "buyer", "text": "70?"},
                  {"turn": 3, "speaker": "seller", "text": "ok"}]}) + "\n", encoding="utf-8")

    rows, unlabelled = build(labels, dialogues)  # no workbook: buyer turns stay unlabelled
    last = rows[-1]
    assert unlabelled == 2
    assert [t["speaker"] for t in last["history"]] == ["buyer", "seller", "buyer"]
    assert last["history"][1] == {"speaker": "seller", "text": "80", "act": "propose", "price": "80"}
    assert "act" not in last["history"][0]
    assert last["target"] == 75.0 and last["walk_away"] == 75.0  # 0.9 x 80 = 72, raised to the target
    assert "deal_agreed" not in last


def _extra_workbook(path, checked):
    import pytest

    Workbook = pytest.importorskip("openpyxl").Workbook

    wb = Workbook()
    wb.remove(wb.active)
    seller = wb.create_sheet("seller")
    seller.append(["checked", "case_id", "message", "gold_act", "gold_prices", "deal_agreed",
                   "needs_clarification", "challenge", "dialogue_index", "turn"])
    seller.append([checked, "cb-test-0002-t1", "90", "propose", "90:new_offer", False, False, None, 2, 1])
    buyer = wb.create_sheet("buyer_turns")
    buyer.append(["checked", "turn_id", "act", "price", "dialogue_index", "turn"])
    buyer.append([checked, "cb-test-0002-t0", "inquire", None, 2, 0])
    challenge = wb.create_sheet("challenge")
    challenge.append(["checked", "case_id", "act", "price"])
    wb.save(path)


def test_extra_workbook_needs_checked_rows_and_merges(tmp_path):
    import pytest
    from scripts.build_gold_jsonl import read_extra

    labels = tmp_path / "labels.csv"
    labels.write_text("case_id,dialogue_index,turn,message,gold_act,gold_prices\n", encoding="utf-8")
    dialogues = tmp_path / "dialogues.jsonl"
    dialogues.write_text(json.dumps({
        "dialogue_index": 2, "list_price": 100.0, "buyer_target": 80.0,
        "turns": [{"turn": 0, "speaker": "buyer", "text": "price?"},
                  {"turn": 1, "speaker": "seller", "text": "90"}]}) + "\n", encoding="utf-8")

    unchecked = tmp_path / "unchecked.xlsx"
    _extra_workbook(unchecked, "")
    with pytest.raises(ValueError, match="not marked checked"):
        read_extra(unchecked)

    checked = tmp_path / "checked.xlsx"
    _extra_workbook(checked, "y")
    rows, unlabelled = build(labels, dialogues, extra=read_extra(checked))
    assert unlabelled == 0
    assert rows[0]["act"] == "propose" and rows[0]["price"] == "90"
    assert rows[0]["history"] == [{"speaker": "buyer", "text": "price?", "act": "inquire"}]
