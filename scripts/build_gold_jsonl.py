"""Build gold.jsonl (evaluation/data.py schema) from the hand-labelled CraigslistBargain test sheets.

From repository root:
  python -m scripts.build_gold_jsonl                          # RQ1 + RQ2 -> artifacts/gold.jsonl
  python -m scripts.build_gold_jsonl --rq1-only               # acts/prices only, no RQ2 workbook
  python -m scripts.build_gold_jsonl --output artifacts/gold.jsonl

Sources (all in data/craigslist_test/):
  labels_barry.csv   seller act (gold_act) and prices (gold_prices "value:role; ...") - the RQ1 gold set
  rq2_review.xlsx    sheet seller_rq2: deal_agreed, needs_clarification, challenge per seller row
                     sheet buyer_turns: act and price of every buyer turn before a labelled seller row
                     sheet challenge: hand-picked RQ2 cases from other dialogues -> --challenge-output
  dialogues.jsonl    full turn text, listing price and the human buyer's target

Each row's history carries act/price on every turn, so retrieval_pipeline.gold_replay rebuilds the
pre-turn state from gold labels. A seller row's price is its first new_offer (propose/counter, falling
back to a reference) or its last reference (accept). target is the dialogue's real buyer target;
walk_away is 0.9 x listing, raised to the target when the target is higher.
Needs openpyxl for the RQ2 workbook.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

DATA = Path("data/craigslist_test")
ACTS = {"propose", "counter", "accept", "reject", "inquire", "vague"}
WALK_AWAY_RATIO = 0.9


def prices(cell, role):
    """Values with the given role from a gold_prices cell such as '200:not_offer; 150:new_offer'."""
    out = []
    for part in (cell or "").split(";"):
        value, _, kind = part.strip().partition(":")
        if kind.strip() == role:
            out.append(value.strip())
    return out


def seller_price(act, cell):
    if act in ("propose", "counter"):
        found = prices(cell, "new_offer") or prices(cell, "reference")
        return found[0] if found else None
    if act == "accept":
        found = prices(cell, "reference")
        return found[-1] if found else None
    return None


def number(value):
    """Excel may store prices as numbers; keep them as plain decimal strings."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def boolean(value, where):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise ValueError(f"{where}: expected TRUE or FALSE, got {value!r}")


def read_sheet(workbook, name):
    ws = workbook[name]
    header = [c.value for c in ws[1]]
    return [dict(zip(header, row)) for row in ws.iter_rows(min_row=2, values_only=True) if row[1]]


def build(labels_path, dialogues_path, workbook_path=None):
    labels = {r["case_id"]: r for r in csv.DictReader(open(labels_path, encoding="utf-8-sig"))}
    dialogues = {d["dialogue_index"]: d for d in map(json.loads, open(dialogues_path, encoding="utf-8"))}
    seller_rq2, buyer = {}, {}
    if workbook_path is not None:
        from openpyxl import load_workbook

        workbook = load_workbook(workbook_path, read_only=True)
        seller_rq2 = {r["case_id"]: r for r in read_sheet(workbook, "seller_rq2")}
        buyer = {r["turn_id"]: r for r in read_sheet(workbook, "buyer_turns")}

    def turn_label(dialogue_index, turn):
        cid = f"cb-test-{dialogue_index:04d}-t{turn['turn']}"
        if turn["speaker"] == "seller":
            row = labels.get(cid)
            return None if row is None else (row["gold_act"], seller_price(row["gold_act"], row["gold_prices"]))
        row = buyer.get(cid)
        return None if row is None else (str(row["act"]).strip(), number(row["price"]))

    rows, unlabelled_history = [], 0
    for cid, label in labels.items():
        act = label["gold_act"].strip()
        if act not in ACTS:
            raise ValueError(f"{cid}: unknown act {act!r}")
        dialogue = dialogues[int(label["dialogue_index"])]
        current = int(label["turn"])
        history, complete = [], True
        for turn in dialogue["turns"]:
            if turn["turn"] >= current:
                break
            entry = {"speaker": turn["speaker"], "text": turn["text"]}
            found = turn_label(dialogue["dialogue_index"], turn)
            if found is None or found[0] not in ACTS:
                complete = False
            else:
                entry["act"] = found[0]
                if found[1] is not None:
                    entry["price"] = found[1]
            history.append(entry)
        if not complete:
            unlabelled_history += 1
        listing = dialogue["list_price"]
        target = dialogue["buyer_target"]
        row = {
            "case_id": cid, "origin": "craigslist_test", "dialogue_index": dialogue["dialogue_index"],
            "turn": current, "history": history, "message": label["message"],
            "act": act, "price": seller_price(act, label["gold_prices"]),
            "listing_price": listing, "target": target,
            "walk_away": max(round(listing * WALK_AWAY_RATIO, 2), target),
        }
        if workbook_path is not None:
            rq2 = seller_rq2.get(cid)
            if rq2 is None:
                raise ValueError(f"{cid}: missing from the seller_rq2 sheet")
            row["needs_clarification"] = boolean(rq2["needs_clarification"], f"{cid}.needs_clarification")
            row["deal_agreed"] = boolean(rq2["deal_agreed"], f"{cid}.deal_agreed")
            row["challenge"] = [t.strip() for t in str(rq2["challenge"] or "").split(",") if t.strip()]
        rows.append(row)
    return rows, unlabelled_history


def build_challenge(workbook_path, dialogues_path):
    """Hand-picked RQ2 cases from other test dialogues; history is unlabelled, so replay predicts it."""
    from openpyxl import load_workbook

    dialogues = {d["dialogue_index"]: d for d in map(json.loads, open(dialogues_path, encoding="utf-8"))}
    rows = []
    for item in read_sheet(load_workbook(workbook_path, read_only=True), "challenge"):
        cid = item["case_id"]
        act = str(item["act"]).strip()
        if act not in ACTS:
            raise ValueError(f"{cid}: unknown act {act!r}")
        dialogue = dialogues[int(item["dialogue_index"])]
        current = int(item["turn"])
        message = next(t["text"] for t in dialogue["turns"] if t["turn"] == current)
        listing, target = dialogue["list_price"], dialogue["buyer_target"]
        rows.append({
            "case_id": cid, "origin": "craigslist_test_challenge", "dialogue_index": dialogue["dialogue_index"],
            "turn": current,
            "history": [{"speaker": t["speaker"], "text": t["text"]} for t in dialogue["turns"] if t["turn"] < current],
            "message": message, "act": act, "price": number(item["price"]),
            "listing_price": listing, "target": target,
            "walk_away": max(round(listing * WALK_AWAY_RATIO, 2), target),
            "needs_clarification": boolean(item["needs_clarification"], f"{cid}.needs_clarification"),
            "deal_agreed": boolean(item["deal_agreed"], f"{cid}.deal_agreed"),
            "challenge": [t.strip() for t in str(item["challenge"] or "").split(",") if t.strip()],
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DATA / "labels_barry.csv")
    parser.add_argument("--dialogues", type=Path, default=DATA / "dialogues.jsonl")
    parser.add_argument("--workbook", type=Path, default=DATA / "rq2_review.xlsx")
    parser.add_argument("--rq1-only", action="store_true", help="skip the RQ2 workbook")
    parser.add_argument("--output", type=Path, default=Path("artifacts/gold.jsonl"))
    parser.add_argument("--challenge-output", type=Path, default=Path("artifacts/gold_challenge.jsonl"),
                        help="hand-picked RQ2 cases (workbook sheet 'challenge'); kept out of --output")
    args = parser.parse_args(argv)
    try:
        rows, unlabelled = build(args.labels, args.dialogues, None if args.rq1_only else args.workbook)
        challenge = [] if args.rq1_only else build_challenge(args.workbook, args.dialogues)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    outputs = [(args.output, rows)] + ([(args.challenge_output, challenge)] if challenge else [])
    for path, items in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in items:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}: {len(rows)} rows; {unlabelled} with incompletely labelled history")
    if challenge:
        print(f"wrote {args.challenge_output}: {len(challenge)} challenge rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
