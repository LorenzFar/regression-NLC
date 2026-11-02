"""Seed reviewable resolver JSONL from CSV gold; never imports resolver decisions.

python -m scripts.resolver_gold_seed --split validation --out evaluation/resolver/review.jsonl
All proposals are silver, not human gold. No CSV is modified. Hard targets and
their ledgers require independent review before use for headline evaluation.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from scripts.gold_csv import read


def prices(cell):
    return [] if not cell.strip() or cell.lower() == "none" else [p.strip() for p in cell.split("|")]


def seed(rows, split, limit=None):
    """Conservative independent seeding; unresolved targets are not manufactured."""
    history, offers, ordinals = [], [], Counter()
    previous = None
    counts, output = Counter(), []
    for row in rows:
        dialogue = row["dialogue"]
        if dialogue != previous:
            history, offers, ordinals = [], [], Counter()
            previous = dialogue
        text, role = row["text"], row["role"].lower()
        note, cell = row.get("gold_note", ""), row.get("gold_price", "").strip()
        amounts = prices(cell)
        act = None
        # Independent lexical seed, NOT an import of resolver rules or parser truth.
        if len(amounts) == 1 and re.search(r"\b(?:how about|i can do|i could do|asking|my price|firm|i offer)\b", text, re.I):
            act = "counter"
        elif cell == "none" and re.search(r"\b(?:condition|how old|mileage|available)\b", text, re.I):
            act = "inquire"
        if note.lower().startswith("uncertain:"):
            counts["uncertain"] += 1
        elif not cell:
            counts["blank"] += 1
        elif role == "seller":
            hard = len(amounts) > 1 or "inferred" in note.lower() or act is None
            record = {"case_id": f"{split}:{dialogue}:{row['turn']}", "dialogue_id": f"{split}:{dialogue}",
                      "split": split, "history": deepcopy(history), "message": text, "speaker": role,
                      "listing_price": float(row["listing_price"]) if row.get("listing_price") else None,
                      "act": act, "price": amounts[0] if len(amounts) == 1 and not hard else None,
                      "act_source": "lexical_seed" if act is not None else "unannotated",
                      "referenced_offer": None, "needs_clarification": False,
                      "deal_agreed": False, "challenge": "review_required" if hard else "single_price",
                      "gold_price": cell, "gold_note": note, "inferred": "inferred" in note.lower(),
                      "label_source": "seeded_rule", "subset": "hard" if hard else "trivial",
                      "review_required": hard, "offers": deepcopy(offers)}
            if not hard:
                record["status"] = "resolved" if amounts else "no_price"
            output.append(record)
            counts["proposed"] += 1
        if text.strip():
            ordinals[role] += 1
            if act == "counter" and len(amounts) == 1 and "inferred" not in note.lower():
                for offer in offers:
                    if offer["owner"] == role:
                        offer["liveness"] = "not_live"
                offers.append({"offer_id": f"{role}-offer-{ordinals[role]}", "owner": role,
                               "turn_index": len(history), "id_source": "minted", "price": amounts[0],
                               "liveness": "unknown", "link": "history"})
            history.append({"speaker": role, "text": text})
        if limit is not None and len(output) >= limit:
            break
    return output, dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.out.suffix.lower() != ".jsonl":
        parser.error("--out must be a new .jsonl file")
    _, rows = read(args.dir / f"{args.split}_turns.csv")
    output, counts = seed(rows, args.split, args.limit)
    with args.out.open("x", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.out), **counts, "label_source": "seeded_rule"}))


if __name__ == "__main__":
    main()
