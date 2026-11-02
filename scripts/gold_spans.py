"""Add a gold_spans column (character offsets of the gold amounts) to the train and validation CSVs.

From repository root:
  python -m scripts.gold_spans            # writes artifacts/train_turns.csv and validation_turns.csv in place
  python -m scripts.gold_spans --dry-run  # only report

gold_spans is JSON: a list of [start, end] pairs, one per occurrence of a gold amount in the text.
  "[]"   the gold says none: no span in this message is a gold price
  ""     not usable: no gold label (unread row) or a gold amount that is not written in the text (typo)
Every occurrence of a repeated amount is marked. test_turns.csv is never touched: test is held out.
Spans come from the extractor's own number reading, rejected spans included, so a gold price the rules
would reject ("price of 2000") is still located.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

from price_reference import explain
from scripts.gold_csv import read, write

COLUMN = "gold_spans"
SPLITS = ("train", "validation")


def spans_for(row):
    """(gold_spans cell, status) for one row."""
    gold = row["gold_price"].strip()
    if not gold:
        return "", "unread"
    if gold == "none":
        return "[]", "none"
    listing = float(row["listing_price"]) if row["listing_price"] else None
    candidates = explain(row["text"], listing)
    found, repeated = [], False
    for amount in (part.strip() for part in gold.split("|")):
        hits = [(c.start, c.end) for c in candidates if c.value == amount]
        if not hits:
            return "", "missing"
        repeated = repeated or len(set(hits)) > 1
        found += hits
    spans = sorted(set(found))
    return json.dumps([list(s) for s in spans]), "repeated" if repeated else "unique"


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for split in SPLITS:
        path = Path(args.dir) / f"{split}_turns.csv"
        fields, rows = read(path)
        counts, examples = Counter(), {"repeated": [], "missing": []}
        for r in rows:
            cell, status = spans_for(r)
            r[COLUMN] = cell
            counts[status] += 1
            if status in examples:
                examples[status].append(r)
        if COLUMN not in fields:
            fields.append(COLUMN)
        print(f"{split}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        for status, rs in examples.items():
            print(f"  {status} rows ({len(rs)}):")
            for r in rs[:40]:
                print(f"    [{r['dialogue']}:{r['turn']}] gold={r['gold_price']} :: {r['text'][:110]!r}")
        if not args.dry_run:
            write(path, fields, rows)


if __name__ == "__main__":
    main()
