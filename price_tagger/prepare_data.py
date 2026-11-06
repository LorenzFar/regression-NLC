"""Turn the turn CSVs into tagger training files (JSONL), one per split.

From repository root:
  python -m price_tagger.prepare_data                       # train + validation -> price_tagger/data/
  python -m price_tagger.prepare_data --splits test         # text only, no labels: do this ONLY for the final run

Per message the file holds:
  pos     character spans of the gold prices (gold_spans)           -> tagged MONEY
  ignore  spans the rules keep as money but the gold does not mark   -> left out of the loss (a budget or a
          market price is money, yet not an offer, so the gold cannot say what the tagger should do)
  everything else, including numbers the rules reject ("6 months", "5:30"), is tagged O. Those negatives are
  the rules' judgement, not ground truth.
  gold    the gold amounts (list of "N.NN"; [] for none; null when the row has no gold label)
  train_ok  false for rows whose gold amount is not written in the text (typos): not used to train.
test rows never carry labels.
"""
import argparse
import json
import sys
from pathlib import Path

from price_reference import explain
from scripts.gold_csv import read

DEFAULT_SPLITS = ("train", "validation")


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def build(row, labelled):
    listing = float(row["listing_price"]) if row["listing_price"] else None
    out = {"id": f"{row['split']}:{row['dialogue']}:{row['turn']}", "text": row["text"], "role": row["role"],
           "category": row["category"], "listing": listing}
    if not labelled:
        return out
    gold_price = row["gold_price"].strip()
    out["gold"] = None if not gold_price else [] if gold_price == "none" else sorted({p.strip() for p in gold_price.split("|")})
    spans = row.get("gold_spans", "")
    pos = [tuple(s) for s in json.loads(spans)] if spans else []
    out["train_ok"] = bool(spans) or not gold_price     # unread rows are all-negative examples
    kept = [(c.start, c.end) for c in explain(row["text"], listing) if c.mention is not None]
    out["pos"] = [list(s) for s in pos]
    out["ignore"] = [list(s) for s in kept if not any(overlaps(s, p) for p in pos)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--out", default="price_tagger/data")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        _, rows = read(Path(args.dir) / f"{split}_turns.csv")
        labelled = split != "test"
        if labelled and rows and "gold_spans" not in rows[0]:
            sys.exit(f"{split}: no gold_spans column; run python -m scripts.gold_spans first")
        rows = [r for r in rows if r["text"].strip()]
        with (Path(args.out) / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(build(r, labelled), ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} messages -> {args.out}/{split}.jsonl" + ("" if labelled else " (text only)"))


if __name__ == "__main__":
    main()
