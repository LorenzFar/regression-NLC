"""Compare the rule-based extractor with the original parser's price column.

From repository root:
  python -m scripts.compare_parser                       # train, 5 examples per category
  python -m scripts.compare_parser --examples 15 --only parser_only
  python -m scripts.compare_parser --csv artifacts/compare_train.csv

The parser's price is NOT ground truth: a disagreement means one side is wrong,
and only reading it tells which. The extractor never sees the parser's column.
Until the resolver exists, "mine" is a placeholder pick: the last plausible
absolute mention in the message.
"""
import argparse
import csv
from decimal import Decimal
from collections import Counter
from pathlib import Path
import random
import sys

from price_reference import explain
from scripts.view_data import load, turns

CATEGORIES = {
    "both_none": "neither found a price",
    "agree": "same price",
    "pick_differs": "parser's price is among my mentions, but my placeholder pick is another",
    "parser_only": "parser has a price, I kept none",
    "mine_only": "I kept a price, parser has none",
    "different": "both have a price, parser's is not among my mentions",
}


def compare(text, listing, parser_price):
    candidates = explain(text, listing)
    kept = [c.mention for c in candidates if c.mention is not None]
    offers = [m for m in kept if m.kind == "absolute" and m.plausible is not False]
    mine = offers[-1].value if offers else None
    same = lambda value: value is not None and abs(Decimal(value) - Decimal(str(parser_price))) < Decimal("0.01")
    if parser_price is None:
        category = "both_none" if mine is None else "mine_only"
    elif same(mine):
        category = "agree"
    elif any(same(m.value) for m in kept):
        category = "pick_differs"
    elif mine is None:
        category = "parser_only"
    else:
        category = "different"
    # When I dropped the parser's number, say why: the fastest way to spot my bugs.
    why = next((f"{c.surface!r}: {c.reject_reason}" for c in candidates
                if c.mention is None and parser_price is not None and same(c.value)), "")
    return category, mine, [m.value for m in kept], why


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--examples", type=int, default=5, help="examples printed per category")
    parser.add_argument("--only", choices=list(CATEGORIES), help="print examples for one category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="write every non-empty turn with its category")
    args = parser.parse_args(argv)

    records = []
    for index, dialogue in enumerate(load(args.split)):
        listing = dialogue["items"]["Price"][0]
        for t in turns(dialogue):
            if not t["text"].strip():
                continue   # end-of-chat UI action, not a message
            category, mine, kept, why = compare(t["text"], listing, t["parser_price"])
            records.append({"split": args.split, "dialogue": index, "turn": t["turn"], "role": t["role"],
                            "listing_price": listing, "text": t["text"], "parser_price": t["parser_price"],
                            "mine": mine, "all_mentions": " ".join(kept), "category": category,
                            "rejected_parser_value": why})

    counts = Counter(r["category"] for r in records)
    print(f"{len(records)} messages in {args.split}\n")
    for name, meaning in CATEGORIES.items():
        print(f"  {name:13} {counts[name]:6}  {counts[name] / len(records):6.1%}   {meaning}")
    priced = len(records) - counts["both_none"]
    print(f"\n  agreement on messages where either side found a price: {counts['agree'] / priced:.1%}")

    rng = random.Random(args.seed)
    for name in [args.only] if args.only else [c for c in CATEGORIES if c not in ("both_none", "agree")]:
        pool = [r for r in records if r["category"] == name]
        print(f"\n=== {name} ({len(pool)}): {CATEGORIES[name]}")
        for r in rng.sample(pool, min(args.examples, len(pool))):
            extra = f"  [I rejected {r['rejected_parser_value']}]" if r["rejected_parser_value"] else ""
            print(f"  [{r['dialogue']}:{r['turn']}] listing={r['listing_price']:g} parser={r['parser_price']} "
                  f"mine={r['mine']} all=[{r['all_mentions']}]{extra}\n      {r['text'][:200]!r}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
