"""Browse CraigslistBargain dialogues from data/raw (train/validation only).

From repository root:
  python -m scripts.view_data                          # first train dialogue
  python -m scripts.view_data --index 42 --split validation
  python -m scripts.view_data --random 3
  python -m scripts.view_data --grep "meet in the middle" --limit 10
  python -m scripts.view_data --export artifacts/train_turns.csv

The test split is deliberately not offered: it may hold the gold-set dialogues.
Parser intent/price columns are the original rule-based parser's output, not gold labels.
"""
import argparse
import csv
from pathlib import Path
import random
import re
import sys

import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


def load(split):
    return pq.read_table(DATA / f"craigslist_bargains-{split}.parquet").to_pylist()


def turns(dialogue):
    """Yield one flat record per turn; empty text marks an end-of-chat UI action."""
    roles = dialogue["agent_info"]["Role"]
    acts = dialogue["dialogue_acts"]
    for i, (agent, text) in enumerate(zip(dialogue["agent_turn"], dialogue["utterance"])):
        price = acts["price"][i]
        yield {"turn": i, "role": roles[agent], "text": text,
               "parser_intent": acts["intent"][i],
               "parser_price": None if price is None or price < 0 else price}


def show(index, dialogue):
    items = dialogue["items"]
    targets = dict(zip(dialogue["agent_info"]["Role"], dialogue["agent_info"]["Target"]))
    print(f"=== dialogue {index}: {items['Title'][0]!r}")
    print(f"    category={items['Category'][0]}  listing={items['Price'][0]}  "
          f"buyer_target={targets.get('buyer')}  seller_target={targets.get('seller')}")
    for t in turns(dialogue):
        text = t["text"] if t["text"].strip() else f"<UI action: {t['parser_intent']}>"
        price = "" if t["parser_price"] is None else f" {t['parser_price']:g}"
        print(f"  {t['turn']:2} {t['role']:6} {text}   [parser: {t['parser_intent'] or '-'}{price}]")
    print()


def export(rows, split, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "dialogue", "turn", "role", "category", "listing_price",
              "text", "parser_intent", "parser_price"]
    # utf-8-sig lets Excel detect the encoding.
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, dialogue in enumerate(rows):
            items = dialogue["items"]
            for t in turns(dialogue):
                writer.writerow({"split": split, "dialogue": index,
                                 "category": items["Category"][0],
                                 "listing_price": items["Price"][0], **t})
    print(f"wrote {path}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--index", type=int, default=0, help="dialogue number to print")
    parser.add_argument("--random", type=int, metavar="N", help="print N random dialogues")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grep", metavar="REGEX", help="print turns matching REGEX (case-insensitive)")
    parser.add_argument("--limit", type=int, default=20, help="max --grep matches to print")
    parser.add_argument("--export", type=Path, metavar="CSV", help="write one row per turn to CSV")
    args = parser.parse_args(argv)

    rows = load(args.split)
    if args.export:
        export(rows, args.split, args.export)
    elif args.grep:
        pattern, shown, total = re.compile(args.grep, re.I), 0, 0
        for index, dialogue in enumerate(rows):
            listing = dialogue["items"]["Price"][0]
            for t in turns(dialogue):
                if t["text"] and pattern.search(t["text"]):
                    total += 1
                    if shown < args.limit:
                        shown += 1
                        print(f"[{index}:{t['turn']}] {t['role']:6} listing={listing:g} "
                              f"parser={t['parser_price']}  {t['text']}")
        print(f"-- {total} matching turns in {args.split} (showing {shown})")
    elif args.random:
        for index in random.Random(args.seed).sample(range(len(rows)), args.random):
            show(index, rows[index])
    else:
        show(args.index, rows[args.index])
    print(f"({len(rows)} dialogues in {args.split})")


if __name__ == "__main__":
    main()
