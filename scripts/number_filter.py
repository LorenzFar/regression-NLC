"""Which messages could contain a price at all? Read-only triage for gold labelling.

From repository root:
  python -m scripts.number_filter artifacts/train_turns.csv              # counts + parser check
  python -m scripts.number_filter artifacts/train_turns.csv --audit 200  # random skipped rows

A message needs reading if its text is non-empty AND contains a digit, a "$", or a spelled
number / money word. This does NOT decide what is a price; it only skips messages that
cannot contain one ("hello, is it available?"). Every spelled number from 1 to 100 and beyond
is built from these words, so any such number trips the filter.
"""
import argparse
import csv
from pathlib import Path
import random
import re
import sys

_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fourty fifty sixty "
    "seventy eighty ninety hundreds? thousands? million grand "
    "bucks? dollars? cents? few couple half dozen free"
).split()
# "no cost" / "no charge" also mean a zero price, so they must be read too.
_EXTRA = r"|\bno (?:cost|charge)\b"
READ_RE = re.compile(r"\d|\$|\b(?:" + "|".join(_WORDS) + r")\b" + _EXTRA, re.I)


def needs_reading(text: str) -> bool:
    return bool(text and text.strip() and READ_RE.search(text))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--audit", type=int, default=0, help="print N random skipped rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    empty = [r for r in rows if not r["text"].strip()]
    read = [r for r in rows if needs_reading(r["text"])]
    skipped = [r for r in rows if r["text"].strip() and not needs_reading(r["text"])]
    print(f"{len(rows)} rows | {len(empty)} empty text | {len(skipped)} skipped (no number) | "
          f"{len(read)} to read ({len(read) / (len(rows) - len(empty)):.0%} of messages)")
    if "parser_price" in rows[0]:
        missed = [r for r in skipped if (r["parser_price"] or "").strip()]
        print(f"parser found a price in a skipped row: {len(missed)}")
        for r in missed[:10]:
            print("  ", r["parser_price"], repr(r["text"][:100]))
    if args.audit:
        print(f"\n--- {args.audit} random skipped rows (scan for any missed number) ---")
        for r in random.Random(args.seed).sample(skipped, min(args.audit, len(skipped))):
            print(f"[{r['dialogue']}:{r['turn']}] {r['text'][:140]!r}")


if __name__ == "__main__":
    main()
