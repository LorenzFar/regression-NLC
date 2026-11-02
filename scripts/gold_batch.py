"""Show the next unlabelled messages to read, with context. Read-only.

From repository root:
  python -m scripts.gold_batch artifacts/train_turns.csv --n 100

Lists messages that need reading (see number_filter) and whose gold_price is still empty,
in file order, each with the two previous messages of the same dialogue and the listing price.
Labelled rows (including "none") never come back, so stopping and resuming is safe.
"""
import argparse
from pathlib import Path
import sys

from scripts.gold_csv import key, read
from scripts.number_filter import needs_reading


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args(argv)

    _, rows = read(args.csv)
    shown, todo, last_dialogue, history = 0, 0, None, []
    for row in rows:
        if row["dialogue"] != last_dialogue:
            last_dialogue, history = row["dialogue"], []
        text = row["text"]
        if text.strip():
            if needs_reading(text) and not (row.get("gold_price") or "").strip() \
                    and not (row.get("gold_note") or "").strip():
                todo += 1
                if shown < args.n:
                    shown += 1
                    print(f"[{key(row)}] {row['role']} | {row['category']} | listing {float(row['listing_price']):g}")
                    for prev in history[-2:]:
                        print(f"     prev  {prev[:150]}")
                    print(f"     TEXT  {text}\n")
            history.append(f"{row['role']}: {text}")
    print(f"-- showed {shown}; {todo} unlabelled messages remain in total")


if __name__ == "__main__":
    main()
