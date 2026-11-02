"""Write labels into gold_price / gold_note of a turn CSV, touching nothing else.

From repository root:
  python -m scripts.gold_apply artifacts/train_turns.csv labels.tsv            # apply
  python -m scripts.gold_apply artifacts/train_turns.csv labels.tsv --dry-run  # validate only
  python -m scripts.gold_apply artifacts/train_turns.csv --status              # progress + format check

labels.tsv: one line per message, tab-separated:   dialogue:turn <TAB> amounts <TAB> note
  amounts = the EXACT text of each money amount, separated by " | ", e.g.  $1.2k | 585
            or  none  (the message was read and has no price)
            or  ?     (not confident: gold_price stays empty, gold_note = "uncertain: ...";
                       such rows must be excluded from scoring, not counted as "no price")
  note    = optional; "delta" for discounts like "$20 off", or "uncertain: why"

Each amount is checked to be an exact substring of the message and is converted by the shared
price_reference.normalise_amount; gold_price ends up as e.g. "1200.00 | 585.00". Invalid lines
are rejected with a reason and everything else is still applied. Before writing, every original
column of every row is verified unchanged. Close the CSV in Excel first.
"""
import argparse
from pathlib import Path
import re
import sys

from price_reference import normalise_amount
from scripts.gold_csv import key, read, write
from scripts.number_filter import needs_reading

FORMAT = re.compile(r"none|\d+\.\d{2}(?: \| \d+\.\d{2})*")


def parse_labels(path: Path):
    labels, errors = {}, []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(" ;; ")
        if len(parts) < 2 or not parts[1].strip():
            errors.append(f"line {number}: need 'dialogue:turn<TAB>amounts'")
        elif parts[0].strip() in labels:
            errors.append(f"line {number}: {parts[0].strip()} appears twice")
        else:
            labels[parts[0].strip()] = (parts[1].strip(), parts[2].strip() if len(parts) > 2 else "")
    return labels, errors


def convert(raw, text):
    """-> (gold_price, error)"""
    if raw.strip() == "?":
        return "", None   # not confident: leave gold_price empty, mark gold_note
    if raw.lower() == "none":
        return "none", None
    values = []
    for piece in (p.strip() for p in raw.split("|")):
        if not piece or piece not in text:
            return None, f"{piece!r} is not an exact piece of the message"
        if piece.lower() in ("free", "no cost", "no charge"):
            values.append("0.00")   # the item's own price is zero
            continue
        value = normalise_amount(piece)
        if value is None:
            return None, f"cannot read {piece!r} as an amount"
        values.append(value)
    return " | ".join(values), None


def status(fields, rows):
    todo = done = bad = stray = unsure = 0
    for row in rows:
        cell = (row.get("gold_price") or "").strip()
        note = (row.get("gold_note") or "").strip()
        if needs_reading(row["text"]):
            if cell:
                done += 1
                bad += not FORMAT.fullmatch(cell)
            elif note.startswith("uncertain"):
                unsure += 1
            else:
                todo += 1
        else:
            stray += bool(cell)
    print(f"to read: {done + unsure + todo} | labelled: {done} | uncertain (gold_price empty, "
          f"exclude from scoring): {unsure} | remaining: {todo} | malformed: {bad} | "
          f"labels on rows that need no reading: {stray}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("labels", type=Path, nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing gold_price")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    fields, rows = read(args.csv)
    if args.status or not args.labels:
        status(fields, rows)
        return

    originals = [dict(r) for r in rows]
    labels, errors = parse_labels(args.labels)
    index = {key(r): i for i, r in enumerate(rows)}
    new_fields = fields + [c for c in ("gold_price", "gold_note") if c not in fields]
    for row in rows:
        row.setdefault("gold_price", "")
        row.setdefault("gold_note", "")

    applied = 0
    for k, (raw, note) in labels.items():
        i = index.get(k)
        if i is None:
            errors.append(f"{k}: no such row")
            continue
        row = rows[i]
        if not needs_reading(row["text"]):
            errors.append(f"{k}: message has no number or money word; it needs no label")
            continue
        if row["gold_price"].strip() and not args.overwrite:
            errors.append(f"{k}: already labelled ({row['gold_price']}); use --overwrite")
            continue
        value, problem = convert(raw, row["text"])
        if problem:
            errors.append(f"{k}: {problem}")
            continue
        if raw.strip() == "?" and not note.startswith("uncertain"):
            note = f"uncertain: {note or 'unsure'}"
        row["gold_price"], row["gold_note"] = value, note
        applied += 1

    protected = [c for c in fields if c not in ("gold_price", "gold_note")]
    for before, after in zip(originals, rows):   # nothing but the two gold columns may differ
        if any(before[c] != after[c] for c in protected):
            sys.exit("ABORTED: an original column would change; nothing written")
    if len(rows) != len(originals):
        sys.exit("ABORTED: row count changed; nothing written")

    if not args.dry_run:
        write(args.csv, new_fields, rows)
    print(f"{'would apply' if args.dry_run else 'applied'} {applied} of {len(labels)} labels; "
          f"{len(errors)} rejected")
    for message in errors[:30]:
        print("  REJECTED", message)
    status(new_fields, rows)


if __name__ == "__main__":
    main()
