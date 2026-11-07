"""Score the original parser and the rule-based extractor against gold_price.

From repository root:
  python -m evaluation.price_extractor.run                        # train + validation (development)
  python -m evaluation.price_extractor.run --split validation
  python -m evaluation.price_extractor.run --split test --final   # held out: report only, once the extractor is frozen
  python -m evaluation.price_extractor.run --json artifacts/price_eval.json
  python -m evaluation.price_extractor.run --split validation --tagger distilbert=validation_predictions.jsonl

Scored rows are those with a gold_price (a number or "none"). Systems compared:
  parser          the original parser_price column (one price or none)
  extractor       every plausible absolute price the extractor keeps (a set); picking one is the resolver's job
  NAME=file       with --tagger NAME=predictions.jsonl: the token tagger's predictions (price_tagger/train.py output)
Rows the number filter judged to hold no price (empty gold_price) are not scored; they only
count as "unread FP" when a system still outputs a price for them.
"""
import argparse
import json
import sys
from pathlib import Path

from price_reference import extract
from scripts.gold_csv import read

from .metrics import score, to_set

SPLITS = ("train", "validation", "test")


def extractor_prices(row):
    listing = float(row["listing_price"]) if row["listing_price"] else None
    kept = [m for m in extract(row["text"], listing) if m.kind == "absolute" and m.plausible is not False]
    return [m.value for m in kept]


SYSTEMS = {
    "parser": lambda row: to_set(row["parser_price"]),
    "extractor": lambda row: frozenset(row["extractor"]),
}


def load_rows(split, directory):
    _, rows = read(Path(directory) / f"{split}_turns.csv")
    scored, unread = [], []
    for r in rows:
        r["extractor"] = extractor_prices(r) if r["text"].strip() else []
        if r["gold_price"].strip():
            r["gold"] = to_set(r["gold_price"])
            scored.append(r)
        elif r["text"].strip():
            unread.append(r)
    return scored, unread


def tagger_system(path):
    predictions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            predictions[p["id"]] = frozenset(p["values"])

    def predict(row):
        key = f"{row['split']}:{row['dialogue']}:{row['turn']}"
        if key not in predictions:
            if not row["text"].strip():
                return frozenset()
            raise SystemExit(f"{path}: no prediction for {key}; the file must cover every split you evaluate")
        return predictions[key]
    return predict


def evaluate(splits, directory="artifacts", taggers=None):
    systems = dict(SYSTEMS)
    for name, path in (taggers or {}).items():
        systems[name] = tagger_system(path)
    scored, unread = [], []
    for split in splits:
        s, u = load_rows(split, directory)
        scored += s
        unread += u
    if not any(r["parser_price"].strip() for r in scored + unread):
        systems.pop("parser")   # e.g. test_turns.csv carries no parser output: nothing to compare with
    report = {"splits": list(splits), "rows_scored": len(scored), "unread_rows": len(unread), "systems": {}}
    for name, predict in systems.items():
        result = score(scored, predict)
        result["unread_false_alarms"] = sum(bool(predict(r)) for r in unread)
        report["systems"][name] = result
    return report


def print_report(report):
    print(f"splits: {', '.join(report['splits'])} | rows scored: {report['rows_scored']} "
          f"| unread rows (no number): {report['unread_rows']}\n")
    print(f"{'system':<16}{'exact':>8}{'prec':>8}{'rec':>8}{'F1':>8}   "
          f"{'found':>6}{'false':>6}{'missed':>7}{'none ok':>8}{'unread FP':>10}")
    for name, r in report["systems"].items():
        o, c = r["overall"], r["overall"]["confusion"]
        print(f"{name:<16}{o['exact_match']:>8.3f}{o['precision']:>8.3f}{o['recall']:>8.3f}{o['f1']:>8.3f}   "
              f"{c['price_found']:>6}{c['false_price']:>6}{c['missed_price']:>7}{c['correct_none']:>8}"
              f"{r['unread_false_alarms']:>10}")
    names = list(report["systems"])
    for g in ("role", "category"):
        print(f"\nby {g} (exact match, F1 per system)")
        print(f"{g:<14}{'n':>6}" + "".join(f"{n:>26}" for n in names))
        for value, first in report["systems"][names[0]][g].items():
            cells = "".join(f"{report['systems'][s][g][value]['exact_match']:>14.3f}"
                            f"{report['systems'][s][g][value]['f1']:>12.3f}" for s in names)
            print(f"{value:<14}{first['n']:>6}{cells}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=SPLITS + ("dev",), default="dev", help='"dev" = train + validation (default)')
    ap.add_argument("--final", action="store_true", help="required for the held-out test split")
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--json", help="also write the full report here")
    ap.add_argument("--tagger", action="append", default=[], metavar="NAME=FILE",
                    help="add a token tagger's predictions as a system (repeatable)")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        ap.error("test is held out for final reporting; pass --final once the extractor is frozen")
    splits = ("train", "validation") if args.split == "dev" else (args.split,)
    taggers = dict(t.split("=", 1) for t in args.tagger)
    report = evaluate(splits, args.dir, taggers)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
