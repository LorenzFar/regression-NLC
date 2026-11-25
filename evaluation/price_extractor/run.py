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
    """(scored, unread). gold_price is the one price its speaker puts forward; a message whose speaker proposes
    several prices has an empty gold_price with an "uncertain:" note and is left out (counted in EXCLUDED)."""
    _, rows = read(Path(directory) / f"{split}_turns.csv")
    scored, unread = [], []
    for r in rows:
        r["extractor"] = extractor_prices(r) if r["text"].strip() else []
        cell = r["gold_price"].strip()
        if cell:
            r["gold"] = to_set(cell)
            scored.append(r)
        elif r["gold_note"].startswith("uncertain"):
            EXCLUDED[0] += 1
        elif r["text"].strip():
            unread.append(r)
    return scored, unread


EXCLUDED = [0]


def tagger_system(path, drop_rule_deltas=False):
    """Predictions of a token tagger. With drop_rule_deltas (a hybrid, reported under its own name) a predicted
    price that overlaps a mention the rules tag as a reduction ("$25 off") is dropped."""
    predictions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            predictions[p["id"]] = p

    def predict(row):
        key = f"{row['split']}:{row['dialogue']}:{row['turn']}"
        if key not in predictions:
            if not row["text"].strip():
                return frozenset()
            raise SystemExit(f"{path}: no prediction for {key}; the file must cover every split you evaluate")
        p = predictions[key]
        if not drop_rule_deltas or not p["spans"]:
            return frozenset(p["values"])
        listing = float(row["listing_price"]) if row["listing_price"] else None
        deltas = [(m.start, m.end) for m in extract(row["text"], listing) if m.kind == "delta"]
        return frozenset(v for (a, b), v in zip(p["spans"], p["values"]) if not any(a < e and s < b for s, e in deltas))
    return predict


def evaluate(splits, directory="artifacts", taggers=None, hybrids=None):
    systems = dict(SYSTEMS)
    for name, path in (taggers or {}).items():
        systems[name] = tagger_system(path)
    for name, path in (hybrids or {}).items():
        systems[name] = tagger_system(path, drop_rule_deltas=True)
    scored, unread = [], []
    for split in splits:
        s, u = load_rows(split, directory)
        scored += s
        unread += u
    if not any(r["parser_price"].strip() for r in scored + unread):
        systems.pop("parser")   # e.g. test_turns.csv carries no parser output: nothing to compare with
    report = {"splits": list(splits), "rows_scored": len(scored), "unread_rows": len(unread),
              "excluded_multi_proposal_rows": EXCLUDED[0], "systems": {}}
    EXCLUDED[0] = 0
    for name, predict in systems.items():
        result = score(scored, predict)
        result["unread_false_alarms"] = sum(bool(predict(r)) for r in unread)
        report["systems"][name] = result
    return report


def print_report(report):
    print(f"excluded (uncertain) rows: {report['excluded_multi_proposal_rows']}")
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
    ap.add_argument("--hybrid", action="append", default=[], metavar="NAME=FILE",
                    help="a tagger's predictions with prices the rules tag as reductions dropped (rules + ML)")
    ap.add_argument("--tagger", action="append", default=[], metavar="NAME=FILE",
                    help="add a token tagger's predictions as a system (repeatable)")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        ap.error("test is held out for final reporting; pass --final once the extractor is frozen")
    splits = ("train", "validation") if args.split == "dev" else (args.split,)
    taggers = dict(t.split("=", 1) for t in args.tagger)
    hybrids = dict(t.split("=", 1) for t in args.hybrid)
    report = evaluate(splits, args.dir, taggers, hybrids)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
