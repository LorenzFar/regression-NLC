"""Score a price extractor against hand-labelled prices in seller messages.

Label format (gold_prices column, and pred_prices in a predictions CSV):
    60:new_offer; 45:reference; 200:not_offer
Values are plain numbers as meant ($1.5k -> 1500, "fifty" -> 50). A blank cell on a
row with gold_act filled means the message has no price. Prediction roles are
optional; role metrics then count unroled predictions as wrong roles.

From repository root:
    python -m evaluation.evaluate_prices data/craigslist_test/labelling_sheet.csv --predictions edric_prices.csv
    python -m evaluation.evaluate_prices data/craigslist_test/labelling_sheet.csv --baseline digits
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import metrics

ROLES = ["new_offer", "reference", "not_offer"]
_DIGITS = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s?(k\b)?", re.I)


class PriceLabelError(ValueError):
    """A price cell cannot be parsed."""


def normalize_value(text):
    raw = text.strip().lower().replace("$", "").replace(",", "")
    scale = Decimal(1000) if raw.endswith("k") else Decimal(1)
    try:
        value = Decimal(raw.rstrip("k").strip()) * scale
    except InvalidOperation:
        raise PriceLabelError(f"not a number: {text!r}") from None
    return format(value.normalize(), "f")


def parse_prices(cell):
    """'60:new_offer; 45' -> [('60', 'new_offer'), ('45', None)]."""
    prices = []
    for part in (cell or "").split(";"):
        if not part.strip():
            continue
        value, _, role = part.partition(":")
        role = role.strip().lower() or None
        if role is not None and role not in ROLES:
            raise PriceLabelError(f"unknown role {role!r}; use one of {', '.join(ROLES)}")
        prices.append((normalize_value(value), role))
    return prices


def digits_baseline(message):
    """Naive reference extractor: every digit run is a new offer."""
    out = []
    for match in _DIGITS.finditer(message):
        value = Decimal(match.group(1).replace(",", "")) * (1000 if match.group(2) else 1)
        out.append((format(value.normalize(), "f"), "new_offer"))
    return out


def _matched(gold, pred, key):
    return sum((Counter(map(key, gold)) & Counter(map(key, pred))).values())


def _prf(tp, n_pred, n_gold):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "predicted": n_pred, "gold": n_gold}


def prf(rows, key, keep=lambda price: True):
    tp = n_pred = n_gold = 0
    for row in rows:
        gold = [p for p in row["gold"] if keep(p)]
        pred = [p for p in row["pred"] if keep(p)]
        tp += _matched(gold, pred, key)
        n_pred, n_gold = n_pred + len(pred), n_gold + len(gold)
    return _prf(tp, n_pred, n_gold)


def score(rows, n_resamples=1000, seed=0):
    value = lambda p: p[0]
    value_role = lambda p: p
    is_offer = lambda p: p[1] == "new_offer"
    by_dialogue = defaultdict(list)
    for row in rows:
        by_dialogue[row["dialogue_index"]].append(row)
    groups = list(by_dialogue.values())

    # Role accuracy on values found by both, pairing equal values in order.
    role_pairs = []
    for row in rows:
        remaining = defaultdict(list)
        for v, role in row["pred"]:
            remaining[v].append(role)
        for v, role in row["gold"]:
            if remaining[v]:
                role_pairs.append((role, remaining[v].pop(0) or "none"))
    return {
        "n_messages": len(rows), "n_dialogues": len(groups),
        "messages_with_gold_price": sum(bool(r["gold"]) for r in rows),
        "value": prf(rows, value),
        "value_f1_ci": metrics.bootstrap_ci(groups, lambda s: prf(s, value)["f1"], n_resamples, seed),
        "value_and_role": prf(rows, value_role),
        "new_offer": prf(rows, value, is_offer),
        "message_exact_match": sum(Counter(map(value, r["gold"])) == Counter(map(value, r["pred"]))
                                   for r in rows) / len(rows),
        "role_accuracy_on_matched_values": (sum(g == p for g, p in role_pairs) / len(role_pairs)
                                            if role_pairs else 0.0),
        "role_confusion": dict(Counter(f"{g}->{p}" for g, p in role_pairs if g != p)),
    }


def load_rows(labels_path, predictions_path=None, baseline=None):
    with Path(labels_path).open(encoding="utf-8-sig", newline="") as stream:
        labelled = [r for r in csv.DictReader(stream) if (r.get("gold_act") or "").strip()]
    if not labelled:
        raise PriceLabelError(f"no labelled rows in {labels_path}")
    predictions = {}
    if predictions_path:
        with Path(predictions_path).open(encoding="utf-8-sig", newline="") as stream:
            predictions = {r["case_id"]: r["pred_prices"] for r in csv.DictReader(stream)}
    rows, problems = [], []
    for row in labelled:
        try:
            gold = parse_prices(row.get("gold_prices"))
            if baseline:
                pred = digits_baseline(row["message"])
            elif row["case_id"] in predictions:
                pred = parse_prices(predictions[row["case_id"]])
            else:
                raise PriceLabelError("no prediction for this case")
        except PriceLabelError as exc:
            problems.append(f"{row['case_id']}: {exc}")
            continue
        rows.append({"case_id": row["case_id"], "dialogue_index": int(row["dialogue_index"]),
                     "message": row["message"], "gold": gold, "pred": pred})
    if problems:
        raise PriceLabelError("\n".join(problems))
    return rows


def markdown(report, rows, max_errors=30):
    s = report["scores"]
    fmt = lambda name, m: (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                           f"{m['tp']} | {m['predicted']} | {m['gold']} |")
    ci = s["value_f1_ci"]
    lines = [
        "# Price extraction evaluation on hand-labelled CraigslistBargain test turns", "",
        f"Messages: {s['n_messages']} ({s['messages_with_gold_price']} with a gold price) from "
        f"{s['n_dialogues']} dialogues. Predictions: {report['prediction_source']}.", "",
        "| Match on | Precision | Recall | F1 | TP | Predicted | Gold |", "|---|---|---|---|---|---|---|",
        fmt("value", s["value"]), fmt("value + role", s["value_and_role"]),
        fmt("new offers only", s["new_offer"]), "",
        f"Value F1 95% CI (dialogue bootstrap): [{ci['low']:.3f}, {ci['high']:.3f}]. "
        f"Messages with exactly the right values: {100 * s['message_exact_match']:.1f}%. "
        f"Role accuracy on matched values: {100 * s['role_accuracy_on_matched_values']:.1f}%.", "",
        "Role confusion (gold->pred): " + (", ".join(f"{k}: {v}" for k, v in s["role_confusion"].items()) or "none"),
        "", f"## Messages with wrong values (up to {max_errors})", "",
        "| Case | Gold | Predicted | Message |", "|---|---|---|---|",
    ]
    show = lambda prices: "; ".join(f"{v}:{r}" if r else v for v, r in prices) or "-"
    wrong = [r for r in rows if Counter(v for v, _ in r["gold"]) != Counter(v for v, _ in r["pred"])]
    lines += [f"| {r['case_id']} | {show(r['gold'])} | {show(r['pred'])} | {r['message'][:90].replace('|', '/')} |"
              for r in wrong[:max_errors]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="CSV with case_id and pred_prices")
    source.add_argument("--baseline", choices=["digits"], help="score a naive reference extractor")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/price-eval"))
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.labels, args.predictions, args.baseline)
    except (PriceLabelError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    source_name = f"baseline {args.baseline}" if args.baseline else str(args.predictions)
    report = {"labels": str(args.labels), "prediction_source": source_name, "scores": score(rows, args.bootstrap)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = markdown(report, rows)
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {args.output / 'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
