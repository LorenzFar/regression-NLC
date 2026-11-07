"""Score the seller-act NLU against hand-labelled CraigslistBargain test turns.

From repository root, after filling gold_act in the labelling sheet:
    python -m evaluation.evaluate_nlu data/craigslist_test/labelling_sheet.csv --checkpoint models/nlu_k2_s0.pt
    python -m evaluation.evaluate_nlu labels.csv --predictions artifacts/nlu-eval/predictions.csv
    python -m evaluation.evaluate_nlu labels.csv --checkpoint models/nlu_k2_s0.pt --second-annotator labels_b.csv

--checkpoint runs the model and saves predictions.csv; --predictions rescores saved
predictions without torch. Writes report.json, report.md and predictions.csv.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import metrics

# Same order as nlu.ACTS; repeated so scoring saved predictions does not import torch.
ACTS = ["propose", "counter", "accept", "reject", "inquire", "vague"]
DEFAULT_DIALOGUES = Path("data/craigslist_test/dialogues.jsonl")
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


class LabelError(ValueError):
    """The labelled sheet contains values the scorer cannot interpret."""


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_gold(path):
    """Rows with a gold_act; blank rows are unlabelled and skipped."""
    gold, problems = [], []
    for line, row in enumerate(read_csv(path), 2):
        act = (row.get("gold_act") or "").strip().lower()
        if not act:
            continue
        if act not in ACTS:
            problems.append(f"line {line} ({row.get('case_id')}): unknown act {act!r}")
            continue
        gold.append({**row, "gold_act": act, "dialogue_index": int(row["dialogue_index"]),
                     "turn": int(row["turn"])})
    if problems:
        raise LabelError("invalid gold_act values; use one of " + ", ".join(ACTS) + "\n" + "\n".join(problems))
    if not gold:
        raise LabelError(f"no labelled rows in {path}; fill the gold_act column first")
    return gold


def load_dialogues(path):
    with Path(path).open(encoding="utf-8") as stream:
        return {d["dialogue_index"]: d for d in map(json.loads, stream)}


def history_for(dialogue, turn):
    """Full oldest-first public history before the turn; NLU keeps its own last-k window."""
    return [(t["speaker"].upper(), t["text"]) for t in dialogue["turns"] if t["turn"] < turn]


def predict_with_checkpoint(rows, dialogues, args):
    from retrieval_pipeline.nlu_adapter import NLUAdapter

    adapter = NLUAdapter(args.checkpoint, model_name=args.model_name, device=args.device,
                         local_files_only=args.offline)
    if adapter.resource_metadata["labels"] != ACTS:
        raise LabelError("checkpoint label order differs from the scorer's ACTS")
    predictions = {}
    for row in rows:
        history = history_for(dialogues[row["dialogue_index"]], row["turn"])
        predictions[row["case_id"]] = adapter.predict(history, row["message"])
    return predictions, adapter.resource_metadata


def load_predictions(path):
    return {row["case_id"]: {act: float(row[f"p_{act}"]) for act in ACTS} for row in read_csv(path)}


def score(rows, threshold=0.7, n_resamples=1000, seed=0):
    """rows need gold_act, probabilities and dialogue_index."""
    for row in rows:
        probabilities = row["probabilities"]
        row["pred_act"] = max(ACTS, key=lambda act: probabilities[act])
        row["confidence"] = probabilities[row["pred_act"]]
        row["correct"] = row["pred_act"] == row["gold_act"]
    gold = [r["gold_act"] for r in rows]
    pred = [r["pred_act"] for r in rows]
    confidences = [r["confidence"] for r in rows]
    correct = [r["correct"] for r in rows]

    majority = Counter(gold).most_common(1)[0][0]
    by_dialogue = defaultdict(list)
    for row in rows:
        by_dialogue[row["dialogue_index"]].append(row)
    groups = list(by_dialogue.values())

    def stat(fn):
        return lambda sample: fn([r["gold_act"] for r in sample], [r["pred_act"] for r in sample])

    accept = metrics.per_class(gold, pred, ACTS)["accept"]
    false_accepts = [r for r in rows if r["pred_act"] == "accept" and r["gold_act"] != "accept"]
    return {
        "n_turns": len(rows), "n_dialogues": len(groups),
        "gold_distribution": dict(Counter(gold)),
        "accuracy": metrics.accuracy(gold, pred),
        "macro_f1": metrics.macro_f1(gold, pred, ACTS),
        "accuracy_ci": metrics.bootstrap_ci(groups, stat(metrics.accuracy), n_resamples, seed),
        "macro_f1_ci": metrics.bootstrap_ci(
            groups, stat(lambda g, p: metrics.macro_f1(g, p, ACTS)), n_resamples, seed),
        "majority_baseline": {"act": majority,
                              "accuracy": metrics.accuracy(gold, [majority] * len(gold)),
                              "macro_f1": metrics.macro_f1(gold, [majority] * len(gold), ACTS)},
        "per_class": metrics.per_class(gold, pred, ACTS),
        "confusion_matrix": {"labels": ACTS, "rows_gold_cols_pred": metrics.confusion_matrix(gold, pred, ACTS)},
        "calibration": metrics.reliability(confidences, correct),
        "coverage": metrics.coverage_curve(confidences, correct, sorted(set(THRESHOLDS + [threshold]))),
        "accept_errors": {
            "precision": accept["precision"], "recall": accept["recall"],
            "false_accepts": len(false_accepts),
            "confident_false_accepts": sum(r["confidence"] >= threshold for r in false_accepts),
            "threshold": threshold,
        },
    }


def agreement(gold_rows, second_path):
    second = {}
    for row in read_csv(second_path):
        act = (row.get("gold_act") or "").strip().lower()
        if act in ACTS:
            second[row["case_id"]] = act
    pairs = [(r["gold_act"], second[r["case_id"]]) for r in gold_rows if r["case_id"] in second]
    if not pairs:
        return {"n_overlap": 0}
    first, other = zip(*pairs)
    return {"n_overlap": len(pairs), "raw_agreement": metrics.accuracy(first, other),
            "cohen_kappa": metrics.cohen_kappa(list(first), list(other)),
            "disagreements": dict(Counter(f"{a}->{b}" for a, b in pairs if a != b))}


def markdown(report, rows, max_errors=30):
    s = report["scores"]
    pct = lambda x: f"{100 * x:.1f}%"
    ci = lambda c: f"[{c['low']:.3f}, {c['high']:.3f}]"
    lines = [
        "# NLU evaluation on hand-labelled CraigslistBargain test turns", "",
        f"Turns: {s['n_turns']} from {s['n_dialogues']} dialogues. Prediction source: {report['prediction_source']}.", "",
        "| Metric | Value | 95% CI (dialogue bootstrap) |", "|---|---|---|",
        f"| Macro-F1 | {s['macro_f1']:.3f} | {ci(s['macro_f1_ci'])} |",
        f"| Accuracy | {s['accuracy']:.3f} | {ci(s['accuracy_ci'])} |",
        f"| Majority baseline ({s['majority_baseline']['act']}) macro-F1 / accuracy | "
        f"{s['majority_baseline']['macro_f1']:.3f} / {s['majority_baseline']['accuracy']:.3f} | |",
        f"| ECE (calibration error) | {s['calibration']['ece']:.3f} | |",
    ]
    if "agreement" in report:
        a = report["agreement"]
        if a["n_overlap"]:
            lines.append(f"| Annotator Cohen's kappa ({a['n_overlap']} shared turns) | {a['cohen_kappa']:.3f} | |")
    lines += ["", "## Per act", "", "| Act | Precision | Recall | F1 | Gold | Predicted |", "|---|---|---|---|---|---|"]
    lines += [f"| {act} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} | {v['predicted']} |"
              for act, v in s["per_class"].items()]
    lines += ["", "## Confusion matrix (rows = gold, columns = predicted)", "",
              "| gold \\ pred | " + " | ".join(ACTS) + " |", "|---" * (len(ACTS) + 1) + "|"]
    lines += [f"| {act} | " + " | ".join(map(str, counts)) + " |"
              for act, counts in zip(ACTS, s["confusion_matrix"]["rows_gold_cols_pred"])]
    lines += ["", "## Gate coverage", "", "| Threshold | Coverage | Kept | Accuracy on kept |", "|---|---|---|---|"]
    lines += [f"| {c['threshold']} | {pct(c['coverage'])} | {c['kept']} | {pct(c['accuracy_on_kept'])} |"
              for c in s["coverage"]]
    lines += ["", "## Reliability", "", "| Confidence bin | n | Mean confidence | Accuracy |", "|---|---|---|---|"]
    lines += [f"| {b['bin']} | {b['n']} | {b['confidence']:.3f} | {b['accuracy']:.3f} |" for b in s["calibration"]["bins"]]
    e = s["accept_errors"]
    lines += ["", "## Accept errors", "",
              f"Accept precision {e['precision']:.3f}, recall {e['recall']:.3f}. "
              f"False accepts: {e['false_accepts']}, of which {e['confident_false_accepts']} "
              f"had confidence >= {e['threshold']} and would pass the gate.", "",
              f"## Most confident mistakes (up to {max_errors})", "",
              "| Case | Gold | Predicted | Confidence | Message |", "|---|---|---|---|---|"]
    mistakes = sorted((r for r in rows if not r["correct"]), key=lambda r: -r["confidence"])[:max_errors]
    lines += [f"| {r['case_id']} | {r['gold_act']} | {r['pred_act']} | {r['confidence']:.2f} | "
              f"{r['message'][:90].replace('|', '/')} |" for r in mistakes]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="labelling sheet CSV with gold_act filled")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="NLU checkpoint to run")
    source.add_argument("--predictions", type=Path, help="saved predictions.csv to rescore")
    parser.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES)
    parser.add_argument("--second-annotator", type=Path, help="another annotator's sheet for Cohen's kappa")
    parser.add_argument("--threshold", type=float, default=0.7, help="gate confidence threshold")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/nlu-eval"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true", help="use only cached tokenizer/config")
    args = parser.parse_args(argv)

    try:
        rows = load_gold(args.labels)
        if args.checkpoint:
            predictions, resources = predict_with_checkpoint(rows, load_dialogues(args.dialogues), args)
            source_name = f"checkpoint {args.checkpoint}"
        else:
            predictions, resources = load_predictions(args.predictions), None
            source_name = f"saved predictions {args.predictions}"
    except (LabelError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    missing = [r["case_id"] for r in rows if r["case_id"] not in predictions]
    if missing:
        parser.exit(2, f"error: no prediction for {len(missing)} labelled cases, e.g. {missing[:5]}\n")
    for row in rows:
        row["probabilities"] = predictions[row["case_id"]]

    report = {"labels": str(args.labels), "prediction_source": source_name, "resources": resources,
              "scores": score(rows, args.threshold, args.bootstrap)}
    if args.second_annotator:
        report["agreement"] = agreement(rows, args.second_annotator)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "dialogue_index", "turn", "gold_act", "pred_act", "confidence", "correct",
                         *(f"p_{act}" for act in ACTS), "message"])
        for r in rows:
            writer.writerow([r["case_id"], r["dialogue_index"], r["turn"], r["gold_act"], r["pred_act"],
                             f"{r['confidence']:.6f}", r["correct"],
                             *(f"{r['probabilities'][act]:.6f}" for act in ACTS), r["message"]])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = markdown(report, rows)
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {args.output / 'report.md'}, report.json and predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
