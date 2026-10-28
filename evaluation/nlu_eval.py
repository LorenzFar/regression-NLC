"""RQ1: evaluate NLU systems on the hand-labelled gold set.

Systems are either checkpoints (``--system k2@0=models/nlu_k2_s0.pt``) or prediction
files (``--predictions parser=parser_preds.jsonl``, rows ``{"case_id", "act"}`` with an
optional ``"probs"``). Names of the form ``label@seed`` are pooled into mean +/- std.

    python -m evaluation.nlu_eval --gold gold.jsonl \\
        --predictions parser=parser.jsonl \\
        --system k0@0=models/k0_s0.pt --system k2@0=models/nlu_k2_s0.pt --output artifacts/rq1.json

The first system (predictions first, then checkpoints) is the baseline for paired tests.
"""
from __future__ import annotations

import argparse
import statistics
import sys

from . import nlu_metrics as m
from .data import load_gold, read_jsonl, write_json


def _act_of(probs):
    return m._top(probs, m.ACTS)[0]


def predictions_from_checkpoint(gold, checkpoint, *, model_name="distilbert-base-uncased", device="cpu"):
    from retrieval_pipeline.nlu_adapter import NLUAdapter

    adapter = NLUAdapter(checkpoint, model_name=model_name, device=device)
    out = []
    for row in gold:
        history = [(t["speaker"].upper(), t["text"]) for t in row.get("history", [])]
        probs = adapter.predict(history, row["message"])
        out.append({"act": _act_of(probs), "probs": probs})
    return out


def predictions_from_file(gold, path):
    by_case = {}
    for row in read_jsonl(path):
        probs = row.get("probs")
        act = row.get("act") or (_act_of(probs) if probs else None)
        if act not in m.ACTS:
            raise ValueError(f"{path}: {row.get('case_id')!r} has no valid act")
        by_case[row["case_id"]] = {"act": act, "probs": probs}
    missing = [g["case_id"] for g in gold if g["case_id"] not in by_case]
    if missing:
        raise ValueError(f"{path}: no prediction for {len(missing)} gold cases, e.g. {missing[:3]}")
    return [by_case[g["case_id"]] for g in gold]


def evaluate_system(gold, preds, *, n_resamples=1000, seed=0) -> dict:
    truth = [g["act"] for g in gold]
    pred = [p["act"] for p in preds]
    report = {
        "n": len(gold),
        "accuracy": m.accuracy(truth, pred),
        "macro_f1": m.bootstrap_ci(m.macro_f1, truth, pred, n_resamples=n_resamples, seed=seed),
        "per_class": m.per_class_prf(truth, pred),
        "confusion_matrix": {"labels": list(m.ACTS), "rows_are": "gold",
                             "matrix": m.confusion_matrix(truth, pred)},
    }
    tags = sorted({t for g in gold for t in g.get("challenge", [])})
    by_tag = {}
    for tag in tags + ["(untagged)"]:
        idx = [i for i, g in enumerate(gold)
               if (tag in g.get("challenge", [])) or (tag == "(untagged)" and not g.get("challenge"))]
        if idx:
            by_tag[tag] = {"n": len(idx),
                           "accuracy": m.accuracy([truth[i] for i in idx], [pred[i] for i in idx])}
    report["by_challenge_tag"] = by_tag
    if all(p.get("probs") for p in preds):
        probs = [p["probs"] for p in preds]
        rc = m.risk_coverage(probs, truth)
        report["calibration"] = {
            "ece": m.expected_calibration_error(probs, truth),
            "nll": m.negative_log_likelihood(probs, truth),
            "brier": m.brier_score(probs, truth),
            "bins": m.calibration_bins(probs, truth),
            "aurc": rc["aurc"],
            "accuracy_at_coverage": {str(c): m.accuracy_at_coverage(rc["curve"], c)
                                     for c in (0.5, 0.8, 0.9, 1.0)},
        }
    return report


def evaluate(gold, systems: dict, *, baseline=None, n_resamples=1000, seed=0) -> dict:
    """``systems`` maps name -> predictions aligned with ``gold``."""
    if not systems:
        raise ValueError("at least one system is required")
    baseline = baseline or next(iter(systems))
    if baseline not in systems:
        raise ValueError(f"unknown baseline {baseline!r}")
    truth = [g["act"] for g in gold]
    report = {"gold_cases": len(gold), "baseline": baseline, "systems": {}, "vs_baseline": {}, "pooled": {}}
    for name, preds in systems.items():
        report["systems"][name] = evaluate_system(gold, preds, n_resamples=n_resamples, seed=seed)
    base = [p["act"] for p in systems[baseline]]
    for name, preds in systems.items():
        if name == baseline:
            continue
        pred = [p["act"] for p in preds]
        report["vs_baseline"][name] = {
            "mcnemar": m.mcnemar_exact(truth, pred, base),
            "macro_f1_diff": m.paired_bootstrap_diff(m.macro_f1, truth, pred, base,
                                                     n_resamples=n_resamples, seed=seed),
        }
    groups = {}
    for name, entry in report["systems"].items():
        groups.setdefault(name.split("@")[0], []).append(entry)
    for label, entries in groups.items():
        f1s = [e["macro_f1"]["estimate"] for e in entries]
        accs = [e["accuracy"] for e in entries]
        report["pooled"][label] = {
            "runs": len(entries),
            "macro_f1_mean": statistics.fmean(f1s),
            "macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else None,
            "accuracy_mean": statistics.fmean(accs),
            "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else None,
        }
    return report


def _pairs(values, flag, parser):
    out = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            parser.error(f"{flag} expects NAME=PATH, got {value!r}")
        if name in out:
            parser.error(f"duplicate system name {name!r}")
        out[name] = path
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--system", action="append", metavar="NAME=CHECKPOINT")
    parser.add_argument("--predictions", action="append", metavar="NAME=FILE")
    parser.add_argument("--baseline")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    files = _pairs(args.predictions, "--predictions", parser)
    checkpoints = _pairs(args.system, "--system", parser)
    if set(files) & set(checkpoints):
        parser.error("a name cannot be both a checkpoint and a prediction file")
    if not files and not checkpoints:
        parser.error("give at least one --system or --predictions")
    try:
        gold = load_gold(args.gold, require=("act", "message"))
        systems = {name: predictions_from_file(gold, path) for name, path in files.items()}
        for name, path in checkpoints.items():
            systems[name] = predictions_from_checkpoint(gold, path, model_name=args.model_name,
                                                        device=args.device)
        report = evaluate(gold, systems, baseline=args.baseline,
                          n_resamples=args.resamples, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
