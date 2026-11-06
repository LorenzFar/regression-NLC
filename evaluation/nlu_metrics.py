"""Classification, calibration, significance and annotation-agreement metrics.

Everything here is deterministic (bootstrap takes an explicit seed) and works on
plain lists, so it can be unit-tested without torch or a checkpoint.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Mapping, Sequence

ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")


def _check(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError("inputs must have the same length")
    if not a:
        raise ValueError("at least one example is required")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred, labels: Sequence[str] = ACTS) -> list[list[int]]:
    """Rows are gold labels, columns are predictions, in ``labels`` order."""
    _check(y_true, y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"unknown label in pair ({t!r}, {p!r})")
        matrix[index[t]][index[p]] += 1
    return matrix


def per_class_prf(y_true, y_pred, labels: Sequence[str] = ACTS) -> dict[str, dict]:
    """Precision/recall/F1/support per label; an empty denominator gives 0.0."""
    matrix = confusion_matrix(y_true, y_pred, labels)
    stats = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return stats


def macro_f1(y_true, y_pred, labels: Sequence[str] | None = None) -> float:
    """Unweighted mean F1. Default labels: those occurring in gold or predictions,
    so an act absent from a small gold set does not silently count as F1 = 0."""
    if labels is None:
        present = set(y_true) | set(y_pred)
        labels = [a for a in ACTS if a in present] + sorted(present - set(ACTS))
    stats = per_class_prf(y_true, y_pred, labels)
    return sum(s["f1"] for s in stats.values()) / len(stats)


# --- calibration ------------------------------------------------------------

def _top(probs: Mapping[str, float], labels: Sequence[str]) -> tuple[str, float]:
    best, best_p = labels[0], probs[labels[0]]
    for label in labels[1:]:
        if probs[label] > best_p:
            best, best_p = label, probs[label]
    return best, best_p


def _check_truth(y_true, labels):
    for t in y_true:
        if t not in labels:
            raise ValueError(f"unknown gold label {t!r}")


def calibration_bins(probs_list, y_true, n_bins: int = 10, labels: Sequence[str] = ACTS):
    """Equal-width reliability bins over top-label confidence; bin i is (i/n, (i+1)/n]."""
    _check(probs_list, y_true)
    _check_truth(y_true, labels)
    bins = [{"lower": i / n_bins, "upper": (i + 1) / n_bins, "count": 0,
             "confidence": 0.0, "accuracy": 0.0} for i in range(n_bins)]
    for probs, truth in zip(probs_list, y_true):
        pred, conf = _top(probs, labels)
        i = min(max(math.ceil(conf * n_bins) - 1, 0), n_bins - 1)
        bins[i]["count"] += 1
        bins[i]["confidence"] += conf
        bins[i]["accuracy"] += pred == truth
    for b in bins:
        if b["count"]:
            b["confidence"] /= b["count"]
            b["accuracy"] /= b["count"]
    return bins


def expected_calibration_error(probs_list, y_true, n_bins: int = 10,
                               labels: Sequence[str] = ACTS) -> float:
    bins = calibration_bins(probs_list, y_true, n_bins, labels)
    n = len(y_true)
    return sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"])


def negative_log_likelihood(probs_list, y_true, labels: Sequence[str] = ACTS,
                            floor: float = 1e-12) -> float:
    _check(probs_list, y_true)
    _check_truth(y_true, labels)
    return -sum(math.log(max(p[t], floor)) for p, t in zip(probs_list, y_true)) / len(y_true)


def brier_score(probs_list, y_true, labels: Sequence[str] = ACTS) -> float:
    _check(probs_list, y_true)
    _check_truth(y_true, labels)
    total = 0.0
    for p, t in zip(probs_list, y_true):
        total += sum((p[label] - (label == t)) ** 2 for label in labels)
    return total / len(y_true)


def risk_coverage(probs_list, y_true, labels: Sequence[str] = ACTS) -> dict:
    """Selective prediction: keep the most confident fraction, report accuracy there.

    ``aurc`` is the mean selective risk (1 - accuracy) over all coverage levels;
    lower is better, and a model whose confidence is uninformative sits at the
    overall error rate.
    """
    _check(probs_list, y_true)
    n = len(y_true)
    tops = [_top(p, labels) for p in probs_list]
    order = sorted(range(n), key=lambda i: -tops[i][1])  # stable on ties
    correct, risk_sum, curve = 0, 0.0, []
    for rank, i in enumerate(order, 1):
        correct += tops[i][0] == y_true[i]
        acc = correct / rank
        curve.append({"coverage": rank / n, "accuracy": acc})
        risk_sum += 1 - acc
    return {"curve": curve, "aurc": risk_sum / n}


def accuracy_at_coverage(curve, coverage: float) -> float:
    n = len(curve)
    k = min(max(math.ceil(coverage * n - 1e-9), 1), n)
    return curve[k - 1]["accuracy"]


# --- significance -------------------------------------------------------------

def bootstrap_ci(metric: Callable, y_true, y_pred, *, n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap over examples."""
    _check(y_true, y_pred)
    rng = random.Random(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(metric([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    stats.sort()
    return {"estimate": metric(y_true, y_pred),
            "low": stats[int(alpha / 2 * n_resamples)],
            "high": stats[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]}


def paired_bootstrap_diff(metric: Callable, y_true, pred_a, pred_b, *,
                          n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for metric(A) - metric(B) using the *same* resampled examples for both."""
    _check(y_true, pred_a)
    _check(y_true, pred_b)
    rng = random.Random(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        t = [y_true[i] for i in idx]
        diffs.append(metric(t, [pred_a[i] for i in idx]) - metric(t, [pred_b[i] for i in idx]))
    diffs.sort()
    le = sum(d <= 0 for d in diffs) / n_resamples
    ge = sum(d >= 0 for d in diffs) / n_resamples
    return {"estimate": metric(y_true, pred_a) - metric(y_true, pred_b),
            "low": diffs[int(alpha / 2 * n_resamples)],
            "high": diffs[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))],
            "p_two_sided": min(1.0, 2 * min(le, ge))}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact two-sided McNemar test on the items where exactly one system is right."""
    _check(y_true, pred_a)
    _check(y_true, pred_b)
    a_only = b_only = 0
    for t, pa, pb in zip(y_true, pred_a, pred_b):
        ca, cb = pa == t, pb == t
        a_only += ca and not cb
        b_only += cb and not ca
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": p}


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Chance-corrected agreement between two annotators (or parser vs human)."""
    _check(labels_a, labels_b)
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n)
                   for c in set(labels_a) | set(labels_b))
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
