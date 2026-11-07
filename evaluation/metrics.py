"""Dependency-free classification, calibration, agreement and bootstrap metrics."""
import random
from collections import Counter


def confusion_matrix(gold, pred, labels):
    """matrix[i][j] counts gold label i predicted as label j."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        matrix[index[g]][index[p]] += 1
    return matrix


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def per_class(gold, pred, labels):
    """Precision, recall, F1 and support per label (0.0 when undefined)."""
    scores = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        scores[label] = {"precision": precision, "recall": recall,
                         "f1": _safe_div(2 * precision * recall, precision + recall),
                         "support": tp + fn, "predicted": tp + fp}
    return scores


def accuracy(gold, pred):
    return _safe_div(sum(g == p for g, p in zip(gold, pred)), len(gold))


def macro_f1(gold, pred, labels):
    """Mean F1 over labels that occur in gold or predictions (as in scikit-learn)."""
    present = [label for label in labels if label in set(gold) | set(pred)]
    scores = per_class(gold, pred, present)
    return _safe_div(sum(scores[label]["f1"] for label in present), len(present))


def cohen_kappa(first, second):
    """Chance-corrected agreement between two annotators' labels."""
    n = len(first)
    if n == 0:
        return 0.0
    observed = accuracy(first, second)
    a, b = Counter(first), Counter(second)
    expected = sum(a[label] * b[label] for label in a) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def reliability(confidences, correct, bins=10):
    """Equal-width confidence bins and expected calibration error (ECE)."""
    n = len(confidences)
    rows, ece = [], 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not members:
            continue
        conf = sum(confidences[i] for i in members) / len(members)
        acc = sum(correct[i] for i in members) / len(members)
        ece += len(members) / n * abs(acc - conf)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(members), "confidence": conf, "accuracy": acc})
    return {"ece": ece, "bins": rows}


def coverage_curve(confidences, correct, thresholds):
    """Share of messages kept at each gate threshold and accuracy on the kept ones."""
    rows = []
    for threshold in thresholds:
        kept = [ok for c, ok in zip(confidences, correct) if c >= threshold]
        rows.append({"threshold": threshold, "coverage": _safe_div(len(kept), len(correct)),
                     "kept": len(kept), "accuracy_on_kept": _safe_div(sum(kept), len(kept))})
    return rows


def bootstrap_ci(groups, statistic, n_resamples=1000, seed=0, level=0.95):
    """Percentile interval, resampling whole groups (dialogues) with replacement.

    groups: list of lists of items; statistic: function(list of items) -> float.
    """
    rng = random.Random(seed)
    values = []
    for _ in range(n_resamples):
        sample = [item for _ in groups for item in rng.choice(groups)]
        values.append(statistic(sample))
    values.sort()
    tail = (1 - level) / 2
    low = values[int(tail * (n_resamples - 1))]
    high = values[int((1 - tail) * (n_resamples - 1))]
    return {"low": low, "high": high, "level": level, "resamples": n_resamples}
