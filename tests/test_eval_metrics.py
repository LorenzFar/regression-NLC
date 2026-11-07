"""Metric correctness: cross-checked against scikit-learn and hand-computed values."""
import math
import random

import pytest
from sklearn import metrics as skm

from evaluation import nlu_metrics as m

ACTS = list(m.ACTS)


def _random_labels(n, seed):
    rng = random.Random(seed)
    truth = [rng.choice(ACTS) for _ in range(n)]
    pred = [t if rng.random() < 0.6 else rng.choice(ACTS) for t in truth]
    return truth, pred


def test_acts_match_the_checkpoint_label_order():
    torch = pytest.importorskip("torch")  # nlu.py imports torch at module level
    pytest.importorskip("transformers")
    import nlu
    assert list(m.ACTS) == nlu.ACTS


def test_f1_precision_recall_and_confusion_match_sklearn():
    truth, pred = _random_labels(400, seed=1)
    assert m.accuracy(truth, pred) == pytest.approx(skm.accuracy_score(truth, pred))
    assert m.macro_f1(truth, pred, ACTS) == pytest.approx(
        skm.f1_score(truth, pred, labels=ACTS, average="macro"))
    assert m.confusion_matrix(truth, pred, ACTS) == skm.confusion_matrix(truth, pred, labels=ACTS).tolist()
    p, r, f, s = skm.precision_recall_fscore_support(truth, pred, labels=ACTS, zero_division=0)
    stats = m.per_class_prf(truth, pred, ACTS)
    for i, act in enumerate(ACTS):
        assert stats[act]["precision"] == pytest.approx(p[i])
        assert stats[act]["recall"] == pytest.approx(r[i])
        assert stats[act]["f1"] == pytest.approx(f[i])
        assert stats[act]["support"] == s[i]


def test_macro_f1_ignores_acts_absent_from_both_gold_and_predictions():
    truth = ["accept", "reject", "accept"]
    pred = ["accept", "reject", "reject"]
    assert m.macro_f1(truth, pred) == pytest.approx(
        skm.f1_score(truth, pred, labels=["accept", "reject"], average="macro"))


def test_unknown_labels_and_length_mismatch_are_errors():
    with pytest.raises(ValueError):
        m.confusion_matrix(["accept"], ["maybe"])
    with pytest.raises(ValueError):
        m.accuracy(["accept"], [])
    with pytest.raises(ValueError):
        m.accuracy([], [])


def test_cohen_kappa_matches_sklearn_and_handles_degenerate_cases():
    a, b = _random_labels(300, seed=2)
    assert m.cohen_kappa(a, b) == pytest.approx(skm.cohen_kappa_score(a, b))
    assert m.cohen_kappa(["x", "x"], ["x", "x"]) == 1.0
    assert m.cohen_kappa(["x", "y"], ["y", "x"]) == pytest.approx(-1.0)


def _probs(top, p):
    rest = (1 - p) / (len(ACTS) - 1)
    return {a: (p if a == top else rest) for a in ACTS}


def test_ece_is_zero_when_confidence_equals_accuracy_and_large_when_overconfident():
    # 10 predictions at confidence 0.8, exactly 8 correct -> perfectly calibrated.
    truth = ["accept"] * 10
    probs = [_probs("accept" if i < 8 else "reject", 0.8) for i in range(10)]
    assert m.expected_calibration_error(probs, truth) == pytest.approx(0.0)
    overconfident = [_probs("reject", 0.99) for _ in range(10)]
    assert m.expected_calibration_error(overconfident, truth) == pytest.approx(0.99)


def test_confidence_of_exactly_one_lands_in_the_last_bin_and_bins_partition_the_data():
    probs = [_probs("accept", 1.0), _probs("accept", 0.1 + 1 / 6)]
    bins = m.calibration_bins(probs, ["accept", "accept"], n_bins=10)
    assert bins[-1]["count"] == 1
    assert sum(b["count"] for b in bins) == 2


def test_nll_and_brier_against_hand_values():
    truth = ["accept", "reject"]
    probs = [_probs("accept", 0.5), _probs("accept", 0.5)]
    rest = 0.5 / 5
    assert m.negative_log_likelihood(probs, truth) == pytest.approx(-(math.log(0.5) + math.log(rest)) / 2)
    first = (0.5 - 1) ** 2 + 5 * rest ** 2
    second = 0.5 ** 2 + (rest - 1) ** 2 + 4 * rest ** 2
    assert m.brier_score(probs, truth) == pytest.approx((first + second) / 2)
    assert m.negative_log_likelihood([{**_probs("accept", 1.0), "reject": 0.0}], ["reject"]) < 40


def test_risk_coverage_rewards_informative_confidence():
    truth = ["accept"] * 4
    informative = [_probs("accept", 0.9), _probs("accept", 0.8), _probs("reject", 0.6), _probs("reject", 0.5)]
    inverted = [_probs("reject", 0.9), _probs("reject", 0.8), _probs("accept", 0.6), _probs("accept", 0.5)]
    good, bad = m.risk_coverage(informative, truth), m.risk_coverage(inverted, truth)
    assert good["aurc"] < bad["aurc"]
    assert good["curve"][-1] == {"coverage": 1.0, "accuracy": 0.5}
    assert m.accuracy_at_coverage(good["curve"], 0.5) == 1.0
    assert m.accuracy_at_coverage(good["curve"], 1.0) == 0.5


def test_mcnemar_exact_known_values():
    truth = ["a"] * 10
    assert m.mcnemar_exact(truth, ["a"] * 10, ["b"] * 10) == {
        "a_only_correct": 10, "b_only_correct": 0, "p_value": pytest.approx(2 / 2 ** 10)}
    assert m.mcnemar_exact(truth, ["a"] * 5 + ["b"] * 5, ["b"] * 5 + ["a"] * 5)["p_value"] == 1.0
    assert m.mcnemar_exact(truth, ["a"] * 10, ["a"] * 10)["p_value"] == 1.0


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    truth, pred = _random_labels(200, seed=3)
    first = m.bootstrap_ci(m.macro_f1, truth, pred, n_resamples=300, seed=5)
    assert first == m.bootstrap_ci(m.macro_f1, truth, pred, n_resamples=300, seed=5)
    assert first["low"] <= first["estimate"] <= first["high"]
    assert first != m.bootstrap_ci(m.macro_f1, truth, pred, n_resamples=300, seed=6)


def test_paired_bootstrap_separates_a_clearly_better_system():
    truth, _ = _random_labels(300, seed=4)
    perfect = list(truth)
    worse = [ACTS[(ACTS.index(t) + 1) % len(ACTS)] if i % 3 == 0 else t for i, t in enumerate(truth)]
    result = m.paired_bootstrap_diff(m.macro_f1, truth, perfect, worse, n_resamples=300, seed=0)
    assert result["low"] > 0 and result["p_two_sided"] < 0.05
    same = m.paired_bootstrap_diff(m.macro_f1, truth, perfect, perfect, n_resamples=100, seed=0)
    assert same["estimate"] == 0 and same["low"] == 0 and same["high"] == 0
