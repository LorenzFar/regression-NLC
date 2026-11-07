"""Separate outcome, candidate and comparable-reference metrics."""
from collections import Counter, defaultdict
from math import sqrt
from fractions import Fraction
import re

from price_reference.types import money
from .data import offer_ref


def _rate(correct, total):
    return correct / total if total else None


def wilson_interval(correct, total):
    """Descriptive 95% case-level binomial interval, not a dialogue-cluster interval."""
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def expected_status(row):
    if row.get("status"):
        return row["status"]
    if row.get("needs_clarification"):
        challenges = row.get("challenge", [])
        challenges = [challenges] if isinstance(challenges, str) else challenges
        return "ambiguous" if set(challenges) & {
            "conditional", "alternatives", "multiple_prices", "price_revision_with_condition"} else "unresolved"
    return "resolved" if row.get("price") is not None else "no_price"


def canonical_id(value):
    match = re.fullmatch(r"(buyer|seller)-(?:offer-)?(\d+)", value or "")
    return (match[1], int(match[2])) if match else None


def reference_match(row, predicted):
    """Return None when identity is not comparable, not an automatic success."""
    target_id = row.get("referenced_offer")
    target = row.get("reference")
    if target is None and target_id:
        matches = [o for o in row.get("offers", []) if offer_ref(o).offer_id == target_id]
        target = matches[0] if len(matches) == 1 else None
    if target is None:
        if not target_id:
            return None
        if not row.get("identity_numbering_confirmed"):
            return None
        return predicted is not None and canonical_id(target_id) is not None and canonical_id(target_id) == canonical_id(predicted.offer_id)
    ref = offer_ref(target)
    if predicted is None:
        return False
    if ref.turn_index is not None and predicted.turn_index is not None:
        return (ref.owner, ref.turn_index) == (predicted.owner, predicted.turn_index)
    # Authoritative state IDs can be compared verbatim without assuming numbering.
    if ref.id_source == predicted.id_source == "state":
        return (ref.owner, ref.offer_id) == (predicted.owner, predicted.offer_id)
    if row.get("identity_numbering_confirmed"):
        return canonical_id(ref.offer_id) is not None and canonical_id(ref.offer_id) == canonical_id(predicted.offer_id)
    return None


def case_result(row, result):
    expected = expected_status(row)
    clarify = result.status in {"unresolved", "ambiguous", "conflict"}
    gold_clarify = bool(row.get("needs_clarification", expected in {"unresolved", "ambiguous", "conflict"}))
    identity = reference_match(row, result.offer) if row.get("referenced_offer") or row.get("reference") else None
    price_target = row.get("price")
    price_ok = result.price == money(price_target) if price_target is not None and not gold_clarify else None
    candidates = row.get("candidates", [])
    targets = {money(c["price"] if isinstance(c, dict) else c) for c in candidates}
    if gold_clarify and price_target is not None:
        targets.add(money(price_target))
    found = {c.price for c in result.candidates}
    return {
        "case_id": row["case_id"], "label_source": row["label_source"], "status_correct": result.status == expected,
        "price_correct": price_ok, "identity_correct": identity,
        "identity_comparable": identity is not None,
        "identity_required": bool(row.get("referenced_offer") or row.get("reference")),
        "clarify": clarify, "gold_clarify": gold_clarify,
        "candidate_found": len(found & targets), "candidate_total": len(targets),
        "liveness_correct": (result.offer_liveness == row["offer_liveness"])
        if row.get("offer_liveness") else None,
        "act_disagreement": result.act_disagreement,
        "correct": result.status == expected and (price_ok is not False)
        and (identity is not False) and clarify == gold_clarify
        and (not targets or targets <= found),
    }


def summarize(cases):
    n = len(cases)
    prices = [r["price_correct"] for r in cases if r["price_correct"] is not None]
    refs = [r["identity_correct"] for r in cases if r["identity_correct"] is not None]
    live = [r["liveness_correct"] for r in cases if r["liveness_correct"] is not None]
    tp = sum(r["clarify"] and r["gold_clarify"] for r in cases)
    counts = {
        "status_accuracy": (sum(r["status_correct"] for r in cases), n),
        "price_exact_match": (sum(prices), len(prices)),
        "reference_accuracy": (sum(refs), len(refs)),
        "liveness_accuracy": (sum(live), len(live)),
        "clarify_precision": (tp, sum(r["clarify"] for r in cases)),
        "clarify_recall": (tp, sum(r["gold_clarify"] for r in cases)),
        "candidate_recall": (sum(r["candidate_found"] for r in cases), sum(r["candidate_total"] for r in cases)),
        "act_disagreement_rate": (sum(r["act_disagreement"] for r in cases), n),
    }
    return {"n": n, "status_accuracy": _rate(sum(r["status_correct"] for r in cases), n),
            "price_exact_match": _rate(sum(prices), len(prices)), "price_n": len(prices),
            "reference_accuracy": _rate(sum(refs), len(refs)), "reference_n": len(refs),
            "identity_not_comparable": sum(r["identity_required"] and not r["identity_comparable"] for r in cases),
            "liveness_accuracy": _rate(sum(live), len(live)), "liveness_n": len(live),
            "clarify_precision": _rate(tp, sum(r["clarify"] for r in cases)),
            "clarify_recall": _rate(tp, sum(r["gold_clarify"] for r in cases)),
            "candidate_recall": _rate(sum(r["candidate_found"] for r in cases), sum(r["candidate_total"] for r in cases)),
            "act_disagreement_rate": _rate(sum(r["act_disagreement"] for r in cases), n),
            "rate_counts": {name: {"numerator": correct, "denominator": total,
                                   "interval_95": wilson_interval(correct, total)}
                            for name, (correct, total) in counts.items()},
            "interval_method": "Wilson 95%; descriptive case/mention-level, not dialogue-cluster adjusted"}


def score(rows, results):
    cases = [case_result(row, result) for row, result in zip(rows, results, strict=True)]
    groups = defaultdict(list)
    for row, case in zip(rows, cases):
        for name, value in (("label_source", row["label_source"]), ("subset", row.get("subset", "unspecified")),
                            ("inferred", str(row.get("inferred", "inferred" in row.get("gold_note", "").lower()))),
                            ("conditions", "alternatives" if expected_status(row) == "ambiguous" else
                             "single_conditional" if row.get("conditions") else "unconditional"),
                            ("zero_price", str(row.get("price") is not None and money(row["price"]) == "0.00"))):
            groups[(name, value)].append(case)
    slices = {}
    for (name, value), members in groups.items():
        slices.setdefault(name, {})[value] = summarize(members)
    return {"overall": summarize(cases), "slices": slices, "cases": cases}


def paired(left, right):
    right = {r["case_id"]: r for r in right}
    counts = Counter()
    for row in left:
        if row["case_id"] in right:
            other = right[row["case_id"]]
            key = ("right" if row["correct"] else "wrong") + "->" + ("right" if other["correct"] else "wrong")
            counts[key] += 1
    return {key: counts[key] for key in ("right->right", "right->wrong", "wrong->right", "wrong->wrong")}


def exact_mcnemar(left, right, endpoint="correct"):
    """Two-sided exact binomial McNemar on paired, nonmissing boolean outcomes.

    Uses integer arithmetic for the tail; avoids scipy and floating-point underflow
    during summation. Cases are the unit, not independent dialogue clusters.
    """
    right_by_id = {case["case_id"]: case for case in right}
    n = lost = gained = 0
    for case in left:
        other = right_by_id.get(case["case_id"])
        if other is None or case.get(endpoint) is None or other.get(endpoint) is None:
            continue
        n += 1
        lost += bool(case[endpoint] and not other[endpoint])
        gained += bool(not case[endpoint] and other[endpoint])
    discordant = lost + gained
    probability = None
    if n:
        if not discordant:
            probability = 1.0
        else:
            term = total = 1
            for k in range(1, min(lost, gained) + 1):
                term = term * (discordant - k + 1) // k
                total += term
            probability = min(1.0, float(Fraction(2 * total, 1 << discordant)))
    return {"endpoint": endpoint, "n": n, "right_to_wrong": lost, "wrong_to_right": gained,
            "discordant_n": discordant, "p_value": probability,
            "status": "no_paired_cases" if not n else "no_discordant_cases" if not discordant else "computed",
            "method": "two-sided exact McNemar; case-level, not dialogue-cluster adjusted; no multiple-testing correction"}
