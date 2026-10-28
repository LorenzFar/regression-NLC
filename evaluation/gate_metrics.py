"""RQ2 metrics: does asking when unsure reduce wrong deal confirmations, and at what cost?

Inputs are replay records (``python -m retrieval_pipeline.replay``) joined by
``case_id`` to gold rows carrying ``needs_clarification`` and ``deal_agreed``.

Conventions, matching the proposal:
* a *confirmation* is a buyer action in ``confirm_actions`` (default ``confirm``);
* a turn on which the gate asks counts as no confirmation, i.e. we assume a
  clarification prevents the wrong confirmation. The seller's answer is not simulated.
"""
from __future__ import annotations

CONFIRM_ACTIONS = frozenset({"confirm"})


def _ratio(num: int, den: int):
    return num / den if den else None


def _join(records, gold, mode):
    usable, errors, missing = {}, 0, 0
    for record in records:
        if record["gate"]["mode"] != mode:
            continue
        if record.get("error") is not None:
            errors += 1
            continue
        row = gold.get(record["case_id"])
        if row is None:
            missing += 1
            continue
        usable[record["case_id"]] = (record, row)
    return usable, errors, missing


def _summarize(rows) -> dict:
    """rows: (asked, confirmed, needs_clarification, deal_agreed) per case."""
    asked = sum(a for a, *_ in rows)
    tp = sum(a and need for a, _, need, _ in rows)
    fp = asked - tp
    ambiguous = sum(need for _, _, need, _ in rows)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, ambiguous)
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))
    return {
        "cases": len(rows),
        "clarifications": asked,
        "incorrect_confirmations": sum(c and not agreed for _, c, _, agreed in rows),
        "correct_confirmations": sum(c and agreed for _, c, _, agreed in rows),
        "unnecessary_clarifications": fp,
        "ambiguous_cases": ambiguous,
        "gate_precision": precision, "gate_recall": recall, "gate_f1": f1,
    }


def gate_report(records, gold: dict, *, modes=("off", "reference_only", "full"),
                confirm_actions=CONFIRM_ACTIONS) -> dict:
    """Metrics for each recorded gate mode, with extra turns measured against ``off``."""
    report = {}
    for mode in modes:
        usable, errors, missing = _join(records, gold, mode)
        if not usable and not errors:
            continue
        rows = []
        for record, row in usable.values():
            action = record["policy"]["action"]
            asked = action == "clarify"
            rows.append((asked, (not asked) and action in confirm_actions,
                         row["needs_clarification"], row["deal_agreed"]))
        report[mode] = {**_summarize(rows), "failed_cases": errors, "cases_without_gold": missing}
    base = report.get("off")
    for mode, entry in report.items():
        if base is not None and mode != "off":
            entry["extra_clarification_turns"] = entry["clarifications"] - base["clarifications"]
            entry["incorrect_confirmations_prevented"] = (
                base["incorrect_confirmations"] - entry["incorrect_confirmations"])
    return report


def sweep_threshold(records, gold: dict, thresholds, *, confirm_actions=CONFIRM_ACTIONS) -> dict:
    """Trade-off curve over the confidence threshold tau, from gate-``off`` records.

    Re-derives the gate from each case's recorded probabilities and price/reference
    flags, and takes the recorded ``off`` policy action whenever the gate stays quiet.
    This assumes the policy only differs from ``off`` by emitting ``clarify``, and
    that low confidence means top probability < tau (the proposal's definition).
    Also reports the expected incorrect confirmations of a *random* gate that asks
    equally often; the real gate should beat it, otherwise confidence is not informative.
    """
    for tau in thresholds:
        if type(tau) not in (int, float) or not 0 <= tau <= 1:
            raise ValueError("thresholds must be numbers between zero and one")
    usable, errors, missing = _join(records, gold, "off")
    if not usable:
        raise ValueError("sweep needs successful gate-off records that match the gold set")

    def run(tau, *, use_state, use_confidence):
        rows = []
        for record, row in usable.values():
            flags = record.get("price_reference") or {}
            base_action = record["policy"]["action"]
            asked = (base_action == "clarify"
                     or flags.get("reference_resolved") is False
                     or (use_state and flags.get("state_conflict") is True)
                     or (use_confidence and max(record["probabilities"].values()) < tau))
            rows.append((asked, (not asked) and base_action in confirm_actions,
                         row["needs_clarification"], row["deal_agreed"]))
        return _summarize(rows)

    # "off" is the no-gate counterfactual: it ignores even the reference flag.
    plain = []
    for record, row in usable.values():
        action = record["policy"]["action"]
        plain.append((action == "clarify", action in confirm_actions, row["needs_clarification"],
                      row["deal_agreed"]))
    off = {**_summarize(plain), "gate": "off"}
    n, base_incorrect = off["cases"], off["incorrect_confirmations"]

    def decorate(entry):
        ask_rate = entry["clarifications"] / n
        entry["extra_clarification_turns"] = entry["clarifications"] - off["clarifications"]
        entry["random_gate_expected_incorrect"] = base_incorrect * (1 - ask_rate)
        return entry

    rows = [off, decorate({**run(0.0, use_state=False, use_confidence=False),
                           "gate": "reference_only"})]
    for tau in sorted(set(thresholds)):
        rows.append(decorate({**run(tau, use_state=True, use_confidence=True),
                              "gate": "full", "tau": tau}))
    return {"cases": n, "failed_cases": errors, "cases_without_gold": missing, "rows": rows}
