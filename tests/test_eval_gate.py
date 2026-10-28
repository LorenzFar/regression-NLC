"""RQ2 metrics on hand-built replay records with known answers."""
import pytest

from evaluation.gate_metrics import gate_report, sweep_threshold


def record(case_id, mode, action, *, top=0.9, resolved=True, conflict=False, error=None):
    base = {"case_id": case_id, "gate": {"mode": mode, "threshold": 0.7, "reasons": []}, "error": error}
    if error:
        return base
    return {**base, "policy": {"action": action},
            "probabilities": {"accept": top, "reject": 1 - top},
            "price_reference": {"reference_resolved": resolved, "state_conflict": conflict}}


# case: (needs_clarification, deal_agreed)
GOLD = {
    "agreed":    {"needs_clarification": False, "deal_agreed": True},
    "ambiguous": {"needs_clarification": True, "deal_agreed": False},
    "unsure":    {"needs_clarification": True, "deal_agreed": False},
    "clear_no":  {"needs_clarification": False, "deal_agreed": False},
}

# Gate off: the buyer confirms three cases; two of those confirmations are wrong.
OFF = [record("agreed", "off", "confirm", top=0.95),
       record("ambiguous", "off", "confirm", top=0.9, resolved=False),
       record("unsure", "off", "confirm", top=0.55),
       record("clear_no", "off", "reject", top=0.9)]


def test_gate_off_counts_wrong_confirmations():
    report = gate_report(OFF, GOLD, modes=("off",))["off"]
    assert report["cases"] == 4
    assert report["incorrect_confirmations"] == 2 and report["correct_confirmations"] == 1
    assert report["clarifications"] == 0 and report["gate_precision"] is None
    assert report["gate_recall"] == 0.0


def test_full_gate_prevents_wrong_confirmations_and_reports_cost():
    full = [record("agreed", "full", "confirm"), record("ambiguous", "full", "clarify"),
            record("unsure", "full", "clarify"), record("clear_no", "full", "clarify")]
    report = gate_report(OFF + full, GOLD)
    entry = report["full"]
    assert entry["incorrect_confirmations"] == 0 and entry["correct_confirmations"] == 1
    assert entry["clarifications"] == 3 and entry["unnecessary_clarifications"] == 1
    assert entry["gate_precision"] == pytest.approx(2 / 3) and entry["gate_recall"] == 1.0
    assert entry["gate_f1"] == pytest.approx(0.8)
    assert entry["extra_clarification_turns"] == 3
    assert entry["incorrect_confirmations_prevented"] == 2


def test_failed_and_ungolded_cases_are_counted_not_silently_dropped():
    records = OFF + [record("boom", "off", None, error={"type": "ContractError"}),
                     record("no_gold", "off", "reject")]
    report = gate_report(records, GOLD, modes=("off",))["off"]
    assert report["cases"] == 4 and report["failed_cases"] == 1 and report["cases_without_gold"] == 1


def test_sweep_is_monotone_and_beats_a_random_gate():
    sweep = sweep_threshold(OFF, GOLD, [0.5, 0.7, 0.9, 1.0])
    rows = sweep["rows"]
    assert rows[0]["gate"] == "off" and rows[0]["incorrect_confirmations"] == 2
    # reference flag alone catches "ambiguous" but not the low-confidence "unsure".
    assert rows[1]["gate"] == "reference_only" and rows[1]["incorrect_confirmations"] == 1
    taus = [r for r in rows if r["gate"] == "full"]
    asks = [r["clarifications"] for r in taus]
    wrong = [r["incorrect_confirmations"] for r in taus]
    assert asks == sorted(asks) and wrong == sorted(wrong, reverse=True)
    by_tau = {r["tau"]: r for r in taus}
    assert by_tau[0.7]["incorrect_confirmations"] == 0 and by_tau[0.7]["clarifications"] == 2
    assert by_tau[0.7]["random_gate_expected_incorrect"] == pytest.approx(2 * (1 - 2 / 4))
    assert by_tau[0.7]["incorrect_confirmations"] < by_tau[0.7]["random_gate_expected_incorrect"]
    assert by_tau[1.0]["clarifications"] == 4 and by_tau[1.0]["unnecessary_clarifications"] == 2


def test_sweep_needs_off_records_and_valid_thresholds():
    with pytest.raises(ValueError, match="gate-off"):
        sweep_threshold([record("agreed", "full", "confirm")], GOLD, [0.5])
    with pytest.raises(ValueError, match="between zero and one"):
        sweep_threshold(OFF, GOLD, [1.5])
