"""Follow-up review: protect generator wording, authority boundaries and reporting."""
import json

import pytest

from evaluation.resolver.data import gold_ledger
from evaluation.resolver.metrics import exact_mcnemar
from evaluation.resolver.run import evaluate, main
from price_reference import CachingNLU, ReferenceObserver, build_ledger, resolve
from price_reference.evidence import collect_mentions
from retrieval_pipeline.contracts import PolicyDecision
from retrieval_pipeline.generator import RetrievalGenerator
from retrieval_pipeline.pipeline import Session


@pytest.mark.parametrize("reason", ["low_confidence", "unresolved_reference", "state_conflict", "conditional_price"])
@pytest.mark.parametrize("specific", [False, True])
def test_every_rendered_clarification_template_is_skipped_by_ledger(reason, specific):
    decision = PolicyDecision("clarify", "90.00" if specific else None,
                              "seller-1" if specific else None, reason)
    # Exercise the actual renderer, so a template wording change cannot hide behind
    # copied strings in this test. Teammate generator source is never modified.
    rendered = RetrievalGenerator().generate(decision, [], "No, 90")
    history = [("seller", "asking 100"), ("buyer", "Would you take 80?"),
               ("seller", "No, 90"), ("buyer", rendered.text)]
    ledger = build_ledger(history)
    buyer = [offer for offer in ledger.offers if offer.ref.owner == "buyer"]
    assert len(buyer) == 1 and buyer[0].price == "80.00"
    assert buyer[0].liveness == "unknown"
    assert not [turn for turn in ledger.ambiguous_turns if turn.ref.owner == "buyer"]


@pytest.mark.parametrize("text", ["Did you say 90? I can do 85", "Is that 90? I offer 85",
                                  "Could you clarify 90? Would you take 85?",
                                  "Are you referring to 90? Could you do 85?",
                                  "Could you clarify 90? I'll take it for 85"])
def test_quoted_inquiry_is_clause_scoped_and_preserves_later_offer(text):
    ledger = build_ledger([("seller", "asking 90"), ("buyer", text)])
    assert ledger.offers[-1].price == "85.00" and ledger.offers[-1].ref.owner == "buyer"
    assert not ledger.ambiguous_turns
    answer = resolve(text, speaker="buyer", act="counter", mentions=collect_mentions(text), ledger=build_ledger([]))
    assert answer.price == "85.00"


@pytest.mark.parametrize("text", ["Could you do 90?", "Would you take 90?", "Fine, I'll take it at 90"])
def test_same_price_offer_is_not_a_clarification(text):
    ledger = build_ledger([("seller", "asking 90"), ("buyer", text)])
    assert ledger.offers[-1].price == "90.00" and ledger.offers[-1].ref.owner == "buyer"


def test_observer_reports_live_tracker_fact_and_separate_advisory_disagreement():
    class NLU:
        def predict(self, history, message):
            return {"accept": 1.0}

    snapshot = Session("fact", history=[("buyer", "60"), ("buyer", "70")], state={
        "buyer_offer_id": "buyer-1", "buyer_price": "60", "agreement": "pending"}).snapshot()
    output = ReferenceObserver(CachingNLU(NLU())).observe(snapshot, "case", "ok deal")
    assert output["status"] == "resolved" and output["price"] == "60.00"
    assert output["offer_liveness"] == "live" and output["liveness_established"]
    assert output["state_history_disagreement"] and output["requires_validation"]
    assert not output["offer_actionable"]  # backward-compatibility hint remains conservative
    assert output["state_history_mismatch"] == output["state_history_disagreement"]


def test_gold_ledger_can_represent_live_state_despite_history_disagreement():
    ledger = gold_ledger({"history": [], "offers": [{
        "offer_id": "buyer-1", "owner": "buyer", "price": "60", "liveness": "live",
        "state_history_disagreement": True}]})
    answer = resolve("ok deal", act="accept", mentions=[], ledger=ledger)
    assert answer.liveness_established and answer.requires_validation


def test_exact_mcnemar_known_tail_symmetry_empty_and_missing_outcomes():
    left = [{"case_id": str(i), "correct": True} for i in range(6)]
    right = [{"case_id": str(i), "correct": False} for i in range(6)]
    test = exact_mcnemar(left, right)
    assert test["n"] == test["discordant_n"] == 6
    assert test["p_value"] == pytest.approx(0.03125)
    assert exact_mcnemar(right, left)["p_value"] == test["p_value"]
    assert exact_mcnemar(left, left)["p_value"] == 1.0
    assert exact_mcnemar([], right)["status"] == "no_paired_cases"
    missing = [{"case_id": "0", "price_correct": None}]
    assert exact_mcnemar(missing, right, "price_correct")["n"] == 0
    balanced = [{"case_id": str(i), "correct": i < 3} for i in range(6)]
    opposite = [{"case_id": str(i), "correct": i >= 3} for i in range(6)]
    assert exact_mcnemar(balanced, opposite)["p_value"] == 1.0


def test_empty_eligibility_and_small_headline_are_explicit_in_cli(tmp_path, capsys):
    row = {"case_id": "silver", "message": "I can do 50", "history": [], "act": "counter",
           "price": "50", "gold_price": "50.00", "offers": [], "label_source": "seeded_rule"}
    path = tmp_path / "silver.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    main(["--gold", str(path), "--conditions", "A"])
    output = json.loads(capsys.readouterr().out)
    assert output["eligibility"]["status"] == "no_eligible_cases"
    assert output["eligibility"]["message"]
    assert output["condition_evaluation_status"]["A"] == {"n": 0, "status": "no_evaluable_cases"}
    headline = output["headline_human_reviewed_hard"]["A"]
    assert headline["status"] == "insufficient_n" and headline["n"] == 0
    assert headline["metrics"] is None


def test_paired_tests_are_reported_by_source_and_endpoint():
    row = {"case_id": "one", "message": "50", "history": [], "act": "counter", "predicted_act": "reject",
           "price": "50", "gold_price": "50.00", "offers": [], "label_source": "ai_proposed"}
    report = evaluate([row])
    tests = report["paired_tests"]["A:B"]["by_label_source"]["ai_proposed"]
    assert tests["price_correct"]["right_to_wrong"] == 1
    assert tests["status_correct"]["n"] == 1
