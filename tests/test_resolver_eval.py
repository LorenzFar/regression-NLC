from dataclasses import asdict
import hashlib
import json

import pytest

from evaluation.resolver.data import (annotated_mention, gold_ledger, load_gold,
                                      mentions_from_gold, verify_freeze)
from evaluation.resolver.metrics import canonical_id, reference_match
from evaluation.resolver.run import evaluate, main
from price_reference import OfferRef
from scripts.resolver_gold_seed import seed


def row(**changes):
    result = {"case_id": "one", "history": [], "message": "I can do 50", "act": "counter",
              "price": "50", "status": "resolved", "needs_clarification": False,
              "gold_price": "50.00", "offers": [], "label_source": "ai_proposed", "subset": "hard"}
    result.update(changes)
    return result


def test_gold_mentions_never_invent_spans_and_exact_occurrence_is_preserved():
    mentions = mentions_from_gold("2O00", "2000.00", inferred=True)
    assert mentions[0].start is None and mentions[0].provenance == "gold_inferred"
    text = "50, then 50"
    mention = annotated_mention(text, {"value": "50.00", "kind": "absolute", "surface": "50", "occurrence_index": 1})
    assert (mention.start, mention.end) == (9, 11)
    assert text[mention.start:mention.end] == mention.surface
    with pytest.raises(ValueError, match="occurrence"):
        annotated_mention(text, {"value": "50.00", "surface": "50", "occurrence_index": 3})
    with pytest.raises(ValueError, match="match"):
        annotated_mention(text, {"value": "50.00", "surface": "60", "start": 0, "end": 2})
    assert mentions_from_gold(text, "none") == ()


def test_gold_loader_requires_actual_review_and_allows_team_extensions(tmp_path):
    path = tmp_path / "gold.jsonl"
    record = row(label_source="human_reviewed", optional_team_extension={"hello": "world"})
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer"):
        load_gold(path)
    record["reviewer"] = "Test reviewer (authored fixture)"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_gold(path)[0]["message"] == record["message"]
    with pytest.raises(ValueError, match="offers"):
        gold_ledger({})


def test_conditions_a_to_d_and_no_gold_act_substitution():
    # Deliberately wrong recorded NLU isolates NLU error from mention/ledger error.
    record = row(message="50", predicted_act="reject")
    report = evaluate([record])
    assert report["conditions"]["A"]["overall"]["price_exact_match"] == 1
    assert report["conditions"]["B"]["overall"]["price_exact_match"] == 0
    assert report["conditions"]["C"]["overall"]["price_exact_match"] == 0
    assert report["conditions"]["D"]["overall"]["price_exact_match"] == 1
    assert report["paired"]["A:B"]["right->wrong"] == 1
    assert report["headline"]["A"]["n"] == 0
    without_nlu = evaluate([row()])
    assert without_nlu["conditions"]["B"]["overall"]["n"] == 0
    assert without_nlu["skipped"]["B"][0]["reason"] == "missing_predicted_act"


def test_ledger_diagnostic_really_uses_different_ledgers():
    record = row(message="ok deal", act="accept", gold_price="none", mentions=[],
                 referenced_offer="buyer-1", history=[], offers=[{
                     "offer_id": "buyer-1", "owner": "buyer", "price": "50.00", "liveness": "live"}])
    report = evaluate([record], conditions=["A", "D"])
    assert report["conditions"]["A"]["predictions"][0]["price"] == "50.00"
    assert report["conditions"]["D"]["predictions"][0]["status"] == "unresolved"
    assert report["conditions"]["A"]["overall"]["reference_accuracy"] == 1


def test_blank_none_uncertain_and_pending_review_are_distinct():
    records = [row(case_id="blank", gold_price=""),
               row(case_id="none", gold_price="none", message="hello", act="vague", price=None, status="no_price"),
               row(case_id="unsure", gold_price="", gold_note="uncertain: damaged amount"),
               row(case_id="pending", review_required=True)]
    report = evaluate(records, conditions=["A"])
    assert report["excluded"] == {"blank_unconfirmed": 1, "uncertain": 1, "pending_review": 1, "silver": 0}
    assert report["conditions"]["A"]["overall"]["n"] == 1
    assert report["conditions"]["A"]["overall"]["status_accuracy"] == 1


def test_identity_comparison_has_coverage_and_does_not_assume_numbering():
    assert canonical_id("buyer-1") == canonical_id("buyer-offer-1")
    record = row(referenced_offer="buyer-1")
    minted = OfferRef("buyer-offer-1", "buyer", 2)
    assert reference_match(record, minted) is None
    assert reference_match({**record, "identity_numbering_confirmed": True}, minted)
    target = {"offer_id": "buyer-99", "owner": "buyer", "turn_index": 2}
    assert reference_match({**record, "reference": target}, minted)
    state = OfferRef("buyer-9", "buyer", None, "state")
    assert reference_match({**record, "reference": asdict(state)}, state)


def test_ambiguous_candidate_metric_does_not_require_an_unsafe_pick():
    record = row(message="50 if pickup, otherwise 55", status="ambiguous", needs_clarification=True,
                 candidates=["50", "55"], mentions=[
                     {"value": "50.00", "surface": "50", "start": 0, "end": 2},
                     {"value": "55.00", "surface": "55", "start": 28, "end": 30}])
    # Locate the second surface explicitly instead of assuming the string length.
    record["mentions"][1].update(start=record["message"].rfind("55"), end=len(record["message"]))
    report = evaluate([record], conditions=["A"])
    assert report["conditions"]["A"]["overall"]["candidate_recall"] == 1
    assert report["conditions"]["A"]["predictions"][0]["price"] is None


def test_seeding_is_silver_preserves_input_and_does_not_claim_hard_targets():
    records = [dict(dialogue="1", turn="0", text="I can do 50", role="seller", listing_price="60", gold_price="50.00", gold_note=""),
               dict(dialogue="1", turn="1", text="50 or 55", role="seller", listing_price="60", gold_price="50.00 | 55.00", gold_note="")]
    before = json.dumps(records)
    output, counts = seed(records, "validation")
    assert json.dumps(records) == before
    assert output[0]["label_source"] == "seeded_rule"
    assert not output[0]["review_required"] and output[1]["review_required"]
    assert output[1]["price"] is None
    assert output[1]["history"][0]["text"] == records[0]["text"]
    assert counts["proposed"] == 2


def test_freeze_manifest_hash_dialogues_and_holdout_guard(tmp_path):
    path = tmp_path / "gold.jsonl"
    record = row(split="holdout", dialogue_id="new-dialogue")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--gold", str(path), "--split", "holdout"])
    manifest = {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dialogue_ids": ["new-dialogue"], "resolver_commit": "authored-test-revision",
                "frozen_at": "2026-11-03", "exposure": "authored test fixture"}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_freeze(path, freeze, [record]) == manifest
    main(["--gold", str(path), "--split", "holdout", "--freeze-manifest", str(freeze), "--conditions", "A"])
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        verify_freeze(path, freeze, [record])


def test_team_shaped_authored_examples_all_conditions():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "evaluation" / "resolver" / "examples.jsonl"
    records = load_gold(path)
    report = evaluate(records)
    for condition in ("A", "B", "C", "D"):
        metrics = report["conditions"][condition]["overall"]
        assert metrics["n"] == 4 and metrics["status_accuracy"] == 1
        assert metrics["price_exact_match"] == metrics["reference_accuracy"] == 1
        assert metrics["candidate_recall"] == 1
    assert all(result["n"] == 0 for result in report["headline"].values())
