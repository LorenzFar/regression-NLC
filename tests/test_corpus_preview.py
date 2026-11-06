"""Real provisional corpus plumbing, not held-out response-quality evaluation."""
import copy
import json
import re
from pathlib import Path

import pytest

from retrieval_pipeline import ContractError, PolicyDecision, RetrievalGenerator
from retrieval_pipeline.bank import approve_candidate, build_bank, load_bank, prepare_candidate, retrieval_context
from retrieval_pipeline.contracts import Turn

ROOT = Path(__file__).resolve().parents[1]
SHORTLIST = ROOT / "examples/craigslist_review_shortlist.jsonl"
BANK = ROOT / "examples/provisional_craigslist_bank.jsonl"
TEMPLATES = [
    "how about you give it to me for {price}", "i would like to purchase for {price}",
    "Would you take {price}?", "Would you agree to sell for {price}?",
    "I can go to {price}", "would you accept {price}", "Can you do {price}",
]


def shortlist():
    return [json.loads(line) for line in SHORTLIST.read_text(encoding="utf-8").splitlines()]


def test_shortlist_remains_unapproved_even_with_provisional_opt_in(tmp_path):
    rows = shortlist()
    assert len(rows) == 12
    ids = {row["dialogue_id"] for row in rows}
    for row in rows:
        assert row["origin"] == "corpus"
        assert row["review_status"] == "agent_shortlisted_needs_human_review"
        assert row["approved"] is False and row["reviewer"] is None
        assert row == prepare_candidate(row, ids)
    for allow in (False, True):
        destination = tmp_path / "bank.jsonl"
        with pytest.raises(ContractError, match="explicit approval required"):
            build_bank(rows, destination, ids, allow_provisional=allow)
        assert not destination.exists()
    with pytest.raises(ContractError, match="unsupported bank schema"):
        RetrievalGenerator(SHORTLIST, allow_provisional=True)


def test_seven_real_sources_preserved_with_honest_local_trial_review():
    rows = load_bank(BANK, allow_provisional=True)
    assert [row["template"] for row in rows] == TEMPLATES
    assert [row["action"] for row in rows] == ["propose"] * 2 + ["counter"] * 5
    original = {row["template_id"]: row for row in shortlist()}
    for row in rows:
        source = original[row["template_id"]]
        for key in ("dialogue_id", "turn_id", "source_event_index", "seller_event_index",
                    "origin", "split", "speaker", "action", "history", "message", "response",
                    "price_spans", "template", "context", "history_window"):
            assert row[key] == source[key]
        assert row["origin"] == "corpus"
        assert row["approved"] is False and row["human_approval"] is False
        assert row["reviewer"] == "agent:Hermes"
        assert row["review_status"] == "agent_reviewed_local_trial"
        assert row["split_status"] == "provisional_official_train"
        for key in ("source_queue_sha256", "raw_train_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", row["agent_review"][key])
            assert row["agent_review"][key] == source["agent_review"][key]
        assert row["context"] == retrieval_context([Turn(**t) for t in row["history"]], row["message"])


def test_strict_default_rejects_bank_and_builder_and_human_approval_masquerade(tmp_path):
    rows = load_bank(BANK, allow_provisional=True)
    ids = {row["dialogue_id"] for row in rows}
    with pytest.raises(ContractError, match="provisional"):
        load_bank(BANK)
    with pytest.raises(ContractError, match="provisional"):
        RetrievalGenerator(BANK)
    with pytest.raises(ContractError, match="provisional"):
        load_bank(BANK, allow_provisional="yes")
    path = tmp_path / "bank.jsonl"
    with pytest.raises(ContractError, match="provisional"):
        build_bank(rows, path, ids)
    assert not path.exists()
    with pytest.raises(ContractError, match="provisional"):
        approve_candidate(rows[0], "Human name cannot erase provisional review")
    candidate = prepare_candidate(rows[0], ids)
    with pytest.raises(ContractError, match="provisional"):
        approve_candidate(candidate, "agent:Hermes")
    build_bank(rows, path, ids, allow_provisional=True)
    assert load_bank(path, allow_provisional=True) == rows


@pytest.mark.parametrize("changes", [
    {"approved": True}, {"human_approval": True}, {"human_approval": None},
    {"reviewer": "Human name"}, {"origin": "authored_fixture"},
    {"review_status": "agent_shortlisted_needs_human_review"},
    {"split_status": "team_allowlist_intersection"}, {"context": "leaked response"},
    {"template": "I accept {price}"}, {"split": "test"},
])
def test_opt_in_still_rejects_inconsistent_or_misleading_artifacts(tmp_path, changes):
    records = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()]
    records[1].update(changes)
    path = tmp_path / "tampered.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    with pytest.raises(ContractError):
        load_bank(path, allow_provisional=True)


def test_response_text_does_not_enter_ranking(tmp_path):
    rows = copy.deepcopy(load_bank(BANK, allow_provisional=True))
    row = rows[0]
    original_context = row["context"]
    response = "zebracomet 8000"
    start = response.index("8000")
    row.update(response=response, price_spans=[dict(start=start, end=len(response), value="8000", validated=True)])
    ids = {r["dialogue_id"] for r in rows}
    prepared = prepare_candidate(row, ids)
    assert prepared["context"] == original_context
    row.update(template=prepared["template"])
    path = tmp_path / "response-sentinel.jsonl"
    build_bank(rows, path, ids, allow_provisional=True)
    generator = RetrievalGenerator(path, allow_provisional=True)
    assert "zebracomet" not in generator.vectorizer.vocabulary_
    result = generator.generate(PolicyDecision("propose", "73.27"), [], "zebracomet")
    assert result.source == "fallback" and result.fallback_reason == "zero_similarity"


def test_preview_real_corpus_hits_exact_replacement_and_fixed_contracts():
    from examples.corpus_contract import preview

    records = preview()
    assert len(records) == 15
    assert len({record["case"] for record in records}) == len(records)
    proofs = [r for r in records if r["purpose"] == "source_context_plumbing_not_held_out"]
    assert len(proofs) == 7
    by_id = {r["template_id"]: r for r in load_bank(BANK, allow_provisional=True)}
    for record in proofs:
        assert record["source"] == "retrieval" and record["origin"] == "corpus"
        assert record["template_id"] == record["expected_template_id"]
        assert record["similarity"] == pytest.approx(1.0)
        row = by_id[record["template_id"]]
        assert record["source_dialogue_id"] == row["dialogue_id"]
        assert record["source_turn_id"] == row["turn_id"]
        assert record["original_response"] == row["response"]
        assert record["text"] == row["template"].format(price="73.27")
        assert row["price_spans"][0]["value"] not in record["text"]
        assert re.findall(r"[0-9]+(?:\.[0-9]+)?", record["text"]) == ["73.27"]
    by_case = {r["case"]: r for r in records}
    for action, text in (("propose", "I can offer 73.27."), ("counter", "Would you take 73.27 instead?")):
        result = by_case["zero_match_" + action]
        assert result["source"] == "fallback" and result["fallback_reason"] == "zero_similarity"
        assert result["text"] == text
    assert by_case["fixed_accept"]["text"] == "I accept your offer of 73.27."
    assert by_case["fixed_confirm"]["text"] == "Thank you for accepting my offer of 73.27."
    for case in ("fixed_accept", "fixed_confirm"):
        assert by_case[case]["fallback_reason"] == "no_compatible_candidate"
        assert by_case[case]["origin"] is None
    assert by_case["fixed_clarify"]["source"] == "clarification"
    assert by_case["fixed_clarify"]["text"] == "Which offer are you referring to?"
    exploratory = [r for r in records if r["purpose"] == "exploratory_not_quality_eval"]
    assert len(exploratory) == 3
    for record in exploratory:
        assert record["source"] == "retrieval" and record["origin"] == "corpus"
        assert record["text"] == by_id[record["template_id"]]["template"].format(price="73.27")
    assert all(r["demo"] == "provisional_corpus_local_trial" and r["human_approval"] is False for r in records)

