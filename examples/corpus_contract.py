"""Offline provisional corpus retrieval: plumbing proofs, not quality evaluation.

Run: .venv/Scripts/python.exe -m examples.corpus_contract
No NLU, LLM, network, policy inference, or human approval is performed.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from retrieval_pipeline import PolicyDecision, RetrievalGenerator, Turn

BANK = Path(__file__).with_name("provisional_craigslist_bank.jsonl")


def preview():
    """Load the real corpus bank with explicit local-trial opt-in."""
    generator = RetrievalGenerator(BANK, allow_provisional=True)
    price = "73.27"  # Exact policy-authorized amount, never a source price.
    requests = [
        ("source_context_" + row["template_id"],
         "source_context_plumbing_not_held_out", PolicyDecision(row["action"], price),
         [Turn(**turn) for turn in row["history"]], row["message"], row["template_id"])
        for row in generator.bank
    ]
    requests += [
        ("zero_match_propose", "fixed_contract", PolicyDecision("propose", price), [], "unrelatedtoken", None),
        ("zero_match_counter", "fixed_contract", PolicyDecision("counter", price), [], "unrelatedtoken", None),
        ("fixed_accept", "fixed_contract", PolicyDecision("accept", price, referenced_offer_id="seller-offer-demo"), [], "unrelatedtoken", None),
        ("fixed_confirm", "fixed_contract", PolicyDecision("confirm", price, referenced_offer_id="buyer-offer-demo"), [], "unrelatedtoken", None),
        ("fixed_clarify", "fixed_contract", PolicyDecision("clarify", clarification_reason="unresolved_reference"), [], "unrelatedtoken", None),
        ("new_seller_1", "exploratory_not_quality_eval", PolicyDecision("propose", price), [], "This coffee table is gorgeous. What would you offer?", None),
        ("new_seller_2", "exploratory_not_quality_eval", PolicyDecision("counter", price), [], "The phone is brand new in the box. I cannot go below 95.", None),
        ("new_seller_3", "exploratory_not_quality_eval", PolicyDecision("counter", price), [], "The bike is in great shape but your offer is too low for me.", None),
    ]
    by_id = {row["template_id"]: row for row in generator.bank}
    records = []
    for case, purpose, decision, history, message, expected_id in requests:
        result = generator.generate(decision, history, message)
        selected = by_id.get(result.template_id, {})
        records.append(dict(
            case=case, purpose=purpose, demo="provisional_corpus_local_trial",
            human_approval=False, review_status="agent_reviewed_local_trial",
            split_status="provisional_official_train", reviewer="agent:Hermes",
            history=[asdict(turn) for turn in history], message=message,
            expected_template_id=expected_id,
            original_response=selected.get("response"), **asdict(result)))
    return records


def main():
    print("PROVISIONAL corpus local trial: 7 agent-reviewed templates, no human approval; "
          "team split/currency alignment pending. Source contexts prove plumbing, "
          "not held-out quality. New seller phrases are exploratory only.", file=sys.stderr)
    for record in preview():
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

