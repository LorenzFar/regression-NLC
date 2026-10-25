"""Proposed teammate contract, backed by authored examples, not corpus results.

Run from the repository root: python -m examples.generator_contract

Public API:
    RetrievalGenerator(bank_path=None, history_window=2)
        .generate(decision: PolicyDecision, history: list[Turn], message: str)
        -> GenerationResult

Amounts are non-negative SGD strings: ASCII digits with an optional one- or
 two-digit fraction. No currency, signs, exponent, whitespace, or rounding.
Spelling is preserved, including trailing/leading zeros. Zero is supported;
policy owns which amounts to authorize. All contracts are frozen dataclasses.
History is oldest-first, excluding the separate current seller message.

propose/counter require price and do not take referenced_offer_id.
accept/confirm require price and a nonblank referenced_offer_id. The reference
component/tracker, not the generator, proves offer ownership and live validity.
reject may carry an optional validated price/reference, but does not end a
session. finished must be a bool; only policy/tracker and pipeline apply it.
Only clarify accepts clarification_reason. Reasons are low_confidence,
unresolved_reference, and state_conflict. Omit BOTH offer fields for generic
questions, or provide BOTH for a specific question; partial details fail.
Offer IDs are opaque and are not copied into wording as possible extra amounts.

Ticket 01 uses fixed fallback/clarification wording only. Fallbacks have
source='fallback', fallback_reason='no_bank', template_id='fixed.<action>'.
Clarifications have source='clarification', no fallback_reason, and a reason/
specificity-bearing template_id. similarity is None. Retrieval arrives in 02;
a non-None bank_path currently raises NotImplementedError, never silently loads.
Invalid requests raise ContractError (ValueError) without mutating inputs.
These are proposed adapter-facing types, not verified teammate compatibility.
"""
from dataclasses import asdict
import json

from retrieval_pipeline import ContractError, PolicyDecision, RetrievalGenerator, Turn


def main():
    generator = RetrievalGenerator()
    history = [Turn("BUYER", "I can offer 70."), Turn("SELLER", "Could you do 90?")]
    decisions = [
        PolicyDecision("propose", "80"),
        PolicyDecision("counter", "80.50"),
        PolicyDecision("accept", "80.50", "seller-offer-1"),
        PolicyDecision("confirm", "80.50", "buyer-offer-1", finished=True),
        PolicyDecision("reject"),
        PolicyDecision("clarify", clarification_reason="low_confidence"),
        PolicyDecision("clarify", clarification_reason="unresolved_reference"),
        PolicyDecision("clarify", "80.50", "buyer-offer-1", "state_conflict"),
    ]
    for decision in decisions:
        result = generator.generate(decision, history, "Can we agree?")
        print(json.dumps({"fixture": True, "decision": asdict(decision), "result": asdict(result)}))
    try:
        generator.generate(PolicyDecision("accept", "80.50"), history, "Yes")
    except ContractError as error:
        print(json.dumps({"fixture": True, "error_type": type(error).__name__, "error": str(error)}))
    else:
        raise AssertionError("Acceptance without an offer identity must fail")


if __name__ == "__main__":
    main()
