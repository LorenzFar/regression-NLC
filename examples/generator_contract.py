"""Demonstrate fixed buyer replies using authored cases."""
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
