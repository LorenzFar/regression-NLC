"""Scripted, labelled fixture demo through the public conversation pipeline."""
import argparse
import json

from .contracts import PolicyDecision
from .fixtures import FixtureCase, ScriptedComponents
from .pipeline import Pipeline, Session


def scripted_cases():
    pending = {"buyer_offer_id": "buyer-1", "buyer_price": "100.25", "agreement": "pending"}
    return {
        "opening": FixtureCase(
            {"propose": 0.90, "counter": 0.02, "accept": 0.02, "reject": 0.02, "inquire": 0.02, "vague": 0.02},
            PolicyDecision("counter", "100.25"),
            price_reference={"price": "110.50", "reference_resolved": True},
            state_after=pending, message="I am asking 110.50.",
        ),
        "uncertain": FixtureCase(
            {"propose": 0.1, "counter": 0.1, "accept": 0.2, "reject": 0.1, "inquire": 0.1, "vague": 0.4},
            PolicyDecision("reject"), price_reference={"reference_resolved": True},
            state_after=pending, message="Maybe.",
        ),
        "decline": FixtureCase(
            {"propose": 0.02, "counter": 0.90, "accept": 0.02, "reject": 0.02, "inquire": 0.02, "vague": 0.02},
            PolicyDecision("reject"), price_reference={"price": "109", "reference_resolved": True},
            state_after=pending, message="No, 109 instead.",
        ),
        "known-acceptance": FixtureCase(
            {"propose": 0.02, "counter": 0.02, "accept": 0.90, "reject": 0.02, "inquire": 0.02, "vague": 0.02},
            PolicyDecision("confirm", "100.25", "buyer-1", finished=True),
            price_reference={"price": "100.25", "referenced_offer_id": "buyer-1", "reference_resolved": True},
            state_after={**pending, "agreement": "confirmed"}, finished=True,
            message="I accept your offer of 100.25.",
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "reference_only", "full"), default="full")
    parser.add_argument("--jsonl", action="store_true", help="Write only diagnostic JSONL to stdout")
    args = parser.parse_args(argv)
    cases = scripted_cases()
    pipeline = Pipeline(**ScriptedComponents(cases, mode=args.mode).adapters())
    session = Session("fixture-demo")
    if not args.jsonl:
        print(f"AUTHORED FIXTURE DEMO (not real policy/gate/reference/NLU evaluation); mode={args.mode}")
    for case_id, case in cases.items():
        try:
            result = pipeline.turn(session, case.message, case_id=case_id, turn_id=case_id)
        except Exception as error:
            if hasattr(error, "trace"):
                print(json.dumps(error.trace))
            raise
        if args.jsonl:
            print(json.dumps(result.trace))
        else:
            print(f"Seller: {case.message}")
            print(f"Buyer [{result.generation.action}; {result.generation.source}]: {result.generation.text}")
            print(f"  gate={result.trace['gate']['reasons']}; finished={result.snapshot.finished}")
    if not args.jsonl:
        print(f"Final fixture state: {json.dumps(session.snapshot().state, sort_keys=True)}")


if __name__ == "__main__":
    main()
