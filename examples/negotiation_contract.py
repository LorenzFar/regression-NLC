"""Worked trace from docs/negotiation.md through the rule-based buyer pipeline.

Without --checkpoint NLU probabilities are scripted; price/reference is the
interim regex adapter. Authored conversations, not evaluation evidence.
"""
import argparse
import json
from pathlib import Path

from retrieval_pipeline.negotiation import RegexPriceReference, build_buyer_pipeline
from retrieval_pipeline.pipeline import Session

ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")
CONVERSATIONS = {
    "worked-trace": [
        ("I'm asking 120 for it.", "propose", 0.91),
        ("Lowest I'd go is 110.", "counter", 0.88),
        ("ok", "accept", 0.55),
        ("Yes, 95 works for me.", "accept", 0.93),
    ],
    "no-live-offer": [("Deal.", "accept", 0.95)],
}


class ScriptedNLU:
    provenance = "fixture:scripted-nlu"

    def __init__(self):
        self.scores = {message: (act, p) for turns in CONVERSATIONS.values() for message, act, p in turns}

    def predict(self, history, message):
        act, probability = self.scores[message]
        rest = (1 - probability) / (len(ACTS) - 1)
        return {label: probability if label == act else rest for label in ACTS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="supplied safe-load NLU checkpoint")
    parser.add_argument("--bank", default=str(Path(__file__).with_name("approved_fixture_bank.jsonl")))
    args = parser.parse_args(argv)
    if args.checkpoint:
        from retrieval_pipeline.nlu_adapter import NLUAdapter

        nlu = NLUAdapter(args.checkpoint)
    else:
        nlu = ScriptedNLU()
    pipeline = build_buyer_pipeline(nlu, RegexPriceReference(), target="80", walk_away="100",
                                    fraction="0.5", bank=args.bank)
    for name, turns in CONVERSATIONS.items():
        session = Session(name)
        print(f"== {name} (target 80, walk-away 100, fraction 0.5, threshold 0.7)")
        for index, (message, _, _) in enumerate(turns, 1):
            result = pipeline.turn(session, message, case_id=f"{name}-{index}", turn_id=f"seller-{index}")
            print(f"Seller: {message}")
            print(f"Buyer [{result.generation.action}; gate={list(result.trace['gate']['reasons'])}]: "
                  f"{result.generation.text}")
        snapshot = session.snapshot()
        print(f"Final: finished={snapshot.finished} state={json.dumps(snapshot.state, sort_keys=True)}")


if __name__ == "__main__":
    main()
