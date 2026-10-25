"""Authored fixture demonstration, NOT a corpus experiment.

Run from the repository: python -m examples.retrieval_contract
Use --export-sources DIRECTORY to exercise the separate review/build CLI stages.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from buyer_agent import PolicyDecision, RetrievalGenerator
from buyer_agent.bank import approve_candidate, build_bank, prepare_candidate

BANK = Path(__file__).with_name("approved_fixture_bank.jsonl")
TRAIN_IDS = {"fixture-camera", "fixture-bicycle"}


def authored_sources():
    """Explicitly authored buyer turns with explicit action and price annotations."""
    rows = []
    for item, context, response in [
        ("camera", "Pristine camera lens", "For the camera, I offer 80."),
        ("bicycle", "Bicycle wheels frame", "For the bicycle, I offer 80."),
    ]:
        start = response.index("80")  # Fixture annotation, not a price extractor.
        rows.append(dict(template_id=f"fixture.counter.{item}", dialogue_id=f"fixture-{item}",
                         turn_id="buyer-2", origin="authored_fixture", split="train", speaker="BUYER",
                         action="counter", history=[], message=context, response=response,
                         price_spans=[dict(start=start, end=start + 2, value="80", validated=True)],
                         mixed_action=False, unresolved_reference=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-sources", type=Path)
    parser.add_argument("--build-fixture", action="store_true",
                        help="rebuild the illustrative fixture-approval artifact, not corpus approval")
    args = parser.parse_args()
    if args.export_sources:
        args.export_sources.mkdir(parents=True, exist_ok=True)
        (args.export_sources / "sources.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in authored_sources()), encoding="utf-8")
        (args.export_sources / "train-ids.json").write_text(json.dumps(sorted(TRAIN_IDS)), encoding="utf-8")
    if args.build_fixture:
        rows = [approve_candidate(prepare_candidate(row, TRAIN_IDS),
                                  "Authored fixture demonstration review (not corpus approval)")
                for row in authored_sources()]
        build_bank(rows, BANK, TRAIN_IDS)
    generator = RetrievalGenerator(BANK)

    for message in ("Pristine camera lens", "Bicycle wheels frame", "unrelatedtoken"):
        result = generator.generate(PolicyDecision("counter", "80.50"), [], message)
        print(json.dumps(dict(message=message, **asdict(result),
                              demo="authored_fixture")))


if __name__ == "__main__":
    main()
