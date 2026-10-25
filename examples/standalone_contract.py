"""Combined standalone demo; authored cases are not corpus or evaluation evidence.

Uses the existing replay CLI and Pipeline.turn, not a second conversation engine.
Without --checkpoint NLU is explicitly scripted; with it only NLU becomes real.
"""
import argparse
from pathlib import Path

from buyer_agent.replay import main as replay_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="supplied safe-load NLU checkpoint")
    parser.add_argument("--bank", default=str(Path(__file__).with_name("approved_fixture_bank.jsonl")))
    parser.add_argument("--output", default="-", help="diagnostic JSONL path, or - for stdout")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    command = [str(Path(__file__).with_name("standalone_cases.jsonl")),
               "--modes", "full", "--bank", args.bank, "--output", args.output,
               "--model-name", args.model_name, "--device", args.device]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    return replay_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
