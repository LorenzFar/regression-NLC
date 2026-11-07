"""RQ2: clarification-gate metrics from replay output and gold labels.

    python -m retrieval_pipeline.replay cases.jsonl --output artifacts/replay.jsonl ...
    python -m evaluation.gate_eval --replay artifacts/replay.jsonl --gold gold.jsonl \\
        --thresholds 0.5 0.6 0.7 0.8 0.9 --output artifacts/rq2.json

Replay must contain the gate modes you want compared (``off`` is required for the sweep).
"""
from __future__ import annotations

import argparse
import sys

from .data import load_gold, read_jsonl, write_json
from .gate_metrics import CONFIRM_ACTIONS, gate_report, sweep_threshold


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", required=True, help="JSONL written by retrieval_pipeline.replay")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--confirm-actions", nargs="+", default=sorted(CONFIRM_ACTIONS),
                        help="buyer actions that count as confirming a deal")
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    try:
        gold_rows = load_gold(args.gold, require=("needs_clarification", "deal_agreed"))
        gold = {row["case_id"]: row for row in gold_rows}
        records = read_jsonl(args.replay)
        confirm = frozenset(args.confirm_actions)
        report = {"gold_cases": len(gold), "confirm_actions": sorted(confirm),
                  "by_mode": gate_report(records, gold, confirm_actions=confirm)}
        if args.thresholds:
            report["sweep"] = sweep_threshold(records, gold, args.thresholds, confirm_actions=confirm)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc!r}", file=sys.stderr)
        return 2
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
