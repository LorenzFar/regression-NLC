"""Compare two annotators' gold_act labels on shared case IDs; no model needed.

From repository root:
    python -m evaluation.agreement sheet_a.csv sheet_b.csv
"""
import argparse
import json
from pathlib import Path

from .evaluate_nlu import LabelError, agreement, load_gold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    try:
        result = agreement(load_gold(args.first), args.second)
    except (LabelError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
