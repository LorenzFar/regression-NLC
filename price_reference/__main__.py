"""Try the extractor on one message.

From repository root:
  python -m price_reference "I can do $500, it's only 2 years old" --listing 600
"""
import argparse
import sys

from .extractor import explain


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--listing", type=float, help="listing price, for the plausibility check")
    args = parser.parse_args(argv)
    candidates = explain(args.text, args.listing)
    if not candidates:
        print("no numbers found")
    for c in candidates:
        if c.mention:
            m = c.mention
            print(f"KEEP   {c.surface!r:22} value={m.value:>9}  kind={m.kind:8} "
                  f"currency={m.has_currency!s:5} plausible={m.plausible}  span={m.start}-{m.end}")
        else:
            print(f"REJECT {c.surface!r:22} value={c.value or '-':>9}  reason: {c.reject_reason}")


if __name__ == "__main__":
    main()
