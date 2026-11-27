"""Pick more CraigslistBargain TEST dialogues for the gold set (v3 extension). Read-only on existing labels.

From repository root:
  python -m scripts.gold_extend sample --n 35 --out data/craigslist_test/v3      # natural set, rare-act weighted
  python -m scripts.gold_extend mine   --out data/craigslist_test/v3             # challenge candidates

sample: chooses whole dialogues not already in the gold set or the challenge sheet. Dialogues whose seller turns
        contain rejection or first-offer cues are preferred (reject and propose are the rarest gold acts), and a
        quarter of the picks are uniform random so the set is not only cue-matched. Writes v3_natural.md (transcripts
        to label) and v3_natural_ids.json. Every seller AND buyer turn of a chosen dialogue must be labelled so
        RQ2 can replay it from gold.
mine:   scans seller turns of the remaining unused dialogues for RQ2 challenge patterns (conditional acceptance,
        reference to an earlier offer, agreement words next to a new price) and writes v3_challenge_candidates.md.
        Candidates are suggestions only: each one is hand-checked before it enters the challenge sheet.

Selection is deterministic (fixed seed). The script never touches labels_barry.csv or rq2_review.xlsx.
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

DATA = Path("data/craigslist_test")
SEED = 425

REJECT = re.compile(r"\b(too low|can'?t (?:do|go|accept|take)|cannot|won'?t|no way|not (?:going|willing|interested)"
                    r"|lowball|firm|nope|sorry|that'?s too)\b|^\s*no\b", re.I)
PROPOSE = re.compile(r"\b(asking|i(?:'m| am) (?:selling|looking for)|price is|sell (?:it )?for)\b", re.I)

AGREE = r"(?:deal|ok(?:ay)?|sure|fine|alright|works|sounds good|accept|agree|yes)"
CHALLENGE = {
    "conditional_acceptance": re.compile(rf"\b{AGREE}\b.*\b(?:if|but|as long as|only if|provided|cash only|pick ?up|today|tomorrow)\b", re.I),
    "reference_to_earlier_offer": re.compile(r"\b(?:that|your|the) (?:offer|price|number|amount)\b|\b(?:earlier|before|previous|original|first) (?:offer|price)\b|\bsame (?:as|price)\b", re.I),
    "counter_with_agreement_words": re.compile(r"\d[\d,.]*k?\b.{0,25}\b(?:(?:and|&) (?:we|you)(?:'ve| have)? (?:got )?a deal|(?:and )?it'?s yours|deal\b)", re.I),
}


def load(data):
    dialogues = {d["dialogue_index"]: d for d in map(json.loads, open(data / "dialogues.jsonl", encoding="utf-8"))}
    used = {int(r["dialogue_index"]) for r in csv.DictReader(open(data / "labels_barry.csv", encoding="utf-8-sig"))}
    try:
        from openpyxl import load_workbook
        ws = load_workbook(data / "rq2_review.xlsx", read_only=True)["challenge"]
        header = [c.value for c in next(ws.iter_rows(max_row=1))]
        col = header.index("dialogue_index")
        used |= {int(r[col]) for r in ws.iter_rows(min_row=2, values_only=True) if r[col] is not None}
    except ImportError:
        print("warning: openpyxl missing, challenge-sheet dialogues not excluded", file=sys.stderr)
    return dialogues, used


def seller_turns(dialogue):
    return [t for t in dialogue["turns"] if t["speaker"] == "seller" and t["text"].strip()]


def transcript(dialogue, mark=None):
    lines = [f"## d{dialogue['dialogue_index']:04d} | {dialogue['category']} | {dialogue['item']} | "
             f"listing {dialogue['list_price']:g} | buyer target {dialogue['buyer_target']:g}"]
    for t in dialogue["turns"]:
        flag = "  <<<" if mark is not None and t["turn"] == mark else ""
        lines.append(f"t{t['turn']:>2} {t['speaker'].upper():6} {t['text'].strip()}{flag}")
    return "\n".join(lines)


def sample(dialogues, used, n, out):
    rng = random.Random(SEED)
    free = sorted(i for i in dialogues if i not in used and 2 <= len(seller_turns(dialogues[i])) <= 8)

    def score(i):
        turns = seller_turns(dialogues[i])
        return 2 * sum(bool(REJECT.search(t["text"])) for t in turns) + sum(bool(PROPOSE.search(t["text"])) for t in turns)

    n_random = n // 4
    weighted = sorted(free, key=lambda i: (-score(i), rng.random()))[: n - n_random]
    rest = [i for i in free if i not in weighted]
    picks = sorted(weighted + rng.sample(rest, n_random))
    out.mkdir(parents=True, exist_ok=True)
    (out / "v3_natural_ids.json").write_text(json.dumps({"seed": SEED, "weighted": sorted(weighted),
                                                         "random": sorted(set(picks) - set(weighted))}, indent=1))
    (out / "v3_natural.md").write_text("\n\n".join(transcript(dialogues[i]) for i in picks) + "\n", encoding="utf-8")
    n_seller = sum(len(seller_turns(dialogues[i])) for i in picks)
    n_buyer = sum(1 for i in picks for t in dialogues[i]["turns"] if t["speaker"] == "buyer" and t["text"].strip())
    print(f"{len(picks)} dialogues ({len(weighted)} weighted, {n_random} random): {n_seller} seller, {n_buyer} buyer turns")
    return picks


def mine(dialogues, used, out):
    taken = set(used)
    ids_file = out / "v3_natural_ids.json"
    if ids_file.exists():
        saved = json.loads(ids_file.read_text())
        taken |= set(saved["weighted"]) | set(saved["random"])
    blocks, counts = [], {k: 0 for k in CHALLENGE}
    for i in sorted(dialogues):
        if i in taken:
            continue
        for t in seller_turns(dialogues[i]):
            if t["turn"] < 2:
                continue
            tags = [k for k, rx in CHALLENGE.items() if rx.search(t["text"])]
            if tags:
                for k in tags:
                    counts[k] += 1
                d = dialogues[i]
                context = [x for x in d["turns"] if x["turn"] < t["turn"]][-3:]
                blocks.append(f"### cb-test-{i:04d}-t{t['turn']} [{', '.join(tags)}] listing {d['list_price']:g}\n"
                              + "\n".join(f"    {x['speaker'].upper():6} {x['text'].strip()}" for x in context)
                              + f"\n >> SELLER {t['text'].strip()}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "v3_challenge_candidates.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"{len(blocks)} candidate turns; by pattern: {counts}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "mine"])
    parser.add_argument("--n", type=int, default=35)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=DATA / "v3")
    args = parser.parse_args(argv)
    dialogues, used = load(args.data)
    if args.command == "sample":
        sample(dialogues, used, args.n, args.out)
    else:
        mine(dialogues, used, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
