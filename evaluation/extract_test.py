"""Extract the CraigslistBargain test split into viewable files and a labelling sheet.

The team NLU (nlu.ipynb) trained and calibrated on the train and validation
splits only, so every test dialogue is unseen. The test split carries no intent
or price annotations; gold labels are added by hand in the labelling sheet.

From repository root:
    python -m evaluation.extract_test
    python -m evaluation.extract_test --parquet path/to/test.parquet --sample-dialogues 60

Needs pandas and pyarrow (only for this extraction step).
"""
import argparse
import csv
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

PARQUET_URL = ("https://huggingface.co/datasets/stanfordnlp/craigslist_bargains/resolve/"
               "refs%2Fconvert%2Fparquet/default/craigslist_bargains-test.parquet")
DEFAULT_OUT = Path("data/craigslist_test")
CONTEXT_TURNS = 3
LABEL_COLUMNS = ["gold_act", "gold_prices", "annotator", "notes"]
SHEET_COLUMNS = ["case_id", "dialogue_index", "turn", "category", "item", "list_price",
                 "context", "message"] + LABEL_COLUMNS


def _first(value):
    return value[0] if len(value) else None


def dialogue_from_record(index, record):
    """Convert one dataset row to a plain dialogue; empty UI-action turns are dropped."""
    roles = list(record["agent_info"]["Role"])
    targets = list(record["agent_info"]["Target"])
    items = record["items"]
    turns = []
    for turn, (agent, text) in enumerate(zip(record["agent_turn"], record["utterance"])):
        if text and text.strip():
            turns.append({"turn": turn, "speaker": roles[int(agent)], "text": text.strip()})
    target = {role: float(value) for role, value in zip(roles, targets)}
    return {
        "dialogue_index": index,
        "category": _first(items["Category"]),
        "item": _first(items["Title"]),
        "list_price": float(_first(items["Price"])),
        "description": _first(items["Description"]),
        "buyer_target": target.get("buyer"),
        "seller_target": target.get("seller"),
        "turns": turns,
    }


def seller_rows(dialogue):
    """One labelling row per seller message, with the last few turns as reading context."""
    rows = []
    for position, turn in enumerate(dialogue["turns"]):
        if turn["speaker"] != "seller":
            continue
        context = dialogue["turns"][max(0, position - CONTEXT_TURNS):position]
        rows.append({
            "case_id": f"cb-test-{dialogue['dialogue_index']:04d}-t{turn['turn']}",
            "dialogue_index": dialogue["dialogue_index"],
            "turn": turn["turn"],
            "category": dialogue["category"],
            "item": dialogue["item"],
            "list_price": dialogue["list_price"],
            "context": "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in context),
            "message": turn["text"],
            **{column: "" for column in LABEL_COLUMNS},
        })
    return rows


def transcript(dialogue):
    lines = [f"## Dialogue {dialogue['dialogue_index']}: {dialogue['item']}",
             f"Category: {dialogue['category']} | List price: {dialogue['list_price']:g} | "
             f"Buyer target: {dialogue['buyer_target']:g} | Seller target: {dialogue['seller_target']:g}",
             ""]
    lines += [f"- **t{t['turn']} {t['speaker'].upper()}:** {t['text']}" for t in dialogue["turns"]]
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    # utf-8-sig so Excel shows non-ASCII item names correctly.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, help="local test parquet; downloaded when omitted")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-dialogues", type=int, default=60,
                        help="dialogues in the labelling sheet; 0 puts every dialogue in it")
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args(argv)
    try:
        import pandas as pd
    except ImportError:
        parser.error("extraction needs pandas and pyarrow: pip install pandas pyarrow")

    args.out.mkdir(parents=True, exist_ok=True)
    parquet = args.parquet
    if parquet is None:
        parquet = args.out / "craigslist_bargains-test.parquet"
        if not parquet.is_file():
            print(f"Downloading {PARQUET_URL}")
            urlretrieve(PARQUET_URL, parquet)
    frame = pd.read_parquet(parquet)
    dialogues = [dialogue_from_record(index, record)
                 for index, record in enumerate(frame.to_dict("records"))]

    with (args.out / "dialogues.jsonl").open("w", encoding="utf-8") as stream:
        for dialogue in dialogues:
            stream.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
    (args.out / "transcripts.md").write_text(
        "# CraigslistBargain test split\n\nUnseen by NLU training. Empty UI-action turns are omitted.\n\n"
        + "\n".join(transcript(d) for d in dialogues), encoding="utf-8")
    all_rows = [row for dialogue in dialogues for row in seller_rows(dialogue)]
    write_csv(args.out / "all_seller_turns.csv", all_rows)

    # Sample whole dialogues so labellers see real context; sorted for readable order.
    chosen = list(range(len(dialogues)))
    if 0 < args.sample_dialogues < len(dialogues):
        chosen = sorted(random.Random(args.seed).sample(chosen, args.sample_dialogues))
    chosen = set(chosen)
    sheet = [row for row in all_rows if row["dialogue_index"] in chosen]
    write_csv(args.out / "labelling_sheet.csv", sheet)

    print(f"{len(dialogues)} dialogues, {len(all_rows)} seller turns -> {args.out}")
    print(f"labelling_sheet.csv: {len(chosen)} dialogues, {len(sheet)} seller turns (seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
