"""Average the token probabilities of several tagger models that share a tokenizer, then decode as usual.

  python ensemble.py --data data --models runs/a/model runs/b/model --split validation --out runs/ens
Needs the same --no-meta setting the models were trained with. Writes <out>/<split>_predictions.jsonl.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

if __package__:
    from .train import collate, decode, encode, read_jsonl, score
else:  # Keep the existing standalone GPU/Jupyter workflow working.
    from train import collate, decode, encode, read_jsonl, score


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", choices=["train", "validation", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    from price_reference import normalise_amount

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.models[0])
    rows = read_jsonl(os.path.join(args.data, f"{args.split}.jsonl"))
    items = [encode(tok, r, not args.no_meta, labelled=False) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(items[i]["input_ids"]))
    probs = {}
    for path in args.models:
        model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
        for k in range(0, len(order), args.batch_size):
            idx = order[k:k + args.batch_size]
            batch = collate([items[i] for i in idx], tok.pad_token_id)
            p = model(input_ids=batch["input_ids"].to(device),
                      attention_mask=batch["attention_mask"].to(device)).logits.softmax(-1).cpu()
            for j, i in enumerate(idx):
                n = len(items[i]["input_ids"])
                probs[i] = probs.get(i, 0) + p[j, :n] / len(args.models)
        del model
    preds = []
    for i, row in enumerate(rows):
        labels = probs[i].argmax(-1).tolist()
        spans = decode(row["text"], items[i], labels)
        values, unreadable = [], 0
        for s, e in spans:
            v = normalise_amount(row["text"][s:e].strip())
            if v is None:
                unreadable += 1
            else:
                values.append(v)
        preds.append({"id": row["id"], "spans": [list(s) for s in spans], "values": values, "unreadable": unreadable})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.split}_predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    if "gold" in rows[0]:
        s = score(rows, preds)
        print(args.split, {k: round(v, 4) for k, v in s.items() if isinstance(v, float)}, s["confusion"])


if __name__ == "__main__":
    main()
