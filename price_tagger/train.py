"""Fine-tune a token classifier (default DistilBERT) to tag money amounts, and score it like the rules.

Runs anywhere with PyTorch + transformers; written for a GPU Jupyter node.
  python train.py --data data --out runs/distilbert --epochs 4
  python train.py --data data --out runs/distilbert --predict test --model runs/distilbert/model   # final run only

Labels per token: O, B (first token of a money amount), I (later tokens). Gold prices are tagged; numbers the
rules keep as money but the gold does not mark are masked out of the loss; everything else is O.
A predicted span becomes a price only if price_reference.normalise_amount can read its text, so the model
finds the amount and the same normaliser reads the value, exactly as for the rules.
Model selection uses validation only. The test split is predicted only with --predict test.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_reference import normalise_amount  # noqa: E402

try:
    from evaluation.price_extractor.metrics import Tally, to_set
except ImportError:
    from metrics import Tally, to_set

O, B, I = 0, 1, 2
MAX_LEN = 256


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def meta(row):
    listing = "" if row["listing"] is None else f" listing {int(round(row['listing']))}"
    return f"{row['role']} {row['category']}{listing}"


def encode(tok, row, use_meta, labelled, ignore_mode="mask"):
    enc = (tok(meta(row), row["text"], truncation="only_second", max_length=MAX_LEN, return_offsets_mapping=True)
           if use_meta else tok(row["text"], truncation=True, max_length=MAX_LEN, return_offsets_mapping=True))
    text_seq = 1 if use_meta else 0
    seq = enc.sequence_ids()
    offsets = enc["offset_mapping"]
    labels, started = [], set()
    pos = [tuple(s) for s in row.get("pos", [])] if labelled else []
    ignore = [tuple(s) for s in row.get("ignore", [])] if labelled else []
    for (a, b), s in zip(offsets, seq):
        if s != text_seq:
            labels.append(-100)
            continue
        hit = next((i for i, (ps, pe) in enumerate(pos) if a < pe and b > ps), None)
        if hit is not None:
            labels.append(I if hit in started else B)
            started.add(hit)
        elif ignore_mode == "mask" and any(a < ie and b > is_ for is_, ie in ignore):
            labels.append(-100)
        else:
            labels.append(O)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels,
            "offsets": offsets, "seq": seq, "text_seq": text_seq}


def collate(batch, pad_id):
    n = max(len(x["input_ids"]) for x in batch)
    pad = lambda seq, v: seq + [v] * (n - len(seq))
    return {"input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in batch]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
            "labels": torch.tensor([pad(x["labels"], -100) for x in batch])}


def decode(text, item, pred):
    """Token labels -> list of (start, end) character spans."""
    spans, cur = [], None
    for (a, b), s, p in zip(item["offsets"], item["seq"], pred):
        if s != item["text_seq"]:
            continue
        if p == B or (p == I and cur is None):
            if cur:
                spans.append(tuple(cur))
            cur = [a, b]
        elif p == I:
            cur[1] = b
        else:
            if cur:
                spans.append(tuple(cur))
            cur = None
    if cur:
        spans.append(tuple(cur))
    return spans


@torch.no_grad()
def predict(model, tok, rows, items, device, batch_size=128):
    model.eval()
    out = []
    order = sorted(range(len(rows)), key=lambda i: len(items[i]["input_ids"]))
    results = {}
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        batch = collate([items[i] for i in idx], tok.pad_token_id)
        logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        preds = logits.argmax(-1).cpu().tolist()
        for i, p in zip(idx, preds):
            results[i] = p[:len(items[i]["input_ids"])]
    for i, row in enumerate(rows):
        spans = decode(row["text"], items[i], results[i])
        values, unreadable = [], 0
        for s, e in spans:
            v = normalise_amount(row["text"][s:e].strip())
            if v is None:
                unreadable += 1
            else:
                values.append(v)
        out.append({"id": row["id"], "spans": [list(s) for s in spans], "values": values, "unreadable": unreadable})
    return out


def score(rows, preds):
    """Same scoring as evaluation.price_extractor: labelled rows only; unread rows count as false alarms."""
    t, alarms, unread, unreadable = Tally(), 0, 0, 0
    for row, p in zip(rows, preds):
        unreadable += p["unreadable"]
        if row.get("gold") is None:
            unread += 1
            alarms += bool(p["values"])
        else:
            t.add(to_set("none" if not row["gold"] else " | ".join(row["gold"])), frozenset(p["values"]))
    s = t.summary()
    s.update({"unread_rows": unread, "unread_false_alarms": alarms, "unreadable_spans": unreadable})
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--unread-fraction", type=float, default=0.15, help="share of the all-negative unread rows to train on")
    ap.add_argument("--ignore-mode", choices=["mask", "negative"], default="mask",
                    help="mask: money the gold does not mark (budgets, deltas) is left out of the loss, so the model "
                         "tags all money; negative: it is tagged O, so the model learns the gold's own sense of price")
    ap.add_argument("--no-meta", action="store_true", help="feed the message alone (no role, category, listing price)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict", choices=["train", "validation", "test"], help="load --model and write predictions for this split; no training")
    ap.add_argument("--model", help="model directory for --predict")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = open(os.path.join(args.out, "train.log"), "a", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    use_meta = not args.no_meta
    if args.predict:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
        rows = read_jsonl(os.path.join(args.data, f"{args.predict}.jsonl"))
        items = [encode(tok, r, use_meta, labelled=False) for r in rows]
        preds = predict(model, tok, rows, items, device)
        if "gold" in rows[0]:
            say("scores on " + args.predict + ": " + json.dumps({k: v for k, v in score(rows, preds).items() if k != "confusion"}))
        path = os.path.join(args.out, f"{args.predict}_predictions.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        say(f"wrote {len(preds)} predictions to {path}")
        return

    say(f"device {device}; model {args.model_name}; meta {use_meta}; args {vars(args)}")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=3, id2label={0: "O", 1: "B", 2: "I"}, label2id={"O": 0, "B": 1, "I": 2}).to(device)

    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val_rows = read_jsonl(os.path.join(args.data, "validation.jsonl"))
    labelled = [r for r in train_rows if r["train_ok"] and r["gold"] is not None]
    unread = [r for r in train_rows if r["train_ok"] and r["gold"] is None]
    random.shuffle(unread)
    train_rows = labelled + unread[:int(len(unread) * args.unread_fraction)]
    say(f"train rows {len(train_rows)} ({len(labelled)} labelled + {len(train_rows) - len(labelled)} unread negatives); val rows {len(val_rows)}")
    train_items = [encode(tok, r, use_meta, True, args.ignore_mode) for r in train_rows]
    val_items = [encode(tok, r, use_meta, False) for r in val_rows]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(train_items) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_items)))
        random.shuffle(order)
        total, n, t0 = 0.0, 0, time.time()
        for k in range(0, len(order), args.batch_size):
            batch = {key: v.to(device) for key, v in collate([train_items[i] for i in order[k:k + args.batch_size]], tok.pad_token_id).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item()
            n += 1
        preds = predict(model, tok, val_rows, val_items, device)
        s = score(val_rows, preds)
        history.append({"epoch": epoch, "train_loss": total / n, **s})
        say(f"epoch {epoch}: loss {total / n:.4f} | val exact {s['exact_match']:.4f} P {s['precision']:.4f} R {s['recall']:.4f} "
            f"F1 {s['f1']:.4f} | false-none {s['confusion']['false_price']} missed {s['confusion']['missed_price']} "
            f"unread-FP {s['unread_false_alarms']} unreadable {s['unreadable_spans']} | {time.time() - t0:.0f}s")
        if s["f1"] > best:
            best = s["f1"]
            model.save_pretrained(os.path.join(args.out, "model"))
            tok.save_pretrained(os.path.join(args.out, "model"))
            with open(os.path.join(args.out, "validation_predictions.jsonl"), "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            say(f"  saved best (val F1 {best:.4f})")
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "best_val_f1": best, "history": history}, f, indent=1)
    say("done")


if __name__ == "__main__":
    main()
