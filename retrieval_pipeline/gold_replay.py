"""Independent gold-style seller turns through the FULL integrated buyer pipeline.

    python -m retrieval_pipeline.gold_replay GOLD.jsonl --checkpoint models/nlu_k2_s0.pt \
        --output artifacts/rq2-replay.jsonl

No fixtures or current-turn gold labels enter inference. Missing listing_price is
estimated as the largest positive absolute money mention extracted without a listing
from history + message (not delta/comparison amounts). This is only a heuristic;
no usable amount means an error record. Supply real listing prices when available.
History uses gold labels only if ALL turns have an act and priced proposal/counter
labels; otherwise the whole history is predicted. Seller acts use their own prefix;
buyer acts use extracted prices, then agreement words, else vague. Predicted prices
use the last plausible absolute mention; ambiguous/reference-only history remains
approximate. Reconstruction quality is separate from current-turn gate quality.
Exit codes: 0 all succeeded; 1 recorded case/resource failures; 2 invalid CLI/setup.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_DOWN
import argparse
import json
from pathlib import Path
import re
import sys

from price_reference.extractor import extract

from .agent import build_agent, load_nlu
from .contracts import Turn
from .negotiation import track_turns
from .pipeline import Session
from .replay import MODES

ACTS = {"propose", "counter", "accept", "reject", "inquire", "vague"}
AGREEMENT = re.compile(r"\b(?:accept|agreed?|deal|yes|okay|ok|works|sounds good)\b", re.I)


def _amount(value, *, positive=False):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError("amount must be finite and positive" if positive else "amount must be finite and nonnegative")
    return format(amount.quantize(Decimal("0.01"), rounding=ROUND_DOWN), "f")


def _restore(row, nlu, target_ratio, walk_away_ratio):
    if not isinstance(row.get("message"), str) or not row["message"].strip():
        raise ValueError("message must be a nonblank string")
    history = row.get("history", [])
    if not isinstance(history, list) or any(
        not isinstance(t, dict) or t.get("speaker") not in {"buyer", "seller"}
        or not isinstance(t.get("text"), str) for t in history
    ):
        raise ValueError("history must contain buyer/seller text turns")
    if "listing_price" in row:
        listing = _amount(row["listing_price"], positive=True)
        listing_source = "explicit"
    else:
        amounts = [Decimal(m.value) for text in [*(t["text"] for t in history), row["message"]]
                   for m in extract(text, None) if m.kind == "absolute" and Decimal(m.value) > 0]
        if not amounts:
            raise ValueError("no listing_price or positive absolute money mention; cannot derive listing")
        listing = _amount(max(amounts), positive=True)
        listing_source = "max_extracted_absolute"
    limits = {key: _amount(row[key] if key in row else Decimal(listing) * Decimal(str(ratio)))
              for key, ratio in (("target", target_ratio), ("walk_away", walk_away_ratio))}
    if Decimal(limits["target"]) > Decimal(limits["walk_away"]):
        raise ValueError("target must not exceed walk_away")
    labelled = all(t.get("act") in ACTS and
                   (t["act"] not in {"propose", "counter"} or t.get("price") is not None)
                   for t in history)
    turns, prefix = [], []
    for turn in history:
        if labelled:
            recorded = {"speaker": turn["speaker"], "act": turn["act"],
                        "price": _amount(turn["price"]) if turn.get("price") is not None else None}
            if "referenced_offer_id" in turn:
                recorded["referenced_offer_id"] = turn["referenced_offer_id"]
        else:
            prices = [m.value for m in extract(turn["text"], Decimal(listing))
                      if m.kind == "absolute" and m.plausible is not False]
            if turn["speaker"] == "seller":
                probabilities = nlu.predict(list(prefix), turn["text"])
                act = max(probabilities, key=probabilities.get)
            elif prices:
                act = "counter" if turns else "propose"
            else:
                act = "accept" if AGREEMENT.search(turn["text"]) else "vague"
            recorded = {"speaker": turn["speaker"], "act": act,
                        "price": prices[-1] if prices else None}
        turns.append(recorded)
        prefix.append((turn["speaker"].upper(), turn["text"]))
    state = track_turns(turns, limits=limits)[-1]
    state["listing_price"] = listing
    return ([Turn(t["speaker"], t["text"]) for t in history], state,
            "gold_labels" if labelled else "predicted", listing_source)


def replay_gold(rows, nlu, *, modes=MODES, threshold=0.7, target_ratio=0.7,
                walk_away_ratio=0.9, bank=None):
    """One fresh pipeline/session per row × mode; failures retain evaluator keys."""
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("modes must be nonempty, unique off/reference_only/full values")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    for ratio in (target_ratio, walk_away_ratio):
        if not Decimal(str(ratio)).is_finite() or Decimal(str(ratio)) < 0:
            raise ValueError("ratios must be finite and nonnegative")
    records, seen = [], set()
    for index, raw in enumerate(rows, 1):
        row = raw if isinstance(raw, dict) else {}
        identity = row.get("case_id")
        valid_id = isinstance(identity, str) and bool(identity.strip())
        case_id = identity if valid_id else f"invalid-line-{index}"
        duplicate = case_id in seen
        seen.add(case_id)
        prepared, preparation_error = None, None
        try:
            if not valid_id:
                raise ValueError("case_id must be a nonblank string")
            if duplicate:
                raise ValueError("duplicate case_id")
            if "input_error" in row:
                raise ValueError(row["input_error"])
            prepared = _restore(row, nlu, target_ratio, walk_away_ratio)
        except Exception as exc:
            preparation_error = exc
        for mode in modes:
            record = {"case_id": case_id, "input_index": index, "origin": row.get("origin"),
                      "gate": {"mode": mode, "threshold": threshold, "reasons": []},
                      "reconstruction_mode": None, "error": None}
            try:
                if preparation_error is not None:
                    raise preparation_error
                history, state, reconstruction, listing_source = prepared
                record.update(reconstruction_mode=reconstruction, listing_price_source=listing_source)
                pipeline = build_agent(nlu, listing_price=state["listing_price"],
                                       **state["limits"], gate_mode=mode, threshold=threshold, bank=bank)
                session = Session(case_id, history=history, state=deepcopy(state))
                record.update(pipeline.turn(session, row["message"], case_id=case_id,
                                            turn_id=f"{case_id}:seller").trace)
            except Exception as exc:
                record.update(getattr(exc, "trace", {}))
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            try:
                record = json.loads(json.dumps(record, allow_nan=False))
            except (TypeError, ValueError) as exc:
                record = {"case_id": case_id, "gate": {"mode": mode, "threshold": threshold, "reasons": []},
                          "reconstruction_mode": record["reconstruction_mode"],
                          "error": {"type": "SerializationError", "message": str(exc)}}
            records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold", help="gold-style JSONL, not scripted replay fixtures")
    parser.add_argument("--checkpoint", default="models/nlu_k2_s0.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--target-ratio", type=float, default=0.7)
    parser.add_argument("--walk-away-ratio", type=float, default=0.9)
    parser.add_argument("--bank")
    args = parser.parse_args(argv)
    if args.output != "-" and Path(args.gold).resolve() == Path(args.output).resolve():
        parser.error("input and output must differ")
    try:
        rows = []
        for index, line in enumerate(Path(args.gold).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                def reject_constant(value):
                    raise ValueError(f"nonfinite JSON number: {value}")
                rows.append(json.loads(line, parse_constant=reject_constant))
            except ValueError as exc:
                rows.append({"case_id": f"invalid-line-{index}", "input_error": str(exc)})
        if not rows:
            raise ValueError("input file is empty")
        # Validate CLI settings before loading the model.
        replay_gold([], None, modes=args.modes, threshold=args.threshold,
                    target_ratio=args.target_ratio, walk_away_ratio=args.walk_away_ratio)
        try:
            nlu = load_nlu(args.checkpoint)
        except Exception as exc:
            # Resource failures must also be real diagnostics, not synthetic inference.
            resource_error = exc
            class UnavailableNLU:
                def predict(self, history, message):
                    raise resource_error
            nlu = UnavailableNLU()
        records = replay_gold(rows, nlu, modes=args.modes, threshold=args.threshold,
                              target_ratio=args.target_ratio, walk_away_ratio=args.walk_away_ratio,
                              bank=args.bank)
        text = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
        if args.output == "-":
            sys.stdout.write(text)
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return int(any(r["error"] is not None for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
