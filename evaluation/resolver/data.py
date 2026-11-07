"""JSONL gold loading. CSV values never acquire invented character offsets."""
import hashlib
import json
from pathlib import Path

from price_reference.types import (AmbiguousTurn, Offer, OfferLedger, OfferRef,
                                   PriceCandidate, ResolverMention, money)


def mentions_from_gold(text, gold_price, *, inferred=False):
    if gold_price is None or str(gold_price).strip().lower() in {"", "none"}:
        return ()
    prices = gold_price if isinstance(gold_price, list) else str(gold_price).split("|")
    return tuple(ResolverMention(money(p.strip() if isinstance(p, str) else p), "unknown",
                                 "gold_inferred" if inferred else "gold_csv_value") for p in prices)


def annotated_mention(text, record):
    record = dict(record)
    occurrence = record.pop("occurrence_index", None)
    # Only a supplied exact surface plus an explicit occurrence may create offsets.
    # A value alone is never searched for in the message.
    if record.get("start") is None and record.get("surface") is not None and occurrence is not None:
        surface = record["surface"]
        if not surface or type(occurrence) is not int or occurrence < 0:
            raise ValueError("invalid gold surface occurrence")
        starts, start = [], 0
        while (start := text.find(surface, start)) >= 0:
            starts.append(start)
            start += len(surface)
        if occurrence >= len(starts):
            raise ValueError("gold surface occurrence not found")
        record["start"] = starts[occurrence]
        record["end"] = starts[occurrence] + len(surface)
    record.setdefault("provenance", "gold_annotated")
    mention = ResolverMention(**record)
    mention.validate(text)
    return mention


def offer_ref(record):
    record = record.get("ref", record)
    return OfferRef(record["offer_id"], record["owner"], record.get("turn_index"),
                    record.get("id_source", "state"))


def gold_ledger(row):
    if "offers" not in row:
        raise ValueError("gold-ledger conditions require an explicit offers list (possibly empty)")
    offers = tuple(Offer(offer_ref(o), money(o["price"]), o.get("liveness", "unknown"),
                         o.get("link", "state_only"),
                         o.get("state_history_disagreement", o.get("state_history_mismatch", False))) for o in row["offers"])
    ambiguous = tuple(AmbiguousTurn(offer_ref(t), tuple(
        PriceCandidate(money(c["price"]), tuple(c["span"]) if c.get("span") else None,
                       c.get("role", "primary"), c.get("condition")) for c in t["candidates"]),
        t.get("reason", "multiple_prices")) for t in row.get("ambiguous_turns", []))
    for turn in ambiguous:
        if turn.ref.turn_index is not None:
            if turn.ref.turn_index >= len(row["history"]):
                raise ValueError("ambiguous gold turn exceeds history bounds")
            history_turn = row["history"][turn.ref.turn_index]
            text = history_turn["text"] if isinstance(history_turn, dict) else history_turn[1]
            for candidate in turn.candidates:
                candidate.validate(text)
    pending = offer_ref(row["pending_acceptance"]) if row.get("pending_acceptance") else None
    return OfferLedger(offers, ambiguous, pending)


def gold_mentions(row):
    if "mentions" in row:
        return tuple(annotated_mention(row["message"], m) for m in row["mentions"])
    if "gold_price" not in row:
        raise ValueError("gold mentions require mentions or gold_price; target price is not mention gold")
    return mentions_from_gold(row["message"], row["gold_price"], inferred=row.get("inferred", False))


def load_gold(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") in seen or not row.get("case_id"):
                raise ValueError(f"duplicate/missing case_id on line {line_number}")
            seen.add(row["case_id"])
            if row.get("label_source") not in {"seeded_rule", "ai_proposed", "human_reviewed"}:
                raise ValueError("gold requires explicit label_source")
            if row["label_source"] == "human_reviewed" and not str(row.get("reviewer", "")).strip():
                raise ValueError("human_reviewed requires reviewer")
            if not isinstance(row.get("message"), str) or not isinstance(row.get("history"), list):
                raise ValueError("gold requires message text and history list")
            if row.get("price") is not None:
                money(row["price"])
            if row.get("status") is not None and row["status"] not in {
                    "resolved", "no_price", "unresolved", "conflict", "ambiguous"}:
                raise ValueError("unknown gold status")
            if "needs_clarification" in row and type(row["needs_clarification"]) is not bool:
                raise ValueError("needs_clarification must be a JSON boolean")
            if "mentions" in row:
                gold_mentions(row)
            if "offers" in row:
                gold_ledger(row)
            rows.append(row)
    return rows


def verify_freeze(path, manifest_path, rows):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    required = {"file_sha256", "dialogue_ids", "resolver_commit", "frozen_at", "exposure"}
    if not required <= manifest.keys() or not all(manifest[k] for k in required):
        raise ValueError("freeze manifest missing hash, dialogue ids, commit, date or exposure")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest["file_sha256"]:
        raise ValueError("holdout file differs from freeze manifest")
    ids = {str(r.get("dialogue_id", "")) for r in rows}
    if "" in ids or ids != set(map(str, manifest["dialogue_ids"])):
        raise ValueError("holdout dialogue IDs differ from manifest")
    return manifest
