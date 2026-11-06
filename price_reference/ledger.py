"""Best-effort historical evidence, not a replacement for committed tracker state."""
from dataclasses import replace
import re

from .evidence import candidate, clause, collect_mentions, demotion, multiple_reason
from .types import AmbiguousTurn, Offer, OfferLedger, OfferRef, money


def _turn(turn):
    if isinstance(turn, dict):
        return turn.get("speaker", turn.get("role", "")), turn["text"]
    if hasattr(turn, "speaker"):
        return turn.speaker, turn.text
    return turn


def _structured_state(history_offers, ambiguous, state):
    """Read the teammate OfferTracker's authoritative JSON offer ledger.

    Schema: offers=[{id, by, terms: {price}, status}], agreement=none/pending/confirmed.
    An explicit offers=[] means no recorded offers, not permission to infer others.
    """
    records = state["offers"]
    if not isinstance(records, list):
        raise ValueError("tracker offers must be a list")
    offers, seen = [], set()
    last_by_owner = {record.get("by"): index for index, record in enumerate(records) if isinstance(record, dict)}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("tracker offers must contain objects")
        owner, offer_id = record.get("by"), record.get("id")
        if owner not in {"buyer", "seller"} or not isinstance(offer_id, str) or not offer_id.strip() or offer_id in seen:
            raise ValueError("tracker offer needs a unique id and buyer/seller owner")
        if not isinstance(record.get("terms"), dict) or "price" not in record["terms"]:
            raise ValueError("tracker offer needs terms.price")
        status = record.get("status")
        if status not in {"live", "superseded", "withdrawn", "rejected", "confirmed", "not_live"}:
            raise ValueError("unknown tracker offer status")
        seen.add(offer_id)
        price = money(record["terms"]["price"])
        owner_history = [offer for offer in history_offers if offer.ref.owner == owner]
        matches = [offer for offer in owner_history if offer.price == price]
        linked = matches[0] if len(matches) == 1 else None
        ref = OfferRef(offer_id, owner, linked.ref.turn_index if linked else None, "state")
        disagreement = False
        if index == last_by_owner[owner]:
            disagreement = bool(owner_history and owner_history[-1].price != price)
            disagreement |= any(turn.ref.owner == owner and (
                ref.turn_index is None or turn.ref.turn_index is None or turn.ref.turn_index > ref.turn_index)
                for turn in ambiguous)
        offers.append(Offer(ref, price, "live" if status == "live" else "not_live",
                            "state_linked_by_price" if linked else "state_only", disagreement))
    live = [offer for offer in offers if offer.liveness == "live"]
    pending = live[0].ref if state.get("agreement") == "pending" and len(live) == 1 and live[0].ref.owner == "seller" else None
    # The complete tracker ledger supersedes reconstructed offer membership. History
    # still supplies unique links and advisory disagreement, never alternative IDs.
    return OfferLedger(tuple(offers), (), pending)


def build_ledger(history, state=None, listing_price=None) -> OfferLedger:
    offers, ambiguous, ordinals = [], [], {"buyer": 0, "seller": 0}
    for index, turn in enumerate(history):
        speaker, text = _turn(turn)
        owner = speaker.lower()
        if owner not in ordinals:
            raise ValueError("history speaker must be buyer or seller")
        ordinals[owner] += 1
        ref = OfferRef(f"{owner}-offer-{ordinals[owner]}", owner, index)
        # Classify each mention, not the entire turn: a clarification can be followed
        # by a genuine proposal whose wording is not in a finite offer lexicon.
        kept = [m for m in collect_mentions(text, listing_price)
                if m.kind == "absolute" and not demotion(text, m, owner)
                and not re.search(r"\bi accept (?:your|that|the) offer\b", clause(text, m)[2], re.I)]
        if not kept:
            continue
        # Supersession is evidence only; withdrawal/rejection is not reconstructed.
        offers = [replace(o, liveness="not_live") if o.ref.owner == owner else o for o in offers]
        if len(kept) == 1:
            offers.append(Offer(ref, kept[0].value))
        else:
            ambiguous.append(AmbiguousTurn(ref, tuple(candidate(text, m) for m in kept), multiple_reason(text)))

    state = state or {}
    if "offers" in state:
        return _structured_state(offers, ambiguous, state)
    for owner in ordinals:
        offer_id = state.get(f"{owner}_offer_id")
        price = state.get(f"{owner}_price")
        if not offer_id or price is None:
            continue
        price = money(price)
        owner_history = [o for o in offers if o.ref.owner == owner]
        last_clean = owner_history[-1] if owner_history else None
        mismatch = last_clean is not None and last_clean.price != price
        matches = [o for o in offers if o.ref.owner == owner and o.price == price]
        # Repeated identical prices do not identify one originating turn.
        linked = matches[0] if len(matches) == 1 else None
        ref = OfferRef(str(offer_id), owner, linked.ref.turn_index if linked else None, "state")
        live = "live" if state.get("agreement") == "pending" else "unknown"
        history_ambiguous = any(t.ref.owner == owner and (
            ref.turn_index is None or t.ref.turn_index is None or t.ref.turn_index > ref.turn_index)
            for t in ambiguous)
        if state.get(f"{owner}_offer_live") is False or state.get("agreement") in {"rejected", "withdrawn", "confirmed"}:
            live = "not_live"
        offers = [replace(o, liveness="not_live") if o.ref.owner == owner else o for o in offers
                  if o is not linked]
        offers.append(Offer(ref, price, live, "state_linked_by_price" if linked else "state_only",
                            mismatch or history_ambiguous))
    return OfferLedger(tuple(offers), tuple(ambiguous))


def latest(ledger, owner):
    """Return (offer, blocked_by_ambiguous_turn). Explicit state is authoritative."""
    offers = [o for o in ledger.offers if o.ref.owner == owner]
    state = [o for o in offers if o.ref.id_source == "state"]
    if state:
        # Tracker facts take precedence; disagreement is exposed for policy validation.
        return state[-1], False
    offer = max(offers, key=lambda o: o.ref.turn_index if o.ref.turn_index is not None else -1, default=None)
    last_index = offer.ref.turn_index if offer and offer.ref.turn_index is not None else -1
    blocked = any(t.ref.owner == owner and (t.ref.turn_index is None or t.ref.turn_index > last_index)
                  for t in ledger.ambiguous_turns)
    return offer, blocked
