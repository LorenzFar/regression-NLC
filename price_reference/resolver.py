"""Pure rules-v1 price and offer-reference resolution."""
from decimal import Decimal
import re

from .evidence import (ACCEPT_WORDING, OFFER_WORDING, POINTER,
                       candidate, clause, conditions, demotion, floor_price, multiple_reason)
from .ledger import latest
from .types import Offer, OfferLedger, PriceCandidate, Resolution, ResolverMention, money

ACTS = {"propose", "counter", "accept", "reject", "inquire", "vague"}


def _reference(offer, **kwargs):
    return Resolution("resolved", offer.price, "referenced_offer", offer=offer.ref,
                      offer_liveness=offer.liveness, state_history_disagreement=offer.state_history_disagreement, **kwargs)


def _problem(reason, offer=None, *, conflict=False, candidates=()):
    return Resolution("conflict" if conflict else "unresolved", reason=reason,
                      context_price=offer.price if offer else None,
                      context_offer=offer.ref if offer else None, candidates=candidates,
                      state_history_disagreement=bool(offer and offer.state_history_disagreement))


def _accept(mentions, ledger, owner, text):
    target, blocked = latest(ledger, owner)
    if any(m.kind == "delta" for m in mentions):
        return _problem("delta_acceptance", target)
    if not mentions:
        if blocked:
            return _problem("ambiguous_history", target)
        if target is None:
            return _problem("no_standing_offer")
        return _reference(target, conditions=conditions(text))
    if len(mentions) > 1:
        return Resolution("ambiguous", candidates=tuple(candidate(text, m) for m in mentions),
                          reason="multiple_accept_prices")
    price = mentions[0].value
    # An explicit price can identify a candidate from an ambiguous historical turn.
    ambiguous_matches = [t for t in ledger.ambiguous_turns if t.ref.owner == owner
                         and sum(c.price == price for c in t.candidates) == 1]
    matches = [o for o in ledger.offers if o.ref.owner == owner and o.price == price]
    if target and target.price == price and not blocked:
        return _reference(target, conditions=conditions(text))
    if ambiguous_matches:
        turn = ambiguous_matches[-1]
        if matches and (turn.ref.turn_index is None or matches[-1].ref.turn_index is None):
            return _problem("reference_identity_ambiguous", target)
        if not matches or turn.ref.turn_index > matches[-1].ref.turn_index:
            return _reference(Offer(turn.ref, price), reason="matched_ambiguous_candidate", conditions=conditions(text))
    if matches:
        return _reference(matches[-1], conditions=conditions(text))
    if target is None and not any(t.ref.owner == owner for t in ledger.ambiguous_turns):
        return _problem("no_standing_offer")
    return _problem("price_mismatch", target, conflict=True,
                    candidates=tuple(candidate(text, m) for m in mentions))


def _delta(text, mention, ledger, speaker):
    if mention.start is None:
        return _problem("delta_unlocated")
    _, _, local = clause(text, mention)
    down = re.search(r"\b(?:off|less|lower|cheaper|discount|knock(?:ed)?|cut|drop(?:ped)?|reduc(?:e|ed|ing))\b", local, re.I)
    up = re.search(r"\b(?:more|extra|higher|another|additional)\b", local, re.I)
    if bool(down) == bool(up):
        return _problem("unknown_direction")
    your = bool(re.search(r"\byour\b", local, re.I))
    my = bool(re.search(r"\bmy\b", local, re.I))
    if your and my:
        return _problem("unknown_base")
    owner = ("buyer" if speaker == "seller" else "seller") if your else speaker
    base, blocked = latest(ledger, owner)
    if base is None or blocked:
        return _problem("unknown_base")
    value = Decimal(base.price) + Decimal(mention.value) * (-1 if down else 1)
    if value < 0:
        return _problem("negative_result", base)
    return Resolution("resolved", money(value), "delta", base_offer=base.ref,
                      source_span=(mention.start, mention.end), conditions=conditions(text, mention),
                      state_history_disagreement=base.state_history_disagreement,
                      reason="zero_price" if value == 0 else None)


def resolve(message: str, *, speaker: str = "seller", act: str | None,
            mentions: list[ResolverMention] | tuple[ResolverMention, ...],
            ledger: OfferLedger) -> Resolution:
    """Resolve evidence without mutating the input ledger or committing any state.

    Unknown liveness identifies an offer, not permission to finalize a deal.
    NLU probability thresholds belong to the downstream gate.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    speaker = speaker.lower()
    if speaker not in {"buyer", "seller"} or act is not None and act not in ACTS:
        raise ValueError("invalid speaker or dialogue act")
    for mention in mentions:
        mention.validate(message)
    clean = [m for m in mentions if not demotion(message, m, speaker)]
    dropped = [demotion(message, m, speaker) for m in mentions if demotion(message, m, speaker)]
    pointer = bool(POINTER.search(message))
    if not clean and any(reason in {"withdrawn", "superseded"} for reason in dropped):
        return Resolution("no_price", reason="withdrawn_price")
    if not clean and re.search(
            r"\b(?:no (?:deal|agreement)|not (?:a )?deal|"
            r"(?:cannot|can['’]t|won['’]t|will not|don['’]t|do not)\s+(?:accept|agree|take))\b",
            message, re.I):
        return Resolution("no_price", reason="refusal")
    if any(m.start is None for m in clean) and len(clean) > 1:
        return Resolution("ambiguous", candidates=tuple(candidate(message, m) for m in clean),
                          reason="mention_unlocated")
    if any(m.kind == "delta" and m.start is None for m in clean):
        return _problem("delta_unlocated")

    # Strong explicit proposal wording overrides a wrong inquire/vague/reject/accept act.
    # Generic acceptance text ("I accept your offer") is not proposal wording.
    wording = any(OFFER_WORDING.search(clause(message, m)[2]) or floor_price(message, m, speaker) for m in clean)
    effective = act
    disagreement = False
    proposal_question = bool(re.search(r"\b(?:(?:do|can|could|would|will) you|(?:would|could)\s+\$?\d|is\s+\$?\d)", message, re.I))
    accepting_existing = (act == "accept" and not proposal_question
                          and not any(floor_price(message, m, speaker) for m in clean)
                          and bool(re.search(r"\baccept\b|\byour offer\b|\bworks for me\b", message, re.I)))
    if clean and wording and not accepting_existing and act is not None and act not in {"propose", "counter"}:
        effective = "counter"
        disagreement = True
    if effective == "accept":
        if not clean and dropped:
            # A wrong accept act cannot turn a refusal/budget-only utterance into a deal.
            return _problem("act_price_disagree")
        return _accept(clean, ledger, "buyer" if speaker == "seller" else "seller", message)

    if act is None and pointer:
        return _problem("act_unknown")
    if clean:
        if len(clean) > 1:
            return Resolution("ambiguous", candidates=tuple(candidate(message, m) for m in clean),
                              reason=multiple_reason(message), act_disagreement=disagreement)
        mention = clean[0]
        if effective in {"inquire", "vague"}:
            return _problem("act_price_disagree")
        if effective == "reject":
            return Resolution("no_price", reason="rejection")
        if mention.kind == "delta":
            if act is None:
                return _problem("act_unknown")
            return _delta(message, mention, ledger, speaker)
        chosen = candidate(message, mention)
        attached = conditions(message, mention)
        return Resolution("resolved", mention.value, "explicit", source_span=chosen.span,
                          candidates=(chosen,), conditions=attached,
                          reason="conditional_price" if attached else None,
                          act_disagreement=disagreement)

    if pointer:
        if effective in {"propose", "counter"}:
            other = "buyer" if speaker == "seller" else "seller"
            owner = other if re.search(r"\byour\b", message, re.I) else speaker
            target, blocked = latest(ledger, owner)
            if blocked:
                return _problem("ambiguous_history", target)
            choices = [o for o in ledger.offers if o.ref.owner == owner]
            by_price = {o.price: o for o in choices}
            if not choices:
                return _problem("no_standing_offer")
            if len(by_price) > 1:
                return Resolution("ambiguous", candidates=tuple(PriceCandidate(p) for p in by_price),
                                  reason="multiple_prior_prices")
            return _reference(target or choices[-1], conditions=conditions(message))
        # A rejection/inquiry may mention an existing offer without making a new one.
        if effective in {"reject", "inquire", "vague"} and (ledger.offers or ledger.ambiguous_turns):
            return Resolution("no_price", reason="pointer_context")
        return _problem("pointer_without_authority")
    if "budget" in dropped and effective in {"propose", "counter"}:
        return _problem("budget_only")
    if effective in {"propose", "counter"} and not dropped:
        return _problem("no_price_stated")
    return Resolution("no_price", reason="context_only" if dropped else None)
