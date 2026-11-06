"""Rule-based buyer gate, policy and tracker; the design is docs/negotiation.md.

State is plain JSON: {"offers": [{id, by, terms: {price}, status}], "agreement"},
plus optional per-item {"limits": {target, walk_away}} that override the policy's.
Only OfferTracker.commit proposes new state; the policy reads it from snapshots.
"""
import re
from copy import deepcopy
from dataclasses import replace
from decimal import ROUND_DOWN, Decimal

from .contracts import ContractError, PolicyDecision
from .generator import RetrievalGenerator
from .pipeline import GateDecision, Pipeline, Transition

CENT = Decimal("0.01")


def _money(value) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_DOWN)


OFFER_ACTS = {"propose", "counter"}
PENDING = "pending_clarification"


def top_act(observation) -> tuple[str, float]:
    act = max(observation.probabilities, key=observation.probabilities.get)
    return act, observation.probabilities[act]


def offer_confidence(observation) -> tuple[str, float]:
    """Top act and its confidence; propose and counter count together.

    The policy treats both as "the seller named a price", so the NLU being
    unsure which of the two it is should not trigger a clarification.
    """
    act, probability = top_act(observation)
    if act in OFFER_ACTS:
        probability = sum(observation.probabilities.get(name, 0.0) for name in OFFER_ACTS)
    return act, probability


def _live(state) -> list[dict]:
    return [offer for offer in state.get("offers", []) if offer["status"] == "live"]


def next_offer_id(state, by) -> str:
    """The id the tracker assigns to the next offer; the policy uses it for accept."""
    return f"{by}-{len(state.get('offers', [])) + 1}"


def _last_price(state, by) -> Decimal | None:
    prices = [offer["terms"]["price"] for offer in state.get("offers", []) if offer["by"] == by]
    return _money(prices[-1]) if prices else None


def _seller_price(observation) -> Decimal | None:
    price = observation.price_reference.get("price")
    return None if price is None else _money(price)


def acceptance_target(state, price_reference) -> dict | None:
    """The single offer a seller acceptance can confirm, or None.

    Exactly one live offer must exist: the buyer's, or the seller offer the
    buyer already accepted (pending). A named id or price must match it.
    Once confirmed, acknowledging the agreed offer stays consistent.
    """
    live = _live(state)
    if len(live) != 1:
        return None
    offer = live[0]
    agreement = state.get("agreement")
    if agreement != "confirmed" and offer["by"] != ("seller" if agreement == "pending" else "buyer"):
        return None
    reference, price = price_reference.get("referenced_offer_id"), price_reference.get("price")
    if reference is not None and reference != offer["id"]:
        return None
    if price is not None and Decimal(str(price)) != Decimal(offer["terms"]["price"]):
        return None
    return deepcopy(offer)


def _add_offer(state, by, price):
    for offer in state["offers"]:
        offer["status"] = "superseded"
    state["offers"].append({"id": next_offer_id(state, by), "by": by,
                            "terms": {"price": str(_money(price))}, "status": "live"})


def _supersede(state, by):
    for offer in state["offers"]:
        if offer["by"] == by:
            offer["status"] = "superseded"


def _pending_offer(state) -> dict | None:
    """The live seller offer the buyer accepted and is waiting on, or None."""
    live = _live(state)
    if state.get("agreement") == "pending" and len(live) == 1 and live[0]["by"] == "seller":
        return live[0]
    return None


def _apply_seller_move(state, act, price):
    if act in {"propose", "counter"} and price is not None:
        _add_offer(state, "seller", price)
        state["agreement"] = "none"
    elif act == "reject":
        _supersede(state, "buyer")
        if _pending_offer(state) is not None:
            _supersede(state, "seller")
            state["agreement"] = "none"


def _apply_buyer_move(state, action, price):
    if action in {"propose", "counter"}:
        live = _live(state)
        # Restating the live offer keeps its id, so references to it stay valid.
        if not (live and live[0]["by"] == "buyer" and _money(live[0]["terms"]["price"]) == _money(price)):
            _add_offer(state, "buyer", price)
        state["agreement"] = "none"
    elif action == "accept":
        state["agreement"] = "pending"
    elif action == "confirm":
        state["agreement"] = "confirmed"
    elif action == "reject":
        _supersede(state, "seller")
        state["agreement"] = "none"


class OfferTracker:
    provenance = "rules:offer-tracker"

    def open(self, snapshot, decision):
        """Buyer speaks first: the listing price is the seller's asking offer."""
        state = deepcopy(snapshot.state)
        state.setdefault("offers", [])
        state.setdefault("agreement", "none")
        if state.get("listing_price") is not None:
            _add_offer(state, "seller", state["listing_price"])
        _apply_buyer_move(state, decision.action, decision.price)
        return Transition(state, False)

    def commit(self, snapshot, observation, gate, decision):
        state = deepcopy(snapshot.state)
        if decision.action == "clarify":
            # Offers stay untouched; only remember what we asked about, so a
            # "yes" on the next turn can confirm that reading.
            act, _ = top_act(observation)
            state[PENDING] = {"act": act, "price": decision.price}
            return Transition(state, False)
        state.pop(PENDING, None)
        state.setdefault("offers", [])
        state.setdefault("agreement", "none")
        act, _ = top_act(observation)
        _apply_seller_move(state, act, _seller_price(observation))
        _apply_buyer_move(state, decision.action, decision.price)
        return Transition(state, decision.action == "confirm")


AFFIRMATION = re.compile(r"^\W*(yes|yeah|yep|yup|correct|right|exactly|that'?s right|indeed|sure)\b", re.I)


class ClarificationFollowUp:
    """A "yes" to our clarifying question confirms the reading we asked about.

    Runs before the gate. If the last buyer turn clarified a seller price and the
    seller now affirms ("yes", or any reply the NLU reads as accept, such as
    "ok") without naming a new price, the observation becomes that price with
    full confidence. Otherwise "ok" would accept our own older offer instead.
    Anything else passes through unchanged.
    """
    provenance = "rules:clarification-follow-up"

    def resolve(self, snapshot, observation):
        pending = snapshot.state.get(PENDING)
        if not pending or pending.get("price") is None:
            return observation
        act, _ = top_act(observation)
        affirmed = AFFIRMATION.match(observation.message) or act == "accept"
        # A price the seller actually wrote is an answer of its own; a price the
        # resolver borrowed from an earlier offer ("ok" -> our 200) is not.
        reference = observation.price_reference
        written = reference.get("price") is not None and reference.get("price_source") != "referenced_offer"
        restated = written and Decimal(reference["price"]) == Decimal(pending["price"])
        if (written and not restated) or not (affirmed or restated):
            return observation
        # We asked "do you mean <price>?", so a yes means the seller named that price:
        # an acceptance if it is our live offer, otherwise a seller offer.
        if pending["act"] == "accept":
            matches = acceptance_target(snapshot.state, {"price": pending["price"]}) is not None
            confirmed = "accept" if matches else "counter"
        else:
            confirmed = pending["act"] if pending["act"] in OFFER_ACTS else "counter"
        probabilities = {act: 1.0 if act == confirmed else 0.0 for act in observation.probabilities}
        reference = {"price": pending["price"], "reference_resolved": True, "status": "resolved",
                     "reason": "confirmed_after_clarification"}
        return replace(observation, probabilities=probabilities, price_reference=reference)


def _apply_recorded_turn(state, turn):
    act, price = turn["act"], turn.get("price")
    if turn["speaker"] == "seller":
        if act != "accept":
            _apply_seller_move(state, act, price)
        elif acceptance_target(state, turn) is not None:
            state["agreement"] = "confirmed"
    elif turn["speaker"] != "buyer":
        raise ContractError("speaker must be buyer or seller")
    elif act == "accept":
        if [offer["by"] for offer in _live(state)] == ["seller"]:
            state["agreement"] = "pending"
    elif act == "reject" or (act in {"propose", "counter"} and price is not None):
        _apply_buyer_move(state, act, price)


def track_turns(turns, *, limits=None) -> list[dict]:
    """Fold gold-labelled {speaker, act, price} turns through the tracker's rules.

    Returns the state before the first turn and after each one, so states[-1]
    is the pre-turn state for replaying the next seller message. Offer ids are
    the tracker's own, and the buyer's turns are recorded rather than decided.
    """
    state = {"offers": [], "agreement": "none", **({"limits": deepcopy(limits)} if limits else {})}
    states = [deepcopy(state)]
    for turn in turns:
        _apply_recorded_turn(state, turn)
        states.append(deepcopy(state))
    return states


def agreement_accuracy(dialogues) -> float | None:
    """Per-turn share where "confirmed" matches a turn's gold deal_agreed; None if unlabelled."""
    scored = [(state["agreement"] == "confirmed") == turn["deal_agreed"]
              for turns in dialogues
              for turn, state in zip(turns, track_turns(turns)[1:]) if "deal_agreed" in turn]
    return sum(scored) / len(scored) if scored else None


def _state_conflict(state, act, reference) -> bool:
    if act == "accept":
        return acceptance_target(state, reference) is None
    if state.get("agreement") == "confirmed":
        return act in {"propose", "counter", "reject"}
    if act == "reject":
        # This also asks about a "no" to a seller offer the buyer just accepted.
        return not any(offer["by"] == "buyer" for offer in _live(state))
    return False


class ConfidenceGate:
    """Clarify on unresolved references, act/state conflicts, or top probability below τ."""
    provenance = "rules:confidence-gate"

    def __init__(self, threshold=0.7, mode="full"):
        if mode not in {"off", "reference_only", "full"}:
            raise ContractError("unknown gate mode")
        if not 0 <= threshold <= 1:
            raise ContractError("gate threshold must be between zero and one")
        self.threshold, self.mode = threshold, mode

    def evaluate(self, snapshot, observation):
        act, probability = offer_confidence(observation)
        reference = observation.price_reference
        reasons = []
        if self.mode != "off" and (
            reference.get("reference_resolved") is False
            or (act in {"propose", "counter"} and reference.get("price") is None)
        ):
            reasons.append("unresolved_reference")
        # "250 if you collect today": the buyer negotiates price only, so it asks.
        if self.mode != "off" and _conditional_price(reference) is not None:
            reasons.append("conditional_price")
        if self.mode == "full":
            if _state_conflict(snapshot.state, act, reference):
                reasons.append("state_conflict")
            if probability < self.threshold:
                reasons.append("low_confidence")
        return GateDecision(self.mode, tuple(reasons), self.threshold)


def _conditional_price(reference) -> str | None:
    """The price the seller tied to a condition, if the resolver found one."""
    for candidate in reference.get("candidates") or []:
        if candidate.get("role") == "conditional":
            return candidate.get("price")
    return None


def _limits(target, walk_away) -> tuple[Decimal, Decimal]:
    if target is None or walk_away is None:
        raise ContractError("target and walk_away must be given together")
    target, walk_away = _money(target), _money(walk_away)
    if not 0 <= target <= walk_away:
        raise ContractError("target must be between zero and walk_away")
    return target, walk_away


class RulePolicy:
    """Open at target; concede a fixed fraction toward the seller, never past walk-away.

    Accepts a seller price within walk-away once the buyer has countered at
    least once and the remaining gap to our next counter is at most close_gap
    of the seller's price: another round would risk the deal for a small saving.
    """
    provenance = "rules:concession-policy"

    def __init__(self, target=None, walk_away=None, fraction="0.5", close_gap="0.05", whole_dollars=False):
        self.limits = None if target is None and walk_away is None else _limits(target, walk_away)
        self.fraction = Decimal(str(fraction))
        if not 0 < self.fraction <= 1:
            raise ContractError("fraction must be in (0, 1]")
        self.close_gap = Decimal(str(close_gap))
        if not 0 <= self.close_gap < 1:
            raise ContractError("close_gap must be in [0, 1)")
        # People haggle in whole dollars; round counters down so 252.50 becomes 252.
        self.whole_dollars = whole_dollars

    def opening(self, snapshot):
        """The buyer's first message: offer the target price."""
        target, _ = self._limits_for(snapshot.state)
        return PolicyDecision("propose", str(target))

    def _limits_for(self, state) -> tuple[Decimal, Decimal]:
        limits = state.get("limits")
        if limits is not None:
            return _limits(limits.get("target"), limits.get("walk_away"))
        if self.limits is None:
            raise ContractError("limits must come from the policy or the session state")
        return self.limits

    def _next_price(self, last_buyer, seller, target, walk_away) -> Decimal:
        if last_buyer is None:
            return target
        price = _money(last_buyer + self.fraction * (seller - last_buyer))
        if self.whole_dollars:
            price = price.quantize(Decimal("1"), rounding=ROUND_DOWN).quantize(CENT)
        return min(walk_away, price)

    def decide(self, snapshot, observation, gate):
        state = snapshot.state
        if gate.reasons:
            conditional = _conditional_price(observation.price_reference)
            if "conditional_price" in gate.reasons and conditional is not None:
                return PolicyDecision("clarify", str(_money(conditional)), next_offer_id(state, "seller"),
                                      clarification_reason="conditional_price")
            seller = _seller_price(observation)
            # Ask about the price the seller just named, if any.
            if seller is not None:
                return PolicyDecision("clarify", str(seller), next_offer_id(state, "seller"),
                                      clarification_reason=gate.reasons[0])
            live = _live(state)
            if len(live) == 1:
                return PolicyDecision("clarify", live[0]["terms"]["price"], live[0]["id"],
                                      clarification_reason=gate.reasons[0])
            return PolicyDecision("clarify", clarification_reason=gate.reasons[0])
        act, _ = top_act(observation)
        offer = acceptance_target(state, observation.price_reference) if act == "accept" else None
        if offer is not None:
            return PolicyDecision("confirm", offer["terms"]["price"], offer["id"], finished=True)
        target, walk_away = self._limits_for(state)
        last_buyer, on_record = _last_price(state, "buyer"), _last_price(state, "seller")
        seller = _seller_price(observation)
        if act in {"propose", "counter"} and seller is not None:
            price = self._next_price(last_buyer, seller, target, walk_away)
            # A repeated price is the seller's final one, so waiting cannot improve it.
            # Close the gap only after at least one real counter, so the buyer
            # never accepts the seller's first reply to its opening.
            countered = sum(offer["by"] == "buyer" for offer in state.get("offers", [])) >= 2
            close = countered and seller <= walk_away and seller - price <= self.close_gap * seller
            if seller <= price or close or (seller == on_record and seller <= walk_away):
                return PolicyDecision("accept", str(seller), next_offer_id(state, "seller"))
            if last_buyer == walk_away:
                return PolicyDecision("reject")
            return PolicyDecision("counter", str(price))
        # No new seller price: a rejection earns one step, anything else changes nothing.
        pending = _pending_offer(state)
        if pending is not None and act != "reject":
            return PolicyDecision("accept", pending["terms"]["price"], pending["id"])
        if act == "reject" and on_record is not None:
            price = self._next_price(last_buyer, on_record, target, walk_away)
            if on_record > price and last_buyer == walk_away:
                return PolicyDecision("reject")
            return PolicyDecision("counter", str(min(on_record, price)))
        price = target if last_buyer is None else min(last_buyer, walk_away)
        if on_record is None:
            return PolicyDecision("propose", str(price))
        return PolicyDecision("counter", str(min(on_record, price)))


class RegexPriceReference:
    """INTERIM stand-in for the team's price/reference resolver: digits only, no offer ids."""
    provenance = "interim:regex-price-reference"

    def __init__(self):
        # nlu imports torch; load it only when this adapter is used.
        from nlu import _PRICE_RE

        self._pattern = _PRICE_RE

    def observe(self, snapshot, case_id, message):
        values = set()
        for match in self._pattern.finditer(message):
            digits = re.search(r"\d[\d,]*(?:\.\d+)?", match.group())
            if digits:
                value = Decimal(digits.group().replace(",", ""))
                values.add(value * 1000 if match.group().rstrip().lower().endswith("k") else value)
        if len(values) == 1:
            return {"price": str(values.pop()), "reference_resolved": True}
        return {"reference_resolved": False} if values else {}


def build_buyer_pipeline(nlu, price_reference, *, target=None, walk_away=None, fraction="0.5",
                         threshold=0.7, mode="full", bank=None) -> Pipeline:
    return Pipeline(nlu=nlu, price_reference=price_reference,
                    gate=ConfidenceGate(threshold, mode),
                    policy=RulePolicy(target, walk_away, fraction),
                    tracker=OfferTracker(), generator=RetrievalGenerator(bank))
