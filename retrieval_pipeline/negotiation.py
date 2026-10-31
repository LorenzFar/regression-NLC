"""Rule-based buyer gate, policy and tracker; the design is docs/negotiation.md.

State is plain JSON: {"offers": [{id, by, terms: {price}, status}], "agreement"}.
Only OfferTracker.commit proposes new state; the policy reads it from snapshots.
"""
import re
from copy import deepcopy
from decimal import ROUND_DOWN, Decimal

from .contracts import ContractError, PolicyDecision
from .generator import RetrievalGenerator
from .pipeline import GateDecision, Pipeline, Transition

CENT = Decimal("0.01")


def _money(value) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_DOWN)


def top_act(observation) -> tuple[str, float]:
    act = max(observation.probabilities, key=observation.probabilities.get)
    return act, observation.probabilities[act]


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
    """
    live = _live(state)
    if len(live) != 1:
        return None
    offer = live[0]
    if offer["by"] != ("seller" if state.get("agreement") == "pending" else "buyer"):
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


class OfferTracker:
    provenance = "rules:offer-tracker"

    def commit(self, snapshot, observation, gate, decision):
        state = deepcopy(snapshot.state)
        if decision.action == "clarify":
            return Transition(state, False)
        state.setdefault("offers", [])
        state.setdefault("agreement", "none")
        act, _ = top_act(observation)
        price = _seller_price(observation)
        if act in {"propose", "counter"} and price is not None:
            _add_offer(state, "seller", price)
            state["agreement"] = "none"
        elif act == "reject":
            _supersede(state, "buyer")
        if decision.action in {"propose", "counter"}:
            _add_offer(state, "buyer", decision.price)
        elif decision.action == "accept":
            state["agreement"] = "pending"
        elif decision.action == "confirm":
            state["agreement"] = "confirmed"
        elif decision.action == "reject":
            _supersede(state, "seller")
        return Transition(state, decision.action == "confirm")


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
        act, probability = top_act(observation)
        reference = observation.price_reference
        reasons = []
        if self.mode != "off" and (
            reference.get("reference_resolved") is False
            or (act in {"propose", "counter"} and reference.get("price") is None)
        ):
            reasons.append("unresolved_reference")
        if self.mode == "full":
            if act == "accept" and acceptance_target(snapshot.state, reference) is None:
                reasons.append("state_conflict")
            if probability < self.threshold:
                reasons.append("low_confidence")
        return GateDecision(self.mode, tuple(reasons), self.threshold)


class RulePolicy:
    """Open at target; concede a fixed fraction toward the seller, never past walk-away."""
    provenance = "rules:concession-policy"

    def __init__(self, target, walk_away, fraction="0.5"):
        self.target, self.walk_away = _money(target), _money(walk_away)
        self.fraction = Decimal(str(fraction))
        if not 0 <= self.target <= self.walk_away:
            raise ContractError("target must be between zero and walk_away")
        if not 0 < self.fraction <= 1:
            raise ContractError("fraction must be in (0, 1]")

    def _next_price(self, last_buyer, seller) -> Decimal:
        if last_buyer is None:
            return self.target
        return min(self.walk_away, _money(last_buyer + self.fraction * (seller - last_buyer)))

    def decide(self, snapshot, observation, gate):
        state = snapshot.state
        if gate.reasons:
            live = _live(state)
            if len(live) == 1:
                return PolicyDecision("clarify", live[0]["terms"]["price"], live[0]["id"],
                                      clarification_reason=gate.reasons[0])
            return PolicyDecision("clarify", clarification_reason=gate.reasons[0])
        act, _ = top_act(observation)
        last_buyer = _last_price(state, "buyer")
        offer = acceptance_target(state, observation.price_reference) if act == "accept" else None
        if offer is not None:
            return PolicyDecision("confirm", offer["terms"]["price"], offer["id"], finished=True)
        seller = _seller_price(observation)
        if act in {"propose", "counter"} and seller is not None:
            price = self._next_price(last_buyer, seller)
            if seller <= price:
                return PolicyDecision("accept", str(seller), next_offer_id(state, "seller"))
            if last_buyer == self.walk_away:
                return PolicyDecision("reject")
            return PolicyDecision("counter", str(price))
        # reject / inquire / vague, or an unpriced offer with the gate off.
        seller = _last_price(state, "seller")
        if seller is not None:
            return PolicyDecision("counter", str(min(seller, self._next_price(last_buyer, seller))))
        return PolicyDecision("propose", str(last_buyer if last_buyer is not None else self.target))


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


def build_buyer_pipeline(nlu, price_reference, *, target, walk_away, fraction="0.5",
                         threshold=0.7, mode="full", bank=None) -> Pipeline:
    return Pipeline(nlu=nlu, price_reference=price_reference,
                    gate=ConfidenceGate(threshold, mode),
                    policy=RulePolicy(target, walk_away, fraction),
                    tracker=OfferTracker(), generator=RetrievalGenerator(bank))
