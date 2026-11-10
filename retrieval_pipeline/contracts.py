"""Immutable pipeline contracts; prices are exact SGD decimal strings."""
import re
from dataclasses import dataclass


PRICE_ACTIONS = {"propose", "counter", "accept", "confirm"}
ACTIONS = PRICE_ACTIONS | {"reject", "clarify", "answer"}


class ContractError(ValueError):
    """A generation request violates the public contract."""


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class OfferContext:
    """Tracker's previous buyer offer and this turn's resolved seller offer.

    A missing current seller price must not be filled from older seller offers.
    These prices support wording only; PolicyDecision.price remains authoritative.
    """
    previous_buyer_price: str | None = None
    current_seller_price: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    price: str | None = None
    referenced_offer_id: str | None = None
    clarification_reason: str | None = None
    finished: bool = False
    offer_context: OfferContext | None = None
    answer_type: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    action: str
    price: str | None
    template_id: str | None
    source: str
    similarity: float | None = None
    fallback_reason: str | None = None
    origin: str | None = None
    source_dialogue_id: str | None = None
    source_turn_id: str | None = None


def validate_public_context(history: list[Turn], message: str) -> None:
    """Validate preceding public turns and the separate current seller message."""
    if not isinstance(history, list) or any(
        not isinstance(turn, Turn)
        or not isinstance(turn.speaker, str) or turn.speaker.lower() not in {"buyer", "seller"}
        or not isinstance(turn.text, str)
        for turn in history
    ):
        raise ContractError("history must be a list of Turn(speaker=str, text=str)")
    if not isinstance(message, str):
        raise ContractError("message must be a string")


def validate_generation_request(decision: PolicyDecision, history: list[Turn], message: str) -> None:
    """Validate policy-authorized terms without choosing or rendering wording."""
    validate_public_context(history, message)
    if not isinstance(decision, PolicyDecision):
        raise ContractError("decision must be a PolicyDecision")
    if not isinstance(decision.action, str) or decision.action not in ACTIONS:
        raise ContractError("unsupported buyer action")
    if type(decision.finished) is not bool:
        raise ContractError("finished must be a boolean supplied by policy/tracker")
    context = decision.offer_context
    if context is not None:
        if not isinstance(context, OfferContext):
            raise ContractError("offer_context must be an OfferContext")
        for price in (context.previous_buyer_price, context.current_seller_price):
            if price is not None and (
                not isinstance(price, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", price)
            ):
                raise ContractError("offer_context prices must be non-negative SGD decimal strings")
    if decision.clarification_reason is not None and decision.action != "clarify":
        raise ContractError("clarification_reason is only supported for clarify")
    if decision.price is not None or decision.action in PRICE_ACTIONS:
        if not isinstance(decision.price, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", decision.price):
            raise ContractError("price must be a non-negative SGD decimal string with at most two fractional digits")
    if decision.action == "answer":
        if not isinstance(decision.answer_type, str) or decision.answer_type not in {
            "current_offer", "interest", "hold_price", "redirect", "no_agreement",
            "accepted_offer", "arrangement", "accepted_arrangement",
        }:
            raise ContractError("answer requires a supported answer_type")
        if decision.answer_type != "redirect" and decision.price is None:
            raise ContractError("priced answer requires the current price")
        if decision.referenced_offer_id is not None or decision.finished:
            raise ContractError("answer cannot reference an offer or finish a session")
    elif decision.answer_type is not None:
        raise ContractError("answer_type is only supported for answer")
    reference = decision.referenced_offer_id
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        raise ContractError("referenced_offer_id must be a nonblank string")
    if decision.action in {"accept", "confirm"} and reference is None:
        raise ContractError("referenced_offer_id is required for accept and confirm")
    if decision.action in {"propose", "counter"} and reference is not None:
        raise ContractError("referenced_offer_id is unsupported for propose and counter")
    if decision.action == "clarify":
        reason = decision.clarification_reason
        if not isinstance(reason, str) or reason not in {
            "low_confidence", "unresolved_reference", "state_conflict", "conditional_price", "unclear_message", "collection_day", "price_mismatch"
        }:
            raise ContractError("clarification_reason must be low_confidence, unresolved_reference, "
                                "state_conflict, conditional_price, unclear_message, collection_day, or price_mismatch")
        if (reference is None) != (decision.price is None):
            raise ContractError("specific clarification requires both referenced_offer_id and price")
