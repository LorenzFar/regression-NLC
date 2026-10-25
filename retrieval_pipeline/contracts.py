"""Proposed immutable handoff contracts; amounts are exact SGD decimal strings."""
from dataclasses import dataclass


class ContractError(ValueError):
    """A generation request violates the proposed public contract."""


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    price: str | None = None
    referenced_offer_id: str | None = None
    clarification_reason: str | None = None
    finished: bool = False


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
