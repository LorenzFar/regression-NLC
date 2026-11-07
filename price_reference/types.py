"""Shared handoff types; amounts are exact decimal strings, as in retrieval_pipeline."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re


@dataclass(frozen=True)
class PriceMention:
    """One money amount found in a message. Says nothing about whether it is an offer.

    start/end are Python character offsets into the message (end exclusive) and
    cover the whole surface form, including "$", "k" or "hundred".
    kind: "absolute" ("$500"), "delta" ("50 less", "$20 off", "reduce it by 200") or
    "reference" (a comparison price: "I paid $600", "retails for 150").
    plausible: whether the amount fits the listing price; None when no listing price.
    """
    value: str
    start: int
    end: int
    surface: str
    kind: str = "absolute"
    has_currency: bool = False
    plausible: bool | None = None


@dataclass(frozen=True)
class Candidate:
    """Every number-like span the extractor looked at, kept or rejected (for debugging)."""
    surface: str
    start: int
    end: int
    value: str | None
    reject_reason: str | None
    mention: PriceMention | None = None


RESOLVER_VERSION = "1"
OWNERS = {"buyer", "seller"}
LIVENESS = {"live", "not_live", "unknown"}


def money(value) -> str:
    """Finite nonnegative decimal, rounded half-up to cents (never through float)."""
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("price must be finite and nonnegative")
    return format(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def _price(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d{2}", value):
        raise ValueError("price must be a nonnegative two-decimal string")


def _span(span):
    if span is not None and (len(span) != 2 or any(type(x) is not int for x in span)
                             or not 0 <= span[0] < span[1]):
        raise ValueError("invalid source span")


@dataclass(frozen=True)
class OfferRef:
    offer_id: str
    owner: str
    turn_index: int | None = None
    id_source: str = "minted"

    def __post_init__(self):
        if not isinstance(self.offer_id, str) or not self.offer_id.strip() or self.owner not in OWNERS:
            raise ValueError("offer needs nonblank id and buyer/seller owner")
        if self.id_source not in {"state", "minted"}:
            raise ValueError("unknown offer id source")
        if self.turn_index is not None and (type(self.turn_index) is not int or self.turn_index < 0):
            raise ValueError("invalid turn index")


@dataclass(frozen=True)
class Offer:
    ref: OfferRef
    price: str
    liveness: str = "unknown"
    link: str = "history"
    state_history_disagreement: bool = False

    @property
    def state_history_mismatch(self):
        """Compatibility alias; disagreement is advisory, not proof of stale state."""
        return self.state_history_disagreement

    def __post_init__(self):
        _price(self.price)
        if self.liveness not in LIVENESS:
            raise ValueError("unknown liveness")
        if self.link not in {"history", "state_linked_by_price", "state_only"}:
            raise ValueError("unknown offer link")
        if self.liveness == "live" and self.link == "history":
            raise ValueError("history alone cannot prove liveness")


@dataclass(frozen=True)
class PriceCandidate:
    price: str
    span: tuple[int, int] | None = None
    role: str = "primary"
    condition: str | None = None

    def __post_init__(self):
        _price(self.price)
        _span(self.span)
        if self.role not in {"primary", "conditional", "alternative"}:
            raise ValueError("unknown candidate role")

    def validate(self, text):
        if self.span is not None and self.span[1] > len(text):
            raise ValueError("candidate span exceeds message bounds")


@dataclass(frozen=True)
class AmbiguousTurn:
    ref: OfferRef
    candidates: tuple[PriceCandidate, ...]
    reason: str = "multiple_prices"

    def __post_init__(self):
        if len(self.candidates) < 2:
            raise ValueError("ambiguous turn needs at least two candidates")


@dataclass(frozen=True)
class OfferLedger:
    offers: tuple[Offer, ...] = ()
    ambiguous_turns: tuple[AmbiguousTurn, ...] = ()
    pending_acceptance: OfferRef | None = None

    def __post_init__(self):
        if self.pending_acceptance is not None:
            live = [offer for offer in self.offers if offer.liveness == "live"]
            if len(live) != 1 or live[0].ref != self.pending_acceptance:
                raise ValueError("pending acceptance must identify the single tracker-live offer")

    @property
    def state_history_disagreement(self):
        return any(o.state_history_disagreement for o in self.offers)

    @property
    def state_history_mismatch(self):
        return self.state_history_disagreement


@dataclass(frozen=True)
class ResolverMention:
    value: str
    kind: str = "absolute"
    provenance: str = "extractor"
    surface: str | None = None
    start: int | None = None
    end: int | None = None
    has_currency: bool | None = None
    plausible: bool | None = None

    def __post_init__(self):
        _price(self.value)
        if self.kind not in {"absolute", "delta", "reference", "unknown"}:
            raise ValueError("unknown mention kind")
        if self.provenance not in {"extractor", "gold_annotated", "gold_csv_value", "gold_inferred"}:
            raise ValueError("unknown mention provenance")
        if self.kind == "unknown" and self.provenance not in {"gold_csv_value", "gold_inferred"}:
            raise ValueError("unknown kind requires CSV/inferred provenance")
        if (self.start is None) != (self.end is None):
            raise ValueError("mention offsets must be both present or both absent")
        if self.start is not None:
            _span((self.start, self.end))
            if self.surface is None:
                raise ValueError("spanned mention needs its original surface")

    def validate(self, text):
        if self.start is not None and (self.end > len(text) or text[self.start:self.end] != self.surface):
            raise ValueError("mention span does not match original text")


def mention_from_price_mention(mention: PriceMention) -> ResolverMention:
    return ResolverMention(mention.value, mention.kind, "extractor", mention.surface,
                           mention.start, mention.end, mention.has_currency, mention.plausible)


@dataclass(frozen=True)
class Resolution:
    status: str
    price: str | None = None
    price_source: str | None = None
    offer: OfferRef | None = None
    offer_liveness: str | None = None
    base_offer: OfferRef | None = None
    source_span: tuple[int, int] | None = None
    candidates: tuple[PriceCandidate, ...] = ()
    conditions: tuple[str, ...] = ()
    act_disagreement: bool = False
    context_price: str | None = None
    context_offer: OfferRef | None = None
    reason: str | None = None
    version: str = RESOLVER_VERSION
    state_history_disagreement: bool = False

    @property
    def state_history_mismatch(self):
        return self.state_history_disagreement

    @property
    def liveness_established(self):
        """Tracker-established liveness only; NOT permission to accept or confirm."""
        return (self.status == "resolved" and self.price_source == "referenced_offer"
                and self.offer_liveness == "live")

    @property
    def offer_actionable(self):
        """Deprecated compatibility eligibility hint, never authorization."""
        return self.liveness_established and not self.requires_validation

    @property
    def requires_validation(self):
        return (self.status in {"unresolved", "conflict", "ambiguous"}
                or self.state_history_disagreement or bool(self.conditions)
                or self.price_source == "referenced_offer" and self.offer_liveness != "live")

    def __post_init__(self):
        if self.status not in {"resolved", "no_price", "unresolved", "conflict", "ambiguous"}:
            raise ValueError("unknown resolution status")
        if (self.price is not None) != (self.status == "resolved"):
            raise ValueError("price must be present iff resolved")
        if self.price is not None:
            _price(self.price)
            if self.price_source not in {"explicit", "delta", "referenced_offer"}:
                raise ValueError("resolved price requires provenance")
        elif self.price_source is not None or self.offer is not None or self.base_offer is not None:
            raise ValueError("unresolved result cannot expose an actionable reference")
        if self.price_source == "referenced_offer" and (self.offer is None or self.offer_liveness not in LIVENESS):
            raise ValueError("referenced price needs identity and liveness")
        if self.price_source != "referenced_offer" and (self.offer is not None or self.offer_liveness is not None):
            raise ValueError("only referenced prices expose a prior offer and liveness")
        if self.price_source == "delta" and self.base_offer is None:
            raise ValueError("delta needs base offer identity")
        if self.status == "ambiguous" and len(self.candidates) < 2:
            raise ValueError("ambiguous result needs at least two candidates")
        if self.context_price is not None:
            _price(self.context_price)
        if (self.context_price is not None or self.context_offer is not None) and self.status not in {
                "unresolved", "ambiguous", "conflict"}:
            raise ValueError("clarification context on non-clarification result")
        _span(self.source_span)
