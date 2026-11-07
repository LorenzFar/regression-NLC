"""Build the full buyer agent from every team component.

    NLU (Farrell) -> price/reference (Edric) -> gate, policy, tracker (Ercong, Weijie)
    -> retrieval generator (Jiong)

The CLI and the web demo both call build_agent, so they run the same pipeline.
"""
from decimal import Decimal
from pathlib import Path

from price_reference import CachingNLU, ReferenceObserver, ReferenceSafetyGate

from .generator import RetrievalGenerator
from .negotiation import ClarificationFollowUp, ConfidenceGate, OfferTracker, RulePolicy
from .pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CORPUS_BANK = ROOT / "examples" / "craigslist_bank.jsonl"
FIXTURE_BANK = ROOT / "examples" / "approved_fixture_bank.jsonl"
DEFAULT_CHECKPOINT = ROOT / "models" / "nlu_k2_s0.pt"


def default_bank() -> Path:
    """Use the corpus bank when it has been built, else the two authored fixtures."""
    return CORPUS_BANK if CORPUS_BANK.exists() else FIXTURE_BANK


def load_nlu(checkpoint=DEFAULT_CHECKPOINT):
    from .nlu_adapter import NLUAdapter  # imports torch, so only load on demand

    return CachingNLU(NLUAdapter(str(checkpoint)))


def build_agent(nlu, *, listing_price, target, walk_away, fraction="0.5", close_gap="0.05",
                gate_mode="full", threshold=0.7, bank=None, generator=None) -> Pipeline:
    """One buyer pipeline for one listing. gate_mode is off, reference_only or full."""
    listing = Decimal(str(listing_price))
    nlu = nlu if isinstance(nlu, CachingNLU) else CachingNLU(nlu)
    if generator is None:
        generator = RetrievalGenerator(str(bank or default_bank()), allow_provisional=True)
    return Pipeline(
        nlu=nlu,
        price_reference=ReferenceObserver(nlu, listing_price=listing),
        gate=ReferenceSafetyGate(ConfidenceGate(threshold, gate_mode)),
        policy=RulePolicy(target, walk_away, fraction, close_gap, whole_dollars=True),
        tracker=OfferTracker(),
        generator=generator,
        follow_up=ClarificationFollowUp(),
    )
