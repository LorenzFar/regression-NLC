"""Safe composition with the teammate's optional negotiation pipeline."""
from .gate import ReferenceSafetyGate
from .observer import CachingNLU, ReferenceObserver


def build_safe_buyer_pipeline(nlu, *, listing_price=None, **kwargs):
    """Build the teammate pipeline with shared NLU and full resolver safety checks.

    Requires retrieval_pipeline.negotiation from the teammate integration. Import
    it lazily so extraction and resolution remain usable before that merge.
    Other keyword arguments are passed to the teammate's build_buyer_pipeline.
    Experimental off/reference_only modes must use that lower-level builder.
    """
    mode = kwargs.pop("mode", "full")
    if mode != "full":
        raise ValueError("build_safe_buyer_pipeline requires mode='full'")
    from retrieval_pipeline.negotiation import build_buyer_pipeline

    shared = nlu if isinstance(nlu, CachingNLU) else CachingNLU(nlu)
    pipeline = build_buyer_pipeline(
        shared, ReferenceObserver(shared, listing_price=listing_price),
        mode="full", **kwargs)
    pipeline.gate = ReferenceSafetyGate(pipeline.gate)
    return pipeline
