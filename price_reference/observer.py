"""Pipeline adapter. Sharing CachingNLU avoids a second model forward pass."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from threading import RLock

from .evidence import collect_mentions
from .ledger import _turn, build_ledger
from .resolver import ACTS, resolve


class CachingNLU:
    def __init__(self, nlu, maxsize=128):
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self.nlu, self.maxsize = nlu, maxsize
        self._cache = OrderedDict()
        self._lock = RLock()
        self.provenance = f"cached:{getattr(nlu, 'provenance', type(nlu).__name__)}"

    def clear(self):
        """Call after changing the underlying model, calibration or inference settings."""
        with self._lock:
            self._cache.clear()

    def predict(self, history, message):
        history = [(speaker.upper(), text) for speaker, text in history]
        key = (tuple(history), message)
        with self._lock:
            if key not in self._cache:
                fn = self.nlu.predict if hasattr(self.nlu, "predict") else self.nlu
                self._cache[key] = deepcopy(fn(history, message))
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
            return deepcopy(self._cache[key])


def act_from_probabilities(probabilities):
    if not probabilities or any(k not in ACTS or not isfinite(v) or not 0 <= v <= 1
                                for k, v in probabilities.items()):
        raise ValueError("NLU must return nonempty finite six-act probabilities")
    return max(probabilities, key=probabilities.get)


class ReferenceObserver:
    provenance = "price-reference:rules-v1"

    def __init__(self, nlu=None, listing_price=None):
        self.nlu, self.listing_price = nlu, listing_price

    def observe(self, snapshot, case_id, message, *, act=None):
        listing = self.listing_price if self.listing_price is not None else snapshot.state.get("listing_price")
        ledger = build_ledger(snapshot.history, snapshot.state, listing)
        mentions = collect_mentions(message, listing)
        if act is None and self.nlu is not None:
            history = [(speaker.upper(), text) for speaker, text in map(_turn, snapshot.history)]
            act = act_from_probabilities(self.nlu.predict(history, message))
        result = resolve(message, speaker="seller", act=act, mentions=mentions, ledger=ledger)
        offer = result.offer
        context = None
        if result.status in {"unresolved", "ambiguous", "conflict"}:
            context = {"price": result.context_price,
                       "referenced_offer_id": result.context_offer.offer_id if result.context_offer else None,
                       "candidates": [asdict(c) for c in result.candidates]}
        return {
            "price": result.price, "referenced_offer_id": offer.offer_id if offer else None,
            "reference_resolved": result.status not in {"unresolved", "ambiguous"},
            "state_conflict": result.status == "conflict", "status": result.status,
            "price_source": result.price_source, "offer_owner": offer.owner if offer else None,
            "offer_id_source": offer.id_source if offer else None,
            "offer_liveness": result.offer_liveness, "conditions": list(result.conditions),
            "liveness_established": result.liveness_established,
            "offer_actionable": result.offer_actionable,
            "requires_validation": result.requires_validation,
            "state_history_disagreement": result.state_history_disagreement,
            "state_history_mismatch": result.state_history_disagreement,
            "ledger_state_history_disagreement": ledger.state_history_disagreement,
            "candidates": [asdict(c) for c in result.candidates],
            "act_disagreement": result.act_disagreement, "clarify_context": context,
            "base_offer": asdict(result.base_offer) if result.base_offer else None,
            "source_span": result.source_span, "reason": result.reason, "act_used": act,
            "mentions": [asdict(m) for m in mentions], "version": result.version,
        }
