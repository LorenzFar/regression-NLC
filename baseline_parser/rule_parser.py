"""Team rule-based parser: seller-act inference with the NLU-compatible API.

History is oldest-first (speaker, text); speaker names are case-insensitive.
Context model lookup: explicit ``context_model``, PARSER_CONTEXT_MODEL, then
context_model.json next to this file. ``relabel=True`` (default) is parser v2, the
rule parser plus the training relabel rule; ``relabel=False`` is the original
He et al. (2018) parser (v1). Output is deterministic, so probabilities are
one-hot and ``conf`` is always 1.0; use the NLU for calibrated confidence.
Optional per-call hints: ``listing_price``, ``item_numbers`` (or ``item_title``
and ``item_description``); a missing hint skips the matching price filter.
"""
import hashlib
import os
import platform
from copy import deepcopy
from pathlib import Path

import baseline_parser as bp

ACTS = list(bp.ACTS)
DEFAULT_CONTEXT_MODEL = str(Path(__file__).resolve().parent / "context_model.json")


class RuleParser:
    """Load the context model once; inference is deterministic."""

    def __init__(self, context_model=None, *, relabel=True, use_context_model=True):
        self.relabel = bool(relabel)
        self.context_model = None
        path = None
        if use_context_model:
            path = Path(context_model or os.environ.get("PARSER_CONTEXT_MODEL") or DEFAULT_CONTEXT_MODEL)
            if not path.is_file():
                raise FileNotFoundError(
                    f"context model not found at {str(path)!r}; set PARSER_CONTEXT_MODEL or pass context_model")
            self.context_model = bp.load_context_model(path)
        digest = None
        if path is not None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self._metadata = {
            "component": "rule_based_parser", "fixture": False,
            "version": "v2" if self.relabel else "v1", "relabel": self.relabel,
            "context_model_path": str(path.resolve()) if path else None,
            "context_model_sha256": digest, "labels": list(ACTS),
            "python": platform.python_version(),
        }

    @property
    def resource_metadata(self):
        """Detached provenance snapshot for the resources actually loaded."""
        return deepcopy(self._metadata)

    def _detail(self, history, message, listing_price=None, item_numbers=None,
                item_title=None, item_description=None):
        """Shared inference for the public API."""
        if item_numbers is None and (item_title or item_description):
            item_numbers = bp.get_item_numbers(item_title, item_description)
        return bp.predict_detail(history or [], message, self.relabel, listing_price=listing_price,
                                 item_numbers=item_numbers, context_model=self.context_model)

    def predict_act(self, history, message, **hints):
        """Return the six-act label."""
        return self._detail(history, message, **hints)["act"]

    def predict(self, history, message, **hints):
        """Return a one-hot probability dictionary over the six acts."""
        act = self.predict_act(history, message, **hints)
        return {a: float(a == act) for a in ACTS}

    def predict_top(self, history, message, **hints):
        detail = self._detail(history, message, **hints)
        act = detail["act"]
        return {"act": act, "conf": 1.0, "probs": {a: float(a == act) for a in ACTS},
                "raw_act": detail["raw_act"], "relabelled": detail["relabelled"]}

    def predict_batch(self, items):
        return [self.predict(history, message) for history, message in items]


_default = None


def _get():
    global _default
    if _default is None:
        _default = RuleParser()
    return _default


def predict(history, message, **hints):
    return _get().predict(history, message, **hints)


def predict_top(history, message, **hints):
    return _get().predict_top(history, message, **hints)


def predict_act(history, message, **hints):
    return _get().predict_act(history, message, **hints)


if __name__ == "__main__":
    history = [("BUYER", "Can you do $40?"), ("SELLER", "I can do 45")]
    for message in ["ok that works", "no way, that's too much", "how about 42?"]:
        print(predict_top(history, message, listing_price=50.0))