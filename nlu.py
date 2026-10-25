"""Team NLU: calibrated seller-act inference with the original history-first API.

History is oldest-first (speaker, text); speaker names are case-insensitive.
Checkpoint lookup: explicit ``ckpt``, NLU_CKPT, then nlu_out/nlu_k2_s0.pt.
Public predict/predict_top retain four-decimal probabilities. The buyer adapter
uses the same inference path without rounding.
"""
import hashlib
import math
import os
from pathlib import Path
import platform
import re
from copy import deepcopy

import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

ACTS = ["propose", "counter", "accept", "reject", "inquire", "vague"]
MAX_LEN = 192
BASE_MODEL = "distilbert-base-uncased"
DEFAULT_CKPT = str(Path(__file__).resolve().parent / "nlu_out" / "nlu_k2_s0.pt")

# Identical to training; do not extend without retraining.
_NUM_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
              r"fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)")
_PRICE_RE = re.compile(
    rf"(\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:k|bucks|dollars?)?\b|"
    rf"\b{_NUM_WORDS}(?:[\s-]{_NUM_WORDS})*\b)", re.I)


def mask_prices(text):
    return _PRICE_RE.sub("<PRICE>", text)


class NLU:
    """Load once from config and checkpoint; never download unused base weights."""

    def __init__(self, ckpt=None, device=None, *, model_name=BASE_MODEL,
                 local_files_only=False):
        path = Path(ckpt or os.environ.get("NLU_CKPT") or DEFAULT_CKPT)
        if not path.is_file():
            raise FileNotFoundError(f"NLU checkpoint not found at {str(path)!r}; set NLU_CKPT or pass ckpt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.T, self.k = checkpoint["T"], checkpoint["k"]
        if (type(self.T) not in (int, float) or not math.isfinite(self.T) or self.T <= 0):
            raise ValueError("checkpoint T must be a finite positive number")
        if type(self.k) is not int or self.k < 0:
            raise ValueError("checkpoint k must be a nonnegative integer")
        if checkpoint["acts"] != ACTS:
            raise ValueError("checkpoint acts must match the supplied six-label order")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False, local_files_only=local_files_only)
        config = AutoConfig.from_pretrained(
            model_name, num_labels=len(ACTS), trust_remote_code=False,
            local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=False)
        self.model.load_state_dict(checkpoint["state"], strict=True)
        self.model.to(self.device).eval()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self._metadata = {
            "component": "supplied_checkpoint_nlu", "fixture": False,
            "checkpoint_path": str(path.resolve()), "checkpoint_sha256": digest,
            "checkpoint_bytes": path.stat().st_size,
            "model_name": str(model_name), "model_type": config.model_type,
            "config_commit": getattr(config, "_commit_hash", None),
            "tokenizer_class": type(self.tok).__name__, "device": str(self.device),
            "labels": list(ACTS), "history_window": self.k, "temperature": self.T,
            "max_length": MAX_LEN, "weights_only": True,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": transformers.__version__,
        }

    @property
    def resource_metadata(self):
        """Detached provenance snapshot for the resources actually loaded."""
        return deepcopy(self._metadata)

    def _encode(self, history, message):
        hist = [(str(w).upper(), t) for w, t in (history or [])]
        ctx = " ".join(f"[{w}] {mask_prices(t)}" for w, t in hist[-self.k:]) if self.k > 0 and hist else None
        return self.tok("[SELLER] " + mask_prices(message), ctx,
                        truncation="longest_first", max_length=MAX_LEN,
                        return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def _predict_probabilities(self, history, message):
        """Shared unrounded inference for the public API and strict buyer adapter."""
        logits = self.model(**self._encode(history, message)).logits
        probabilities = (logits / self.T).softmax(-1).squeeze(0)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("NLU produced nonfinite probabilities; check checkpoint weights")
        return dict(zip(ACTS, probabilities.cpu().tolist()))

    def predict(self, history, message):
        """Return the original four-decimal calibrated probability dictionary."""
        return {act: round(probability, 4)
                for act, probability in self._predict_probabilities(history, message).items()}

    def predict_top(self, history, message):
        probs = self.predict(history, message)
        act = max(probs, key=probs.get)
        return {"act": act, "conf": probs[act], "probs": probs}

    def predict_batch(self, items):
        return [self.predict(history, message) for history, message in items]


_default = None


def _get():
    global _default
    if _default is None:
        _default = NLU()
    return _default


def predict(history, message):
    return _get().predict(history, message)


def predict_top(history, message):
    return _get().predict_top(history, message)


if __name__ == "__main__":
    history = [("BUYER", "Can you do $40?"), ("SELLER", "I can do 45")]
    for message in ["ok that works", "no way, that's too much", "how about 42?"]:
        print(predict_top(history, message))
