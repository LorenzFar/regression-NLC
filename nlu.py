"""
nlu.py - CS425 Team 2 NLU stage (importable, no notebook needed).

COPY THE NLU_OUT FROM THE FOLDER I SENT IN THE GROUP!!

Quick start
-----------
    from nlu import predict, predict_top

    history = [("BUYER", "Can you do $40?"), ("SELLER", "I can do 45")]
    predict(history, "ok that works")
    # {'propose': 0.0014, 'counter': 0.0062, 'accept': 0.9785, ...}

    predict_top(history, "ok that works")
    # {'act': 'accept', 'conf': 0.9785, 'probs': {...}}

Notes
-----
* history is a list of (speaker, text), oldest first, speaker is "BUYER" or
  "SELLER" (case-insensitive). Only the last k turns are used (k comes from
  the checkpoint, k=2 for nlu_k2_s0.pt). An empty list is fine.
* message is the SELLER's latest message (raw text; prices are masked inside).
* Returned probabilities are temperature-calibrated (T stored in the checkpoint).
* The checkpoint is looked up in this order:
    1. the `ckpt` argument / NLU(ckpt=...)
    2. env var NLU_CKPT
    3. nlu_out/nlu_k2_s0.pt next to this file
"""
import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ACTS = ["propose", "counter", "accept", "reject", "inquire", "vague"]
MAX_LEN = 192
BASE_MODEL = "distilbert-base-uncased"
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "nlu_out", "nlu_k2_s0.pt")

# Price masking, identical to training (do not change without retraining)
_NUM_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
              r"fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)")
_PRICE_RE = re.compile(
    rf"(\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:k|bucks|dollars?)?\b|"
    rf"\b{_NUM_WORDS}(?:[\s-]{_NUM_WORDS})*\b)", re.I)


def mask_prices(text):
    return _PRICE_RE.sub("<PRICE>", text)


class NLU:
    """Loads one checkpoint once; call .predict(history, message)."""

    def __init__(self, ckpt=None, device=None):
        ckpt = ckpt or os.environ.get("NLU_CKPT") or DEFAULT_CKPT
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"NLU checkpoint not found at {ckpt!r}. Put nlu_k2_s0.pt in nlu_out/ "
                "or set the NLU_CKPT environment variable.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.device)
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(ACTS))
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.T, self.k = float(ck["T"]), int(ck["k"])

    def _encode(self, history, message):
        hist = [(str(w).upper(), t) for w, t in (history or [])]
        ctx = " ".join(f"[{w}] {mask_prices(t)}" for w, t in hist[-self.k:]) \
            if self.k > 0 and hist else None
        return self.tok("[SELLER] " + mask_prices(message), ctx,
                        truncation="longest_first", max_length=MAX_LEN,
                        return_tensors="pt").to(self.device)

    @torch.no_grad()
    def predict(self, history, message):
        """Return {act: calibrated probability} for the seller's message."""
        logits = self.model(**self._encode(history, message)).logits
        p = (logits / self.T).softmax(-1).squeeze(0).cpu().numpy()
        return dict(zip(ACTS, p.round(4).tolist()))

    def predict_top(self, history, message):
        """Return {'act', 'conf', 'probs'}; conf is the top calibrated probability."""
        probs = self.predict(history, message)
        act = max(probs, key=probs.get)
        return {"act": act, "conf": probs[act], "probs": probs}

    def predict_batch(self, items):
        """items: list of (history, message). Returns a list of prob dicts."""
        return [self.predict(h, m) for h, m in items]


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
    h = [("BUYER", "Can you do $40?"), ("SELLER", "I can do 45")]
    for msg in ["ok that works", "no way, that's too much", "how about 42?"]:
        r = predict_top(h, msg)
        print(f"{r['act']:8s} {r['conf']:.2f} | {msg}")
    r = predict_top([], "ok that works")
    print(f"{r['act']:8s} {r['conf']:.2f} | ok that works (no history)")