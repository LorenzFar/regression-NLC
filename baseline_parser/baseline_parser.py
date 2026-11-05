"""Rule-based parser baseline for RQ1: a Python 3 port of the He et al. (2018) parser.

``predict(history, message)`` returns one of six acts: propose, counter, accept,
reject, inquire, vague. History is oldest-first (speaker, text); speaker names are
case-insensitive. ``relabel=False`` is the original parser mapped to our acts
(version 1); ``relabel=True`` adds the training relabel rule (version 2).

Differences from the original parser:
1. Tokenising is NLTK word_tokenize on lowercased text, as in the original; "$" is
   glued back onto its number (STICK_DOLLAR_SIGN) so "$50" is one token.
2. Price filters follow PriceTracker.link_entity. A number without "$" needs a left
   and right neighbour seen beside "$" prices in training (context_model), must not
   exceed 1.5x the listing price, and must not appear in the item title or
   description unless it equals the listing price. The context model is trained on
   the train split only. A missing listing price or item numbers skips that filter.
3. Turns the original calls "intro" or "inform" have no match among our six acts, so
   they become ``unmapped`` ("vague" by default).
4. A "nan" token crashes the original; here it is treated as a non-number.

Relabel rule (version 2), the same two fixes as the training notebook:
- an "accept" with no price in the previous turn becomes "vague";
- a short agreement ("okay", "sounds good") right after a price becomes "accept".
The notebook reads the previous turn's price from the dataset price field; here it is
found with a regex on the text, since gold rows have no price field.

Command line:
    python baseline_parser.py gold.jsonl preds.jsonl --context-model context_model.json [--relabel]
    python baseline_parser.py --selftest

Gold rows: {"case_id", "history", "message", "listing_price", "item_title",
"item_description"}. Each output line is {"case_id": ..., "act": ...}.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize

ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")

# Identical to the training notebook; "intro" and "inform" are deliberately unmapped.
INTENT_MAP = {
    "init-price": "propose", "counter-price": "counter", "insist": "counter",
    "agree": "accept", "accept": "accept", "disagree": "reject", "reject": "reject",
    "inquiry": "inquire", "vague-price": "vague", "unknown": "vague",
}

# Rule tables, copied unchanged from the original parser.
QUESTION_WORDS = {"what", "when", "where", "why", "which", "who", "whose", "how", "do", "did",
                  "does", "are", "is", "would", "will", "can", "could", "any"}
NEG_WORDS = {"no", "not", "n't"}
GREETING_WORDS = {"hi", "hello", "hey", "hiya", "howdy"}
GREETING_PATTERNS = [r"how are you", r"interested in"]
PRICE_PATTERNS = [r"come down", r"(highest|lowest)", r"go (lower|higher)", r"too (high|low)"]
AGREEMENT_PATTERNS = [
    r"^that works[.!]*$", r"^great[.!]*$", r"^(ok|okay)[.!]*$", r"^great, thanks[.!]*$",
    r"^deal[.!]*$", r"^[\w ]*have a deal[\w ]*$", r"^i can do that[.]*$",
]

# Relabel rule; identical to the training notebook, do not change without retraining.
_NUM_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|"
              r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)")
NOTEBOOK_PRICE_RE = re.compile(
    rf"(\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:k|bucks|dollars?)?\b|"
    rf"\b{_NUM_WORDS}(?:[\s-]{_NUM_WORDS})*\b)", re.I)
NOTEBOOK_AGREE_RE = re.compile(
    r"\b(ok(ay)?|deal|sounds good|works for me|that works|fine|sure|agreed?|you got it|done)\b", re.I)


# NLTK splits "$50" into "$" and "50", but the price rules need one token, so
# the sign is glued back. Set to False to see raw NLTK output.
STICK_DOLLAR_SIGN = True


def _is_number(s):
    return re.match(r"[.,0-9]+", s) is not None


def _stick_dollar_sign(tokens):
    out, i = [], 0
    while i < len(tokens):  # '$', '5' -> '$5' ; '5', '$' -> '5$'
        tok = tokens[i]
        if tok == "$":
            if i < len(tokens) - 1 and _is_number(tokens[i + 1]):
                out.append(tok + tokens[i + 1])
                i += 2
                continue
            if i > 0 and out and _is_number(tokens[i - 1]):
                out[-1] += tok
                i += 1
                continue
        out.append(tok)
        i += 1
    return out


def tokenize(utterance: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        utterance = utterance.lower()
    try:
        tokens = word_tokenize(utterance)
    except LookupError:  # first run: fetch the tokenizer data, then retry
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(utterance)
    return _stick_dollar_sign(tokens) if STICK_DOLLAR_SIGN else tokens


# Price linking.
class Price:
    """A token recognised as a price; ``type`` is set later from the dialogue state."""
    __slots__ = ("surface", "value", "type")

    def __init__(self, surface, value, type="price"):
        self.surface, self.value, self.type = surface, value, type


def _process_string(token: str) -> str:
    token = re.sub(r"[\$\,]", "", token)
    try:
        if token.endswith("k"):
            token = str(float(token.replace("k", "")) * 1000)
    except ValueError:
        pass
    return token


def train_context_model(utterances) -> dict:
    """Left/right neighbour counts around '$'-marked numbers (original PriceTracker.train).

    Pass the raw utterances of the *training* split only.
    """
    context = {"left": defaultdict(int), "right": defaultdict(int)}
    for text in utterances:
        tokens = ["<s>"] + tokenize(text) + ["</s>"]
        for i, token in enumerate(tokens):
            if token[0] == "$" or token[-1] == "$":
                context["left"][tokens[i - 1]] += 1
                context["right"][tokens[i + 1]] += 1
    return context


def save_context_model(context, path):
    Path(path).write_text(json.dumps({k: dict(v) for k, v in context.items()}), encoding="utf-8")


def load_context_model(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"left": defaultdict(int, data["left"]), "right": defaultdict(int, data["right"])}


def get_item_numbers(title, description):
    """Numbers in the item title and description (original ``get_kb_numbers``).

    ``description`` is a list of strings (as in the original KB) or a single string.
    """
    if isinstance(description, str):
        description = [description]
    title = tokenize(re.sub(r"[^\w0-9\.,]", " ", title or ""))
    description = tokenize(re.sub(r"[^\w0-9\.,]", " ", " ".join(description or [])))
    numbers = set()
    for token in chain(title, description):
        try:
            numbers.add(float(_process_string(token)))
        except ValueError:
            continue
    return numbers


def link_prices(tokens, *, context_model=None, listing_price=None, item_numbers=None):
    """Replace price-like tokens by ``Price`` objects (original ``link_entity``)."""
    padded = ["<s>"] + list(tokens) + ["</s>"]
    out = []
    for i in range(1, len(padded) - 1):
        token = padded[i]
        number = None
        try:
            number = float(_process_string(token))
            has_dollar = token[0] == "$" or token[-1] == "$"
            if not math.isfinite(number):
                number = None
            elif (not has_dollar and context_model is not None
                  and not (padded[i - 1] in context_model["left"]
                           and padded[i + 1] in context_model["right"])):
                number = None
            elif not has_dollar and listing_price is not None and number > 1.5 * listing_price:
                number = None
            elif (not has_dollar and item_numbers and number != listing_price
                  and number in item_numbers):
                number = None
        except ValueError:
            number = None
        out.append(token if number is None else Price(token, number))
    return out


# Tagging and intent rules, from the original parser.
def _is_question(tokens):
    if not tokens:
        return False
    first, last = tokens[0], tokens[-1]
    return last == "?" or (isinstance(first, str) and first in QUESTION_WORDS)


def _is_greeting(tokens, raw):
    if any(isinstance(t, str) and t in GREETING_WORDS for t in tokens):
        return True
    return any(re.search(p, raw, re.IGNORECASE) for p in GREETING_PATTERNS)


def _is_negative(tokens):
    return any(isinstance(t, str) and t in NEG_WORDS for t in tokens)


def _is_vague_price(raw):  # the original search is case-sensitive
    return any(re.search(p, raw) for p in PRICE_PATTERNS)


def _is_agreement(raw):  # matched on the raw, unstripped text
    return any(re.match(p, raw, re.IGNORECASE) for p in AGREEMENT_PATTERNS)


def _tag(raw, tokens):
    tags = []
    if _is_question(tokens):
        tags.append("question")
    if _is_greeting(tokens, raw):
        tags.append("greeting")
    if _is_negative(tokens):
        tags.append("negative")
    if _is_vague_price(raw):
        tags.append("vague-price")
    if _is_agreement(raw):
        tags.append("agree")
    types = list({t.type for t in tokens if isinstance(t, Price)})
    tags.extend(types)
    if types:
        tags.append("has_price")
    return tags


def _classify(tags, time, curr_price, my_act):
    if time == 0 and "has_price" not in tags:
        return "intro"
    if "has_price" in tags and curr_price is None:
        return "init-price"
    if "has_price" not in tags and "vague-price" in tags:
        return "vague-price"
    if "price" in tags or "listing_price" in tags:
        return "counter-price"
    if "partner_price" in tags:
        return "insist"
    if tags == ["question"]:
        return "inquiry"
    if "agree" in tags or "my_price" in tags:
        return "agree"
    if not tags and my_act == "inquiry":
        return "inform"
    if tags == ["negative"]:
        return "disagree"
    return "unknown"


def _compare(x, y, inc):
    r = 0 if x == y else (-1 if x < y else 1)
    return r * inc


def _proposed_price(tokens, speaker, speaker_prev_price):
    """Which price the speaker meant when several are mentioned (original rule)."""
    inc = 1.0 if speaker == "seller" else -1.0
    candidates = []
    for t in tokens:
        if not isinstance(t, Price):
            continue
        if ((speaker_prev_price is None or _compare(t.value, speaker_prev_price, inc) <= 0)
                and (speaker == "seller" or t.type != "listing_price")):
            candidates.append(t.value)
    if not candidates:
        return None
    return candidates[max(range(len(candidates)), key=lambda i: inc * candidates[i])]


def classify_dialogue(turns, *, listing_price=None, item_numbers=None, context_model=None):
    """Original parser intent for every ``(speaker, text)`` turn, read by the other side."""
    state = {"price": {"buyer": None, "seller": None}, "curr": None,
             "act": {"buyer": "<start>", "seller": "<start>"}}
    intents = []
    for time, (speaker, text) in enumerate(turns):
        reader = "buyer" if speaker == "seller" else "seller"
        tokens = link_prices(tokenize(text), context_model=context_model,
                             listing_price=listing_price, item_numbers=item_numbers)
        for t in tokens:
            if isinstance(t, Price):
                if listing_price is not None and t.value == listing_price:
                    t.type = "listing_price"
                elif t.value == state["price"][reader]:
                    t.type = "my_price"
                elif t.value == state["price"][speaker]:
                    t.type = "partner_price"
                else:
                    t.type = "price"
        intent = _classify(_tag(text, tokens), time, state["curr"], state["act"][reader])
        price = _proposed_price(tokens, speaker, state["price"][speaker])
        if price is not None:
            state["price"][speaker] = price
            state["curr"] = price
        state["act"][speaker] = intent
        intents.append(intent)
    return intents


# Public API.
def _normalise(history):
    turns = []
    for turn in history or []:
        speaker, text = (turn["speaker"], turn["text"]) if isinstance(turn, dict) else turn
        speaker = str(speaker).lower()
        if speaker not in ("buyer", "seller") or not isinstance(text, str):
            raise ValueError("history turns must be (BUYER|SELLER, text) tuples or {speaker, text} dicts")
        turns.append((speaker, text))
    return turns


def predict_detail(history, message, relabel=False, *, speaker="seller", unmapped="vague",
                   listing_price=None, item_numbers=None, context_model=None) -> dict:
    """Return the parser intent, the six-act label and whether relabelling changed it."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a nonempty string")
    if unmapped not in ACTS:
        raise ValueError(f"unmapped must be one of {list(ACTS)}")
    turns = _normalise(history)
    intent = classify_dialogue(turns + [(speaker.lower(), message)], listing_price=listing_price,
                               item_numbers=item_numbers, context_model=context_model)[-1]
    act = INTENT_MAP.get(intent, unmapped)
    raw_act, relabelled = act, False
    if relabel:
        prev_has_price = bool(turns) and NOTEBOOK_PRICE_RE.search(turns[-1][1]) is not None
        if act == "accept" and not prev_has_price:
            act = "vague"
        elif (act in ("vague", "inquire") and prev_has_price and len(message.split()) <= 6
              and NOTEBOOK_AGREE_RE.search(message) and "?" not in message):
            act = "accept"
        relabelled = act != raw_act
    return {"intent": intent, "act": act, "raw_act": raw_act, "relabelled": relabelled}


def predict(history, message, relabel=False, **kwargs) -> str:
    """Six-act label. ``relabel=False`` is the raw parser; ``True`` is version 2."""
    return predict_detail(history, message, relabel, **kwargs)["act"]


# Command line and self-test.
_SELFTEST_DIALOGUE = [  # first training dialogue in the notebook; expected = dataset intent
    ("buyer", "Hi, not sure if the charger would work for my car. Can you sell it to me for $5?", "init-price"),
    ("seller", "It will work, i have never seen a car without a cigarette lighter port.", "unknown"),
    ("buyer", "Still, can I buy it for $5? I'm on a tight budge", "insist"),
    ("seller", "I think the lowest I would want to go is 8. ", "counter-price"),
    ("buyer", "How about $6 and I pick it up myself? It'll save you shipping to me.", "counter-price"),
    ("seller", "7, and we have a deal.", "counter-price"),
    ("buyer", "Eh, fine. $7.", "agree"),
]


def selftest() -> int:
    got = classify_dialogue([(s, t) for s, t, _ in _SELFTEST_DIALOGUE])
    want = [e for *_, e in _SELFTEST_DIALOGUE]
    for (speaker, text, expected), actual in zip(_SELFTEST_DIALOGUE, got):
        print(f"{'ok ' if actual == expected else 'BAD'} {speaker:6s} expected={expected:13s} got={actual:13s} | {text[:50]}")
    checks = [
        predict([("BUYER", "Can you do $40?"), ("SELLER", "I can do 45")], "ok") == "accept",
        predict([], "ok") == "vague",  # nothing to accept; also hits the 'intro' rule
        predict([("BUYER", "Can you do $40?")], "okay sounds good", relabel=False) == "vague",
        predict([("BUYER", "Can you do $40?")], "okay sounds good", relabel=True) == "accept",
    ]
    ok = got == want and all(checks)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold", nargs="?", help="gold JSONL (case_id, history, message)")
    parser.add_argument("output", nargs="?", help="predictions JSONL to write")
    parser.add_argument("--relabel", action="store_true", help="version 2: apply the notebook relabel rule")
    parser.add_argument("--context-model", help="context_model.json trained on the train split")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.gold or not args.output:
        parser.error("gold and output are required (or use --selftest)")
    context_model = load_context_model(args.context_model) if args.context_model else None
    if context_model is None:
        print("warning: no --context-model, so the bare-number context filter is off", file=sys.stderr)
    rows = []
    for number, line in enumerate(Path(args.gold).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                listing_price = row.get("listing_price")
                item_numbers = (get_item_numbers(row.get("item_title"), row.get("item_description"))
                                if row.get("item_title") or row.get("item_description") else None)
                rows.append({"case_id": row["case_id"],
                             "act": predict(row.get("history", []), row["message"], args.relabel,
                                            listing_price=listing_price, item_numbers=item_numbers,
                                            context_model=context_model)})
            except (ValueError, KeyError) as exc:
                print(f"{args.gold}:{number}: {exc!r}", file=sys.stderr)
                return 2
    Path(args.output).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} predictions to {args.output} (relabel={args.relabel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())