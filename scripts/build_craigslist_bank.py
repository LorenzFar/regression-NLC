"""Build the buyer response bank from CraigslistBargain training dialogues.

    python scripts/build_craigslist_bank.py artifacts/craigslist/parsed.json \
        --output examples/craigslist_bank.jsonl \
        --heldout-output artifacts/craigslist/heldout_cases.jsonl

Input is the official parsed training file (He et al., 2018), whose messages carry
rule-parser intents and prices. A buyer message that directly answers a seller
message becomes a template when:

  * its parser intent maps to a buyer action (table INTENT_ACTIONS below);
  * propose/counter contain exactly one number, equal to the parser's price;
    accept/confirm contain no number or only that price; reject contains none;
  * it never mentions words from its own item listing, so it reads naturally
    for any item; and
  * it passes bank.prepare_candidate (no conditions, word prices or symbols).

Labels come from the parser, not from people, so rows are marked
review_status="parser_labelled" and load only with allow_provisional=True.
A fixed share of dialogues (by UUID hash) is held out for
scripts/evaluate_generator.py and never enters the bank.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_pipeline.bank import build_bank, prepare_candidate
from retrieval_pipeline.contracts import ContractError

NUMBER = re.compile(r"\d+(?:\.\d+)?")
WORD = re.compile(r"[a-z]+")
# Common words that may appear in listings but are fine in generic replies.
GENERIC = {"good", "great", "nice", "deal", "price", "offer", "condition", "would", "could",
           "that", "this", "with", "have", "your", "will", "just", "like", "work", "works",
           "sounds", "sure", "okay", "thanks", "thank", "much", "more", "take", "make",
           "really", "very", "well", "only", "still", "need", "want", "pick", "time"}


def buyer_action(intent, previous_seller_intent):
    """Map the parser's buyer intent to a buyer action, or None to skip."""
    if intent == "init-price":
        return "propose"
    if intent in {"counter-price", "insist"}:
        return "counter"
    if intent == "agree":
        # Agreeing after the seller agreed confirms the buyer's own offer.
        return "confirm" if previous_seller_intent == "agree" else "accept"
    if intent == "disagree":
        return "reject"
    return None


def price_spans(response, action, parser_price):
    """Return validated spans for the parser price, or None if ineligible."""
    numbers = list(NUMBER.finditer(response))
    if action == "reject":
        return [] if not numbers else None
    if not numbers and action in {"accept", "confirm"}:
        return []
    if len(numbers) != 1 or parser_price is None or float(numbers[0].group()) != float(parser_price):
        return None
    match = numbers[0]
    return [{"start": match.start(), "end": match.end(), "value": match.group(), "validated": True}]


def item_words(scenario):
    item = scenario["kbs"][0]["item"]
    text = " ".join([item.get("Title", ""), *item.get("Description", [])]).lower()
    return {word for word in WORD.findall(text) if len(word) > 3} - GENERIC


def common_title_words(dialogues, min_titles=15):
    """Words used in many listing titles (bike, iphone, bedroom...) name items."""
    counts = Counter()
    for dialogue in dialogues:
        title = dialogue["scenario"]["kbs"][0]["item"].get("Title", "").lower()
        counts.update(set(WORD.findall(title)))
    return {word for word, n in counts.items() if n >= min_titles and len(word) > 2} - GENERIC


# Wording checks on top of the parser label. They are cheap filters, not semantics.
EXTRA_TERMS = re.compile(r"\b(pick|picking|today|tonight|tomorrow|truck|bring|come|venmo|paypal|"
                         r"week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                         r"asap|inside|friend|throw|meet(?! in the middle))\b", re.I)
NEGATIVE = re.compile(r"n't\b|\b(no|not|cannot|too|sorry|unfortunately|pass|afford)\b", re.I)
REJECTION = re.compile(r"\b(can'?t|cannot|won'?t|too (high|much|expensive|steep)|no thanks|"
                       r"not (interested|willing|able)|out of my (budget|range)|unreasonable)\b", re.I)
POSITIVE = re.compile(r"\b(deal|great|awesome|good|perfect|sweet|wonderful|thanks?|yes|ok|okay|"
                      r"problem|worries|fine|sure|sign me up)\b", re.I)
AGREEMENT = re.compile(r"\b(works for me|sounds good|works great|i'?ll take it|it is|is good)\b", re.I)
# "Final offer" wording would be false whenever the policy concedes again later.
COMMITMENT = re.compile(r"\b(final|last|highest|max|maximum|most i can|best i can|firm|"
                        r"can'?t go (above|higher|over|any)|cannot go|no higher|won'?t go)\b", re.I)


def wording_problem(action, response):
    if EXTRA_TERMS.search(response):
        return "extra_terms"
    if action == "reject" and (not REJECTION.search(response) or POSITIVE.search(response)):
        return "not_a_rejection"
    if action in {"accept", "confirm"} and ("?" in response or NEGATIVE.search(response)
                                            or re.search(r"\b(like to offer|how about|would you)\b", response, re.I)):
        return "not_an_agreement"
    if action in {"propose", "counter"} and AGREEMENT.search(response):
        return "sounds_like_agreement"
    if action in {"propose", "counter"} and COMMITMENT.search(response):
        return "claims_final_offer"
    return None


def in_heldout(uuid, percent):
    return int(hashlib.md5(uuid.encode()).hexdigest(), 16) % 100 < percent


def dialogue_turns(dialogue):
    """Yield (event_index, history, seller_turn, buyer_turn) for buyer replies to sellers."""
    roles = {index: kb["personal"]["Role"] for index, kb in enumerate(dialogue["scenario"]["kbs"])}
    messages = []
    for index, event in enumerate(dialogue["events"]):
        if event["action"] == "message" and isinstance(event["data"], str) and event["data"].strip():
            metadata = event.get("metadata") or {}
            messages.append({"index": index, "speaker": roles[event["agent"]],
                             "text": " ".join(event["data"].split()),
                             "intent": metadata.get("intent"), "price": metadata.get("price")})
    for position in range(1, len(messages)):
        seller, buyer = messages[position - 1], messages[position]
        if seller["speaker"] == "seller" and buyer["speaker"] == "buyer":
            history = [{"speaker": m["speaker"], "text": m["text"]} for m in messages[:position - 1]]
            yield buyer["index"], history, seller, buyer


def make_row(dialogue, event_index, history, seller, buyer, history_window):
    """A labelled buyer turn with its price span, or (None, reason)."""
    action = buyer_action(buyer["intent"], seller["intent"])
    # The parser rarely tags refusals; a clear refusal without numbers also counts.
    if action is None and REJECTION.search(buyer["text"]) and not NUMBER.search(buyer["text"]):
        action = "reject"
    if action is None:
        return None, "unmapped_intent"
    # "$80" -> "80": the slot holds a bare decimal; the symbol is dropped.
    response = re.sub(r"\$\s*(?=\d)", "", buyer["text"])
    words = response.split()
    if not 3 <= len(words) <= 25:
        return None, "length"
    spans = price_spans(response, action, buyer["price"])
    if spans is None:
        return None, "price_mismatch"
    row = {
        "template_id": f"craigslist:{dialogue['uuid']}:{event_index}",
        "dialogue_id": dialogue["uuid"], "turn_id": str(event_index),
        "origin": "corpus", "split": "train", "speaker": "BUYER", "action": action,
        "parser_intent": buyer["intent"], "category": dialogue["scenario"].get("category"),
        "mixed_action": False, "unresolved_reference": False, "price_spans": spans,
        # Only the turns retrieval reads are stored; context is identical.
        "history": history[-history_window:] if history_window else [],
        "message": seller["text"], "response": response, "original_response": buyer["text"],
        "review_status": "parser_labelled", "reviewer": "parser:he2018-rules",
        "split_status": "provisional_official_train", "human_approval": False,
    }
    return row, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parsed", help="official parsed training JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-output", help="JSONL of held-out cases for evaluation")
    parser.add_argument("--heldout-percent", type=int, default=10)
    parser.add_argument("--history-window", type=int, default=2)
    args = parser.parse_args(argv)

    dialogues = json.loads(Path(args.parsed).read_text(encoding="utf-8"))
    train_ids = sorted(d["uuid"] for d in dialogues if not in_heldout(d["uuid"], args.heldout_percent))
    allowed = set(train_ids)
    item_nouns = common_title_words(dialogues)
    counts, bank, heldout, seen = Counter(), [], [], set()
    for dialogue in dialogues:
        for event_index, history, seller, buyer in dialogue_turns(dialogue):
            counts["buyer_replies"] += 1
            row, reason = make_row(dialogue, event_index, history, seller, buyer, args.history_window)
            if row is None:
                counts[f"skip:{reason}"] += 1
                continue
            if dialogue["uuid"] not in allowed:
                # Held-out references keep item-specific wording: people write that way.
                heldout.append({**row, "history": history, "split": "heldout"})
                continue
            if set(WORD.findall(row["response"].lower())) & (item_words(dialogue["scenario"]) | item_nouns):
                counts["skip:item_specific"] += 1
                continue
            problem = wording_problem(row["action"], row["response"])
            if problem:
                counts[f"skip:{problem}"] += 1
                continue
            try:
                row = prepare_candidate(row, allowed, args.history_window)
            except ContractError:
                counts["skip:bank_rules"] += 1
                continue
            key = (row["action"], row["template"].lower())
            if key in seen:
                counts["skip:duplicate_template"] += 1
                continue
            seen.add(key)
            row.update(approved=False, reviewer="parser:he2018-rules")
            bank.append(row)

    build_bank(bank, args.output, train_ids, args.history_window, allow_provisional=True)
    if args.heldout_output:
        Path(args.heldout_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.heldout_output).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in heldout), encoding="utf-8")
    counts.update(f"bank:{row['action']}" for row in bank)
    print(json.dumps({"dialogues": len(dialogues), "bank_dialogues": len(train_ids),
                      "templates": len(bank), "heldout_cases": len(heldout),
                      "counts": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
