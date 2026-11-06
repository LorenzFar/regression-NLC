"""Corpus bank builder rules and the generator behaviour that depends on them."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_craigslist_bank as builder  # noqa: E402
from retrieval_pipeline import PolicyDecision, RetrievalGenerator, Turn  # noqa: E402
from retrieval_pipeline.bank import load_bank  # noqa: E402
from retrieval_pipeline.contracts import ContractError  # noqa: E402

BANK = ROOT / "examples" / "craigslist_bank.jsonl"


@pytest.mark.parametrize(("intent", "previous", "action"), [
    ("init-price", "intro", "propose"), ("counter-price", "counter-price", "counter"),
    ("insist", "counter-price", "counter"), ("agree", "counter-price", "accept"),
    ("agree", "agree", "confirm"), ("disagree", "counter-price", "reject"), ("inquiry", "intro", None),
])
def test_parser_intents_map_to_buyer_actions(intent, previous, action):
    assert builder.buyer_action(intent, previous) == action


def test_only_the_parser_price_becomes_the_slot():
    assert builder.price_spans("How about 80?", "counter", 80.0) == [
        {"start": 10, "end": 12, "value": "80", "validated": True}]
    assert builder.price_spans("80 or 90?", "counter", 80.0) is None  # two numbers
    assert builder.price_spans("How about 85?", "counter", 80.0) is None  # parser disagrees
    assert builder.price_spans("Deal!", "accept", None) == []  # agreement needs no number
    assert builder.price_spans("Not for 80", "reject", None) is None


@pytest.mark.parametrize(("action", "text", "problem"), [
    ("counter", "How about {price} if I pick it up today?", "extra_terms"),
    ("reject", "no problem!", "not_a_rejection"),
    ("accept", "Would you do {price}?", "not_an_agreement"),
    ("counter", "{price} works for me", "sounds_like_agreement"),
    ("counter", "Could we meet in the middle at {price}?", None),
    ("reject", "Sorry, that's too much for me.", None),
])
def test_wording_filters(action, text, problem):
    assert builder.wording_problem(action, text) == problem


def test_built_bank_is_provisional_and_loads_only_with_opt_in():
    with pytest.raises(ContractError):
        load_bank(BANK)
    rows = load_bank(BANK, allow_provisional=True)
    assert {row["action"] for row in rows} == {"propose", "counter", "accept", "confirm", "reject"}
    assert all(row["review_status"] == "parser_labelled" and row["origin"] == "corpus" for row in rows)
    manifest = json.loads(BANK.read_text(encoding="utf-8").splitlines()[0])
    assert {row["dialogue_id"] for row in rows} <= set(manifest["train_dialogue_ids"])


def test_corpus_wording_keeps_the_policy_price_and_does_not_repeat():
    generator = RetrievalGenerator(str(BANK), allow_provisional=True)
    history, message = [], "I'm asking 300 for it."
    seen = set()
    for price in ("200.00", "220.50", "235.00"):
        result = generator.generate(PolicyDecision("counter", price), history, message)
        assert result.source == "retrieval" and result.price == price
        shown = price[:-3] if price.endswith(".00") else price
        assert shown in result.text
        assert result.template_id not in seen
        seen.add(result.template_id)
        history += [Turn("seller", message), Turn("buyer", result.text)]


def test_confirm_can_use_accept_wording():
    generator = RetrievalGenerator(str(BANK), allow_provisional=True)
    result = generator.generate(PolicyDecision("confirm", "90.00", "buyer-1", finished=True), [], "ok deal")
    assert result.source == "retrieval"
    assert next(r for r in generator.bank if r["template_id"] == result.template_id)["action"] in {"accept", "confirm"}
