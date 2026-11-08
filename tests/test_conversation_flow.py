"""Buyer opening, clarification follow-up and offer confidence through build_agent."""
from pathlib import Path

import pytest

from retrieval_pipeline.agent import build_agent
from retrieval_pipeline.contracts import ContractError
from retrieval_pipeline.pipeline import Session

BANK = Path(__file__).resolve().parents[1] / "examples" / "craigslist_bank.jsonl"
ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")


class FakeNLU:
    """Fixed probabilities per message; unknown messages are confident vague."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, history, message):
        probabilities = dict.fromkeys(ACTS, 0.0)
        probabilities.update(self.scores.get(message, {"vague": 1.0}))
        return probabilities


def agent(scores):
    return build_agent(FakeNLU(scores), listing_price="300", target="200", walk_away="260", bank=BANK)


def session():
    return Session("s", state={"listing_price": "300"})


def test_buyer_opens_with_a_greeting_and_the_target_price():
    pipeline, live = agent({}), session()
    result = pipeline.open(live, context="City road bike, aluminium frame")
    assert result.generation.action == "propose" and result.generation.price == "200.00"
    assert result.generation.text.startswith("Hi! I saw your listing.") and "200" in result.generation.text
    offers = live.snapshot().state["offers"]
    assert [(o["by"], o["terms"]["price"], o["status"]) for o in offers] == [
        ("seller", "300.00", "superseded"), ("buyer", "200.00", "live")]
    with pytest.raises(ContractError):
        pipeline.open(live, context="again")


def test_seller_ok_after_the_opening_confirms_the_buyer_offer():
    pipeline, live = agent({"ok deal": {"accept": 0.95}}), session()
    pipeline.open(live, context="City road bike")
    result = pipeline.turn(live, "ok deal", case_id="c")
    assert (result.generation.action, result.generation.price) == ("confirm", "200.00")
    assert live.snapshot().finished


def test_propose_counter_split_is_not_low_confidence():
    pipeline, live = agent({"I'm asking 300 for it.": {"propose": 0.51, "counter": 0.45, "vague": 0.04}}), session()
    result = pipeline.turn(live, "I'm asking 300 for it.", case_id="c")
    assert list(result.trace["gate"]["reasons"]) == []
    assert result.generation.action == "counter"


def test_yes_after_a_clarification_confirms_the_price_we_asked_about():
    scores = {"I can do 250.": {"counter": 0.4, "vague": 0.6 - 0.01, "inquire": 0.01}, "yes": {"accept": 0.9}}
    pipeline, live = agent(scores), session()
    asked = pipeline.turn(live, "I can do 250.", case_id="c1")
    assert asked.generation.action == "clarify" and asked.generation.price == "250.00"
    assert "250" in asked.generation.text
    answer = pipeline.turn(live, "yes", case_id="c2")
    assert answer.trace["follow_up_applied"] is True
    # The seller's 250 is now on record and the buyer responds to it.
    assert answer.generation.action == "counter" and answer.generation.price == "200.00"
    state = live.snapshot().state
    assert "pending_clarification" not in state
    assert [o["terms"]["price"] for o in state["offers"] if o["by"] == "seller"] == ["250.00"]


def test_ok_after_a_clarification_confirms_the_asked_price_not_our_old_offer():
    scores = {"I can do 280.": {"counter": 0.4, "vague": 0.6}, "ok": {"accept": 0.95}}
    pipeline, live = agent(scores), session()
    pipeline.open(live, context="City road bike")
    pipeline.turn(live, "I can do 280.", case_id="c1")
    answer = pipeline.turn(live, "ok", case_id="c2")
    # Without the follow-up, "ok" would confirm the buyer's opening 200.
    assert answer.generation.action != "confirm"
    assert [o["terms"]["price"] for o in live.snapshot().state["offers"] if o["by"] == "seller"][-1] == "280.00"


def test_conditional_price_gets_a_question_about_that_price():
    pipeline, live = agent({"250 if you can collect today": {"accept": 0.63, "counter": 0.35}}), session()
    pipeline.open(live, context="City road bike")
    result = pipeline.turn(live, "250 if you can collect today", case_id="c")
    assert "conditional_price" in result.trace["gate"]["reasons"]
    assert result.generation.text == "Would you accept 250 without that condition?"


def test_a_new_price_after_a_clarification_is_read_normally():
    scores = {"I can do 250.": {"vague": 0.6, "counter": 0.4}, "no, 240": {"counter": 0.95}}
    pipeline, live = agent(scores), session()
    pipeline.turn(live, "I can do 250.", case_id="c1")
    answer = pipeline.turn(live, "no, 240", case_id="c2")
    assert answer.trace["follow_up_applied"] is False
    assert answer.trace["price_reference"]["price"] == "240.00"
