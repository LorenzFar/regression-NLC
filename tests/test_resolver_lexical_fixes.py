"""D1-D6 review regressions; protect real offers while removing comparison bounds."""
import pytest

from evaluation.resolver.run import evaluate
from price_reference import CachingNLU, ReferenceObserver, build_ledger, resolve
from price_reference.evidence import collect_mentions
from retrieval_pipeline.pipeline import Session

ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")


def run(text, act="counter", history=(), speaker="seller"):
    return resolve(text, speaker=speaker, act=act, mentions=collect_mentions(text), ledger=build_ledger(history))


@pytest.mark.parametrize("text", ["What do you mean? I can go 85",
                                  "What do you mean by as-is? 85 is my max",
                                  "Could you clarify 90? I could go 85",
                                  "What do you mean? 85 cash today",
                                  "I accept your offer of 90. How about 85 instead?"])
def test_d1_later_genuine_offer_survives_without_a_whole_turn_lexicon_gate(text):
    history = [("seller", "asking 100"), ("buyer", "Would you take 80?"),
               ("seller", "No, 90"), ("buyer", text)]
    ledger = build_ledger(history)
    buyer = [offer for offer in ledger.offers if offer.ref.owner == "buyer"]
    assert [offer.price for offer in buyer] == ["80.00", "85.00"]
    assert buyer[-1].ref.offer_id == "buyer-offer-2"
    answer = run("ok deal", "accept", history)
    assert answer.price == "85.00" and answer.offer == buyer[-1].ref


@pytest.mark.parametrize("text", ["What do you mean? I can go 85", "What do you mean by as-is? 85 is my max"])
@pytest.mark.parametrize("act", ACTS)
def test_d1_current_message_offer_survives_act_errors(text, act):
    assert run(text, act, speaker="buyer").price == "85.00"


@pytest.mark.parametrize("text", ["Can you go higher than 450?", "Could you do better than 450?",
                                  "Can you do any better than 450?", "Would you go up from 450?",
                                  "Can you go lower than 450?", "Could you do less than 450?",
                                  "Can you go above 450?", "Can you go below 450?"])
@pytest.mark.parametrize("act", ACTS)
def test_d2_comparison_bound_is_never_a_proposed_price(text, act):
    answer = run(text, act, [("buyer", "450")])
    assert answer.price is None
    assert not build_ledger([("seller", text)]).offers


def test_d2_comparison_question_never_supersedes_seller_offer():
    ledger = build_ledger([("seller", "asking 100"), ("buyer", "85"),
                           ("seller", "Can you go higher than 85?")])
    assert [offer.price for offer in ledger.offers if offer.ref.owner == "seller"] == ["100.00"]
    assert ledger.offers[0].liveness == "unknown"
    answer = run("Can you go higher than 450? I can do 480")
    assert answer.price == "480.00"


@pytest.mark.parametrize("text", ["I won't accept anything less than 450", "I'm not taking less than 450",
                                  "I won't accept any less than 450", "I'm not accepting anything below 450",
                                  "I cannot take any amount under 450"])
@pytest.mark.parametrize("act", ACTS)
def test_d3_seller_floors_are_not_demoted_as_comparison_bounds_or_refusals(text, act):
    answer = run(text, act)
    assert answer.price == "450.00" and answer.status == "resolved"
    assert build_ledger([("seller", text)]).offers[0].price == "450.00"


@pytest.mark.parametrize("text", ["Is 450 ok? No, it isn't", "I can do 450. No, that won't work for me.",
                                  "450 or 500? No, it is not acceptable."])
@pytest.mark.parametrize("act", ACTS)
def test_d4_explicit_later_cancellation_never_resolves_withdrawn_price(text, act):
    answer = run(text, act, [("seller", "450"), ("buyer", "450")])
    assert answer.price is None and answer.status == "no_price"
    assert not build_ledger([("seller", text)]).offers


@pytest.mark.parametrize("text", ["Is 450 ok? No, make it 500", "I can do 450. No, how about 500 instead?"])
@pytest.mark.parametrize("act", ACTS)
def test_d4_replacement_price_survives_cancellation_of_earlier_price(text, act):
    assert run(text, act).price == "500.00"
    assert [offer.price for offer in build_ledger([("seller", text)]).offers] == ["500.00"]


def test_d4_item_condition_negation_does_not_cancel_price():
    answer = run("I can do 450. No, it isn't broken.")
    assert answer.price == "450.00"


class ActNLU:
    def __init__(self, act):
        self.act = act

    def predict(self, history, message):
        return {self.act: 1.0}


def test_d5_ledger_disagreement_does_not_contaminate_unrelated_explicit_offer():
    snapshot = Session("new", history=[("buyer", "60"), ("buyer", "70")], state={
        "buyer_offer_id": "buyer-1", "buyer_price": "60", "agreement": "pending"}).snapshot()
    output = ReferenceObserver(CachingNLU(ActNLU("counter"))).observe(snapshot, "c", "I can do 90")
    assert output["price"] == "90.00" and output["ledger_state_history_disagreement"]
    assert not output["state_history_disagreement"] and not output["requires_validation"]
    referenced = ReferenceObserver(CachingNLU(ActNLU("accept"))).observe(snapshot, "c", "ok deal")
    assert referenced["price"] == "60.00" and referenced["state_history_disagreement"]
    assert referenced["requires_validation"]


def test_d5_other_owners_disagreement_does_not_contaminate_validated_reference():
    snapshot = Session("other", history=[("seller", "90"), ("seller", "100"), ("buyer", "60")], state={
        "seller_offer_id": "seller-1", "seller_price": "90", "buyer_offer_id": "buyer-1",
        "buyer_price": "60", "agreement": "pending"}).snapshot()
    output = ReferenceObserver(CachingNLU(ActNLU("accept"))).observe(snapshot, "c", "ok deal")
    assert output["ledger_state_history_disagreement"] and not output["state_history_disagreement"]
    assert output["liveness_established"] and not output["requires_validation"]


def test_d6_pairwise_metrics_remain_visible_when_another_condition_is_unavailable():
    row = {"case_id": "one", "history": [], "message": "I can do 50", "act": "counter",
           "price": "50", "gold_price": "50.00", "offers": [], "label_source": "ai_proposed"}
    report = evaluate([row], conditions=("A", "C", "D"))
    assert report["common_subset"]["n"] == 0  # all-selected intersection is intentionally unchanged
    comparison = report["pairwise_subsets"]["A:D"]
    assert comparison["n"] == 1 and comparison["case_ids"] == ["one"]
    for condition in ("A", "D"):
        metrics = comparison["conditions"][condition]["by_label_source"]["ai_proposed"]
        assert metrics["price_exact_match"] == 1 and metrics["price_n"] == 1
    assert report["pairwise_subsets"]["D:C"]["n"] == 0
