"""Independent-review counterexamples, including cases where proposed fixes overreach."""
from dataclasses import asdict

import pytest

from evaluation.resolver.metrics import wilson_interval
from evaluation.resolver.run import evaluate
from price_reference import (AmbiguousTurn, CachingNLU, Offer, OfferLedger, OfferRef,
                             PriceCandidate, ReferenceObserver, build_ledger, resolve)
from price_reference.evidence import collect_mentions
from retrieval_pipeline.contracts import PolicyDecision, Turn
from retrieval_pipeline.fixtures import FixtureCase, ScriptedComponents
from retrieval_pipeline.pipeline import Pipeline, Session
from scripts.resolver_gold_seed import seed

ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")


def result(text, act="counter", history=(), state=None, speaker="seller"):
    return resolve(text, speaker=speaker, act=act, mentions=collect_mentions(text),
                   ledger=build_ledger(history, state))


@pytest.mark.parametrize("clarification", [
    "Could you clarify what you mean about the offer of 90.00?",
    "Are you referring to the offer of 90.00?",
    "Could you clarify the status of the offer of 90.00?",
    "What do you mean by 90?",
    "Is your price 90?",
    "You are asking 90, right?",
])
def test_clarification_and_quoted_price_never_become_phantom_buyer_offers(clarification):
    history = [("seller", "asking 100"), ("buyer", "Would you take 80?"),
               ("seller", "No, 90"), ("buyer", clarification)]
    ledger = build_ledger(history)
    assert [o.price for o in ledger.offers if o.ref.owner == "buyer"] == ["80.00"]
    answer = result("ok deal", "accept", history)
    assert answer.price == "80.00" and answer.offer.offer_id == "buyer-offer-1"
    assert not answer.offer_actionable and answer.requires_validation


def test_equal_price_can_be_a_genuine_new_bid_and_turn_ids_stay_stable():
    history = [("seller", "asking 90"), ("buyer", "I offer 90")]
    assert build_ledger(history).offers[-1].ref.owner == "buyer"
    history += [("buyer", "Could you clarify the offer of 90?"), ("buyer", "I offer 85")]
    assert build_ledger(history).offers[-1].ref.offer_id == "buyer-offer-3"


@pytest.mark.parametrize("text", ["It's free of scratches", "It's free from damage",
                                  "It's free shipping", "It's free parking", "It's free with the purchase"])
@pytest.mark.parametrize("act", ACTS)
def test_nonprice_free_never_creates_zero(text, act):
    answer = result(text, act)
    assert answer.price != "0.00"
    assert all(m.value != "0.00" for m in collect_mentions(text))


@pytest.mark.parametrize("text", ["It's free", "free to a good home", "You can have it for free"])
def test_item_free_remains_a_valid_zero(text):
    assert result(text).price == "0.00"


@pytest.mark.parametrize("text", ["I'll deliver it for free shipping at 450", "It's free of scratches, 450 firm"])
def test_free_service_or_condition_does_not_hide_the_real_price(text):
    answer = result(text)
    assert answer.price == "450.00"


@pytest.mark.parametrize("text,price", [("I can't do less than 450", "450.00"),
                                      ("No way I can do less than 500", "500.00"),
                                      ("I won't go below 450", "450.00"),
                                      ("I can't accept less than 450", "450.00"),
                                      ("No less than 450", "450.00"),
                                      ("I have it listed for 500", "500.00"),
                                      ("I have 500 firm", "500.00")])
@pytest.mark.parametrize("act", ACTS)
def test_seller_floor_and_asking_price_survive_act_errors(text, price, act):
    assert result(text, act).price == price


def test_buyer_budget_is_still_not_an_offer():
    assert result("I have 500", "counter", speaker="buyer").price is None
    assert result("I only have 500", "counter", speaker="seller").price is None
    ledger = build_ledger([("buyer", "I have 500"), ("seller", "I have 500 firm")])
    assert len(ledger.offers) == 1 and ledger.offers[0].ref.owner == "seller"


@pytest.mark.parametrize("text", ["I don't accept 400", "I do not accept 400", "I don't want 400",
                                  "I do not pay 400", "I'm not interested at 400", "No deal at 400",
                                  "I'm not accepting 400"])
@pytest.mark.parametrize("act", ACTS)
def test_negated_amount_is_never_an_offer(text, act):
    answer = result(text, act, [("buyer", "400")])
    assert answer.price is None


@pytest.mark.parametrize("text,act", [("What's the deal with the dent?", "inquire"),
                                      ("It's a great deal", "vague"),
                                      ("that price is too high", "reject")])
def test_ordinary_deal_talk_or_rejection_does_not_request_clarification(text, act):
    assert result(text, act, [("seller", "500")]).status == "no_price"


@pytest.mark.parametrize("text", ["Will you take 450?", "Do you accept 450?", "Could you do 450?",
                                  "Can you go 450?", "Would 450 work?", "Is 450 ok?",
                                  "450 works for me", "Take it for 450"])
@pytest.mark.parametrize("act", ACTS)
def test_extended_price_proposals_are_not_erased_by_nlu(text, act):
    assert result(text, act, [("buyer", "450")]).price == "450.00"


@pytest.mark.parametrize("text", ["If you're interested, I can do 450", "I bought it when new, 450 now"])
def test_conversational_condition_does_not_attach_to_price(text):
    answer = result(text)
    assert answer.price == "450.00" and not answer.conditions


@pytest.mark.parametrize("text", ["If you collect today, I can do 450", "I can do 450, if you collect today"])
def test_material_condition_is_preserved_across_its_comma(text):
    answer = result(text)
    assert answer.price == "450.00"
    assert tuple(c.lower() for c in answer.conditions) == ("if you collect today",)
    assert answer.requires_validation and not answer.offer_actionable


@pytest.mark.parametrize("text,act,owner", [("that works if you collect today", "accept", "buyer"),
                                          ("same as before if you collect today", "counter", "seller")])
def test_conditions_on_implicit_references_also_require_validation(text, act, owner):
    state = {f"{owner}_price": "450", f"{owner}_offer_id": f"{owner}-1", "agreement": "pending"}
    answer = result(text, act, [(owner, "450")], state)
    assert answer.price == "450.00" and answer.conditions == ("if you collect today",)
    assert not answer.offer_actionable and answer.requires_validation


@pytest.mark.parametrize("agreement,expected", [("pending", "live"), ("confirmed", "not_live"),
                                               ("withdrawn", "not_live"), ("rejected", "not_live")])
def test_tracker_agreement_lifecycle(agreement, expected):
    ledger = build_ledger([("buyer", "60")], {"buyer_price": "60", "buyer_offer_id": "buyer-1", "agreement": agreement})
    answer = resolve("ok deal", act="accept", mentions=[], ledger=ledger)
    assert answer.offer_liveness == expected
    assert answer.offer_actionable == (expected == "live")


def test_state_history_disagreement_preserves_tracker_fact_but_requires_validation():
    history = [("buyer", "60"), ("buyer", "70")]
    state = {"buyer_price": "60", "buyer_offer_id": "buyer-1", "agreement": "pending"}
    answer = result("ok deal", "accept", history, state)
    assert answer.price == "60.00" and answer.offer_liveness == "live"
    assert answer.state_history_disagreement and answer.requires_validation
    assert answer.liveness_established and not answer.offer_actionable
    explicit = result("I accept 60", "accept", history, state)
    assert explicit.price == "60.00" and not explicit.offer_actionable and explicit.requires_validation


def test_later_ambiguous_history_does_not_override_authoritative_state():
    history = [("buyer", "60"), ("buyer", "70 or 80")]
    state = {"buyer_price": "60", "buyer_offer_id": "buyer-1", "agreement": "pending"}
    answer = result("ok deal", "accept", history, state)
    assert answer.price == "60.00" and answer.liveness_established
    assert answer.state_history_disagreement and answer.requires_validation
    assert not result("I accept 60", "accept", history, state).offer_actionable


def test_unknown_turn_index_is_not_treated_as_zero_when_comparing_references():
    offer = Offer(OfferRef("buyer-unlocated", "buyer", None, "minted"), "60.00", "unknown", "history")
    ambiguity = AmbiguousTurn(OfferRef("buyer-offer-2", "buyer", 2),
                              (PriceCandidate("60.00"), PriceCandidate("70.00")))
    answer = resolve("ok 60 works", act="accept", mentions=collect_mentions("ok 60 works"),
                     ledger=OfferLedger((offer,), (ambiguity,)))
    assert answer.reason == "reference_identity_ambiguous" and answer.price is None


def test_minor_type_errors_are_explicit():
    with pytest.raises(ValueError, match="nonblank"):
        OfferRef(7, "buyer")
    with pytest.raises(ValueError, match="bounds"):
        PriceCandidate("50.00", (0, 99)).validate("50")


@pytest.mark.parametrize("live", [True, False])
def test_policy_checks_liveness_and_validation_not_just_price(live):
    class NLU:
        def predict(self, history, message):
            return {"accept": 1.0}

    nlu = CachingNLU(NLU())
    cases = {"c": FixtureCase({"accept": 1.0}, PolicyDecision("confirm", "60", "buyer-1"))}
    adapters = ScriptedComponents(cases).adapters(nlu=nlu)
    adapters["price_reference"] = ReferenceObserver(nlu=nlu)

    class ValidatingPolicy:
        def decide(self, snapshot, observation, gate):
            reference = observation.price_reference
            if (reference["status"] == "resolved" and reference["liveness_established"]
                    and not reference["requires_validation"] and not gate.reasons):
                return PolicyDecision("confirm", reference["price"], reference["referenced_offer_id"])
            return PolicyDecision("clarify", clarification_reason="unresolved_reference")

    adapters["policy"] = ValidatingPolicy()
    state = {"buyer_price": "60", "buyer_offer_id": "buyer-1", "agreement": "pending"} if live else {}
    output = Pipeline(**adapters).turn(Session("live", history=[Turn("buyer", "60")], state=state), "ok deal", case_id="c")
    assert output.generation.action == ("confirm" if live else "clarify")


def test_silver_is_opt_in_and_each_condition_reports_identical_common_cases():
    base = {"history": [], "message": "I can do 50", "act": "counter", "price": "50.00",
            "status": "resolved", "gold_price": "50.00", "offers": [], "subset": "hard"}
    rows = [{**base, "case_id": "silver", "label_source": "seeded_rule", "act_source": "lexical_seed"},
            {**base, "case_id": "reviewed", "label_source": "human_reviewed", "reviewer": "authored-test",
             "predicted_act": "counter"},
            {**base, "case_id": "missing_prediction", "label_source": "ai_proposed"}]
    report = evaluate(rows)
    assert report["excluded"]["silver"] == 1
    assert report["common_subset"]["case_ids"] == ["reviewed"]
    assert report["headline"]["A"]["n"] == 1 and not report["headline"]["A"]["publishable"]
    assert report["headline"]["A"]["metrics"] is None
    included = evaluate(rows, include_silver=True, headline_min_n=1)
    assert included["conditions"]["A"]["slices"]["label_source"]["seeded_rule"]["n"] == 1
    assert included["headline"]["A"]["publishable"]
    rates = included["headline"]["A"]["metrics"]["rate_counts"]
    assert rates["price_exact_match"]["denominator"] == 1
    assert rates["price_exact_match"]["interval_95"][0] < 1
    assert wilson_interval(0, 0) is None


def test_seeder_lowercase_team_schema_and_act_provenance():
    rows = [dict(dialogue="1", turn="0", text="I can do 50", role="buyer", listing_price="60", gold_price="50.00", gold_note=""),
            dict(dialogue="1", turn="1", text="I can do 55", role="seller", listing_price="60", gold_price="55.00", gold_note="")]
    output, _ = seed(rows, "validation")
    assert output[0]["history"][0]["speaker"] == "buyer"
    assert output[0]["act_source"] == "lexical_seed"
