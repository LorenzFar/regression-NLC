"""Authored cases: these tests are not claims of corpus annotation accuracy."""
from dataclasses import replace

import pytest

from price_reference import (AmbiguousTurn, Offer, OfferLedger, OfferRef,
                             PriceCandidate, Resolution, ResolverMention,
                             build_ledger, resolve)
from price_reference.evidence import collect_mentions
from price_reference.types import money


def run(text, act="counter", history=(), state=None, ledger=None, speaker="seller"):
    return resolve(text, speaker=speaker, act=act, mentions=collect_mentions(text),
                   ledger=ledger if ledger is not None else build_ledger(history, state))


@pytest.mark.parametrize("text,act,price,status", [
    ("I can do 80", "counter", "80.00", "resolved"),
    ("would you take 50?", "inquire", "50.00", "resolved"),
    ("how about 67?", "vague", "67.00", "resolved"),
    ("is 50 right?", "inquire", None, "unresolved"),
    ("what's the condition?", "inquire", None, "no_price"),
    ("No, 109 instead", "reject", "109.00", "resolved"),
    ("1700 is too much", "reject", None, "no_price"),
    ("your 60 is too low", "reject", None, "no_price"),
    ("I only have 140", "vague", None, "no_price"),
    ("I only have 140", "counter", None, "unresolved"),
    ("I paid 600 new, 450 firm", "counter", "450.00", "resolved"),
    ("it's free", "propose", "0.00", "resolved"),
    ("$0", "counter", "0.00", "resolved"),
    ("0", "counter", "0.00", "resolved"),
    ("I can't do 50. I can do 60", "counter", "60.00", "resolved"),
    ("I only have 40 but I can do 50", "counter", "50.00", "resolved"),
    ("I am asking $600. I can take $50 off", "counter", None, "ambiguous"),
])
def test_decision_table(text, act, price, status):
    result = run(text, act)
    assert (result.price, result.status) == (price, status)


@pytest.mark.parametrize("act", ["propose", "counter", "accept", "reject", "inquire", "vague"])
@pytest.mark.parametrize("text", ["would you take 50?", "how about 50?"])
def test_offer_wording_is_a_soft_act_override(act, text):
    result = run(text, act)
    assert result.price == "50.00"
    assert result.act_disagreement == (act not in {"propose", "counter"})


def test_ledger_state_liveness_ids_and_repeated_price_identity():
    history = [("BUYER", "hello"), ("buyer", "I only have 30"),
               ("BUYER", "I offer 50"), ("seller", "I paid 70 originally"),
               ("buyer", "60")]
    ledger = build_ledger(history)
    assert [o.price for o in ledger.offers] == ["50.00", "60.00"]
    assert [o.ref.offer_id for o in ledger.offers] == ["buyer-offer-3", "buyer-offer-4"]
    assert [o.liveness for o in ledger.offers] == ["not_live", "unknown"]
    changed = list(history)
    changed[1] = ("buyer", "25")
    assert build_ledger(changed).offers[-1].ref == ledger.offers[-1].ref
    state = {"buyer_offer_id": "buyer-9", "buyer_price": "60", "agreement": "pending"}
    offer = build_ledger(history, state).offers[-1]
    assert (offer.ref.offer_id, offer.ref.turn_index, offer.liveness) == ("buyer-9", 4, "live")
    assert build_ledger([], state).offers[0].ref.turn_index is None
    repeated = build_ledger([("buyer", "60"), ("buyer", "60")], state)
    assert repeated.offers[-1].ref.turn_index is None


@pytest.mark.parametrize("text,price,live,status", [
    ("ok deal", "60.00", "unknown", "resolved"),
    ("ok 55 works", "55.00", "not_live", "resolved"),
    ("ok 70 works", None, None, "conflict"),
])
def test_acceptance_identity_and_mismatch(text, price, live, status):
    result = run(text, "accept", [("buyer", "55"), ("buyer", "60")])
    assert (result.price, result.offer_liveness, result.status) == (price, live, status)
    if status == "conflict":
        assert result.offer is None and result.context_price == "60.00"


def test_ambiguous_turn_blocks_implicit_reference_but_explicit_price_identifies():
    history = [("buyer", "50"), ("buyer", "55 or 60")]
    ledger = build_ledger(history)
    assert len(ledger.offers) == len(ledger.ambiguous_turns) == 1
    assert run("ok deal", "accept", ledger=ledger).reason == "ambiguous_history"
    result = run("ok 55 works", "accept", ledger=ledger)
    assert result.price == "55.00" and result.offer.turn_index == 1
    assert result.reason == "matched_ambiguous_candidate"


def test_implicit_accept_without_offer_and_own_repeat():
    assert run("ok deal", "accept").reason == "no_standing_offer"
    assert run("that works", "vague").status == "unresolved"
    result = run("same as before", history=[("seller", "45"), ("buyer", "42")])
    assert result.price == "45.00" and result.offer.owner == "seller"
    assert run("same as before", history=[("seller", "50"), ("seller", "45")]).status == "ambiguous"
    assert run("your previous offer", history=[("seller", "45"), ("buyer", "42")]).price == "42.00"


@pytest.mark.parametrize("text,status", [("I can't do 50", "unresolved"),
                                        ("I cannot accept 50", "no_price"),
                                        ("I only have 50", "unresolved")])
def test_wrong_accept_act_never_overrides_negative_evidence(text, status):
    result = run(text, "accept", [("buyer", "50")])
    assert result.status == status and result.price is None


def test_accept_wording_does_not_create_or_override_a_new_offer():
    history = [("buyer", "50"), ("seller", "I accept your offer of 50")]
    assert [o.ref.owner for o in build_ledger(history).offers] == ["buyer"]
    result = run("I can accept your offer of 50", "accept", [("buyer", "50")])
    assert result.price_source == "referenced_offer" and result.offer.owner == "buyer"


@pytest.mark.parametrize("text", ["no deal", "I can't accept your offer", "I won't take that offer"])
def test_negated_acceptance_is_never_a_deal_even_with_wrong_act(text):
    result = run(text, "accept", [("buyer", "50")])
    assert result.status == "no_price" and result.price is None


def test_acceptance_delta_is_not_misread_as_an_absolute_price_and_condition_is_retained():
    result = run("I accept 50 off", "accept", [("buyer", "50")])
    assert result.price is None and result.reason == "delta_acceptance"
    conditional = run("deal if you collect today", "accept", [("buyer", "50")])
    assert conditional.conditions == ("if you collect today",)


@pytest.mark.parametrize("text,history,price,reason", [
    ("50 less", [("seller", "100")], "50.00", None),
    ("100 less than your offer", [("buyer", "300")], "200.00", None),
    ("50 off", [("seller", "50")], "0.00", "zero_price"),
    ("60 off", [("seller", "50")], None, "negative_result"),
    ("$20 more", [], None, "unknown_base"),
    ("another 10", [("seller", "50")], "60.00", None),
])
def test_delta(text, history, price, reason):
    result = run(text, history=history)
    assert (result.price, result.reason) == (price, reason)
    if price is not None:
        assert result.base_offer is not None and result.price_source == "delta"


@pytest.mark.parametrize("text", ["50 if pickup, otherwise 55", "otherwise 55, 50 if pickup",
                                  "Oops I meant 1400, how about 1350?"])
def test_alternatives_and_revision_never_pick_first_or_last(text):
    result = run(text)
    assert result.status == "ambiguous" and result.price is None and len(result.candidates) == 2


def test_single_condition_retained():
    result = run("50 if you collect today", "propose")
    assert result.price == "50.00" and result.conditions == ("if you collect today",)
    assert len(result.candidates) == 1


@pytest.mark.parametrize("text,status", [("hello", "no_price"), ("that price", "unresolved"),
                                         ("50", "resolved"), ("50 or 60", "ambiguous"),
                                         ("I only have 50", "no_price")])
def test_act_free(text, status):
    assert run(text, None).status == status


def test_types_and_no_fabricated_gold_spans():
    gold = ResolverMention("50.00", "unknown", "gold_csv_value")
    assert resolve("a malformed amount", act="counter", mentions=[gold], ledger=OfferLedger()).price == "50.00"
    both = [gold, replace(gold, value="60.00")]
    assert resolve("unlocated", act="counter", mentions=both, ledger=OfferLedger()).reason == "mention_unlocated"
    with pytest.raises(ValueError):
        ResolverMention("50.00", start=0)
    with pytest.raises(ValueError):
        ResolverMention("50.00", kind="unknown")
    mention = ResolverMention("50.00", surface="50", start=0, end=2)
    with pytest.raises(ValueError, match="span"):
        resolve("60", act="counter", mentions=[mention], ledger=OfferLedger())
    with pytest.raises(ValueError):
        Offer(OfferRef("b", "buyer"), "50.00", "live")
    with pytest.raises(ValueError):
        Resolution("conflict", price="50.00")
    with pytest.raises(ValueError):
        Resolution("ambiguous")
    with pytest.raises(ValueError):
        AmbiguousTurn(OfferRef("b", "buyer"), (PriceCandidate("50.00"),))
    assert money("1.005") == "1.01"
    assert money("0.10") == "0.10"
