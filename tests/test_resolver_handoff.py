"""Public handoff contracts; teammate source and authored fixtures are not edited."""
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import pytest

from evaluation.resolver.data import gold_ledger

from price_reference import (CachingNLU, ReferenceObserver, ReferenceSafetyGate,
                             build_ledger, resolve)
from price_reference.evidence import collect_mentions
from retrieval_pipeline.contracts import PolicyDecision
from retrieval_pipeline.fixtures import FixtureGate
from retrieval_pipeline.generator import RetrievalGenerator
from retrieval_pipeline.pipeline import GateDecision, Observation, Session
from retrieval_pipeline.replay import _snapshot, fixture_pipeline_factory


class FixedNLU:
    def __init__(self, act):
        self.act = act

    def predict(self, history, message):
        return {self.act: 1.0}


def result(text, act, *, state=None):
    return resolve(text, act=act, mentions=collect_mentions(text),
                   ledger=build_ledger([("buyer", "Can you do 40?"), ("seller", "I can do 45")], state))


@pytest.mark.parametrize("text", ["ok that works", "that works", "ok deal", "that price"])
@pytest.mark.parametrize("act", ["counter", "propose"])
def test_bare_pointer_cannot_turn_acceptance_wording_into_an_own_offer(text, act):
    answer = result(text, act)
    assert answer.status == "unresolved" and answer.price is None and answer.act_disagreement


def test_explicit_repeat_and_acceptance_keep_different_owners():
    repeat = result("same as before", "counter")
    assert repeat.price == "45.00" and repeat.offer.owner == "seller"
    assert result("my previous offer", "counter").offer.owner == "seller"
    assert result("your previous offer", "counter").offer.owner == "buyer"
    accepted = result("ok that works", "accept")
    assert accepted.price == "40.00" and accepted.offer.owner == "buyer"


def test_existing_team_standalone_fixture_clarifies_with_real_resolver():
    path = Path(__file__).resolve().parents[1] / "examples" / "standalone_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    case = next(row for row in cases if row["case_id"] == "standalone-unresolved-reference")
    pipeline = fixture_pipeline_factory([case])("full", .7)
    shared = CachingNLU(pipeline.nlu)
    pipeline.nlu = shared
    pipeline.price_reference = ReferenceObserver(shared)
    session = Session.from_snapshot("handoff", _snapshot(case))
    output = pipeline.turn(session, case["message"], case_id=case["case_id"], turn_id=case["turn_id"])
    assert output.trace["gate"]["reasons"] == ("unresolved_reference",)
    assert output.generation.action == "clarify"
    assert output.trace["price_reference"]["price"] is None
    assert output.snapshot.state == case["snapshot"]["state"]


@pytest.mark.parametrize("value", ["109", "100.25", "0"])
def test_canonical_prices_satisfy_the_existing_generator_contract(value):
    answer = result(f"I can do {value}", "counter")
    assert Decimal(answer.price) == Decimal(value)
    rendered = RetrievalGenerator().generate(PolicyDecision("counter", answer.price), [], "offer")
    assert rendered.price == answer.price


def tracker_state(*, agreement="none", live_owner="buyer", live_price="95.00"):
    return {"offers": [
        {"id": "seller-1", "by": "seller", "terms": {"price": "120.00"}, "status": "superseded"},
        {"id": "buyer-2", "by": "buyer", "terms": {"price": "80.00"}, "status": "superseded"},
        {"id": f"{live_owner}-3", "by": live_owner, "terms": {"price": live_price}, "status": "live"}],
        "agreement": agreement}


def test_structured_tracker_state_preserves_ids_statuses_and_input():
    state = tracker_state()
    original = json.dumps(state)
    ledger = build_ledger([("seller", "120"), ("buyer", "80"), ("buyer", "95")], state)
    assert [offer.ref.offer_id for offer in ledger.offers] == ["seller-1", "buyer-2", "buyer-3"]
    assert [offer.liveness for offer in ledger.offers] == ["not_live", "not_live", "live"]
    assert ledger.offers[-1].ref.turn_index == 2
    answer = resolve("I accept 95", act="accept", mentions=collect_mentions("I accept 95"), ledger=ledger)
    assert answer.offer.offer_id == "buyer-3" and answer.liveness_established
    assert json.dumps(state) == original


def test_structured_state_supersedes_history_membership_even_when_empty():
    assert build_ledger([("buyer", "95")], {"offers": []}).offers == ()
    ledger = build_ledger([("buyer", "90")], tracker_state())
    assert ledger.offers[-1].ref.turn_index is None
    assert ledger.offers[-1].state_history_disagreement
    assert ledger.offers[-1].liveness == "live"


def test_pending_seller_offer_acknowledgment_targets_the_offer_buyer_already_accepted():
    state = tracker_state(agreement="pending", live_owner="seller", live_price="45.00")
    answer = result("ok that works", "accept", state=state)
    assert answer.price == "45.00" and answer.offer.offer_id == "seller-3"
    assert answer.liveness_established
    assert result("I accept 40", "accept", state=state).status == "conflict"


def test_pending_agreement_can_be_used_in_gold_ledger_evaluation():
    ledger = build_ledger([], tracker_state(agreement="pending", live_owner="seller", live_price="45.00"))
    record = {"history": [], **asdict(ledger)}
    restored = gold_ledger(record)
    answer = resolve("ok that works", act="accept", mentions=[], ledger=restored)
    assert answer.price == "45.00" and answer.offer == ledger.pending_acceptance


@pytest.mark.parametrize("state", [{"offers": {}}, {"offers": [None]}, {"offers": [{"id": "b", "by": "buyer"}]}])
def test_malformed_tracker_state_does_not_fall_back_to_guessed_history(state):
    with pytest.raises(ValueError):
        build_ledger([("buyer", "95")], state)


def test_safety_gate_catches_conflict_even_when_delegate_ignores_resolver_flag():
    class StateOnlyGate:
        def evaluate(self, snapshot, observation):
            return GateDecision("full", (), .7)

    snapshot = Session("gate").snapshot()
    observation = Observation("case", "I accept 90", {"accept": 1.0}, {
        "status": "conflict", "state_conflict": True, "reference_resolved": True,
        "price": None, "referenced_offer_id": None})
    assert ReferenceSafetyGate(StateOnlyGate()).evaluate(snapshot, observation).reasons == ("state_conflict",)


@pytest.mark.parametrize("flag", ["requires_validation", "act_disagreement"])
def test_full_safety_gate_blocks_unhandled_validation_and_act_disagreement(flag):
    snapshot = Session("validate").snapshot()
    observation = Observation("case", "offer", {"counter": 1.0}, {
        "status": "resolved", "price": "45.00", "reference_resolved": True, flag: True})
    assert ReferenceSafetyGate(FixtureGate("full")).evaluate(snapshot, observation).reasons == ("unresolved_reference",)
    assert ReferenceSafetyGate(FixtureGate("off")).evaluate(snapshot, observation).reasons == ()
    assert ReferenceSafetyGate(FixtureGate("reference_only")).evaluate(snapshot, observation).reasons == ()
