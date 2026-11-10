"""Rule-based tracker, gate and policy with scripted NLU and price/reference inputs."""
from copy import deepcopy
import random

import pytest

from retrieval_pipeline import negotiation
from retrieval_pipeline.contracts import ContractError, PolicyDecision
from retrieval_pipeline.negotiation import (
    ConfidenceGate,
    OfferTracker,
    RulePolicy,
    acceptance_target,
    build_buyer_pipeline,
)
from retrieval_pipeline.pipeline import GateDecision, Observation, Session, Snapshot
from retrieval_pipeline.replay import replay_cases

ACTS = ("propose", "counter", "accept", "reject", "inquire", "vague")


def offer(id, by, price, status="live"):
    return {"id": id, "by": by, "terms": {"price": price}, "status": status}


def say(act, price=None, p=0.9, **reference):
    if price is not None:
        reference = {"price": price, "reference_resolved": True, **reference}
    return act, p, reference


def probabilities(act, p):
    rest = (1 - p) / (len(ACTS) - 1)
    return {label: p if label == act else rest for label in ACTS}


class Script:
    """Scripted NLU and price/reference: one seller turn per message."""
    provenance = "fixture:scripted"

    def __init__(self, turns):
        self.turns = {f"m{index}": turn for index, turn in enumerate(turns)}

    def predict(self, history, message):
        act, p, _ = self.turns[message]
        return probabilities(act, p)

    def observe(self, snapshot, case_id, message):
        return deepcopy(self.turns[message][2])


def talk(*turns, state=None, target="80", walk_away="100", mode="full", threshold=0.7):
    script = Script(turns)
    pipeline = build_buyer_pipeline(script, script, target=target, walk_away=walk_away,
                                    mode=mode, threshold=threshold)
    session = Session("test", state=state)
    results = [pipeline.turn(session, message, case_id=message, turn_id=message)
               for message in script.turns]
    return results, session.snapshot()


def moves(results):
    return [(result.generation.action, result.generation.price) for result in results]


def live(snapshot):
    return [item for item in snapshot.state["offers"] if item["status"] == "live"]


# --- state tracker


def test_new_offers_supersede_older_ones_and_store_term_value_pairs():
    _, snapshot = talk(say("propose", "120"), say("counter", "110"))
    assert snapshot.state == {"agreement": "none", "offers": [
        offer("seller-1", "seller", "120.00", "superseded"),
        offer("buyer-2", "buyer", "80.00", "superseded"),
        offer("seller-3", "seller", "110.00", "superseded"),
        offer("buyer-4", "buyer", "95.00"),
    ]}


def test_clarification_leaves_state_unchanged():
    state = {"offers": [offer("buyer-1", "buyer", "95.00")], "agreement": "none"}
    results, snapshot = talk(say("accept", p=0.5), state=state)
    assert moves(results) == [("clarify", "95.00")]
    # Offers are untouched; the tracker only notes what was asked about.
    pending = snapshot.state.pop("pending_clarification")
    assert pending == {"act": "accept", "price": "95.00", "reason": "low_confidence"}
    assert snapshot.state == state and not snapshot.finished


def test_seller_rejection_supersedes_the_buyer_offer():
    state = {"offers": [offer("buyer-1", "buyer", "95.00")], "agreement": "none"}
    decision = PolicyDecision("reject")
    observation = Observation("c", "m", probabilities("reject", 0.9), {})
    transition = OfferTracker().commit(Snapshot(state=state), observation, GateDecision("full"), decision)
    assert transition.state["offers"] == [offer("buyer-1", "buyer", "95.00", "superseded")]
    assert transition.finished is False


def test_seller_withdrawing_an_accepted_offer_clears_the_agreement():
    results, snapshot = talk(say("propose", "70"), say("reject"), mode="off")
    assert moves(results)[0] == ("accept", "70.00")
    assert snapshot.state["agreement"] == "none"
    assert [item["by"] for item in live(snapshot)] == ["buyer"]


def test_gate_asks_before_a_no_can_undo_an_accepted_offer():
    results, snapshot = talk(say("propose", "70"), say("reject"), say("accept"))
    assert moves(results) == [("accept", "70.00"), ("clarify", "70.00"), ("confirm", "70.00")]
    assert results[1].trace["gate"]["reasons"] == ("state_conflict",)
    assert results[1].generation.text == "Could you clarify the status of the offer of 70?"
    assert snapshot.state["agreement"] == "confirmed" and snapshot.finished


def test_new_seller_price_after_acceptance_clears_the_agreement():
    results, snapshot = talk(say("propose", "70"), say("counter", "85"))
    assert moves(results) == [("accept", "70.00"), ("counter", "80.00")]
    assert snapshot.state["agreement"] == "none"


def test_buyer_counter_clears_a_pending_agreement():
    state = {"offers": [offer("seller-1", "seller", "70.00")], "agreement": "pending"}
    observation = Observation("c", "m", probabilities("vague", 0.9), {})
    transition = OfferTracker().commit(Snapshot(state=state), observation, GateDecision("full"),
                                       PolicyDecision("counter", "60.00"))
    assert transition.state["agreement"] == "none"
    assert transition.state["offers"][-1] == offer("buyer-2", "buyer", "60.00")


def test_restating_the_live_buyer_offer_adds_no_new_offer():
    _, snapshot = talk(say("propose", "120"), say("inquire"), say("vague"))
    assert [item["id"] for item in snapshot.state["offers"]] == ["seller-1", "buyer-2"]
    assert live(snapshot) == [offer("buyer-2", "buyer", "80.00")]


def test_confirmation_finishes_the_session():
    results, snapshot = talk(say("propose", "120"), say("accept", "80"))
    assert moves(results)[-1] == ("confirm", "80.00")
    assert results[-1].trace["policy"]["referenced_offer_id"] == "buyer-2"
    assert snapshot.state["agreement"] == "confirmed" and snapshot.finished


BUYER_LIVE = {"offers": [offer("buyer-1", "buyer", "95.00")], "agreement": "none"}
SELLER_LIVE = {"offers": [offer("seller-1", "seller", "70.00")], "agreement": "none"}


@pytest.mark.parametrize("state, reference, expected", [
    ({}, {}, None),
    (BUYER_LIVE, {}, "buyer-1"),
    (BUYER_LIVE, {"price": "95"}, "buyer-1"),
    (BUYER_LIVE, {"price": "90"}, None),
    (BUYER_LIVE, {"referenced_offer_id": "buyer-1"}, "buyer-1"),
    (BUYER_LIVE, {"referenced_offer_id": "seller-9"}, None),
    (SELLER_LIVE, {}, None),
    ({**SELLER_LIVE, "agreement": "pending"}, {}, "seller-1"),
    ({**SELLER_LIVE, "agreement": "confirmed"}, {}, "seller-1"),
    ({**BUYER_LIVE, "agreement": "pending"}, {}, None),
    ({"offers": [offer("buyer-1", "buyer", "95.00"), offer("buyer-2", "buyer", "96.00")],
      "agreement": "none"}, {}, None),
    ({"offers": [offer("buyer-1", "buyer", "95.00", "superseded")], "agreement": "none"}, {}, None),
])
def test_acceptance_needs_exactly_one_matching_live_offer(state, reference, expected):
    target = acceptance_target(state, reference)
    assert (None if target is None else target["id"]) == expected


def test_track_turns_returns_the_state_before_and_after_each_labelled_turn():
    states = negotiation.track_turns([
        {"speaker": "seller", "act": "propose", "price": "120"},
        {"speaker": "buyer", "act": "counter", "price": "90"},
        {"speaker": "seller", "act": "inquire"},
        {"speaker": "seller", "act": "accept", "price": "90"},
    ], limits={"target": "80", "walk_away": "100"})
    assert states[0] == {"offers": [], "agreement": "none", "limits": {"target": "80", "walk_away": "100"}}
    assert [state["agreement"] for state in states] == ["none", "none", "none", "none", "confirmed"]
    assert states[-1]["offers"][-1] == offer("buyer-2", "buyer", "90.00")


def test_track_turns_needs_a_seller_answer_after_the_buyer_accepts():
    states = negotiation.track_turns([
        {"speaker": "seller", "act": "propose", "price": "70"},
        {"speaker": "buyer", "act": "accept", "price": "70"},
        {"speaker": "seller", "act": "accept"},
    ])
    assert [state["agreement"] for state in states] == ["none", "none", "pending", "confirmed"]


def test_track_turns_keeps_an_acceptance_that_matches_no_offer_unconfirmed():
    states = negotiation.track_turns([
        {"speaker": "buyer", "act": "accept"},
        {"speaker": "seller", "act": "accept"},
        {"speaker": "buyer", "act": "propose", "price": "90"},
        {"speaker": "seller", "act": "accept", "price": "85"},
    ])
    assert [state["agreement"] for state in states] == ["none"] * 5


def test_track_turns_rejects_an_unknown_speaker():
    with pytest.raises(ContractError, match="speaker"):
        negotiation.track_turns([{"speaker": "agent", "act": "propose", "price": "90"}])


def test_agreement_accuracy_scores_only_labelled_turns():
    dialogue = [
        {"speaker": "buyer", "act": "propose", "price": "90"},
        {"speaker": "seller", "act": "vague", "deal_agreed": True},
        {"speaker": "seller", "act": "accept", "deal_agreed": True},
        {"speaker": "seller", "act": "vague", "deal_agreed": True},
    ]
    assert negotiation.agreement_accuracy([dialogue]) == pytest.approx(2 / 3)
    assert negotiation.agreement_accuracy([[{"speaker": "seller", "act": "vague"}]]) is None


# --- confidence gate


def evaluate(act, p=0.9, *, state=None, mode="full", threshold=0.7, **reference):
    gate = ConfidenceGate(threshold, mode)
    snapshot = Snapshot(state=state or {})
    return gate.evaluate(snapshot, Observation("c", "m", probabilities(act, p), reference)).reasons


@pytest.mark.parametrize("mode, expected", [
    ("off", ()),
    ("reference_only", ("unresolved_reference",)),
    ("full", ("unresolved_reference", "state_conflict", "low_confidence")),
])
def test_gate_modes_and_reason_order(mode, expected):
    assert evaluate("accept", 0.5, mode=mode, reference_resolved=False) == expected


def test_gate_asks_below_the_threshold_only():
    assert evaluate("inquire", 0.7, threshold=0.7) == ()
    assert evaluate("inquire", 0.69, threshold=0.7) == ("low_confidence",)


@pytest.mark.parametrize("act", ["propose", "counter"])
def test_gate_treats_an_offer_without_a_price_as_unresolved(act):
    assert evaluate(act, mode="reference_only") == ("unresolved_reference",)
    assert evaluate(act, mode="reference_only", price="90") == ()


@pytest.mark.parametrize("act, state, conflict", [
    ("accept", {}, True),
    ("accept", BUYER_LIVE, False),
    ("reject", {}, True),
    ("reject", SELLER_LIVE, True),
    ("reject", BUYER_LIVE, False),
    ("reject", {**SELLER_LIVE, "agreement": "pending"}, True),
    ("reject", {**BUYER_LIVE, "agreement": "confirmed"}, True),
    ("counter", {**BUYER_LIVE, "agreement": "confirmed"}, True),
    ("counter", BUYER_LIVE, False),
    ("inquire", {}, False),
    ("vague", {**BUYER_LIVE, "agreement": "confirmed"}, False),
])
def test_gate_flags_acts_that_conflict_with_the_state(act, state, conflict):
    expected = ("state_conflict",) if conflict else ()
    assert evaluate(act, state=state, price="95") == expected


def test_gate_names_a_conflict_on_a_different_price():
    assert evaluate("accept", state=BUYER_LIVE, price="90") == ("state_conflict",)


@pytest.mark.parametrize("kwargs", [{"mode": "sometimes"}, {"threshold": 1.5}, {"threshold": -0.1}])
def test_gate_rejects_invalid_settings(kwargs):
    with pytest.raises(ContractError):
        ConfidenceGate(**kwargs)


def test_ambiguous_acceptance_confirms_with_the_gate_off_and_asks_with_it_on():
    turns = (say("propose", "120"), say("accept", p=0.55))
    assert moves(talk(*turns, mode="off")[0])[-1] == ("confirm", "80.00")
    assert moves(talk(*turns, mode="reference_only")[0])[-1] == ("confirm", "80.00")
    assert moves(talk(*turns, mode="full")[0])[-1] == ("clarify", "80.00")


# --- policy


def test_buyer_opens_at_target_then_concedes_a_fixed_fraction():
    results, _ = talk(say("propose", "120"), say("counter", "110"))
    assert moves(results) == [("counter", "80.00"), ("counter", "95.00")]


def test_concession_uses_the_configured_fraction():
    policy = RulePolicy("80", "100", fraction="0.25")
    state = {"offers": [offer("buyer-1", "buyer", "80.00")], "agreement": "none"}
    observation = Observation("c", "m", probabilities("counter", 0.9), {"price": "120"})
    decision = policy.decide(Snapshot(state=state), observation, GateDecision("full"))
    assert (decision.action, decision.price) == ("counter", "90.00")


def test_buyer_accepts_a_first_price_at_or_below_target():
    results, snapshot = talk(say("propose", "80"))
    assert moves(results) == [("accept", "80.00")]
    assert results[0].trace["policy"]["referenced_offer_id"] == "seller-1"
    assert snapshot.state["agreement"] == "pending"


def test_buyer_accepts_a_price_at_or_below_its_last_offer():
    results, _ = talk(say("propose", "120"), say("counter", "78"))
    assert moves(results)[-1] == ("accept", "78.00")


@pytest.mark.parametrize("price", ["90", "100"])
def test_buyer_accepts_a_price_the_seller_repeats_within_walk_away(price):
    results, snapshot = talk(say("propose", price), say("propose", price))
    assert moves(results) == [("counter", "80.00"), ("accept", f"{price}.00")]
    assert snapshot.state["agreement"] == "pending"


def test_buyer_keeps_negotiating_while_the_seller_keeps_moving():
    results, _ = talk(say("propose", "120"), say("counter", "110"), say("counter", "104"))
    assert moves(results) == [("counter", "80.00"), ("counter", "95.00"), ("counter", "99.50")]


def test_buyer_accepts_when_the_seller_is_within_walk_away_and_the_gap_is_small():
    # 97 vs our next counter 93: a 4% gap (under close_gap 5%) is not worth another round.
    results, snapshot = talk(say("propose", "99"), say("counter", "98"), say("counter", "97"))
    assert moves(results) == [("counter", "80.00"), ("counter", "89.00"), ("accept", "97.00")]
    assert snapshot.state["agreement"] == "pending"


def test_firm_price_above_walk_away_is_rejected_once_there_is_no_room_left():
    results, snapshot = talk(*[say("propose", "102") for _ in range(6)])
    assert moves(results) == [("counter", "80.00"), ("counter", "91.00"), ("counter", "96.50"),
                              ("counter", "99.25"), ("counter", "100.00"), ("reject", None)]
    assert not snapshot.finished and snapshot.state["agreement"] == "none"


@pytest.mark.parametrize("act", ["inquire", "vague"])
def test_buyer_does_not_concede_to_a_question_or_vague_message(act):
    results, _ = talk(say("propose", "120"), say(act), say(act))
    follow_up = ("clarify", None) if act == "vague" else ("answer", "80.00")
    assert moves(results) == [("counter", "80.00"), follow_up, follow_up]


def test_seller_rejection_without_a_price_earns_one_concession_step():
    results, _ = talk(say("propose", "120"), say("reject"), say("reject"))
    assert moves(results) == [("counter", "80.00"), ("counter", "100.00"), ("reject", None)]


def test_inquiry_without_a_current_offer_redirects_without_opening():
    assert moves(talk(say("inquire"))[0]) == [("answer", None)]
    assert moves(talk(say("vague"), say("inquire"))[0]) == [("clarify", None), ("answer", None)]


def test_inquiry_does_not_create_an_offer_from_an_old_seller_price():
    state = {"offers": [offer("seller-1", "seller", "70.00", "superseded")], "agreement": "none"}
    assert moves(talk(say("inquire"), state=state)[0]) == [("answer", None)]


def test_question_after_acceptance_keeps_the_deal_and_it_can_still_close():
    results, snapshot = talk(say("propose", "70"), say("inquire"), say("vague"), say("accept"))
    assert moves(results) == [("accept", "70.00"), ("answer", "70.00"), ("clarify", None),
                              ("confirm", "70.00")]
    assert [result.trace["policy"]["referenced_offer_id"] for result in results] == ["seller-1", None, None, "seller-1"]
    assert [item["id"] for item in snapshot.state["offers"]] == ["seller-1"]
    assert snapshot.state["agreement"] == "confirmed" and snapshot.finished


def test_seller_restating_the_accepted_price_can_still_close():
    results, snapshot = talk(say("propose", "75"), say("propose", "75"), say("accept"))
    assert moves(results) == [("accept", "75.00"), ("accept", "75.00"), ("confirm", "75.00")]
    assert snapshot.finished


def test_policy_never_confirms_an_acceptance_with_no_target_even_with_the_gate_off():
    state = {"offers": [offer("buyer-1", "buyer", "95.00")], "agreement": "none"}
    assert moves(talk(say("accept", "90"), state=state, mode="off")[0]) == [("propose", "95.00")]
    assert moves(talk(say("accept"), mode="off")[0]) == [("propose", "80.00")]


def test_clarification_names_the_live_offer_when_there_is_one():
    state = {"offers": [offer("buyer-4", "buyer", "95.00")], "agreement": "none"}
    results, _ = talk(say("accept", p=0.55), state=state)
    assert results[0].generation.text == "Could you clarify whether you mean 95?"
    assert results[0].trace["policy"]["referenced_offer_id"] == "buyer-4"


def test_generated_reply_contains_only_the_policy_price():
    results, _ = talk(say("propose", "120.50"), say("counter", "110.25"))
    assert results[-1].generation.price == "95.12"
    assert "95.12" in results[-1].generation.text
    assert "110.25" not in results[-1].generation.text


@pytest.mark.parametrize("kwargs", [
    {"target": "101", "walk_away": "100"},
    {"target": "-1", "walk_away": "100"},
    {"target": "80", "walk_away": "100", "fraction": "0"},
    {"target": "80", "walk_away": "100", "fraction": "1.5"},
    {"target": "80"},
    {"walk_away": "100"},
])
def test_policy_rejects_invalid_settings(kwargs):
    with pytest.raises(ContractError):
        RulePolicy(**kwargs)


def test_limits_in_the_state_override_the_configured_limits():
    state = {"limits": {"target": "10", "walk_away": "20"}}
    results, snapshot = talk(say("propose", "30"), say("propose", "30"), say("propose", "30"), state=state)
    assert moves(results) == [("counter", "10.00"), ("counter", "20.00"), ("reject", None)]
    assert snapshot.state["limits"] == {"target": "10", "walk_away": "20"}


def test_policy_needs_limits_from_somewhere():
    observation = Observation("c", "m", probabilities("counter", 0.9), {"price": "120"})
    with pytest.raises(ContractError, match="limits"):
        RulePolicy().decide(Snapshot(), observation, GateDecision("full"))


def test_invalid_limits_in_the_state_are_rejected():
    observation = Observation("c", "m", probabilities("counter", 0.9), {"price": "120"})
    snapshot = Snapshot(state={"limits": {"target": "100", "walk_away": "80"}})
    with pytest.raises(ContractError):
        RulePolicy().decide(snapshot, observation, GateDecision("full"))


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("mode", ["off", "reference_only", "full"])
def test_invariants_hold_for_any_sequence_of_seller_turns(seed, mode):
    rng = random.Random(f"{seed}-{mode}")
    target = rng.choice(["50", "80", "99.99"])
    walk_away = rng.choice(["100", "100.01", "150"])
    turns = []
    for _ in range(25):
        price = rng.choice([None, "1", "60", "79.99", "80", "90", "99.99", "100", "100.01", "120", "500"])
        reference = {}
        if rng.random() < 0.1:
            reference["reference_resolved"] = False
        if rng.random() < 0.1:
            reference["referenced_offer_id"] = rng.choice(["buyer-2", "seller-1", "missing"])
        turns.append(say(rng.choice(ACTS), price, rng.uniform(0.3, 1), **reference))
    script = Script(turns)
    pipeline = build_buyer_pipeline(script, script, target=target, walk_away=walk_away, mode=mode)
    session = Session("random")
    limit = float(walk_away)
    for message in script.turns:
        result = pipeline.turn(session, message, case_id=message, turn_id=message)
        decision, snapshot = result.trace["policy"], session.snapshot()
        state = snapshot.state
        if decision["action"] != "clarify":
            lives = live(snapshot)
            assert len(lives) <= 1
            assert state["agreement"] in {"none", "pending", "confirmed"}
            assert all(float(item["terms"]["price"]) <= limit for item in state["offers"] if item["by"] == "buyer")
        if decision["action"] in {"propose", "counter"}:
            assert lives == [item for item in lives if item["by"] == "buyer"]
            assert lives and lives[0]["terms"]["price"] == decision["price"]
            assert state["agreement"] == "none"
        if decision["action"] == "accept":
            assert float(decision["price"]) <= limit
            assert state["agreement"] == "pending"
            assert [item["id"] for item in lives] == [decision["referenced_offer_id"]]
            assert lives[0]["terms"]["price"] == decision["price"]
        if decision["action"] == "confirm":
            assert state["agreement"] == "confirmed"
            assert [item["id"] for item in lives] == [decision["referenced_offer_id"]]
            assert lives[0]["terms"]["price"] == decision["price"]
        assert snapshot.finished == (decision["action"] == "confirm")
        if snapshot.finished:
            break


# --- integration


def test_worked_trace_from_the_design_doc():
    results, snapshot = talk(say("propose", "120", 0.91), say("counter", "110", 0.88),
                             say("accept", p=0.55), say("accept", "95", 0.93))
    assert moves(results) == [("counter", "80.00"), ("counter", "95.00"), ("clarify", "95.00"),
                              ("confirm", "95.00")]
    assert results[2].trace["gate"]["reasons"] == ("low_confidence",)
    assert snapshot.state["agreement"] == "confirmed" and snapshot.finished


def replay_case(case_id, turns, message, limits):
    return {"case_id": case_id, "conversation_id": case_id, "turn_id": "t", "origin": "test",
            "reconstruction_mode": "gold_labels", "message": message,
            "snapshot": {"history": [], "state": negotiation.track_turns(turns, limits=limits)[-1]}}


def replay_records(thresholds=(0.7,), modes=("off", "reference_only", "full")):
    # Each case comes from a different listing, so it carries its own limits in the state.
    cheap, costly = {"target": "10", "walk_away": "20"}, {"target": "800", "walk_away": "1000"}
    buyer_offer = [{"speaker": "seller", "act": "propose", "price": "30"},
                   {"speaker": "buyer", "act": "counter", "price": "15"}]
    cases = [replay_case("clear", buyer_offer, "m0", cheap),
             replay_case("ambiguous", buyer_offer, "m1", cheap),
             replay_case("no-offer", [], "m2", cheap),
             replay_case("unpriced", buyer_offer, "m3", cheap),
             replay_case("priced", [], "m4", costly)]
    script = Script([say("accept", "15", 0.95), say("accept", p=0.55), say("accept", p=0.95),
                     say("counter", p=0.9), say("propose", "1200", 0.9)])
    factory = lambda mode, threshold: build_buyer_pipeline(script, script, mode=mode, threshold=threshold)
    return {threshold: replay_cases(cases, factory, modes=modes, threshold=threshold)
            for threshold in thresholds}


def test_replay_runs_each_case_with_its_own_limits_in_every_gate_mode():
    records = replay_records()[0.7]
    assert all(record["error"] is None for record in records) and len(records) == 15
    actions = {(record["case_id"], record["gate"]["mode"]): record["policy"]["action"] for record in records}
    assert actions["clear", "off"] == actions["clear", "full"] == "confirm"
    assert actions["ambiguous", "off"] == "confirm" and actions["ambiguous", "full"] == "clarify"
    assert actions["no-offer", "reference_only"] == "propose" and actions["no-offer", "full"] == "clarify"
    assert actions["unpriced", "off"] == "counter" and actions["unpriced", "reference_only"] == "clarify"
    priced = next(record for record in records if record["case_id"] == "priced")
    assert priced["policy"]["price"] == "800.00"


def test_threshold_sweep_only_adds_clarifications_as_tau_rises():
    sweep = replay_records(thresholds=(0.0, 0.5, 0.7, 0.9, 1.0), modes=("full",))
    asked = [sum(record["policy"]["action"] == "clarify" for record in sweep[threshold])
             for threshold in (0.0, 0.5, 0.7, 0.9, 1.0)]
    assert asked == [2, 2, 3, 3, 5]


def test_regex_price_reference_stands_in_for_the_resolver():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    reference = negotiation.RegexPriceReference()
    assert reference.observe(None, "c", "I can do 1,200 dollars") == {"price": "1200", "reference_resolved": True}
    assert reference.observe(None, "c", "5k is my price") == {"price": "5000", "reference_resolved": True}
    assert reference.observe(None, "c", "120 or 130") == {"reference_resolved": False}
    assert reference.observe(None, "c", "ok") == {}


def test_negotiation_example_runs(capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from examples.negotiation_contract import main

    main([])
    output = capsys.readouterr().out
    assert "Final: finished=True" in output and '"agreement": "confirmed"' in output
    assert "state_conflict" in output
