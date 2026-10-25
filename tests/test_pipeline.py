"""Public pipeline behavior; all collaborators except rendering are authored fixtures."""
import pytest

from retrieval_pipeline.contracts import PolicyDecision, Turn
from retrieval_pipeline.fixtures import FixtureCase, ScriptedComponents
from retrieval_pipeline.pipeline import Pipeline, Session


@pytest.mark.parametrize("reason", [None, "low_confidence", "unresolved_reference"])
def test_gate_reason_must_match_policy_clarification_without_committing(reason):
    from retrieval_pipeline.contracts import ContractError

    cases = {"offer": FixtureCase(
        {"counter": 1.0}, PolicyDecision("counter", "100"),
        price_reference={"state_conflict": True},
    )}
    adapters = ScriptedComponents(cases).adapters()

    class WrongReasonPolicy:
        def decide(self, snapshot, observation, gate):
            return PolicyDecision("clarify", clarification_reason=reason)

    adapters["policy"] = WrongReasonPolicy()
    session = Session("reason-mismatch", state={"agreement": "pending"})
    before = session.snapshot()
    with pytest.raises(ContractError, match="clarification_reason") as error:
        Pipeline(**adapters).turn(session, "offer", case_id="offer", turn_id="s1")
    assert session.snapshot() == before
    assert error.value.trace["gate"]["reasons"] == ("state_conflict",)
    assert error.value.trace["state_after"] == error.value.trace["state_before"]
    assert "generation" not in error.value.trace


def test_policy_can_choose_any_supplied_gate_reason():
    cases = {"offer": FixtureCase(
        {"counter": 0.4}, PolicyDecision("counter", "100"),
        price_reference={"state_conflict": True},
    )}
    adapters = ScriptedComponents(cases).adapters()

    class ChooseSecondReason:
        def decide(self, snapshot, observation, gate):
            return PolicyDecision("clarify", clarification_reason="low_confidence")

    adapters["policy"] = ChooseSecondReason()
    result = Pipeline(**adapters).turn(Session("reason-choice"), "offer", case_id="offer")
    assert result.trace["gate"]["reasons"] == ("state_conflict", "low_confidence")
    assert result.generation.text == "Could you clarify what you mean?"
    assert result.trace["policy"]["clarification_reason"] == "low_confidence"


@pytest.mark.parametrize("origin", ["authored_fixture", "corpus"])
def test_pipeline_reports_selected_template_provenance(tmp_path, origin):
    from retrieval_pipeline.bank import approve_candidate, build_bank, prepare_candidate
    from retrieval_pipeline.generator import RetrievalGenerator

    # Synthetic source records exercise both supported provenance labels, not real corpus data.
    rows = []
    for name in ("camera", "bicycle"):
        source = dict(
            template_id=name, dialogue_id=f"train-{name}", turn_id=f"{name}-buyer-2",
            origin=origin, split="train", speaker="BUYER", action="counter",
            history=[], message=f"Pristine {name}", response="I offer 80.",
            price_spans=[dict(start=8, end=10, value="80", validated=True)],
            mixed_action=False, unresolved_reference=False,
        )
        rows.append(approve_candidate(prepare_candidate(
            source, {"train-camera", "train-bicycle"}), "Test semantic reviewer"))
    path = tmp_path / "approved.jsonl"
    build_bank(rows, path, {"train-camera", "train-bicycle"})
    cases = {"offer": FixtureCase(
        {"counter": 1.0}, PolicyDecision("counter", "79.25"), message="Pristine bicycle")}
    result = Pipeline(**ScriptedComponents(cases).adapters(),
                      generator=RetrievalGenerator(path)).turn(
        Session("provenance"), "Pristine bicycle", case_id="offer")
    assert result.generation.text == "I offer 79.25."
    assert result.generation.source == "retrieval"
    assert result.generation.template_id == "bicycle"
    assert result.generation.origin == origin
    assert result.generation.source_dialogue_id == "train-bicycle"
    assert result.generation.source_turn_id == "bicycle-buyer-2"
    assert result.trace["generation"]["origin"] == origin
    assert result.trace["generation"]["source_dialogue_id"] == "train-bicycle"
    assert result.trace["generation"]["source_turn_id"] == "bicycle-buyer-2"


@pytest.mark.parametrize("decision", [PolicyDecision("reject"),
    PolicyDecision("clarify", clarification_reason="low_confidence")])
def test_fixed_pipeline_wording_has_no_corpus_provenance(decision):
    cases = {"offer": FixtureCase({"counter": 1.0}, decision)}
    result = Pipeline(**ScriptedComponents(cases).adapters()).turn(
        Session("fixed"), "offer", case_id="offer")
    assert result.generation.source in {"fallback", "clarification"}
    assert result.generation.origin is None
    assert result.generation.source_dialogue_id is None
    assert result.generation.source_turn_id is None
    assert result.trace["generation"]["origin"] is None
    assert result.trace["generation"]["source_dialogue_id"] is None
    assert result.trace["generation"]["source_turn_id"] is None


def test_scripted_counter_keeps_exact_price_and_public_history():
    components = ScriptedComponents({"offer": FixtureCase(
        probabilities={"counter": 0.923456789}, message="105.25?",
        price_reference={"price": "105.25", "reference_resolved": True},
        decision=PolicyDecision("counter", "100.25"),
        state_after={"buyer_offer": "100.25"},
    )})
    session = Session("conversation-1", history=[Turn("buyer", "Hello")])
    result = Pipeline(**components.adapters()).turn(session, "105.25?", case_id="offer", turn_id="s1")
    assert result.generation.text == "Would you take 100.25 instead?"
    assert session.snapshot().history == (
        Turn("buyer", "Hello"), Turn("seller", "105.25?"),
        Turn("buyer", "Would you take 100.25 instead?"),
    )
    assert session.snapshot().state == {"buyer_offer": "100.25"}
    assert result.trace["probabilities"]["counter"] == 0.923456789
    assert result.trace["generation"]["source"] == "fallback"
    assert result.trace["components"]["policy"].startswith("fixture:")
    assert result.trace["session_id"] == "conversation-1"

def test_three_gate_modes_preserve_ablation_and_block_disputed_agreement():
    from retrieval_pipeline.contracts import ContractError
    from retrieval_pipeline.pipeline import Transition

    for mode in ("off", "reference_only", "full"):
        cases = {"yes": FixtureCase(
            probabilities={"accept": 0.6999999999},
            price_reference={"reference_resolved": False, "state_conflict": True},
            decision=PolicyDecision("confirm", "100", "buyer-1", finished=True),
            state_after={"agreement": "confirmed"}, finished=True,
        )}
        session = Session(mode, state={"agreement": "pending"})
        result = Pipeline(**ScriptedComponents(cases, mode=mode).adapters()).turn(
            session, "yes", case_id="yes")
        if mode == "off":
            # Deliberately wrong fixture confirmation must survive the ablation.
            assert result.generation.action == "confirm"
            assert session.snapshot().finished is True
        else:
            assert result.generation.action == "clarify"
            assert result.trace["gate"]["reasons"][0] == "unresolved_reference"
            assert session.snapshot().state == {"agreement": "pending"}
            assert not session.snapshot().finished

    adapters = ScriptedComponents(cases).adapters()
    class DisobedientPolicy:
        def decide(self, snapshot, observation, gate):
            return PolicyDecision("confirm", "100", "buyer-1")
    adapters["policy"] = DisobedientPolicy()
    session = Session("bad-policy")
    with pytest.raises(ContractError, match="gate"):
        Pipeline(**adapters).turn(session, "yes", case_id="yes")
    assert session.snapshot().history == ()

    class DisputedTracker:
        def commit(self, snapshot, observation, gate, decision):
            return Transition({"agreement": "confirmed"}, True)
    adapters = ScriptedComponents(cases).adapters()
    adapters["tracker"] = DisputedTracker()
    with pytest.raises(ContractError, match="clarification"):
        Pipeline(**adapters).turn(session, "yes", case_id="yes")
    assert session.snapshot().state == {}

def test_missing_fields_fail_even_with_injected_renderer_and_leave_session_retryable():
    from retrieval_pipeline.contracts import ContractError, GenerationResult

    class PermissiveRenderer:
        def generate(self, decision, history, message):
            return GenerationResult("unsafe", decision.action, decision.price, None, "fixture")

    for decision in (PolicyDecision("counter"), PolicyDecision("accept", "100"),
                     PolicyDecision("confirm", referenced_offer_id="b1"),
                     PolicyDecision("clarify"), PolicyDecision("counter", "1.234")):
        cases = {"offer": FixtureCase({"counter": 1.0}, decision)}
        session = Session("invalid", state={"agreement": "pending"})
        before = session.snapshot()
        with pytest.raises(ContractError) as error:
            Pipeline(**ScriptedComponents(cases).adapters(), generator=PermissiveRenderer()).turn(
                session, "offer", case_id="offer", turn_id="s1")
        assert session.snapshot() == before
        assert error.value.trace["error"]["type"] == "ContractError"
        assert error.value.trace["state_after"] == error.value.trace["state_before"]

def test_generation_and_commit_failures_are_atomic_and_retry_applies_once():
    from retrieval_pipeline.contracts import ContractError, GenerationResult

    cases = {"offer": FixtureCase({"counter": 1.0}, PolicyDecision("counter", "100.25"),
                                  state_after={"offers": ["100.25"]})}
    session = Session("retry", state={"offers": []})
    before = session.snapshot()
    class FailingGenerator:
        def generate(self, decision, history, message):
            history.append(Turn("buyer", "unsent"))
            raise RuntimeError("render failed")
    with pytest.raises(RuntimeError, match="render failed"):
        Pipeline(**ScriptedComponents(cases).adapters(), generator=FailingGenerator()).turn(
            session, "offer", case_id="offer", turn_id="1")
    assert session.snapshot() == before

    class FailingTracker:
        def commit(self, snapshot, observation, gate, decision):
            snapshot.state["offers"].append("unsent")
            raise RuntimeError("commit failed")
    adapters = ScriptedComponents(cases).adapters()
    adapters["tracker"] = FailingTracker()
    with pytest.raises(RuntimeError, match="commit failed") as error:
        Pipeline(**adapters).turn(session, "offer", case_id="offer", turn_id="1")
    assert session.snapshot() == before
    assert error.value.trace["generation"]["text"] == "Would you take 100.25 instead?"
    assert error.value.trace["state_after"] == error.value.trace["state_before"]

    class IncorrectGenerator:
        def generate(self, decision, history, message):
            return GenerationResult("I accept 999.", "accept", "999", None, "fixture")
    with pytest.raises(ContractError, match="generation"):
        Pipeline(**ScriptedComponents(cases).adapters(), generator=IncorrectGenerator()).turn(
            session, "offer", case_id="offer", turn_id="1")
    assert session.snapshot() == before
    Pipeline(**ScriptedComponents(cases).adapters()).turn(session, "offer", case_id="offer", turn_id="1")
    assert session.snapshot().applied_turn_ids == ("1",)
    assert len(session.snapshot().history) == 2
    assert session.snapshot().state == {"offers": ["100.25"]}

def test_nlu_receives_full_uppercase_prior_history_without_current_and_duplicate_ids_fail():
    from retrieval_pipeline.contracts import ContractError
    cases = {"same text": FixtureCase({"reject": 1.0}, PolicyDecision("reject"))}
    calls = []
    class RecordingNLU:
        provenance = "fixture:recording-nlu"
        def predict(self, history, message):
            calls.append((history[:], message))
            return {"reject": 1.0}
    history = [Turn("buyer", "old offer"), Turn("seller", "old reply"), Turn("buyer", "recent offer")]
    adapters = ScriptedComponents(cases).adapters()
    adapters["nlu"] = RecordingNLU()
    pipeline = Pipeline(**adapters)
    session = Session("history", history=history)
    first = pipeline.turn(session, "same text", case_id="same text", turn_id="one")
    assert calls == [([("BUYER", "old offer"), ("SELLER", "old reply"), ("BUYER", "recent offer")], "same text")]
    assert first.snapshot.history[:3] == tuple(history)
    with pytest.raises(ContractError, match="already applied"):
        pipeline.turn(session, "same text", case_id="same text", turn_id="one")
    assert len(calls) == 1
    pipeline.turn(session, "same text", case_id="same text", turn_id="two")
    assert len(session.snapshot().history) == 7
    assert session.snapshot().applied_turn_ids == ("one", "two")
    assert len(calls[1][0]) == 5
    assert session.snapshot().finished is False

def test_reference_only_does_not_apply_full_gate_confidence_or_state_checks():
    for reason, price_reference in (("low_confidence", {"reference_resolved": True}),
                                     ("state_conflict", {"state_conflict": True})):
        cases = {"case": FixtureCase({"accept": 0.6999999999},
                 PolicyDecision("confirm", "100", "buyer-1"), price_reference=price_reference)}
        for mode in ("off", "reference_only", "full"):
            result = Pipeline(**ScriptedComponents(cases, mode=mode).adapters()).turn(
                Session(mode), "case", case_id="case")
            assert result.generation.action == ("clarify" if mode == "full" else "confirm")
            if mode == "full":
                assert result.trace["gate"]["reasons"][0] == reason
                assert result.trace["gate"]["threshold"] == 0.7
    # Full precision, not display rounding: exact threshold is permitted.
    cases = {"case": FixtureCase({"reject": 0.7}, PolicyDecision("reject"))}
    result = Pipeline(**ScriptedComponents(cases).adapters()).turn(Session("exact"), "case", case_id="case")
    assert result.generation.action == "reject"


def test_reconstruction_sessions_and_fixture_data_are_detached_and_terminal_is_explicit():
    from retrieval_pipeline.contracts import ContractError
    state = {"offers": ["100"], "agreement": "pending"}
    case = FixtureCase({"accept": 1.0}, PolicyDecision("confirm", "100", "buyer-1", finished=True),
                       state_after={"offers": ["100"], "agreement": "confirmed"}, finished=True)
    components = ScriptedComponents({"yes": case})
    case.state_after["offers"].append("fixture mutation")
    original = Session("recorded", history=[Turn("buyer", "I offer 100.")], state=state)
    state["offers"].append("external mutation")
    recorded = original.snapshot()
    left = Session.from_snapshot("left", recorded)
    right = Session.from_snapshot("right", recorded)
    recorded.state["offers"].append("snapshot mutation")
    result = Pipeline(**components.adapters()).turn(left, "yes", case_id="yes")
    assert left.snapshot().state == {"offers": ["100"], "agreement": "confirmed"}
    assert right.snapshot() == original.snapshot()
    assert len(right.snapshot().history) == 1
    result.snapshot.state["offers"].append("result mutation")
    result.trace["state_after"]["state"]["offers"].append("trace mutation")
    assert left.snapshot().state["offers"] == ["100"]
    with pytest.raises(ContractError, match="finished"):
        Pipeline(**components.adapters()).turn(left, "yes", case_id="yes")
    assert len(left.snapshot().history) == 3
    Pipeline(**components.adapters()).turn(Session("new"), "yes", case_id="yes")


def test_invalid_tracker_transition_does_not_publish_or_infer_terminal_state():
    from retrieval_pipeline.contracts import ContractError
    from retrieval_pipeline.pipeline import Transition
    cases = {"case": FixtureCase({"reject": 1.0}, PolicyDecision("reject"))}
    session = Session("invalid-tracker")
    for transition in (Transition({}, "yes"), Transition([], False), None):
        class InvalidTracker:
            def commit(self, snapshot, observation, gate, decision):
                return transition
        adapters = ScriptedComponents(cases).adapters()
        adapters["tracker"] = InvalidTracker()
        with pytest.raises(ContractError, match="transition"):
            Pipeline(**adapters).turn(session, "case", case_id="case")
        assert session.snapshot().history == ()

def test_demo_and_contract_example_use_public_pipeline_and_emit_labelled_jsonl():
    import json
    import subprocess
    import sys

    run = subprocess.run([sys.executable, "-m", "retrieval_pipeline.demo", "--jsonl"],
                         capture_output=True, text=True, check=True)
    traces = [json.loads(line) for line in run.stdout.splitlines()]
    assert [trace["policy"]["action"] for trace in traces] == ["counter", "clarify", "reject", "confirm"]
    assert traces[1]["state_after"]["state"] == traces[0]["state_after"]["state"]
    assert traces[2]["state_after"]["finished"] is False
    assert traces[3]["state_after"]["finished"] is True
    assert traces[0]["generation"]["price"] == "100.25"
    for trace in traces:
        assert trace["components"]["tracker"].startswith("fixture:")
        assert set(trace["timing_ms"]) >= {"nlu", "price_reference", "gate", "policy", "generation", "tracker"}
        assert trace["case_id"] and trace["session_id"]
        assert trace["error"] is None
    example = subprocess.run([sys.executable, "-m", "examples.conversation_contract"],
                             capture_output=True, text=True, check=True)
    assert "independent sessions" in example.stdout
