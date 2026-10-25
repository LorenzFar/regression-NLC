"""Public matched-context replay seam; authored data is not gold evidence."""
from copy import deepcopy
import json
import pytest

from buyer_agent.contracts import PolicyDecision
from buyer_agent.fixtures import FixtureCase, ScriptedComponents
from buyer_agent.pipeline import Pipeline
from buyer_agent.replay import replay_cases


def case(case_id="one", message="Maybe."):
    return {"case_id": case_id, "conversation_id": "recorded-1", "turn_id": case_id,
            "origin": "authored_fixture", "reconstruction_mode": "authored_snapshot",
            "message": message,
            "snapshot": {"history": [{"speaker": "buyer", "text": "I offer 100."}],
                         "state": {"agreement": "pending", "offers": ["100"]}},
            "fixture": {"probabilities": {"accept": 0.6999999999},
                        "decision": {"action": "confirm", "price": "100", "referenced_offer_id": "b1"},
                        "price_reference": {"reference_resolved": True},
                        "state_after": {"agreement": "confirmed"}, "finished": True}}


def factory(mode, threshold):
    scripted = FixtureCase({"accept": 0.6999999999}, PolicyDecision("confirm", "100", "b1"),
                           price_reference={"reference_resolved": True},
                           state_after={"agreement": "confirmed"}, finished=True, message="Maybe.")
    return Pipeline(**ScriptedComponents({"one": scripted}, mode=mode, threshold=threshold).adapters())


def test_variants_receive_same_recorded_snapshot_and_leave_input_untouched():
    original = case()
    before = deepcopy(original)
    records = replay_cases([original], factory)
    assert len(records) == 3
    assert [r["gate"]["mode"] for r in records] == ["off", "reference_only", "full"]
    assert all(r["state_before"] == records[0]["state_before"] for r in records)
    assert records[0]["state_before"]["history"] == [{"speaker": "buyer", "text": "I offer 100."}]
    assert [r["policy"]["action"] for r in records] == ["confirm", "confirm", "clarify"]
    assert records[2]["state_after"]["state"] == {"agreement": "pending", "offers": ["100"]}
    assert all(r["error"] is None for r in records)
    assert original == before


def test_independent_cases_use_their_own_recorded_context_and_stable_ids():
    from buyer_agent.replay import fixture_pipeline_factory
    first, second = case(), case("two", "Yes.")
    second["snapshot"]["history"].append({"speaker": "seller", "text": "Recorded reply, not generated."})
    second["snapshot"]["state"] = {"agreement": "recorded-second"}
    rows = [first, second]
    records = replay_cases(rows, fixture_pipeline_factory(rows), modes=("off",))
    assert len(records) == 2
    assert records[1]["state_before"]["state"] == {"agreement": "recorded-second"}
    assert records[1]["state_before"]["history"] == second["snapshot"]["history"]
    assert [r["case_id"] for r in records] == ["one", "two"]
    assert records[0]["session_id"] != records[1]["session_id"]
    again = replay_cases(rows, fixture_pipeline_factory(rows), modes=("off",))
    assert [r["session_id"] for r in records] == [r["session_id"] for r in again]


def test_failures_are_one_diagnostic_per_variant_and_do_not_stop_later_cases():
    broken = case("broken")
    broken["snapshot"]["finished"] = True
    missing = case("missing")
    del missing["reconstruction_mode"]
    records = replay_cases([broken, missing, case()], factory)
    assert len(records) == 9
    assert all(r["error"] for r in records[:6])
    assert all(r["error"] is None for r in records[6:])
    assert "finished" in records[0]["error"]["message"]
    assert records[0]["state_after"] == records[0]["state_before"]
    assert "reconstruction_mode" in records[3]["error"]["message"]
    assert records[3]["gate"]["mode"] == "off"
    json.dumps(records, allow_nan=False)


def test_invalid_modes_and_nonfinite_or_out_of_range_thresholds_fail_before_execution():
    for modes in (("unknown",), (), ("off", "off"), "full"):
        with pytest.raises(ValueError, match="mode"):
            replay_cases([case()], factory, modes=modes)
    for threshold in (-0.1, 1.1, float("nan"), float("inf"), "0.7", True):
        with pytest.raises(ValueError, match="threshold"):
            replay_cases([case()], factory, threshold=threshold)


def test_nonserializable_and_nonfinite_component_outputs_are_explicit_failures():
    for bad in (float("nan"), float("inf"), object()):
        def bad_factory(mode, threshold):
            pipeline = factory(mode, threshold)
            class BadReference:
                def observe(self, snapshot, case_id, message):
                    return {"diagnostic": bad}
            pipeline.price_reference = BadReference()
            return pipeline
        records = replay_cases([case()], bad_factory)
        assert len(records) == 3
        assert all(r["error"]["type"] == "SerializationError" for r in records)
        json.dumps(records, allow_nan=False)


def test_reference_gate_and_trace_provenance_are_visible_without_hidden_gold_claims():
    from buyer_agent.replay import fixture_pipeline_factory
    row = case()
    row["origin"] = "recorded_corpus"
    row["reconstruction_mode"] = "owner_supplied_predicted_snapshot"
    row["fixture"]["price_reference"] = {"reference_resolved": False, "state_conflict": True}
    records = replay_cases([row], fixture_pipeline_factory([row]))
    assert [r["gate"]["reasons"] for r in records] == [[], ["unresolved_reference"],
        ["unresolved_reference", "state_conflict", "low_confidence"]]
    assert all(r["origin"] == "recorded_corpus" for r in records)
    assert all(r["components"]["nlu"] == "fixture:scripted-nlu" for r in records)
    assert records[0]["probabilities"]["accept"] == 0.6999999999
    assert records[0]["policy"]["price"] == "100"
    assert records[1]["generation"]["source"] == "clarification"
    assert records[0]["generation"]["template_id"] == "fixed.confirm"
    assert "generation" in records[0]["timing_ms"]


def test_jsonl_cli_writes_success_and_line_errors_with_nonzero_exit(tmp_path):
    from buyer_agent.replay import main
    source, output = tmp_path / "cases.jsonl", tmp_path / "trace.jsonl"
    source.write_text(json.dumps(case()) + "\nnot-json\n", encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 1
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert all(r["error"] is None for r in records[:3])
    assert all(r["case_id"] == "invalid-line-2" and r["error"] for r in records[3:])
    source.write_text(json.dumps(case()) + "\n", encoding="utf-8")
    assert main([str(source), "--output", str(output), "--modes", "full"]) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_cli_invalid_utf8_is_an_input_setup_failure(tmp_path, capsys):
    from buyer_agent.replay import main

    source, output = tmp_path / "cases.jsonl", tmp_path / "trace.jsonl"
    source.write_bytes(b"\xff\n")
    assert main([str(source), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "decode" in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_invalid_fixture_and_duplicate_identity_do_not_hide_valid_cases(tmp_path):
    from buyer_agent.replay import main
    broken = case("broken")
    broken["fixture"]["decision"] = {"action": "counter"}
    source, output = tmp_path / "cases.jsonl", tmp_path / "trace.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in [broken, case(), case()]), encoding="utf-8")
    assert main([str(source), "--output", str(output), "--modes", "off"]) == 1
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert records[0]["error"]
    assert records[1]["error"] is None
    assert "duplicate case_id" in records[2]["error"]["message"]


def test_injected_nlu_allows_independent_cases_with_same_text_and_different_fixture_probabilities():
    from buyer_agent.replay import fixture_pipeline_factory

    rows = [case("first"), case("second")]
    rows[1]["fixture"]["probabilities"] = {"reject": 0.2}
    rows[1]["fixture"]["decision"] = {"action": "reject"}
    rows[1]["fixture"]["state_after"] = {"agreement": "pending", "case": "second"}
    rows[1]["fixture"]["finished"] = False
    before = deepcopy(rows)
    calls = []

    class ConstantNLU:
        provenance = "adapter:test-constant-nlu"
        def predict(self, history, message):
            calls.append((history, message))
            return {"accept": 0.95}

    records = replay_cases(rows, fixture_pipeline_factory(rows, nlu=ConstantNLU()))
    assert len(records) == 6
    assert all(r["error"] is None for r in records)
    assert [r["policy"]["action"] for r in records] == ["confirm"] * 3 + ["reject"] * 3
    assert all(r["probabilities"] == {"accept": 0.95} for r in records)
    assert all(r["components"]["nlu"] == "adapter:test-constant-nlu" for r in records)
    assert all(r["state_after"]["state"] == {"agreement": "confirmed"} for r in records[:3])
    assert all(r["state_after"]["state"] == {"agreement": "pending", "case": "second"}
               for r in records[3:])
    assert calls == [([("BUYER", "I offer 100.")], "Maybe.")] * 6
    assert rows == before


@pytest.mark.parametrize("bad_first", [True, False])
def test_malformed_message_does_not_poison_unrelated_fixture_rows(bad_first):
    from buyer_agent.replay import fixture_pipeline_factory

    broken, valid = case("broken", ["not", "text"]), case()
    rows = [broken, valid] if bad_first else [valid, broken]
    before = deepcopy(rows)
    records = replay_cases(rows, fixture_pipeline_factory(rows))
    errors = [r for r in records if r["case_id"] == "broken"]
    successes = [r for r in records if r["case_id"] == "one"]
    assert len(errors) == len(successes) == 3
    assert all(r["error"]["type"] == "ValueError" for r in errors)
    assert all("message must be" in r["error"]["message"] for r in errors)
    assert all(r["error"] is None for r in successes)
    assert [r["policy"]["action"] for r in successes] == ["confirm", "confirm", "clarify"]
    assert rows == before


@pytest.mark.parametrize("bad_first", [True, False])
@pytest.mark.parametrize("probabilities", [
    [], {}, {1: 0.5}, {"accept": True}, {"accept": "0.5"},
    {"accept": -0.1}, {"accept": 1.1}, {"accept": float("nan")},
    {"accept": float("inf")},
])
def test_malformed_probabilities_do_not_poison_shared_fixture_message(bad_first, probabilities):
    from buyer_agent.replay import fixture_pipeline_factory

    broken, valid = case("broken"), case()
    broken["fixture"]["probabilities"] = probabilities
    rows = [broken, valid] if bad_first else [valid, broken]
    records = replay_cases(rows, fixture_pipeline_factory(rows))
    errors = [r for r in records if r["case_id"] == "broken"]
    successes = [r for r in records if r["case_id"] == "one"]
    assert len(errors) == len(successes) == 3
    assert all(r["error"] is not None for r in errors)
    assert all(r["error"] is None for r in successes)
    assert [r["policy"]["action"] for r in successes] == ["confirm", "confirm", "clarify"]


@pytest.mark.parametrize("probabilities", [[], {}, {"accept": True}, {"accept": "unused"}])
def test_injected_nlu_does_not_validate_unused_fixture_probabilities(probabilities):
    from buyer_agent.replay import fixture_pipeline_factory

    class ConstantNLU:
        def predict(self, history, message):
            return {"accept": 0.95}

    rows = [case("first"), case("second")]
    rows[1]["fixture"]["probabilities"] = probabilities
    records = replay_cases(rows, fixture_pipeline_factory(rows, nlu=ConstantNLU()))
    assert len(records) == 6
    assert all(r["error"] is None for r in records)
    assert all(r["probabilities"] == {"accept": 0.95} for r in records)


def test_fixture_factory_reuses_pipelines_only_for_matching_mode_and_threshold():
    from buyer_agent.replay import fixture_pipeline_factory

    build = fixture_pipeline_factory([case()])
    off = build("off", 0.7)
    full = build("full", 0.7)
    stricter = build("full", 0.9)
    assert off is not full and full is not stricter
    assert full.gate.threshold == 0.7 and stricter.gate.threshold == 0.9
    assert build("off", 0.7) is off
    assert build("full", 0.7) is full
    assert build("full", 0.9) is stricter


def test_reused_fixture_pipeline_gets_fresh_sessions_on_every_replay(monkeypatch):
    from buyer_agent.replay import fixture_pipeline_factory

    rows = [case(), case("two", "Yes.")]
    rows[1]["turn_id"] = rows[0]["turn_id"]
    rows[1]["snapshot"] = rows[0]["snapshot"]
    before = deepcopy(rows)
    build = fixture_pipeline_factory(rows)
    sessions = []
    original_turn = Pipeline.turn

    def record_session(self, session, *args, **kwargs):
        sessions.append(session)
        return original_turn(self, session, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "turn", record_session)
    first = replay_cases(rows, build)
    # Terminal state and applied turn IDs from the previous run must not leak.
    assert sessions[0].snapshot().finished is True
    assert sessions[0].snapshot().applied_turn_ids == ("one",)
    again = replay_cases(rows, build)
    assert all(r["error"] is None for r in first + again)
    assert all(r["state_before"] == first[0]["state_before"] for r in first + again)
    assert len({id(session) for session in sessions}) == len(sessions) == 12
    first[0]["state_before"]["state"]["offers"].append("mutated trace")
    assert again[0]["state_before"]["state"]["offers"] == ["100"]
    assert sessions[0].snapshot().state == {"agreement": "confirmed"}
    assert rows == before


def test_generic_factory_is_still_called_for_each_case_and_mode():
    from buyer_agent.replay import fixture_pipeline_factory

    rows = [case(), case("two", "Yes.")]
    calls = []

    def independent_factory(mode, threshold):
        pipeline = fixture_pipeline_factory(rows)(mode, threshold)
        calls.append(pipeline)
        return pipeline

    records = replay_cases(rows, independent_factory)
    assert all(r["error"] is None for r in records)
    assert len(calls) == len(records) == 6
    assert len({id(pipeline) for pipeline in calls}) == 6


def test_fixture_only_replay_still_rejects_ambiguous_duplicate_messages():
    from buyer_agent.replay import fixture_pipeline_factory

    rows = [case("first"), case("second")]
    rows[1]["fixture"]["probabilities"] = {"reject": 0.2}
    records = replay_cases(rows, fixture_pipeline_factory(rows))
    assert len(records) == 6
    assert all(r["error"]["type"] == "ContractError" for r in records)
    assert all("ambiguous fixture NLU message" in r["error"]["message"] for r in records)


def test_fixture_factory_can_replace_only_nlu_at_the_public_replay_seam():
    from buyer_agent.replay import fixture_pipeline_factory
    calls = []
    class RealShapeNLU:
        provenance = "adapter:test-recording-nlu"
        resource_metadata = {"fixture": False, "checkpoint": "test-resource"}
        def predict(self, history, message):
            calls.append((history, message))
            return {"accept": 0.95}
    rows = [case()]
    records = replay_cases(rows, fixture_pipeline_factory(rows, nlu=RealShapeNLU()))
    assert calls == [([("BUYER", "I offer 100.")], "Maybe.")] * 3
    assert all(r["policy"]["action"] == "confirm" for r in records)
    assert records[2]["components"]["nlu"] == "adapter:test-recording-nlu"
    assert records[2]["components"]["policy"] == "fixture:scripted-policy"
    assert records[2]["resources"]["nlu"]["fixture"] is False
