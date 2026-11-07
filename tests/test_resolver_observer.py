import pytest

from price_reference import CachingNLU, ReferenceObserver
from retrieval_pipeline.contracts import ContractError, PolicyDecision, Turn
from retrieval_pipeline.fixtures import FixtureCase, ScriptedComponents
from retrieval_pipeline.pipeline import Pipeline, Session


class CountingNLU:
    def __init__(self, probabilities=None):
        self.calls = []
        self.probabilities = probabilities or {"counter": 1.0}

    def predict(self, history, message):
        self.calls.append((history, message))
        return dict(self.probabilities)


def test_shared_cache_copy_safety_eviction_and_clear():
    model = CountingNLU()
    cache = CachingNLU(model, maxsize=1)
    prediction = cache.predict([("seller", "hi")], "50")
    prediction["counter"] = 0
    assert cache.predict([("SELLER", "hi")], "50") == {"counter": 1.0}
    assert len(model.calls) == 1
    cache.predict([], "60")
    cache.predict([("seller", "hi")], "50")
    assert len(model.calls) == 3
    cache.clear()
    cache.predict([("seller", "hi")], "50")
    assert len(model.calls) == 4


@pytest.mark.parametrize("message,probs,history,state,reason", [
    ("ok deal", {"accept": 1.0}, (), {}, "unresolved_reference"),
    ("ok 70 works", {"accept": 1.0}, (Turn("buyer", "60"),), {}, "state_conflict"),
    ("50 if pickup, otherwise 55", {"counter": 1.0}, (), {}, "unresolved_reference"),
    ("I can do 50", {"counter": .4, "propose": .3, "vague": .3}, (), {}, "low_confidence"),
])
def test_fixture_pipeline_clarification_blocks_priced_action(message, probs, history, state, reason):
    model = CountingNLU(probs)
    nlu = CachingNLU(model)
    cases = {"case": FixtureCase(probs, PolicyDecision("counter", "50"), message=message)}
    adapters = ScriptedComponents(cases).adapters(nlu=nlu)
    adapters["price_reference"] = ReferenceObserver(nlu=nlu)
    session = Session("fixture", history=history, state=state)
    result = Pipeline(**adapters).turn(session, message, case_id="case")
    assert result.generation.action == "clarify"
    assert result.trace["gate"]["reasons"] == (reason,)
    assert result.snapshot.state == state
    assert len(model.calls) == 1
    observation = result.trace["price_reference"]
    if reason == "low_confidence":
        assert observation["price"] == "50.00" and observation["reference_resolved"]
    else:
        assert observation["price"] is None and observation["referenced_offer_id"] is None
    if reason == "state_conflict":
        assert observation["reference_resolved"] and observation["state_conflict"]


def test_naive_policy_cannot_act_when_gate_requires_clarification():
    nlu = CachingNLU(CountingNLU({"counter": .4, "propose": .3, "vague": .3}))
    cases = {"case": FixtureCase({"counter": 1}, PolicyDecision("counter", "50"))}
    adapters = ScriptedComponents(cases).adapters(nlu=nlu)
    adapters["price_reference"] = ReferenceObserver(nlu=nlu)

    class NaivePolicy:
        def decide(self, snapshot, observation, gate):
            return PolicyDecision("counter", observation.price_reference["price"])

    adapters["policy"] = NaivePolicy()
    with pytest.raises(ContractError, match="gate clarification"):
        Pipeline(**adapters).turn(Session("naive"), "I can do 50", case_id="case")


def test_reference_only_mode_conflict_limitation_and_act_free():
    observer = ReferenceObserver(CachingNLU(CountingNLU({"accept": 1.0})))
    snapshot = Session("ref", history=[Turn("buyer", "60")]).snapshot()
    output = observer.observe(snapshot, "case", "ok 70 works")
    from retrieval_pipeline.fixtures import FixtureGate
    from retrieval_pipeline.pipeline import Observation
    gate = FixtureGate("reference_only").evaluate(snapshot, Observation("case", "ok 70 works", {"accept": 1}, output))
    assert gate.reasons == () and output["price"] is None  # deliberately not a production-safety claim
    assert ReferenceObserver().observe(snapshot, "case", "hello")["status"] == "no_price"


def test_state_identity_and_unknown_liveness_are_preserved():
    observer = ReferenceObserver(CachingNLU(CountingNLU({"accept": 1.0})))
    session = Session("state", state={"buyer_offer_id": "buyer-9", "buyer_price": "60", "agreement": "pending"})
    output = observer.observe(session.snapshot(), "case", "ok deal")
    assert output["referenced_offer_id"] == "buyer-9" and output["offer_liveness"] == "live"
    history = Session("history", history=[Turn("buyer", "60")])
    assert observer.observe(history.snapshot(), "case", "ok deal")["offer_liveness"] == "unknown"
