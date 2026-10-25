"""Run: python -m examples.conversation_contract (from the repository root)."""
from retrieval_pipeline.contracts import PolicyDecision, Turn
from retrieval_pipeline.fixtures import FixtureCase, ScriptedComponents
from retrieval_pipeline.pipeline import Pipeline, Session


def main():
    cases = {"recorded-turn": FixtureCase(
        probabilities={"accept": 0.6999999999},
        decision=PolicyDecision("confirm", "100.25", "buyer-1", finished=True),
        price_reference={"reference_resolved": True},
        state_after={"agreement": "confirmed"}, finished=True,
        message="Sounds good.",
    )}
    # Reconstruct from recorded context, not a previous replay variant's output.
    recorded = Session("recorded", history=[Turn("buyer", "I can offer 100.25.")],
                       state={"agreement": "pending", "buyer_offer_id": "buyer-1"}).snapshot()
    outcomes = {}
    for mode in ("off", "reference_only", "full"):
        session = Session.from_snapshot(f"recorded:{mode}", recorded)
        pipeline = Pipeline(**ScriptedComponents(cases, mode=mode).adapters())
        result = pipeline.turn(session, "Sounds good.", case_id="recorded-turn", turn_id="seller-2")
        outcomes[mode] = result.generation.action
        assert len(recorded.history) == 1
    assert outcomes == {"off": "confirm", "reference_only": "confirm", "full": "clarify"}
    print(f"AUTHORED FIXTURE: independent sessions from recorded context: {outcomes}")


if __name__ == "__main__":
    main()
