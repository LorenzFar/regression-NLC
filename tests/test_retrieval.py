"""Actual TF-IDF retrieval through the generation seam, no mocked ranker."""
import pytest

from retrieval_pipeline import ContractError, PolicyDecision, RetrievalGenerator, Turn
from retrieval_pipeline.bank import prepare_candidate, approve_candidate, build_bank, load_bank


def reviewed(template_id, message, response, history=None, action="counter"):
    spans = []
    if "80" in response:
        start = response.index("80")
        spans = [dict(start=start, end=start + 2, value="80", validated=True)]
    return approve_candidate(prepare_candidate(dict(
        template_id=template_id, dialogue_id="fixture", turn_id=template_id,
        origin="authored_fixture", split="train", speaker="BUYER", action=action,
        history=history or [], message=message, response=response, price_spans=spans,
        mixed_action=False, unresolved_reference=False,
    ), {"fixture"}), "Fixture author — semantic review")


def bank_path(tmp_path, rows):
    path = tmp_path / "bank.jsonl"
    build_bank(rows, path, {"fixture"})
    return path


def test_context_selects_compatible_wording_without_changing_exact_price(tmp_path):
    path = bank_path(tmp_path, [
        reviewed("camera", "Pristine camera lens", "For the camera, I offer 80."),
        reviewed("bicycle", "Bicycle wheels frame", "For the bicycle, I offer 80."),
        reviewed("accept", "Pristine camera lens", "I accept 80.", action="accept"),
    ])
    generator = RetrievalGenerator(path)
    camera = generator.generate(PolicyDecision("counter", "80.50"), [], "Pristine camera lens")
    bicycle = generator.generate(PolicyDecision("counter", "80.50"), [], "Bicycle wheels frame")
    assert camera.template_id == "camera"
    assert bicycle.template_id == "bicycle"
    assert camera.text == "For the camera, I offer 80.50."
    assert bicycle.text == "For the bicycle, I offer 80.50."
    assert camera.source == bicycle.source == "retrieval"
    assert camera.similarity > 0
    assert camera.action == bicycle.action == "counter"
    assert camera.price == bicycle.price == "80.50"


def test_window_is_shared_and_candidate_response_does_not_leak(tmp_path):
    history = [dict(speaker="SELLER", text="ancient secretfuture"),
               dict(speaker="BUYER", text="camera lens 70"),
               dict(speaker="SELLER", text="camera lens pristine")]
    path = bank_path(tmp_path, [
        reviewed("camera", "firm 100", "For the camera, I offer 80.", history),
        reviewed("bike", "firm 200", "For the bike, I offer 80.",
                 [dict(speaker="BUYER", text="bicycle wheel"),
                  dict(speaker="SELLER", text="bicycle wheel")]),
    ])
    stored = load_bank(path)[0]
    assert stored["context"] == "camera lens  price \ncamera lens pristine\nfirm  price "
    assert "ancient" not in stored["context"]
    assert "I offer" not in stored["context"]
    result = RetrievalGenerator(path).generate(PolicyDecision("counter", "79.25"),
        [Turn("SELLER", "bicycle wheel " * 100), Turn("BUYER", "camera lens 3"),
         Turn("SELLER", "camera lens pristine")], "firm 4")
    assert result.template_id == "camera"
    assert result.text == "For the camera, I offer 79.25."
    with pytest.raises(ContractError, match="history_window"):
        RetrievalGenerator(path, history_window=0)


def test_stable_ties_fallbacks_and_clarification_are_observable(tmp_path):
    rows = [reviewed("z", "camera", "I can offer 80."),
            reviewed("a", "camera", "My offer is 80.")]
    path = bank_path(tmp_path, rows)
    gen = RetrievalGenerator(path)
    assert gen.generate(PolicyDecision("counter", "80.00"), [], "camera").template_id == "a"
    build_bank(list(reversed(rows)), path, {"fixture"})
    assert RetrievalGenerator(path).generate(PolicyDecision("counter", "80.00"), [], "camera").template_id == "a"
    absent = gen.generate(PolicyDecision("confirm", "80.00", "buyer-1"), [], "camera")
    zero = gen.generate(PolicyDecision("counter", "80.00"), [], "unrelatedtoken")
    assert (absent.source, absent.action, absent.fallback_reason) == ("fallback", "confirm", "no_compatible_candidate")
    assert absent.text == "Thank you for accepting my offer of 80.00."
    assert (zero.source, zero.action, zero.fallback_reason) == ("fallback", "counter", "zero_similarity")
    clarified = gen.generate(PolicyDecision("clarify", clarification_reason="low_confidence"), [], "camera")
    assert clarified.source == "clarification"
    assert clarified.similarity is None
    empty = bank_path(tmp_path, [])
    assert RetrievalGenerator(empty).generate(PolicyDecision("reject"), [], "camera").source == "fallback"


def test_rejection_never_gains_a_price_slot(tmp_path):
    path = bank_path(tmp_path, [reviewed("reject", "camera", "I must decline.", action="reject")])
    result = RetrievalGenerator(path).generate(PolicyDecision("reject", "80.50", "seller-1"), [], "camera")
    assert (result.text, result.template_id, result.price) == ("I must decline.", "reject", "80.50")
