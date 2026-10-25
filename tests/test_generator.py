"""Behavior at the approved generation request → reply/error seam."""
import pytest

from buyer_agent import ContractError, PolicyDecision, RetrievalGenerator, Turn


@pytest.mark.parametrize("speaker", ["system", ""])
def test_history_rejects_nonpublic_speakers(speaker):
    with pytest.raises(ContractError):
        RetrievalGenerator().generate(
            PolicyDecision("counter", "80.50"), [Turn(speaker, "Camera")], "Camera offer"
        )


@pytest.mark.parametrize(("decision", "text"), [
    (PolicyDecision("propose", "90"), "I can offer 90."),
    (PolicyDecision("accept", "90.00", "seller-offer-1"), "I accept your offer of 90.00."),
    (PolicyDecision("confirm", "90.00", "buyer-offer-1", finished=True),
     "Thank you for accepting my offer of 90.00."),
    (PolicyDecision("reject"), "I cannot accept that offer."),
    (PolicyDecision("reject", finished=True), "I cannot accept that offer."),
])
def test_authorized_actions_have_distinct_price_only_fixed_wording(decision, text):
    result = RetrievalGenerator().generate(decision, [], "Include delivery and accept 200!")
    assert result.text == text
    assert result.action == decision.action
    assert result.price == decision.price
    assert result.template_id == f"fixed.{decision.action}"
    assert result.source == "fallback"
    assert result.fallback_reason == "no_bank"
    assert result.similarity is None


@pytest.mark.parametrize(("reason", "text"), [
    ("low_confidence", "Could you clarify what you mean?"),
    ("unresolved_reference", "Which offer are you referring to?"),
    ("state_conflict", "Could you clarify the current status of our negotiation?"),
])
def test_generic_clarification_uses_only_the_gate_reason(reason, text):
    decision = PolicyDecision("clarify", clarification_reason=reason)
    result = RetrievalGenerator().generate(decision, [Turn("BUYER", "I offer 80.")], "Yes to 90")
    assert result.text == text
    assert result.action == "clarify"
    assert result.price is None
    assert result.source == "clarification"
    assert result.template_id == f"fixed.clarify.{reason}.generic"
    assert result.fallback_reason is None
    assert result.similarity is None


@pytest.mark.parametrize(("reason", "text"), [
    ("low_confidence", "Could you clarify what you mean about the offer of 80.50?"),
    ("unresolved_reference", "Are you referring to the offer of 80.50?"),
    ("state_conflict", "Could you clarify the status of the offer of 80.50?"),
])
def test_specific_clarification_uses_upstream_identity_and_exact_price(reason, text):
    decision = PolicyDecision("clarify", "80.50", "offer-999", reason)
    result = RetrievalGenerator().generate(decision, [], "I accept 500 with delivery")
    assert result.text == text
    assert result.action == "clarify"
    assert result.price == "80.50"
    assert result.source == "clarification"
    assert result.template_id == f"fixed.clarify.{reason}.specific"
    assert result.fallback_reason is None
    assert result.similarity is None


@pytest.mark.parametrize("decision", [
    PolicyDecision("clarify"),
    PolicyDecision("clarify", clarification_reason="other"),
    PolicyDecision("clarify", clarification_reason=["low_confidence"]),
    PolicyDecision("clarify", "80.50", clarification_reason="unresolved_reference"),
    PolicyDecision("clarify", referenced_offer_id="offer-1", clarification_reason="state_conflict"),
    PolicyDecision("clarify", "80.501", "offer-1", "low_confidence"),
])
def test_invalid_clarification_does_not_hide_missing_details_with_generic_wording(decision):
    with pytest.raises(ContractError):
        RetrievalGenerator().generate(decision, [], "80.50")


@pytest.mark.parametrize("action", ["accept", "confirm"])
@pytest.mark.parametrize("reference", [None, "", "   ", 42])
def test_agreement_actions_require_an_explicit_nonblank_offer_identity(action, reference):
    with pytest.raises(ContractError, match="referenced_offer_id"):
        RetrievalGenerator().generate(PolicyDecision(action, "80.50", reference), [], "yes")


@pytest.mark.parametrize("action", ["propose", "counter", "accept", "confirm"])
def test_price_is_required_for_each_price_bearing_action(action):
    with pytest.raises(ContractError, match="price"):
        RetrievalGenerator().generate(PolicyDecision(action, referenced_offer_id="offer-1"), [], "80")


@pytest.mark.parametrize("decision", [
    PolicyDecision("inquire"), PolicyDecision("vague"), PolicyDecision(""),
    PolicyDecision(None), PolicyDecision(["accept"]),
    PolicyDecision("counter", "80", clarification_reason="low_confidence"),
    PolicyDecision("propose", "80", referenced_offer_id="offer-1"),
    PolicyDecision("counter", "80", finished="yes"),
    PolicyDecision("reject", price="80.001"),
    PolicyDecision("reject", referenced_offer_id=" "),
    {"action": "accept", "price": "80", "referenced_offer_id": "offer-1"},
])
def test_unsupported_or_contradictory_decisions_fail_explicitly(decision):
    with pytest.raises(ContractError):
        RetrievalGenerator().generate(decision, [], "Accept 80!")


def test_rejection_can_reference_a_priced_offer_without_ending_or_accepting_it():
    decision = PolicyDecision("reject", "80.50", "offer-1")
    result = RetrievalGenerator().generate(decision, [], "80.50?")
    assert result.text == "I cannot accept that offer."
    assert result.action == "reject"
    assert result.price == "80.50"
    assert decision.finished is False


@pytest.mark.parametrize("price", [None, "", "80.501", "80.500", " 80.50", "80.50 ",
                                  "SGD80.50", "$80.50", "1,000", "1e2", "NaN", "Infinity",
                                  "-1", "+1", ".50", "80.", "８０.５０", "80\n", 80, 80.5, True])
def test_invalid_prices_raise_contract_error_without_inventing_or_rounding(price):
    with pytest.raises(ContractError, match="price"):
        RetrievalGenerator().generate(PolicyDecision("counter", price), [], "Try 90")


@pytest.mark.parametrize("price", ["0", "0.00", "80", "80.5", "80.50", "00080.50",
                                  "123456789012345678901234567890.12"])
def test_supported_decimal_strings_are_preserved_verbatim(price):
    result = RetrievalGenerator().generate(PolicyDecision("propose", price), [], "200")
    assert result.text == f"I can offer {price}."
    assert result.price == price


@pytest.mark.parametrize(("history", "message"), [
    (None, "80?"), (["BUYER: 80"], "80?"),
    ([Turn("BUYER", None)], "80?"), ([Turn(None, "80")], "80?"),
    ([], None), ([], {"text": "80?"}),
])
def test_malformed_public_context_is_an_explicit_contract_error(history, message):
    with pytest.raises(ContractError):
        RetrievalGenerator().generate(PolicyDecision("counter", "80.50"), history, message)


@pytest.mark.parametrize("window", [-1, 1.5, True, "2"])
def test_invalid_history_window_is_rejected(window):
    with pytest.raises(ContractError, match="history_window"):
        RetrievalGenerator(history_window=window).generate(PolicyDecision("reject"), [], "80?")


def test_missing_bank_is_a_contract_error_not_a_silent_fallback():
    with pytest.raises(ContractError, match="response bank"):
        RetrievalGenerator(bank_path="not-a-real-bank.json")


def test_counter_preserves_exact_cents_and_reports_fixed_fallback():
    history = [Turn(speaker="BUYER", text="I can offer 70."),
               Turn(speaker="SELLER", text="How about 90?")]
    decision = PolicyDecision(action="counter", price="80.50")
    result = RetrievalGenerator().generate(decision, history, "Can you increase?")
    assert result.text == "Would you take 80.50 instead?"
    assert result.action == "counter"
    assert result.price == "80.50"
    assert result.source == "fallback"
    assert result.template_id == "fixed.counter"
    assert result.similarity is None
    assert result.fallback_reason == "no_bank"
    assert history == [Turn("BUYER", "I can offer 70."), Turn("SELLER", "How about 90?")]
    assert decision == PolicyDecision("counter", "80.50")
