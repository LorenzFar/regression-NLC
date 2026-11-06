"""Render upstream-authorized buyer decisions without selecting negotiated terms."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .contracts import ContractError, GenerationResult, PolicyDecision, Turn, validate_generation_request


class RetrievalGenerator:
    """Retrieve reviewed wording; policy/tracker own offers and session state.

    History is oldest-first and excludes the separate current seller message.
    Banks are validated and TF-IDF statistics fitted once, on train contexts only.
    Provisional agent-reviewed banks require explicit allow_provisional=True.
    """

    def __init__(self, bank_path=None, history_window=2, *, allow_provisional=False):
        # Keep the public bank CLI executable without eager package pre-import.
        from .bank import load_bank

        if type(history_window) is not int or history_window < 0:
            raise ContractError("history_window must be a non-negative integer")
        self.bank_path = bank_path
        self.history_window = history_window
        self.bank = ([] if bank_path is None else load_bank(
            bank_path, history_window, allow_provisional=allow_provisional))
        self.vectorizer = TfidfVectorizer()
        contexts = [row["context"] for row in self.bank]
        # Tokenless contexts are valid but cannot support lexical retrieval.
        self.features = (self.vectorizer.fit_transform(contexts)
                         if any(self.vectorizer.build_analyzer()(c) for c in contexts) else None)

    def generate(self, decision: PolicyDecision, history: list[Turn], message: str) -> GenerationResult:
        """Return authorized wording, or fail without changing inputs/state.

        finished is validated, not inferred or applied here. A live offer's
        identity/ownership is an upstream reference/tracker responsibility.
        """
        validate_generation_request(decision, history, message)
        if decision.action == "clarify":
            generic = {
                "low_confidence": "Could you clarify what you mean?",
                "unresolved_reference": "Which offer are you referring to?",
                "state_conflict": "Could you clarify the current status of our negotiation?",
            }
            specific = {
                "low_confidence": "Could you clarify what you mean about the offer of {price}?",
                "unresolved_reference": "Are you referring to the offer of {price}?",
                "state_conflict": "Could you clarify the status of the offer of {price}?",
            }
            reason = decision.clarification_reason
            reference = decision.referenced_offer_id
            kind = "generic" if reference is None else "specific"
            text = generic[reason] if reference is None else specific[reason].format(price=_display(decision.price))
            return GenerationResult(
                text=text, action=decision.action, price=decision.price,
                template_id=f"fixed.clarify.{reason}.{kind}", source="clarification",
            )
        # Accept and confirm both agree to one price, so they share wording.
        pool = {"accept": {"accept", "confirm"}, "confirm": {"accept", "confirm"}}.get(
            decision.action, {decision.action})
        # Do not repeat wording the buyer already used in this conversation.
        said = {_shape(turn.text) for turn in history if turn.speaker.lower() == "buyer"}
        compatible = [i for i, row in enumerate(self.bank)
                      if row["action"] in pool and _shape(row["template"]) not in said]
        reason = "no_bank" if self.bank_path is None else "no_compatible_candidate"
        if compatible:
            reason = "zero_similarity"
            if self.features is not None:
                from .bank import retrieval_context

                context = retrieval_context(history, message, self.history_window)
                scores = cosine_similarity(self.vectorizer.transform([context]), self.features[compatible])[0]
                ranked = sorted(zip(compatible, scores), key=lambda pair: (-pair[1], self.bank[pair[0]]["template_id"]))
                index, score = ranked[0]
                if score > 0:
                    row = self.bank[index]
                    text = row["template"].format(price=_display(decision.price))
                    return GenerationResult(text=text, action=decision.action, price=decision.price,
                                            template_id=row["template_id"], source="retrieval",
                                            similarity=float(score), origin=row["origin"],
                                            source_dialogue_id=row["dialogue_id"],
                                            source_turn_id=row["turn_id"])
        templates = {
            "propose": "I can offer {price}.",
            "counter": "Would you take {price} instead?",
            "accept": "I accept your offer of {price}.",
            "confirm": "Thank you for accepting my offer of {price}.",
            "reject": "I cannot accept that offer.",
        }
        return GenerationResult(
            text=templates[decision.action].format(price=_display(decision.price)),
            action=decision.action,
            price=decision.price,
            template_id=f"fixed.{decision.action}",
            source="fallback",
            fallback_reason=reason,
        )


def _display(price):
    """Write whole amounts as people do ("80", not "80.00"); the value is unchanged."""
    return price[:-3] if price is not None and price.endswith(".00") else price


def _shape(text):
    """Wording with numbers and slots blanked, for spotting repeated templates."""
    return re.sub(r"\{price\}|\d+(?:\.\d+)?", "#", text).strip().lower()
