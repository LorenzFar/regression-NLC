"""Render upstream-authorized buyer decisions without selecting negotiated terms."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .contracts import ContractError, GenerationResult, PolicyDecision, Turn


class RetrievalGenerator:
    """Retrieve reviewed wording; policy/tracker own offers and session state.

    History is oldest-first and excludes the separate current seller message.
    Banks are validated and TF-IDF statistics fitted once, on train contexts only.
    """

    def __init__(self, bank_path=None, history_window=2):
        # Keep the public bank CLI executable without eager package pre-import.
        from .bank import load_bank

        if type(history_window) is not int or history_window < 0:
            raise ContractError("history_window must be a non-negative integer")
        self.bank_path = bank_path
        self.history_window = history_window
        self.bank = [] if bank_path is None else load_bank(bank_path, history_window)
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
        if not isinstance(history, list) or any(
            not isinstance(turn, Turn)
            or not isinstance(turn.speaker, str) or turn.speaker.lower() not in {"buyer", "seller"}
            or not isinstance(turn.text, str)
            for turn in history
        ):
            raise ContractError("history must be a list of Turn(speaker=str, text=str)")
        if not isinstance(message, str):
            raise ContractError("message must be a string")
        if not isinstance(decision, PolicyDecision):
            raise ContractError("decision must be a PolicyDecision")
        if not isinstance(decision.action, str) or decision.action not in {
            "propose", "counter", "accept", "confirm", "reject", "clarify"
        }:
            raise ContractError("unsupported buyer action")
        if type(decision.finished) is not bool:
            raise ContractError("finished must be a boolean supplied by policy/tracker")
        if decision.clarification_reason is not None and decision.action != "clarify":
            raise ContractError("clarification_reason is only supported for clarify")
        if decision.price is not None or decision.action in {"propose", "counter", "accept", "confirm"}:
            if not isinstance(decision.price, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", decision.price):
                raise ContractError("price must be a non-negative SGD decimal string with at most two fractional digits")
        reference = decision.referenced_offer_id
        if reference is not None and (not isinstance(reference, str) or not reference.strip()):
            raise ContractError("referenced_offer_id must be a nonblank string")
        if decision.action in {"accept", "confirm"} and reference is None:
            raise ContractError("referenced_offer_id is required for accept and confirm")
        if decision.action in {"propose", "counter"} and reference is not None:
            raise ContractError("referenced_offer_id is unsupported for propose and counter")
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
            if not isinstance(reason, str) or reason not in generic:
                raise ContractError("clarification_reason must be low_confidence, unresolved_reference, or state_conflict")
            if (reference is None) != (decision.price is None):
                raise ContractError("specific clarification requires both referenced_offer_id and price")
            kind = "generic" if reference is None else "specific"
            text = generic[reason] if reference is None else specific[reason].format(price=decision.price)
            return GenerationResult(
                text=text, action=decision.action, price=decision.price,
                template_id=f"fixed.clarify.{reason}.{kind}", source="clarification",
            )
        compatible = [i for i, row in enumerate(self.bank)
                      if row["action"] == decision.action]
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
                    text = row["template"].format(price=decision.price)
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
            text=templates[decision.action].format(price=decision.price),
            action=decision.action,
            price=decision.price,
            template_id=f"fixed.{decision.action}",
            source="fallback",
            fallback_reason=reason,
        )
