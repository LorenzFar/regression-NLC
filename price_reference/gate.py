"""Add resolver checks to a collaborator gate without changing its implementation."""
from dataclasses import replace


class ReferenceSafetyGate:
    """Compose with a gate that may only inspect price/id rather than resolver status.

    Full mode additionally blocks resolver conflicts, required validation and act
    disagreement. Off/reference_only retain the documented ablation semantics;
    use full mode for deployment. This wrapper never approves disputed conditions
    or alters policy/tracker state.
    """

    def __init__(self, gate):
        self.gate = gate
        self.provenance = f"reference-safety:{getattr(gate, 'provenance', type(gate).__name__)}"

    def evaluate(self, snapshot, observation):
        decision = self.gate.evaluate(snapshot, observation)
        if decision.mode == "off":
            return decision
        reference = observation.price_reference
        reasons = set(decision.reasons)
        if reference.get("reference_resolved") is False:
            reasons.add("unresolved_reference")
        if decision.mode == "full":
            if reference.get("state_conflict") is True or reference.get("status") == "conflict":
                reasons.add("state_conflict")
            elif reference.get("requires_validation") or reference.get("act_disagreement"):
                reasons.add("unresolved_reference")
        ordered = [reason for reason in ("unresolved_reference", "state_conflict", "low_confidence") if reason in reasons]
        ordered.extend(reason for reason in decision.reasons if reason not in ordered)
        return replace(decision, reasons=tuple(ordered))
