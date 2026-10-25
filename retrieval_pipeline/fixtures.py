"""AUTHORED FIXTURES ONLY: scripted decisions, not autonomous negotiation logic."""
from copy import deepcopy
from dataclasses import dataclass, field

from .contracts import ContractError, PolicyDecision
from .pipeline import GateDecision, Transition


@dataclass(frozen=True)
class FixtureCase:
    probabilities: dict[str, float]
    decision: PolicyDecision
    price_reference: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)
    finished: bool = False
    message: str | None = None


class ScriptedComponents:
    """Case IDs key policy/reference/tracker; exact messages key fixture NLU.

    Ambiguous duplicate messages fail when building fixture NLU adapters.
    Supply adapters(nlu=actual_nlu) to skip fixture NLU construction entirely.
    """

    def __init__(self, cases: dict[str, FixtureCase], *, mode="full", threshold=0.7):
        if mode not in {"off", "reference_only", "full"}:
            raise ContractError("unknown fixture gate mode")
        if not 0 <= threshold <= 1:
            raise ContractError("fixture threshold must be between zero and one")
        self._cases = deepcopy(cases)
        self.mode = mode
        self.threshold = threshold

    def adapters(self, nlu=None):
        if nlu is None:
            messages = {}
            for case_id, case in self._cases.items():
                message = case.message if case.message is not None else case_id
                if message in messages and messages[message] != case.probabilities:
                    raise ContractError("ambiguous fixture NLU message")
                messages[message] = deepcopy(case.probabilities)
            nlu = FixtureNLU(messages)
        return {
            "nlu": nlu,
            "price_reference": FixturePriceReference(self._cases),
            "gate": FixtureGate(self.mode, self.threshold),
            "policy": FixturePolicy(self._cases),
            "tracker": FixtureTracker(self._cases),
        }


class FixtureNLU:
    provenance = "fixture:scripted-nlu"

    def __init__(self, messages):
        self.messages = deepcopy(messages)

    def predict(self, history, message):
        return deepcopy(self.messages[message])


class FixturePriceReference:
    provenance = "fixture:scripted-price-reference"

    def __init__(self, cases):
        self.cases = deepcopy(cases)

    def observe(self, snapshot, case_id, message):
        return deepcopy(self.cases[case_id].price_reference)


class FixtureGate:
    """Illustrative threshold/reference/state logic, NOT the teammate's gate."""
    provenance = "fixture:deterministic-gate"

    def __init__(self, mode="full", threshold=0.7):
        if mode not in {"off", "reference_only", "full"}:
            raise ContractError("unknown fixture gate mode")
        if not 0 <= threshold <= 1:
            raise ContractError("fixture threshold must be between zero and one")
        self.mode, self.threshold = mode, threshold

    def evaluate(self, snapshot, observation):
        reasons = []
        if self.mode != "off":
            if observation.price_reference.get("reference_resolved") is False:
                reasons.append("unresolved_reference")
        if self.mode == "full":
            if observation.price_reference.get("state_conflict") is True:
                reasons.append("state_conflict")
            if max(observation.probabilities.values(), default=0) < self.threshold:
                reasons.append("low_confidence")
        return GateDecision(self.mode, tuple(reasons), self.threshold)


class FixturePolicy:
    provenance = "fixture:scripted-policy"

    def __init__(self, cases):
        self.cases = deepcopy(cases)

    def decide(self, snapshot, observation, gate):
        if gate.reasons:
            return PolicyDecision("clarify", clarification_reason=gate.reasons[0])
        return deepcopy(self.cases[observation.case_id].decision)


class FixtureTracker:
    provenance = "fixture:scripted-tracker"

    def __init__(self, cases):
        self.cases = deepcopy(cases)

    def commit(self, snapshot, observation, gate, decision):
        if decision.action == "clarify":
            return Transition(deepcopy(snapshot.state), False)
        case = self.cases[observation.case_id]
        return Transition(deepcopy(case.state_after), case.finished)
