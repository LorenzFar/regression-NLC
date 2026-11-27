"""Run each turn against detached state and publish successful transitions."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Protocol

from .contracts import ContractError, GenerationResult, PolicyDecision, Turn, validate_generation_request
from .generator import RetrievalGenerator


@dataclass(frozen=True)
class Snapshot:
    history: tuple[Turn, ...] = ()
    state: dict = field(default_factory=dict)
    finished: bool = False
    applied_turn_ids: tuple[str, ...] = ()


class Session:
    """Single-writer session; snapshots and restored sessions never share mutable state."""

    def __init__(self, session_id: str, *, history=(), state=None, finished=False):
        self.session_id = session_id
        self._snapshot = deepcopy(Snapshot(tuple(history), state or {}, finished))

    def snapshot(self) -> Snapshot:
        return deepcopy(self._snapshot)

    @classmethod
    def from_snapshot(cls, session_id: str, snapshot: Snapshot):
        session = cls(session_id)
        session._snapshot = deepcopy(snapshot)
        return session


@dataclass(frozen=True)
class Observation:
    case_id: str
    message: str
    probabilities: dict[str, float]
    price_reference: dict


@dataclass(frozen=True)
class GateDecision:
    mode: str
    reasons: tuple[str, ...] = ()
    threshold: float | None = None


@dataclass(frozen=True)
class Transition:
    state: dict
    finished: bool = False


@dataclass(frozen=True)
class TurnResult:
    generation: GenerationResult
    snapshot: Snapshot
    trace: dict


class NLU(Protocol):
    def predict(self, history: list[tuple[str, str]], message: str) -> dict[str, float]: ...


class PriceReference(Protocol):
    def observe(self, snapshot: Snapshot, case_id: str, message: str) -> dict: ...


class Gate(Protocol):
    def evaluate(self, snapshot: Snapshot, observation: Observation) -> GateDecision: ...


class Policy(Protocol):
    def decide(self, snapshot: Snapshot, observation: Observation, gate: GateDecision) -> PolicyDecision: ...


class Tracker(Protocol):
    def commit(self, snapshot: Snapshot, observation: Observation,
               gate: GateDecision, decision: PolicyDecision) -> Transition: ...


class Generator(Protocol):
    def generate(self, decision: PolicyDecision, history: list[Turn], message: str) -> GenerationResult: ...


class FollowUp(Protocol):
    def resolve(self, snapshot: Snapshot, observation: Observation) -> Observation: ...


# The only state a clarification may record: what it asked about.
PENDING_KEY = "pending_clarification"


def _without_pending(state: dict) -> dict:
    return {key: value for key, value in state.items() if key != PENDING_KEY}


class Pipeline:
    def __init__(self, *, nlu: NLU, price_reference: PriceReference, gate: Gate,
                 policy: Policy, tracker: Tracker, generator: Generator | None = None,
                 follow_up: FollowUp | None = None):
        self.nlu = nlu
        self.price_reference = price_reference
        self.gate = gate
        self.policy = policy
        self.tracker = tracker
        self.generator = generator if generator is not None else RetrievalGenerator()
        self.follow_up = follow_up

    def open(self, session: Session, *, context: str) -> TurnResult:
        """Let the buyer speak first in an empty session.

        context is the public listing text; the generator uses it to pick an
        opening line. Policy chooses the offer, the tracker records it.
        """
        before = session.snapshot()
        if before.history or before.finished or "opening" in before.applied_turn_ids:
            raise ContractError("the buyer can only open an empty, unfinished session")
        decision = self.policy.opening(deepcopy(before))
        validate_generation_request(decision, [], context)
        generation = self.generator.generate(decision, [], context, opening=True)
        if generation.action != decision.action or generation.price != decision.price:
            raise ContractError("generation must preserve the authorized action and price")
        transition = self.tracker.open(deepcopy(before), decision)
        after = Snapshot((Turn("buyer", generation.text),), deepcopy(transition.state), False,
                         before.applied_turn_ids + ("opening",))
        trace = {"session_id": session.session_id, "turn_id": "opening", "policy": asdict(decision),
                 "generation": asdict(generation), "state_before": asdict(before),
                 "state_after": asdict(after), "error": None}
        session._snapshot = after
        return TurnResult(generation, deepcopy(after), trace)

    def turn(self, session: Session, message: str, *, case_id: str,
             turn_id: str | None = None) -> TurnResult:
        before = session.snapshot()
        trace = {
            "case_id": case_id, "session_id": session.session_id, "turn_id": turn_id,
            "components": {name: getattr(component, "provenance", f"adapter:{type(component).__module__}.{type(component).__qualname__}")
                           for name, component in vars(self).items()},
            "state_before": asdict(before), "state_after": asdict(before),
            "timing_ms": {}, "error": None,
        }

        def timed(name, call):
            start = perf_counter()
            try:
                return call()
            finally:
                trace["timing_ms"][name] = (perf_counter() - start) * 1000

        try:
            if before.finished:
                raise ContractError("finished session requires a new or restored session")
            if turn_id is not None and turn_id in before.applied_turn_ids:
                raise ContractError("turn_id already applied")
            if not isinstance(message, str):
                raise ContractError("message must be a string")
            probabilities = timed("nlu", lambda: self.nlu.predict(
                [(turn.speaker.upper(), turn.text) for turn in before.history], message))
            price_reference = timed("price_reference", lambda: self.price_reference.observe(
                deepcopy(before), case_id, message))
            observation = Observation(case_id, message, deepcopy(probabilities), deepcopy(price_reference))
            if self.follow_up is not None:
                resolved = timed("follow_up", lambda: self.follow_up.resolve(deepcopy(before), observation))
                trace["follow_up_applied"] = resolved != observation
                observation = resolved
            trace["probabilities"] = deepcopy(observation.probabilities)
            trace["price_reference"] = deepcopy(observation.price_reference)
            gate = timed("gate", lambda: self.gate.evaluate(deepcopy(before), deepcopy(observation)))
            trace["gate"] = asdict(gate)
            decision = timed("policy", lambda: self.policy.decide(
                deepcopy(before), deepcopy(observation), deepcopy(gate)))
            trace["policy"] = asdict(decision)
            if gate.reasons and decision.action != "clarify":
                raise ContractError("gate clarification requires an authorized clarify decision")
            if gate.reasons and decision.clarification_reason not in gate.reasons:
                raise ContractError("clarification_reason must be one of the supplied gate reasons")
            # Enforce the public request contract even with an injected generator.
            timed("contract_validation", lambda: validate_generation_request(
                decision, list(before.history), message))
            generation = timed("generation", lambda: self.generator.generate(
                decision, list(before.history), message))
            if (not isinstance(generation, GenerationResult)
                or generation.action != decision.action or generation.price != decision.price
                or not isinstance(generation.text, str) or not generation.text.strip()):
                raise ContractError("generation must preserve the authorized action and price with nonempty text")
            trace["generation"] = asdict(generation)
            transition = timed("tracker", lambda: self.tracker.commit(
                deepcopy(before), deepcopy(observation), deepcopy(gate), decision))
            if (not isinstance(transition, Transition) or not isinstance(transition.state, dict)
                or type(transition.finished) is not bool):
                raise ContractError("tracker transition requires dict state and boolean finished")
            if decision.action == "clarify" and (
                _without_pending(transition.state) != _without_pending(before.state)
                or transition.finished or decision.finished
            ):
                raise ContractError("clarification cannot commit disputed state or finish a session")
            if decision.action == "answer" and (
                transition.state != before.state or transition.finished or decision.finished
            ):
                raise ContractError("answer cannot change tracker state or finish a session")
            after = Snapshot(
                before.history + (Turn("seller", message), Turn("buyer", generation.text)),
                deepcopy(transition.state), transition.finished,
                before.applied_turn_ids + ((turn_id,) if turn_id is not None else ()),
            )
            trace["state_after"] = asdict(after)
            result = TurnResult(generation, deepcopy(after), deepcopy(trace))
            session._snapshot = after
            return result
        except Exception as exc:
            trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
            # Preserve the original error type and make failed-turn diagnostics public.
            exc.trace = deepcopy(trace)
            raise
