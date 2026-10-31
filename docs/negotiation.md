# Buyer dialogue manager: state tracker, confidence gate, policy

Design doc for the rule-based components that replace the scripted gate, policy and tracker in `retrieval_pipeline/fixtures.py`. This doc is the source of truth for the implementation; update it when behaviour changes.

- **Module:** `retrieval_pipeline/negotiation.py` (new): `OfferTracker`, `ConfidenceGate`, `RulePolicy`, `RegexPriceReference` (interim), `build_buyer_pipeline`.
- **Unchanged:** `pipeline.py`, `generator.py`, `contracts.py`, fixtures and replay.
- **Owner:** Ercong (see README handoffs).

## 1. Where each component sits

One seller turn runs through `Pipeline.turn`:

```
seller message
  -> NLU               predict(history, message) -> {act: prob}        existing
  -> Price/reference   observe(snapshot, case_id, message) -> dict      interim (Edric)
  -> ConfidenceGate    evaluate(snapshot, observation) -> GateDecision  NEW
  -> RulePolicy        decide(snapshot, observation, gate) -> PolicyDecision  NEW
  -> RetrievalGenerator generate(decision, history, message)            existing
  -> OfferTracker      commit(snapshot, observation, gate, decision) -> Transition  NEW
```

`Pipeline.turn` publishes the new state and both public turns together, and only after generation and the transition both validate. Any component can read `snapshot.state`. Only `OfferTracker.commit` proposes a new one. The pipeline already rejects a policy decision that ignores the gate's reasons, and a clarification that changes state or finishes the session.

Seller acts (NLU labels): `propose`, `counter`, `accept`, `reject`, `inquire`, `vague`.
Buyer actions (generator contract): `propose`, `counter`, `accept`, `confirm`, `reject`, `clarify`.

`price_reference` keys read by these components:

| Key | Type | Meaning |
|---|---|---|
| `price` | decimal string, optional | Price resolved from the seller message |
| `reference_resolved` | bool, optional | `False` means the message refers to an offer that could not be resolved |
| `referenced_offer_id` | string, optional | Offer the seller message refers to |

## 2. State tracker (`OfferTracker`)

### State shape

Offers are stored as term-value pairs, plus the agreement status. The state is plain JSON so replay snapshots keep working.

```json
{"offers": [
   {"id": "seller-1", "by": "seller", "terms": {"price": "120.00"}, "status": "superseded"},
   {"id": "buyer-2",  "by": "buyer",  "terms": {"price": "80.00"},  "status": "live"}
 ],
 "agreement": "none"}
```

- `agreement` is `none`, `pending` (the buyer accepted a seller offer and is waiting for the seller to acknowledge) or `confirmed`.
- `{}` is the initial state: no offers, `agreement="none"`.
- A new offer marks every live offer `superseded`, so at most one offer is live after each commit.
- Offer ids are `f"{by}-{len(offers) + 1}"`. They are deterministic, so the state needs no counter. One helper, `next_offer_id(state, by)`, is shared by the tracker and the policy.
- To add a term later, add a key to `terms`. No other stage changes.

### Update rules (`commit`)

If `decision.action == "clarify"`, return the state unchanged with `finished=False`.

Otherwise apply two steps in order:

**1. Seller move** (NLU top act + resolved price)

| Seller act | Effect |
|---|---|
| `propose` / `counter` with a price | Append a live seller offer (supersedes older ones); `agreement = "none"` |
| `reject` | Supersede the live buyer offer |
| `accept`, `inquire`, `vague` | No change to offers |

**2. Buyer decision**

| Buyer action | Effect |
|---|---|
| `propose` / `counter` | Append a live buyer offer at `decision.price` |
| `accept` | `agreement = "pending"` |
| `confirm` | `agreement = "confirmed"`, `finished = True` |
| `reject` | Supersede the live seller offer |

### Main rule: `acceptance_target(state, price_reference) -> offer | None`

A seller acceptance confirms a deal only when **exactly one** live offer exists for it to refer to. The offer returned must be either:

- the buyer's live offer, or
- the live seller offer the buyer already accepted (`agreement == "pending"`).

It returns `None` when there are zero or several live offers, when the live offer isn't one the seller can accept, or when `price_reference` names a different `referenced_offer_id` or a different `price`. Prices are compared as `Decimal`, so `"95"` matches `"95.00"`.

The gate and the policy both call this one function, so the rule is defined in one place.

## 3. Confidence gate (`ConfidenceGate`)

`ConfidenceGate(threshold=0.7, mode="full").evaluate(snapshot, observation) -> GateDecision`

The gate runs before the policy and returns the reasons to ask for clarification. Empty reasons let the policy negotiate. The checks run in this order:

| # | Reason | Condition | Modes |
|---|---|---|---|
| 1 | `unresolved_reference` | `reference_resolved is False`, or the top act is `propose`/`counter` and no price was resolved | `reference_only`, `full` |
| 2 | `state_conflict` | The top act is `accept` and `acceptance_target(...)` is `None` | `full` |
| 3 | `low_confidence` | The NLU's top probability is below τ | `full` |

`mode` (`off` / `reference_only` / `full`) mirrors `FixtureGate`, so the existing three-mode replay compares gate settings without new code. The gate uses full-precision probabilities.

## 4. Policy (`RulePolicy`)

`RulePolicy(target, walk_away, fraction).decide(snapshot, observation, gate) -> PolicyDecision`

The policy is given a target and a walk-away price. It accepts, rejects, or counters. A counter concedes a fixed fraction of the gap toward the seller's price and never goes past the walk-away price. All money uses `Decimal`. Concessions round down to the cent, and prices are emitted as two-decimal strings (`"95.00"`) that pass the generator's price check.

### Concession formula

```
b = buyer's last offered price (live or superseded), or None
p = seller's price

c = target                                    if b is None
c = min(walk_away, b + fraction * (p - b))    otherwise
```

### Decision tree

| Situation | Decision |
|---|---|
| Gate has reasons | `clarify` with `gate.reasons[0]`. If a live offer exists, attach its id and price, so the generator uses the offer-specific template |
| Seller `accept` | `confirm(offer.price, offer.id, finished=True)` with the offer from `acceptance_target` |
| Seller `propose`/`counter` at `p`, and `p <= c` | `accept(p, next_offer_id(state, "seller"))`. This covers `p <= target`, and the buyer never offers more than the seller asks |
| Seller `propose`/`counter` at `p`, and `b == walk_away` | `reject` (no room left) |
| Seller `propose`/`counter` at `p`, otherwise | `counter(c)` |
| Seller `reject` / `inquire` / `vague`, with a seller price on record | The same concession step toward the latest seller price |
| Seller `reject` / `inquire` / `vague`, with no seller price on record | `propose(b or target)` |

Invariants:

- No buyer price is ever above `walk_away`.
- The buyer opens at `target`.
- `finished` is set only by `confirm`. A buyer `reject` does not end the session.

## 5. Generator (`RetrievalGenerator`, already built)

The generator filters the response bank to buyer utterances of the chosen action. It ranks them by TF-IDF cosine similarity against the last two turns plus the current message, and fills in the policy's price in the `{price}` slot. Clarifications use fixed templates, one for each gate reason. A valid request with no match falls back to a fixed template for the same action.

The pipeline checks that the generated action and price equal the decision, so the wording cannot change the terms of the deal.

## 6. Integration

| Piece | What it does |
|---|---|
| `build_buyer_pipeline(nlu, price_reference, *, target, walk_away, fraction="0.5", threshold=0.7, mode="full", bank=None)` | The single entry point. Returns `Pipeline(nlu, price_reference, gate=ConfidenceGate(threshold, mode), policy=RulePolicy(target, walk_away, fraction), tracker=OfferTracker(), generator=RetrievalGenerator(bank))` |
| `RegexPriceReference` | Interim adapter until Edric's resolver lands, and labelled as such. Reuses `nlu._PRICE_RE`. One bare decimal → `{"price": ..., "reference_resolved": True}`. Several prices → `{"reference_resolved": False}`. No price → `{}`. The replacement only needs the same `observe()` method |
| Replay | `replay_cases(cases, lambda mode, t: build_buyer_pipeline(..., mode=mode, threshold=t))` works unchanged. Cases for the real components need state in the `offers`/`agreement` shape above. The existing fixture cases use the old ad-hoc shape and stay on the fixtures |
| `examples/negotiation_contract.py` | Runs the worked trace below through `build_buyer_pipeline`. By default it uses stub NLU probabilities. With `--checkpoint PATH` it uses `NLUAdapter`. Prints buyer replies, gate reasons and the final state |
| README | Adds these components to the code map, updates the "Still scripted" line, and marks the policy/gate/tracker handoff done |

```python
from retrieval_pipeline.negotiation import RegexPriceReference, build_buyer_pipeline
from retrieval_pipeline.nlu_adapter import NLUAdapter
from retrieval_pipeline.pipeline import Session

pipeline = build_buyer_pipeline(
    NLUAdapter("models/nlu_k2_s0.pt"), RegexPriceReference(),
    target="80", walk_away="100", fraction="0.5",
    bank="examples/approved_fixture_bank.jsonl",
)
session = Session("demo")
result = pipeline.turn(session, "I'm asking 120 for it.", case_id="t1", turn_id="s1")
```

## 7. Worked trace

Target 80, walk-away 100, fraction 0.5, τ 0.7.

| Turn | Seller message | NLU top act | Gate | Buyer decision | State after |
|---|---|---|---|---|---|
| t1 | "I'm asking 120 for it." | propose 0.91 | none | `counter 80.00` (opens at target) | seller-1 120 superseded; buyer-2 80 live; agreement none |
| t2 | "Lowest I'd go is 110." | counter 0.88 | none | `counter 95.00` = min(100, 80 + 0.5 × 30) | seller-3 110 superseded; buyer-4 95 live |
| t3 | "ok" | accept 0.55 | `low_confidence` | `clarify` on buyer-4: "Could you clarify what you mean about the offer of 95.00?" | unchanged |
| t4 | "Yes, 95 works for me." | accept 0.93 | none | `confirm 95.00` on buyer-4, finished | agreement confirmed; finished true |

Variations at t4:

- No live buyer offer → `state_conflict` clarification.
- "Deal at 90" (the price doesn't match buyer-4) → `state_conflict`.
- The seller holds at 102 after the buyer has reached 100 → buyer `reject`.

## 8. Verification

- `python -m pytest -q`: the existing suites still pass. No new tests are planned in this pass.
- `python -m examples.negotiation_contract`: the run ends with `agreement=confirmed` and `finished=True`, no buyer price is above walk-away, and an "ok" with no live buyer offer gets a `state_conflict` clarification.
- `python -m examples.negotiation_contract --checkpoint models/nlu_k2_s0.pt`: inspect the traces with the real NLU.
