# Reference resolver, rules-v1

The extractor reads amounts in one message. The resolver determines the message's
price or identifies a prior offer. The tracker owns committed offers, liveness and
agreement; policy owns the buyer's response. Resolving an offer is not authorization
to accept it. Existing CSVs, NLU and teammate pipeline files are not changed here.

## Core usage

```python
from price_reference import build_ledger, resolve
from price_reference.evidence import collect_mentions

history = [("buyer", "Would you take $50?")]
ledger = build_ledger(history)
result = resolve("I accept your offer of $50", speaker="seller", act="accept",
                 mentions=collect_mentions("I accept your offer of $50"), ledger=ledger)
assert result.price == "50.00"
assert result.offer_liveness == "unknown"
```

All money uses nonnegative Decimal strings with two decimal places. Conversion
rounds half-up (1.005 -> 1.01), not through binary float. Free is 0.00; a delta can
produce zero, but a negative price cannot resolve. Source offsets are Python
character indices, end exclusive, covering the original surface. No typo-corrected
or context-inferred amount acquires a fictitious span.

## Evidence and rules

An offer proposes an amount for the item or states a seller's asking price.
Budgets, refused amounts, comparison values and labelled fees/deposits are context.
Buyer "I have 500" is a budget; seller "I have it listed for 500"/"I have 500
firm" is an asking price. Strong budget wording (only/just have, budget, afford)
remains context for either side. Seller "I can't do less than 450" states a floor,
not a refusal of 450. Negated acceptance/pay/want cues demote their local amounts.
Floors include "I won't accept anything less than 450" and "I'm not taking less
than 450", "I can't go any lower than 450", "Not going lower than 450", and
"I can't let it go for less than 450". Comparison bounds ("Can you go higher/better than 450?", "up from
450") do not propose 450 and do not create historical offers. Defined seller
floors are protected from that filtering; a delta's amount is not a comparison
bound merely because its following words contain "less than".
Offer questions ("would you take 50?", "how about 50?") remain proposals even if NLU
says inquire/vague/reject/accept; `act_disagreement` records this override. Explicit
acceptance wording is matched against the other side's offers. `kind="reference"`
on an extractor mention means a comparison price, not a pointer to a prior offer.
An acceptance-like pointer ("ok that works", "ok deal") paired with a propose/counter
act is contradictory evidence, not an implicit repeat of the speaker's own price:
it requests clarification. Unowned "that price" is also unresolved. Explicit own
repeats ("same as before", "my previous offer") and "your previous offer" retain
their distinct owners. The team's standalone-unresolved-reference case therefore
clarifies with its scripted counter act without changing the teammate fixture.

One clear price resolves. Multiple surviving prices, alternatives, ranges and
corrections remain ambiguous; rules-v1 does not guess the first or last price.
A single conditional price retains both the price and condition; policy must
decide whether the condition is satisfied. A proposal without a number is only
resolved as a reference when a pointer identifies a unique prior price.

Acceptances without a written amount target the other side's latest clean offer.
A later ambiguous turn blocks that implicit target in history-only reconstruction;
authoritative tracker state takes precedence when present. An explicit price may identify
an older offer or one candidate of an ambiguous turn; unmatched prices conflict.
Pointers with an unknown dialogue act clarify. Act-free mode only resolves a
single explicit non-demoted amount and never infers a reference from offer presence.
Exception supported by tracker state: if the buyer already accepted a seller's
single live offer (agreement=pending), a seller acknowledgment confirms that offer.
OfferLedger.pending_acceptance records this authoritative reference; it is not
inferred just from "ok" or from the seller having once stated a price.

Deltas use local direction words. "Your" targets the other side, "my" or an
unqualified delta targets the speaker's latest clean offer. Missing, conflicting or
ambiguous bases/directions clarify. The base offer identity is preserved separately.

Only item-price wording creates a zero for "free" (e.g. "it's free", "you can
have it for free", "free to a good home"). Free shipping, parking, or "free of
scratches" are not zero offers. Generic "great deal"/"what's the deal with the
dent?" is not an acceptance pointer. Material pickup/payment conditions are
retained even across their separating comma; conversational "if interested" and
historical "when new" are not attached to the current price.
An explicit referential cancellation in the immediately following sentence
("Is 450 ok? No, it isn't") withdraws that price. An explicit replacement
("No, make it 500") preserves the replacement rather than choosing the old price.
This is scoped lexical handling, not general discourse inference; item-condition
negation such as "it isn't broken" does not cancel a price.

Mixed explicit plus delta statements ("I'll do 450, that's 50 off") remain
ambiguous in rules-v1. Listing price is not silently used as an unqualified delta
base; it may differ from the standing offer against which the speaker is discounting.

## Offer identity and liveness

History IDs use the ordinal of the owner's *turn*: buyer-offer-3 need not be the
third buyer offer. This is a local scheme, not an assertion about teammate numbering.
State IDs are echoed verbatim. A state offer links to history only when exactly
one owner+price match exists. Repeated equal amounts do not prove originating turn.
Unknown history indices stay None. The ledger accepts Turn objects, speaker/text
dicts and (speaker, text) tuples; roles are case-insensitive.

History alone cannot prove an offer is live. Superseded prices are not_live;
otherwise history liveness is unknown. Flat state buyer_offer_id/buyer_price and
seller equivalents are supported; agreement="pending" establishes live under this
adapter's documented default. Explicit `<owner>_offer_live=False`, withdrawn or
rejected/confirmed state marks not_live. A state price differing from that owner's
latest clean historical offer, or contradicted by a later ambiguous historical turn,
is flagged state_history_disagreement, not assumed stale: an unseen withdrawal could
legitimately explain the discrepancy. This flag is advisory: it does not change
tracker-reported liveness, suppress the state offer or override reference selection.
It sets requires_validation so policy must check the tracker before acting.
Legacy flat-key behavior remains supported. Structured tracker support follows.

The teammate branch's structured tracker schema is also supported:

```json
{"offers": [{"id": "buyer-4", "by": "buyer", "terms": {"price": "95.00"}, "status": "live"}], "agreement": "none"}
```

An explicit offers list is the complete authoritative membership, including when
empty; it takes precedence over legacy flat keys/history reconstruction. IDs are
echoed verbatim (their global numbering is not reconstructed), status=live stays
live, and superseded/withdrawn/rejected records stay not_live. History only supplies
unique owner+price links and advisory disagreement. Malformed structured state raises
instead of silently falling back to guessed history. For this schema agreement is
distinct from offer status: confirmed does not rewrite a recorded live status.
A sole live seller offer with agreement=pending establishes pending_acceptance.

Clarification turns (including all fixed generator clarification templates) and
quoted-price inquiries do not create new offers. Merely matching the other side's
price does not suppress a genuine proposal: "I offer 90" still creates an offer.
Filtering is clause-scoped: "Did you say 90? I can do 85" yields the offer 85.
There is no whole-turn clarification/proposal-lexicon gate: "What do you mean?
I can go 85", "85 is my max" and even an otherwise unrecognized price clause
remain available as offer evidence after the quoted inquiry is demoted. Accepted
other-side offers are excluded per clause in the ledger, preserving later proposals.
Regression tests render every generic and specific clarification through the actual
RetrievalGenerator rather than relying only on copied strings, guarding wording drift.
Skipped turns still count toward turn-ordinal IDs; offer membership and numbering
are separate issues.

Policy must validate unknown liveness against its tracker before accept/confirm.
An identified not_live price may be reopened as a new proposal, not automatically
accepted. The resolver reports evidence but does not silently perform that policy
conversion. Rejection/withdrawal without structured state is not reconstructed.

`Resolution.liveness_established` reports only that the identified reference is
tracker-established live. It says nothing about conditions, gate outcomes or
whether a deal may be confirmed. `requires_validation` is true for clarification
statuses, unknown/not_live referenced offers, conditions or state/history disagreement.
An explicit new offer is not an established *acceptance reference*; policy may still
propose/counter at its price. The observer's state_history_disagreement and
requires_validation apply to the selected price/reference/base only. A separate
ledger_state_history_disagreement reports unrelated ledger concerns for diagnostics;
it does not automatically require validation of an independent new explicit price.
Before accept/confirm, policy must require resolved status, no gate reasons,
tracker-established live validity, satisfied conditions and its own tracker check;
it must handle requires_validation explicitly. Liveness is necessary, not sufficient.
These fields are advisory, NOT structural enforcement; a policy that ignores them
is not protected. Integration checks are limited to the documented revision and
cases below; full-mode wiring must still be verified after the teammate merge.

Compatibility only: the old offer_actionable field/property remains a deprecated,
conservative eligibility hint (live reference and no outstanding validation).
state_history_mismatch remains a deprecated read-only property/output alias for
state_history_disagreement; gold loading accepts either spelling. New callers should
use the factual liveness_established and advisory disagreement/validation fields.

## Pipeline wiring and clarification

```python
from price_reference import CachingNLU, ReferenceObserver
shared_nlu = CachingNLU(actual_nlu)
observer = ReferenceObserver(nlu=shared_nlu, listing_price=None)
# Pipeline(nlu=shared_nlu, price_reference=observer, gate=..., policy=..., tracker=...)
```

The shared bounded LRU uses exactly the pipeline's uppercase history tuples and
message. It returns detached prediction dicts. Clear it after changing model or
calibration settings. NLU must be in inference mode; the wrapper does not configure
a model or assert that stochastic predictions are deterministic. It does not cache
resolution/state. Listing price comes from the constructor or state listing_price;
absence means no plausibility hint. No NLU means act-free mode.

`price` exists iff resolver status is resolved. `referenced_offer_id` exists only
when that resolved price refers to a prior offer; an explicit new offer has no
prior ID. A delta uses `base_offer`. Unresolved/ambiguous output sets
reference_resolved=False. Conflict sets reference_resolved=True and
state_conflict=True so the fixture gate preserves the conflict reason.
Non-resolved output withholds actionable price/ID but may expose clarify_context.

Low NLU confidence belongs to the gate. A resolved price remains evidence even
when that gate requests clarification. Pipeline.turn rejects a policy that ignores
gate reasons. The authored fixture tests prove this integration contract; they
do not establish blanket compatibility with every teammate revision. Fixture reference_only mode ignores
conflicts, so it is not sufficient to enforce all resolver outcomes. A custom
policy must inspect status/liveness/conditions, not merely price presence.

## Teammate integration and the difference from scripted fixtures

"109" and "109.00" are numerically identical Decimal amounts and both satisfy
the generator's at-most-two-decimal contract. Canonical resolver output stays at
two decimal places. Scripted fixture probabilities/reference outputs are authored
demonstration inputs, not authoritative gold for every real component.

After the teammate negotiation module is integrated, use the safe builder without
editing its implementation. It shares the NLU and installs the full safety gate:

```python
from price_reference import build_safe_buyer_pipeline

pipeline = build_safe_buyer_pipeline(actual_nlu, target="80", walk_away="100")
```

The helper requires full mode and imports the teammate negotiation module lazily;
it is available after that module is merged, not in the current standalone branch.
It does not change the teammate's builder or its defaults. Existing manually wired
pipelines still need ReferenceSafetyGate around their gate. Experimental ablations
must use the lower-level teammate builder explicitly.

ReferenceSafetyGate composes the collaborator gate's decision with resolver conflict,
act-disagreement and validation checks in full mode. In particular, a gate that
recomputes acceptance from state but ignores the resolver's state_conflict flag
must not confirm just because the resolver withheld its conflicting price/id.
The wrapper adds state_conflict, causing the existing pipeline guard to require
clarification. Conditions/history-only validity still need validation; the wrapper
conservatively clarifies rather than treating a known price as authorization.
Off/reference_only remain intentional ablation modes, not full production safety.
It does not modify tracker state or decide that a condition is satisfied.

A one-time integration check against origin/feature/policy-gate at bd97c25 used
the actual negotiation module in memory with scripted NLU (no teammate files
copied or changed). It verified counter 80 -> counter 95 -> low-confidence clarify
-> confirm 95, pending seller-offer acknowledgment, and conflict -> clarification
with the wrapper. The same conflict confirmed incorrectly without the wrapper.
These are contract checks at that revision, not real-NLU or corpus accuracy; repeat
against the final merged teammate implementation. Portable tests also substitute
the real resolver into the unchanged standalone fixture and exercise the structured
state schema. Reviewed hard gold and an unseen holdout are still required.

## Gold and review workflow

```powershell
.venv/Scripts/python.exe -m scripts.resolver_gold_seed --split validation --out evaluation/resolver/review.jsonl
.venv/Scripts/python.exe -m evaluation.resolver.run --gold evaluation/resolver/review.jsonl --split validation --conditions A D --include-silver --json evaluation/resolver/report.json
```

Seeding never overwrites a file or modifies a CSV. It independently proposes simple
lexical targets and a silver ledger from earlier single-price offer-like rows.
Seeded acts carry act_source="lexical_seed"; history speakers are lowercase to
match the team schema (the NLU adapter uppercases them only at its own boundary).
These are hypotheses, not verified ground truth. Hard rows have review_required=true
and are excluded until a reviewer supplies act, status, price, reference identity,
conditions and an independently checked ledger, then clears review_required.
Gold mentions must also include refused/comparison/delta amounts when those are
needed as resolver inputs; CSV gold_price alone does not supply all mentions.

Every JSONL row includes case_id, history, message, act, price, referenced_offer,
needs_clarification, deal_agreed, challenge and label_source. Extensions include
split, dialogue_id, status, subset, offers, ambiguous_turns, mentions, conditions,
offer_liveness, inferred and reviewer. Label sources: seeded_rule, ai_proposed,
human_reviewed (requires a real reviewer's name). Never promote machine annotations
to human_reviewed automatically. Raw message text is preserved.

Annotated mentions provide value, kind, original surface and start/end, or surface
plus an explicit zero-based occurrence_index. The loader verifies exact original
characters. CSV value-only mentions have unknown kind and no spans; several
unlocated mentions remain ambiguous and unlocated deltas cannot be computed.
Inferred values retain provenance=gold_inferred. Neither "none" nor an empty cell
is an inferred numerical zero: none is a reviewed negative, blanks require review,
and uncertain notes exclude a row. The extractor's separate unread-FP report does
not automatically distinguish why a cell is blank.

Offers contain ref (offer_id, owner, turn_index, id_source), price, liveness and
link. Flat forms are also accepted by the loader. For authoritative gold offers,
use link=state_only or state_linked_by_price if asserting live validity. Offers=[]
explicitly records an empty gold ledger; missing offers is not silently equivalent.

## Evaluation and limits

Run the four small, authored contract examples (not model or corpus results):

```powershell
.venv/Scripts/python.exe -m evaluation.resolver.run --gold evaluation/resolver/examples.jsonl --split validation
```

These adapt the team template shape with explicit input speaker and recorded
fixture acts. Team template-001's message accepts a seller offer, so its current
speaker must be buyer; treating every team example as a seller message changes
its meaning. The production observer still receives seller messages, as required
by Pipeline.turn. Always annotate speaker when importing mixed-role gold.

Conditions: A gold mentions/act/ledger; B gold mentions/ledger with predicted act;
C predicted mentions/act/ledger; D gold mentions/act with predicted ledger.
Use --nlu module:factory for a real NLU or provide recorded predicted_act or
nlu_probabilities per row. Missing predictions are reported as skipped, never
replaced with the gold act. A is oracle-upstream, not a guaranteed ceiling.

Metrics separately measure status, price, comparable reference identity, liveness,
clarification precision/recall, candidate recall and act disagreement. Conditional
gold prices on clarification rows are candidate targets, not forced unsafe picks.
Identity compares owner+turn_index when available; authoritative state IDs can be
compared verbatim. Alias-number comparison requires identity_numbering_confirmed.
Uncomparable identities are counted separately. Headline metrics require hard,
human_reviewed rows. Silver, inferred and conditional slices are reported separately.
Paired condition tables describe observed changes, not causal error percentages.

Silver is excluded from ALL conditions by default. --include-silver opts into
non-independent lexical-seed diagnostics, never reviewed accuracy: seeder and
resolver share linguistic cues even though they do not import each other's decisions.
The terminal prints results by label_source, not an unqualified mixed-source overall.
The full JSON retains an explicitly labelled diagnostic aggregate for inspection.
Rates include their own numerators, denominators and descriptive Wilson 95% intervals.
These are case/mention-level intervals, not adjusted for dialogue clustering; use a
dialogue-level resampling analysis for inferential claims about correlated turns.
Headline publication requires at least --headline-min-n hard human-reviewed cases
(default 100, a reporting policy rather than a guarantee of representativeness).
Below that minimum the headline reports only eligibility/counts, no accuracy.
The headline status is insufficient_n until it is reportable. If every row was
excluded, eligibility explicitly says no_eligible_cases; each empty condition says
no_evaluable_cases rather than presenting null rates as a result.
The common_subset report evaluates every requested condition on exactly the same
case IDs, alongside per-condition skips and paired tables. An empty intersection
is reported honestly when some predictions or ledgers are missing.
The pairwise_subsets report complements that all-selected intersection: it gives
shared case IDs, n and per-source metrics for each compared pair. Thus A/D metrics
remain available when C has no NLU predictions, without silently redefining the
common_subset or comparing different populations. Empty pairs remain explicitly n=0.
Pairwise shared cases additionally receive two-sided exact McNemar tests for
status correctness and price correctness, separately per label source, with n and
discordant counts. Missing price targets are excluded from that endpoint only.
These tests are exploratory, case-level, not dialogue-cluster adjusted and not
corrected for multiple comparisons. They do not establish causal error attribution.

Train/validation/test labels were development-exposed. Holdout requires a freeze
manifest containing file_sha256, dialogue_ids, resolver_commit, frozen_at, exposure.
The runner verifies the file hash and IDs; the manifest alone cannot prove unseen
provenance or disjointness. Source new dialogues, establish disjointness and freeze
before reading them; do not claim untouched status for already exposed dialogues.
No holdout run or human adjudication is fabricated by implementation tests.

Rules-v1 deliberately does not repair malformed amounts, infer omitted thousands,
handle every pragmatic cue, or replace tracker lifecycle logic. Test failures and
reviewed validation errors should drive extensions. Consider a learned ranker only
after >=500 reviewed training cases, below-target (proposed 85%) multi-candidate
dev performance caused by selection errors, and a measured validation benefit.
