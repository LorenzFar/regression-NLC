# Buyer negotiation agent

A buyer chatbot for CS425 that negotiates price with a human seller. You play the seller; the bot is the buyer. It makes the first offer, reads each of your messages, decides what to do with simple rules, and replies using sentences written by real buyers in the CraigslistBargain dataset. When it isn't sure what you meant, it asks instead of guessing.

**Working end to end:** checkpoint NLU (Farrell), price extraction and reference resolution (Edric, `price_reference/`), rule-based tracker, gate and policy (Ercong, Weijie, `retrieval_pipeline/negotiation.py`), and retrieval generation from a 546-template CraigslistBargain bank (Jiong). `retrieval_pipeline/agent.py` connects them; the web demo and the RQ2 replay both use it. Gold-set results are still pending (see Remaining work).

## How it works

Every seller message goes through six steps:

| Step | Owner | What it does | Example |
|---|---|---|---|
| 1. Understand | Farrell | DistilBERT labels the message as propose, counter, accept, reject, inquire or vague, using the last few turns as context | "Lowest I'd go is 270" → counter (95%) |
| 2. Find the price | Edric | Extracts the price, resolves "same as before", and flags prices with conditions | "250 if you collect today" → 250, conditional |
| 3. Decide whether to ask | Weijie | The clarification gate asks when the NLU is under 70% sure, a reference or condition is unclear, or the message conflicts with the offers | 51% sure → "Could you clarify…?" |
| 4. Decide what to do | Weijie | Rules with a target and walk-away price choose accept, counter or reject, and the price | Seller 265, bot offered 240 → counter 252 |
| 5. Write the reply | Jiong | Picks the real buyer sentence that best fits the conversation and inserts the policy's price | "That seems a little pricey. Would you be willing to accept 252?" |
| 6. Remember the deal | Ercong | Tracks every offer and whether a deal is agreed; confirms only when it's clear which offer was accepted | seller 300, buyer 240, seller 265 … |

The wording only fills in the price the rules chose, so it can never change the terms of the deal.

## How we evaluate it

| Question | How | Status |
|---|---|---|
| RQ1: does the NLU do better with earlier messages? | Rule parser vs NLU without history vs NLU with 1–3 turns, macro-F1 on the hand-labelled gold set, 3 seeds | Needs gold set and extra checkpoints |
| RQ2: does asking stop wrong deal confirmations without too many extra questions? | Same seller turns with the gate off / reference only / full (`retrieval_pipeline/gold_replay.py` → `evaluation/gate_eval.py`) | Runs end to end; needs gold set. On 12 authored practice cases: off 2 wrong confirmations, full 0 with 6 questions |
| Price finding | Exact-match price and correct-reference rate (`evaluation/price_extractor`, `evaluation/resolver`) | In repo |
| Reply wording | 489 held-out real buyer turns: overlap with the real reply, variety, price kept (`scripts/evaluate_generator.py`) | Retrieval 0.216 vs fixed sentences 0.205 overlap; variety 0.249 vs 0.011; price kept 100% |
| Rules | Tests: never above walk-away, price unchanged, correct deal state | Passing |

## Chatbot demo

You play the seller; the bot is the buyer and opens with an offer at its target price. Each buyer reply shows the NLU probabilities, extracted price, gate decision, policy action, the corpus sentence the wording came from, the offer ledger, and what the bot would have said with the gate off.

```powershell
$env:HF_HUB_OFFLINE = "1"   # after the DistilBERT tokenizer is cached once
.\.venv\Scripts\python.exe -m demo_app.server --port 8000
# open http://127.0.0.1:8000
```

`--scripted-nlu` swaps in a keyword stand-in (labelled in the header) when torch or the checkpoint is unavailable. `--bank PATH` picks another response bank.

### Things to try (presentation walkthrough)

You play the seller and type your own replies; the buttons under the chat are only examples. Click any buyer message to see, step by step, why it said that.

Use the **City road bike** preset (listed 300; buyer target 240, walk-away 280), gate **Full**, τ 0.7, concession fraction 0.5. Every reply below was produced by the real checkpoint.

| You type (seller) | Buyer replies | What to point out |
|---|---|---|
| *(start a new negotiation)* | Hi! I saw your listing. Would you be willing to take 240? | The buyer moves first, at its target. Ledger: seller-1 300 (listing), buyer-2 240 |
| `That's too low. This bike is pretty new. How about 265?` | That seems a little pricey. Would you be willing to accept 252? | Counter halfway from 240 to 265, rounded down to whole dollars. The wording is a real CraigslistBargain buyer line |
| `That's way too low, 265 still` | yeah i could do 265, that sounds great! | The seller repeated 265, which is within walk-away, so the policy treats it as final and accepts |
| `250 if you can collect today` | Could you clarify whether 250 depends on that condition? | The resolver finds a price tied to a condition; the buyer only negotiates price, so the gate asks |
| `No, I would want 250` | Great then 250 it is then. | A plain 250 is cheaper than the 265 already accepted, so it accepts |
| `ok` | Ok thanks, We'll do 250. | Confirmed, deal banner, input locked |

Clarification (start a new negotiation for each row group):

| You type | Buyer replies | What to point out |
|---|---|---|
| `Deal!` | Okay I can do 240 thanks. | Confirms the opening offer, because there is exactly one live offer |
| `I can do 270.` then `yes` | Are you referring to the offer of 270? → How about 255? | The NLU reads "I can do 270" as accepting a price that is not on the table, so it asks. "yes" confirms 270 and the buyer counters it |
| `270 if you can collect today` then `no, 270 either way` | Could you clarify whether 270 depends on that condition? → How about 255? | Restating the price it asked about counts as an answer |

RQ2 in one exchange: start one negotiation with gate **Full** and one with gate **Off**, and type `I can do 280.` then `ok` in both.

| Gate | After `I can do 280.` | After `ok` |
|---|---|---|
| Off | That seems a little pricey. Would you be willing to accept 240? | Ok, 240 it is. → deal at 240, which the seller never agreed to |
| Full | Are you referring to the offer of 280? | How about 260? (280 is now on record); a further `ok` confirms 260 |

### Settings explained

- **Buyer target:** the buyer's opening offer and the price it hopes for.
- **Walk-away:** the most it will ever pay. It never offers above this and rejects once it has reached it and the seller still wants more.
- **Concession fraction:** each counter moves this share of the way from the buyer's last offer to the seller's latest price (0.5 = halfway). Counters are rounded down to whole dollars. The buyer accepts when the seller is at or below its next counter, when the seller repeats a price within walk-away, or, after it has countered at least once, when the gap left is at most 5% of the seller's price.
- **Clarification gate:** decides whether to ask instead of acting. *Full* asks when the NLU's confidence is below τ, when a reference ("same as before") or a conditional price is unresolved, or when the message conflicts with the offers on the table. *Reference only* asks only about references and conditions. *Off* never asks (the RQ2 baseline).
- **Confidence threshold τ:** the gate asks when the NLU's top probability is below τ. Propose and counter are added together, since both mean "the seller named a price".

## What the integration added

| Area | Change | Where |
|---|---|---|
| Integration | All team branches merged; one factory builds NLU → price/reference → gate → policy → generator → tracker | `retrieval_pipeline/agent.py` |
| Response bank | 546 buyer templates from CraigslistBargain train dialogues, labelled by the He et al. rule parser, 10% of dialogues held out | `scripts/build_craigslist_bank.py`, `examples/craigslist_bank.jsonl` |
| Generator | Accept and confirm share wording; no repeated templates within a conversation; whole prices print as `80`; opening line with greeting | `retrieval_pipeline/generator.py` |
| Buyer opens | Buyer offers its target first; listing price recorded as the seller's asking offer | `Pipeline.open`, `RulePolicy.opening`, `OfferTracker.open` |
| Policy | After at least one counter, accept a seller price within walk-away when the gap to the next counter is ≤ 5% (`close_gap`); demo counters in whole dollars (`whole_dollars`) | `RulePolicy` |
| Gate | Confidence for offers = P(propose) + P(counter) | `offer_confidence`, `ConfidenceGate` |
| Clarification | Ask about the seller's named price; conditional prices ("250 if you collect today") get their own question; the buyer remembers the asked price, and a following "yes", "ok" or restated price confirms it | `ClarificationFollowUp`, `pending_clarification`, `conditional_price` |
| Evaluation | Generator comparison on held-out turns; full-pipeline RQ2 replay for the gold set | `scripts/evaluate_generator.py`, `retrieval_pipeline/gold_replay.py` |
| Demo | Local chat UI: equal-height scrolling panels, step-by-step explanation of each reply, gate-off comparison, `?say=msg1\|msg2` replays a scripted conversation | `demo_app/` |

Design details for the negotiation changes: `docs/negotiation.md` → Integration changes.

## Setup and run

Use Python 3.11. From the repository root in PowerShell:

```powershell
# Copy mode avoids OneDrive hardlink errors.
$env:UV_LINK_MODE = "copy"
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe "torch>=2.14" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -e ".[test,nlu]"

# No checkpoint needed for these scripted examples and tests.
.\.venv\Scripts\python.exe -m examples.standalone_contract
.\.venv\Scripts\python.exe -m examples.retrieval_contract
.\.venv\Scripts\python.exe -m pytest -q
```

NLU follows the team's requirements: `torch>=2.14` and `transformers>=5.19`. Linux/macOS uses `.venv/bin/python`. Run examples from the checkout; they are not installed command-line tools.

For real NLU, copy the team's `nlu_k2_s0.pt` to `models/`. Its SHA-256 is `41758d4d44553f6d83af6589c788de4dc7eacadd16bf28907f87909e0d416e36`. Load only trusted checkpoints. First loading needs the DistilBERT config/tokenizer from Hugging Face; later runs can use its cache. `--model-name PATH` accepts a compatible local config/tokenizer directory.

```powershell
.\.venv\Scripts\python.exe -m examples.standalone_contract --checkpoint models/nlu_k2_s0.pt --output artifacts/standalone-demo.jsonl
.\.venv\Scripts\python.exe -m retrieval_pipeline.replay examples/standalone_cases.jsonl --checkpoint models/nlu_k2_s0.pt --bank examples/approved_fixture_bank.jsonl --output artifacts/standalone-replay.jsonl
```

The demo covers two retrieval matches, one fixed fallback, and one clarification. Replay runs the cases in three gate modes. `--checkpoint` replaces only scripted NLU; unfinished components remain scripted. Missing resources produce an error, not synthetic predictions. Models and generated traces stay out of Git.

## Code map

| File | Responsibility |
|---|---|
| `nlu.py` | Team NLU: masking, checkpoint loading, seller-act predictions |
| `retrieval_pipeline/nlu_adapter.py` | Pipeline NLU adapter and resource metadata |
| `retrieval_pipeline/contracts.py` | Public turn, buyer action, and reply types |
| `retrieval_pipeline/generator.py` | Action-compatible retrieval and fixed wording |
| `retrieval_pipeline/bank.py` | Prepare, approve, validate, and load response banks |
| `retrieval_pipeline/pipeline.py` | Run one turn and publish state/history together |
| `retrieval_pipeline/fixtures.py` | Scripted collaborators for examples and tests |
| `retrieval_pipeline/replay.py` | Independent recorded cases and JSONL export |
| `examples/`, `tests/` | Runnable contracts, authored cases, and regression tests |
| `retrieval_pipeline/negotiation.py` | Offer tracker, recorded-dialogue tracking, confidence gate, concession policy, `build_buyer_pipeline` |
| `docs/negotiation.md` | Design and worked trace for the tracker, gate, and policy |
| `price_reference/` | Price extraction, offer ledger, reference resolver, resolver safety gate |
| `baseline_parser/` | Rule-based parser from He et al. (2018), the RQ1 baseline |
| `retrieval_pipeline/agent.py` | `build_agent`: all components in one pipeline, used by the demo and RQ2 replay |
| `retrieval_pipeline/gold_replay.py` | Replays gold seller turns through the full pipeline in each gate mode |
| `scripts/build_craigslist_bank.py` | Builds the corpus response bank and the held-out generator cases |
| `scripts/evaluate_generator.py` | Compares fixed, random and retrieved wording on held-out turns |
| `demo_app/` | Local web chat demo (standard library server, one HTML page) |
| `evaluation/` | RQ1/RQ2 metrics, price-extractor and resolver evaluation |

## Generator contract

```python
from retrieval_pipeline import PolicyDecision, RetrievalGenerator

generator = RetrievalGenerator("examples/approved_fixture_bank.jsonl")
result = generator.generate(
    PolicyDecision(action="counter", price="80.50"),
    history=[], message="Pristine camera lens",
)
print(result.text)
```

`history` is an oldest-first list of `Turn(speaker, text)`, excluding the current message. Speakers are `buyer` or `seller` (case-insensitive for generation and bank preparation; replay snapshots use lowercase). Full history is retained. Retrieval uses the last two turns plus the current message by default; NLU has its own checkpoint-defined window.

| Decision field | Rule |
|---|---|
| `action` | `propose`, `counter`, `accept`, `confirm`, `reject`, or `clarify` |
| `price` | Nonnegative SGD decimal string, at most two decimal places. Required for propose/counter/accept/confirm. Never rounded. |
| `referenced_offer_id` | Nonblank offer ID for accept/confirm. Upstream validates ownership and whether the offer remains valid. |
| `clarification_reason` | For clarify: `low_confidence`, `unresolved_reference`, or `state_conflict`. Offer-specific clarification needs both offer ID and price. |
| `finished` | Supplied by policy; the tracker transition decides whether the session ends. |

Retrieval filters by action, then ranks preceding contexts with TF-IDF/cosine similarity. Context numbers are masked; response text is not. Positive-score ties use template ID. Only the authorized price fills the template. Clarification uses fixed wording. A valid request without a match gets a same-action fallback; invalid requests raise `ContractError`.

Results include `text`, `action`, `price`, `template_id`, `source`, `similarity`, `fallback_reason`, and provenance (`origin`, `source_dialogue_id`, `source_turn_id`). `source` is `retrieval`, `fallback`, or `clarification`; retrieval does not imply corpus origin.

## Pipeline contract

```python
from retrieval_pipeline.pipeline import Pipeline, Session

pipeline = Pipeline(nlu=nlu, price_reference=price_reference, gate=gate,
                    policy=policy, tracker=tracker, generator=generator)
result = pipeline.turn(Session("conversation-1"), seller_message,
                       case_id="case-1", turn_id="seller-1")
```

| Component | Method |
|---|---|
| NLU | `predict(history, message) -> dict[str, float]` |
| Price/reference | `observe(snapshot, case_id, message) -> dict` |
| Gate | `evaluate(snapshot, observation) -> GateDecision` |
| Policy | `decide(snapshot, observation, gate) -> PolicyDecision` |
| Generator | `generate(decision, history, message) -> GenerationResult` |
| Tracker | `commit(snapshot, observation, gate, decision) -> Transition` |
| Follow-up (optional) | `resolve(snapshot, observation) -> Observation`, runs before the gate |

`pipeline.open(session, context=listing_text)` lets the buyer speak first: it calls `policy.opening(snapshot)` and `tracker.open(snapshot, decision)`, and the generator picks an opening line. A clarification may only add `pending_clarification` to the state; offers and agreement stay unchanged.

NLU receives uppercase `(speaker, text)` history tuples. The gate gets full-precision probabilities. `Transition` contains a state dictionary and boolean `finished`. Collaborators compute from detached inputs; `commit` computes a proposed local transition, not an external database write.

State and both public turns are published only after generation and transition validation succeed. Failure leaves the session unchanged and exposes `exception.trace`. Clarification preserves agreement state and cannot finish a session. Policy must select one of the gate's requested clarification reasons. Finished sessions reject new turns. Explicit turn IDs prevent duplicate application; failed turns can be retried. Sessions support one writer, not concurrent or durable storage.

## Response bank

Use only buyer responses from an authoritative training-dialogue allowlist. Never include validation/test responses. The process is **prepare → human review → approve → build**:

```powershell
.\.venv\Scripts\python.exe -m examples.retrieval_contract --export-sources artifacts/bank-work
.\.venv\Scripts\python.exe -m retrieval_pipeline.bank prepare artifacts/bank-work/sources.jsonl artifacts/bank-work/candidates.jsonl --train-ids artifacts/bank-work/train-ids.json
# Inspect candidates and save only selected rows as reviewed.jsonl.
.\.venv\Scripts\python.exe -m retrieval_pipeline.bank approve artifacts/bank-work/reviewed.jsonl artifacts/bank-work/approved.jsonl --reviewer "Reviewer name"
.\.venv\Scripts\python.exe -m retrieval_pipeline.bank build artifacts/bank-work/approved.jsonl artifacts/bank-work/bank.jsonl --train-ids artifacts/bank-work/train-ids.json
```

The exported sources above are authored examples. Corpus sources need stable `template_id`, `dialogue_id`, `turn_id`, `origin="corpus"`, `split="train"`, `speaker="BUYER"`, explicit buyer `action`, preceding `history`, seller `message`, buyer `response`, and explicit false `mixed_action`/`unresolved_reference` flags. Priced actions need one validated `{start, end, value, validated: true}` price span; reject needs none. Offsets use Python characters with an exclusive end. The value must match the bare decimal substring exactly.

Only that span becomes `{price}`. The builder excludes multiple prices, unsupported terms, and mixed actions. These checks do not prove meaning: a person must review real templates. Approval records that decision; it does not perform review. Loading checks training IDs, source consistency, approvals, unique IDs, and context window. Corrupt banks fail explicitly. Use the same `--history-window` in preparation, building, and generation.

### CraigslistBargain response bank

`examples/craigslist_bank.jsonl` holds 546 buyer templates (158 propose, 264 counter, 106 accept, 6 confirm, 12 reject) from the official training dialogues. Build it from the official parsed train file (`artifacts/craigslist/parsed.json`, see the script docstring for the URL):

```powershell
.\.venv\Scripts\python.exe scripts\build_craigslist_bank.py artifacts\craigslist\parsed.json --output examples\craigslist_bank.jsonl --heldout-output artifacts\craigslist\heldout_cases.jsonl
.\.venv\Scripts\python.exe scripts\evaluate_generator.py artifacts\craigslist\heldout_cases.jsonl --output artifacts\generator-eval.json
```

A buyer message that directly answers a seller message becomes a template when the He et al. parser intent maps to a buyer action, its only number equals the parser's price (agreement may have no number), it has 3 to 25 words, it does not name items or extra terms (pickup, days, "final offer", "can't go higher"), and it passes `bank.prepare_candidate`. 10% of dialogues, chosen by UUID hash, are held out and never enter the bank. Labels come from the parser, so rows carry `review_status="parser_labelled"` and load only with `allow_provisional=True` (`build_agent` sets it). Human review of the bank is still open.

Generator behaviour: retrieval ranks same-action templates by TF-IDF similarity of the last two turns plus the seller message; accept and confirm share wording; templates the buyer already used in the conversation are skipped; the buyer's opening adds "Hi! I saw your listing." and skips templates that react to a seller turn ("Ok.", "Sounds great!"); whole prices print as `80`, not `80.00` (the decision price is unchanged).

Held-out results (489 buyer turns; every system gets the human buyer's action and price):

| System | Token F1 vs human reply | Distinct-2 | Price kept |
|---|---:|---:|---:|
| Fixed fallback sentences | 0.205 | 0.011 | 100% |
| Random template, same action | 0.207 | 0.280 | 100% |
| TF-IDF retrieval (used) | 0.216 | 0.249 | 100% |
| TF-IDF uni+bigram | 0.211 | 0.258 | 100% |

Retrieval gives varied, human-sounding wording while always stating the policy's price. Word overlap with the human reply barely separates the systems, so context similarity adds little measurable gain on this metric; a small human rating would be the better test.

### Provisional corpus retrieval (local trial only)

`examples/provisional_craigslist_bank.jsonl` contains **7 real Craigslist buyer
turns (2 propose, 5 counter)** selected from the preserved 12-row
`examples/craigslist_review_shortlist.jsonl`. Original responses, public context,
dialogue/event IDs, numeric spans and source-file hashes are retained. Only the
validated numeric substring becomes `{price}`; response text never enters ranking.

The bank is **agent-reviewed, not human-approved**: `reviewer="agent:Hermes"`,
`review_status="agent_reviewed_local_trial"`, `approved=false`,
`human_approval=false`, `split_status="provisional_official_train"`.
`build_bank`, `load_bank`, and `RetrievalGenerator` require explicit
`allow_provisional=True` for these rows; their strict defaults and the human
approval CLI reject them. Opt-in does not approve the untouched shortlist.

```powershell
.\.venv\Scripts\python.exe -m examples.corpus_contract
.\.venv\Scripts\python.exe -m pytest -q tests/test_corpus_preview.py tests/test_bank.py
```

The preview actually loads this corpus bank. Seven original pre-response contexts
prove propose/counter retrieval and exact replacement with policy price `73.27`;
these are **deterministic plumbing proofs, not held-out quality evaluation**.
It also shows zero-match fallbacks for both actions, fixed accept/confirm/clarify
wording, and three new seller phrases labeled exploratory. No checkpoint, LLM or
network is needed. Source prices remain in provenance only, never rendered replies.
Team split alignment, human semantic review, and currency/SGD suitability remain
pending. Local activation is not production approval or a claim of improved quality;
that requires a matched held-out evaluation with the team-aligned split.

## Replay

Replay runs recorded seller turns as independent test cases. Each restores its own pre-turn history/state. Generated replies never become input to the next recorded case.

See `examples/standalone_cases.jsonl` for the input shape. Each row needs nonblank `case_id`, `conversation_id`, `turn_id`, `origin`, `reconstruction_mode`, and `message`; a `snapshot` with public `history` and tracker `state`; and a `fixture` with scripted results. The CLI needs `fixture`; a real `replay_cases(cases, pipeline_factory)` caller can supply its own components.

Gate modes are `off`, `reference_only`, and `full`; threshold defaults to `0.7`. The scripted full gate checks unresolved references, state conflicts, and low confidence. Real factories must keep unrelated settings fixed and isolate mutable state between cases.

Output has one JSONL record per case/mode, including failures. Traces contain component identities, probabilities, policy, state before/after, generation/provenance, timing, and errors. Exit codes: **0** all cases succeeded; **1** at least one failed; **2** invalid command/input-file setup. Malformed rows report errors without stopping valid rows. A generated candidate in a failed trace was not committed. Fixtures are not gold evaluation data.

## Evaluation (RQ1 and RQ2)

`evaluation/` turns gold labels and the files this repo already writes into the numbers for the report. It needs no checkpoint to run its own tests. Schema and field meanings: `evaluation/data.py`; example rows: `examples/gold_template.jsonl` (authored placeholders, not gold data).

```powershell
# RQ1: baseline predictions file + one checkpoint per variant/seed ("label@seed" pools seeds into mean +/- std)
.\.venv\Scripts\python.exe -m evaluation.nlu_eval --gold gold.jsonl --predictions parser=parser_preds.jsonl --system k0@0=models/nlu_k0_s0.pt --system k2@0=models/nlu_k2_s0.pt --output artifacts/rq1.json

# RQ2: replay gold seller turns through the FULL pipeline in all gate modes, then score them
.\.venv\Scripts\python.exe -m retrieval_pipeline.gold_replay gold.jsonl --checkpoint models/nlu_k2_s0.pt --output artifacts/replay.jsonl
.\.venv\Scripts\python.exe -m evaluation.gate_eval --replay artifacts/replay.jsonl --gold gold.jsonl --thresholds 0.5 0.6 0.7 0.8 0.9 --output artifacts/rq2.json
```

RQ1 reports accuracy, macro-F1 with bootstrap CI, per-class F1, confusion matrix, per-challenge-tag accuracy, calibration (ECE, NLL, Brier, risk-coverage), exact McNemar and a paired bootstrap of the macro-F1 difference against the baseline. RQ2 reports incorrect and correct confirmations, clarification count, extra turns, unnecessary clarifications, gate precision/recall/F1 and a tau sweep with the expected result of a random gate that asks equally often. The sweep re-derives the gate from recorded probabilities and flags, so it assumes the policy differs from `off` only by emitting `clarify`. It does not include the resolver checks or tracker state conflicts of the integrated gate, so report `by_mode` for the real comparison.

`gold_replay` rebuilds the pre-turn tracker state with `track_turns`: from gold `act`/`price` labels when every history turn has them, otherwise from predictions (documented in the module). Gold rows may carry `listing_price`, `target` and `walk_away`; otherwise the listing price is estimated from the largest amount mentioned. `examples/challenge_demo.jsonl` has 12 authored cases (`origin="authored_demo"`, not gold) for showing the mechanics: with the real checkpoint, gate off made 2 incorrect confirmations, full gate made 0 at the cost of 6 clarifications (1 unnecessary).

## Remaining work

| Owner | Needed |
|---|---|
| Barry | Hand-labelled gold set (with history act/price labels and listing prices) to run RQ1 and RQ2 for real |
| Farrell | NLU training script and the k0/k1/k3 checkpoints over three seeds for RQ1 |
| Jiong | Human check of the parser-labelled response bank (optional; the rule parser labels are the documented source) |

## Domain language

- **Seller act:** interpreted seller intent; it does not authorize a buyer response.
- **Buyer action:** policy-authorized response, including price or clarification request.
- **Agreement state:** the tracker's assessment of a pending or confirmed deal.
- **Acceptance:** the buyer agrees to a particular seller offer.
- **Confirmation:** the buyer acknowledges seller acceptance of a particular buyer offer.
- **Rejection:** declines an offer; it need not end negotiation.
- **Finished session:** policy/tracker ended negotiation, with or without agreement.
- **Response template:** reusable buyer wording with an authorized price slot when needed.
- **Response bank:** approved buyer response templates from training conversations.
