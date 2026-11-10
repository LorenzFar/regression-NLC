# Buyer negotiation agent

A local chatbot for CS425 that negotiates prices. You play the seller, and the bot plays the buyer. It starts with an offer, responds to your prices, and asks questions when an offer is unclear.

The project combines a trained NLU model, price and offer-reference rules, a negotiation policy, and a bank of buyer replies from CraigslistBargain. It runs locally without an LLM API.

## How it works

1. **Read the message:** predict whether the seller is making an offer, accepting, rejecting, asking a question, or being vague.
2. **Resolve the price:** extract amounts and identify which offer the seller means.
3. **Check whether to ask:** ask for clarification when confidence is low, an offer is unclear, or a price has a condition.
4. **Choose a response:** use price rules to counter, accept, reject, or confirm a deal; answer seller inquiries that pass the gate.
5. **Write the reply:** select a matching response template and insert the chosen price.
6. **Update the conversation:** record offers and whether a deal has been confirmed.

The policy chooses the action and price. Retrieval chooses negotiation wording. For seller `inquire` messages that pass the gate, policy selects a fixed answer type and the tracker’s current price. General price questions restate the offer; interest and increase questions do not concede. Collection/pickup/delivery questions redirect to price before interest handling. Agreement questions say there is no agreement yet, or restate the pending accepted seller price with “I’ve accepted …”, never the older buyer offer. Other questions redirect to the current price; without one, they ask what price the seller would accept. Answers do not change offers or agreement. Purchase method is fixed to **self-collection**; collection dates are not scheduled. Narrow collection-price patterns are supported, not general delivery or scheduling language. “Final” prices still use the normal negotiation rules, not automatic acceptance.

## Chatbot demo

Follow [Setup and run](#setup-and-run) first if this is a new checkout. Then start the server below. Restart it after code changes and begin a new negotiation.

You type as the seller. Click a buyer reply to see the predicted intent, price, clarification decision, buyer action, selected template, and recorded offers.

```powershell
$env:HF_HUB_OFFLINE = "1"   # after the DistilBERT tokenizer is cached once
.\.venv\Scripts\python.exe -m demo_app.server --port 8000
# open http://127.0.0.1:8000
```

`--scripted-nlu` swaps in a keyword stand-in (labelled in the header) when torch or the checkpoint is unavailable. `--bank PATH` picks another response bank.

### Demo conversations

Use the **City road bike** preset: listing **300**, target **240**, walk-away **280**, concession fraction **0.5**, gate **Full**, confidence threshold **0.7**. Start a **new negotiation for each row**. The buyer opens at 240. Send each numbered seller message separately, waiting for the reply before the next one.

These conversations were checked through the live frontend with `models/nlu_k2_s0.pt` and the current response bank; the updated different-price clarification below was rechecked directly through the same pipeline, not the frontend. Check the **action and price** in the diagnostics: retrieved wording can sound like agreement even when the action is `counter`.

#### Price negotiation

| Scenario | Seller messages, in order | Expected buyer actions |
|---|---|---|
| Accept the opening offer | 1. `Deal!` | Confirm 240 and end the conversation. |
| Offer a lower price | 1. `I am asking 230.`<br>2. `Okay, deal.` | Accept 230 → confirm 230. |
| Raise the offer gradually | 1. `I am asking 300.`<br>2. `I could do 290.`<br>3. `Okay, deal.` | Counter 270 → counter 280 → confirm 280. |
| Reach the buyer's maximum | 1. `I am asking 400.`<br>2. `I am asking 350.` | Counter 280 → reject the higher price. Rejection does not close the chat. |
| Repeat a price within the maximum | 1. `I am asking 270.`<br>2. `I am asking 270.`<br>3. `Okay, deal.` | Counter 255 → accept 270 → confirm 270. |
| Accept a small remaining gap | 1. `I am asking 270.`<br>2. `I am asking 275.`<br>3. `Okay, deal.` | Counter 255 → accept 275 → confirm 275. After its first counter, the next calculated offer is close enough to 275. |

#### Clarify which offer is meant

| Scenario | Seller messages, in order | Expected buyer response |
|---|---|---|
| General clarification | 1. `Maybe.` | “Could you clarify what you mean?” No offer changes. |
| Explicit acceptance | 1. `I accept your offer.` | Confirm 240 and end the conversation. |
| Refer to an earlier offer | 1. `I could do 270.`<br>2. `Actually, let’s go with the earlier offer.`<br>3. `I mean 240.` | Counter 255 → “Which offer are you referring to?” → accept 240. |
| Refer to another price | 1. `I agree to the other price`<br>2. `I mean 240.` | Ask which offer → accept 240. |
| Name a different price while agreeing | 1. `Deal at 300`<br>2. `yes` (or `no`) | “Are you asking for 300?” → `yes` sends seller price 300 through the normal price rules (counter 270); `no` asks “What price are you asking for?” Offers stay unchanged while clarifying. |
| Answer a question about the seller's price | 1. `I can do 280.`<br>2. `ok`<br>3. `ok` | Ask whether the seller means 280 → counter 260 → confirm 260. The first `ok` answers the question; the second accepts the counteroffer. |

Clarification leaves the recorded offers unchanged. `Maybe.` asks “Could you clarify what you mean?” in Full mode. `I accept your offer.` confirms a matching live buyer offer. After the buyer **accepts** a seller price, `Okay, deal.` can be used to confirm it.

#### Self-collection and conditional prices

| Scenario | Seller message | Expected buyer action |
|---|---|---|
| Collection or delivery | `260 if you collect, or 270 delivered` | Use collection price 260 → counter 250. |
| Collection costs more | `290 if you collect, or 270 delivered` | Use collection price 290 → counter 265. Do not choose the delivery price. |
| Two available prices | `I can take 260 or 270` | Use 260 → counter 250. |
| Two final prices | `My final price is 260 or 270` | Use 260 → counter 250. |
| Final collection price | `260 for collection, final price` | Counter 250. “Final” does not force acceptance. |
| Repeat a final price | Send `My final price is 270` twice | Counter 255 → accept 270 under the repeated-price rule. |

These exact forms are supported. Other ways of giving multiple prices may still need clarification.

The bot negotiates price only. It does not promise a collection time.

| Scenario | Seller messages, in order | Expected buyer response |
|---|---|---|
| Remove the condition | 1. `250 if you can collect today`<br>2. `yes` | “Would you accept 250 if I collect on another day?” → counter 245. `yes` makes 250 available for another-day collection; it does not automatically complete a deal. |
| Keep the condition, then name another price | 1. `250 if you can collect today`<br>2. `no`<br>3. `I can do 260.` | Ask about another day → “What price would you accept for self-collection on another day?” → counter 250. No collection date is promised. |
| Give a new price immediately | 1. `250 if you can collect today`<br>2. `No, I can do 260.` | Ask about another day → counter 250 using the new seller price of 260. |
| Repeat the conditional offer | 1. `250 if you can collect today`<br>2. `250 if you collect today` | Ask about another day again. It does not treat the repeated price as unconditional. |

#### Compare the gate on and off

Start two fresh conversations and send `I can do 280.` followed by `ok` in each.

| Gate | Observed result |
|---|---|
| **Full** | Asks about 280, then counters at 260. Another `ok` confirms 260. |
| **Off** | Counters at 240 again, then confirms 240. It loses the seller's intended price of 280. |

#### Current limits to try

These examples show gaps, rather than features to present as working reliably. Start a fresh conversation for each row.

| Seller message | Current behaviour |
|---|---|
| `I could do 260 or 270.` | Asks about the buyer's 240 instead of asking which seller price is intended. `I mean 260.` recovers with a counter at 250. |
| `Can you increase your offer?` | If labelled `inquire` and the gate passes: answers with the current offer and asks what price would work; no increase. |
| `No.` | Counters at 270 using the listing price of 300. A bare rejection does not end the chat. |

Some bank replies contain personal stories, budget claims, or agreement wording that does not fit the chosen action. Price and state checks do not guarantee appropriate wording. Collection times, delivery arrangements, and general questions about the item are outside the current demo's scope.

Direct `k3_s1` smoke checks: `What price would you like?`, `What price would work for you?`, and `What price do you want?` restate 240. `Have we agreed?` says there is no agreement yet. After `I am asking 270.` twice, a price/agreement question answers “I’ve accepted 270”; `Deal.` or `Okay, deal.` can still confirm 270. `Are you interested in pickup?` is instead labelled `vague` and clarified. After pending acceptance plus an inquiry, `I accept your offer.` was also labelled `vague`; no inquiry keyword override bypasses the NLU or gate.

### Settings explained

- **Target:** the buyer's opening price.
- **Walk-away:** the highest price the buyer will pay.
- **Concession fraction:** how far each counter moves toward the seller's price. At 0.5, it moves halfway. The demo rounds down to whole dollars and caps the result at walk-away.
- **Gate mode:** **Full** checks confidence, offer references, conditions, and conflicting offers. **Reference only** skips the confidence and state-conflict checks. **Off** skips clarification.
- **Confidence threshold:** how sure the model must be before acting; the default is 0.7. The probabilities for propose and counter are added because both describe an offer.

The buyer may accept a repeated seller price within its maximum. After at least one counteroffer, it may also accept a price within its maximum when:

```text
seller's price − buyer's next calculated offer ≤ 5% × seller's price
```

Accepting a seller offer and confirming a deal are separate actions. A rejection declines the offer; it does not necessarily end the conversation.

## Project files

- `demo_app/`: local web chat and turn-by-turn diagnostics.
- `retrieval_pipeline/`: connects the components, chooses buyer actions, tracks offers, and generates replies.
- `price_reference/`: extracts prices and resolves offer references.
- `nlu.py`: loads the trained model and predicts seller intent.
- `examples/`: response banks and sample conversations.
- `evaluation/` and `scripts/`: data preparation, replay, and evaluation tools.
- `tests/`: automated checks.

See [negotiation details](docs/negotiation.md) for the policy and state tracker.

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
| `action` | `propose`, `counter`, `accept`, `confirm`, `reject`, `clarify`, or `answer` |
| `price` | Nonnegative SGD decimal string, at most two decimal places. Required for propose/counter/accept/confirm. Never rounded. |
| `referenced_offer_id` | Nonblank offer ID for accept/confirm. Upstream validates ownership and whether the offer remains valid. |
| `clarification_reason` | For clarify: `low_confidence`, `unresolved_reference`, `state_conflict`, `conditional_price`, `collection_day`, `price_mismatch`, or `unclear_message`. Offer-specific clarification needs both offer ID and price. |
| `finished` | Supplied by policy; the tracker transition decides whether the session ends. |
| `answer_type` | For answer: `current_offer`, `interest`, `hold_price`, `redirect`, `no_agreement`, `accepted_offer`, `arrangement`, or `accepted_arrangement`. Policy selects the type and current/accepted price; answers preserve tracker state. |
| `offer_context` | Optional `OfferContext` containing the previous buyer price and the current resolved seller price. Used to check increase and midpoint wording. |

Retrieval filters by action, then ranks preceding contexts with TF-IDF/cosine similarity. Context numbers are masked; response text is not. Positive-score ties use template ID. Only the authorized price fills the template. Increase and midpoint wording use the recorded buyer offer and the resolved seller price, rather than parsing conversation text again. Without this context, those templates are skipped; other templates remain available. Clarification uses fixed wording. A valid request without a match gets a same-action fallback; invalid requests raise `ContractError`.

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

Earlier generator results (489 held-out buyer turns; each system receives the recorded buyer action and price). These results predate the latest wording checks:

| System | Token F1 vs human reply | Distinct-2 | Numeric-price check |
|---|---:|---:|---:|
| Fixed fallback sentences | 0.205 | 0.011 | 100% |
| Random template, same action | 0.207 | 0.280 | 100% |
| TF-IDF retrieval (used) | 0.216 | 0.249 | 100% |
| TF-IDF uni+bigram | 0.211 | 0.258 | 100% |

The numeric-price check confirms that any price written in a reply matches the policy; it also permits replies with no price. Word overlap barely separates the systems. These scores do not measure persuasion or prove that every reply fits the conversation.

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

- Run RQ1 and RQ2 on the hand-labelled evaluation set, with offer history and listing prices.
- Add the remaining NLU context-window checkpoints and training seeds for RQ1.
- Review response-bank wording for unsupported claims and replies that do not fit the chosen action.

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
