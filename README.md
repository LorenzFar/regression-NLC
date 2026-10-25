# Buyer negotiation agent

Buyer-side, price-only negotiation for CS425. NLU reads the seller message. Policy authorizes a buyer action and price. The generator selects wording; the tracker owns agreement state.

**Working:** checkpoint NLU, template retrieval, pipeline, and replay tests. **Still scripted:** price/reference, policy, gate, and tracker. Example templates are authored fixtures, not a response bank from the training corpus. These examples do not measure model accuracy or negotiation success.

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
.\.venv\Scripts\python.exe -m buyer_agent.replay examples/standalone_cases.jsonl --checkpoint models/nlu_k2_s0.pt --bank examples/approved_fixture_bank.jsonl --output artifacts/standalone-replay.jsonl
```

The demo covers two retrieval matches, one fixed fallback, and one clarification. Replay runs the cases in three gate modes. `--checkpoint` replaces only scripted NLU; unfinished components remain scripted. Missing resources produce an error, not synthetic predictions. Models and generated traces stay out of Git.

## Code map

| File | Responsibility |
|---|---|
| `nlu.py` | Team NLU: masking, checkpoint loading, seller-act predictions |
| `buyer_agent/nlu.py` | Pipeline NLU adapter and resource metadata |
| `buyer_agent/contracts.py` | Public turn, buyer action, and reply types |
| `buyer_agent/generator.py` | Action-compatible retrieval and fixed wording |
| `buyer_agent/bank.py` | Prepare, approve, validate, and load response banks |
| `buyer_agent/pipeline.py` | Run one turn and publish state/history together |
| `buyer_agent/fixtures.py` | Scripted collaborators for examples and tests |
| `buyer_agent/replay.py` | Independent recorded cases and JSONL export |
| `examples/`, `tests/` | Runnable contracts, authored cases, and regression tests |

## Generator contract

```python
from buyer_agent import PolicyDecision, RetrievalGenerator

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
from buyer_agent.pipeline import Pipeline, Session

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

NLU receives uppercase `(speaker, text)` history tuples. The gate gets full-precision probabilities. `Transition` contains a state dictionary and boolean `finished`. Collaborators compute from detached inputs; `commit` computes a proposed local transition, not an external database write.

State and both public turns are published only after generation and transition validation succeed. Failure leaves the session unchanged and exposes `exception.trace`. Clarification preserves agreement state and cannot finish a session. Policy must select one of the gate's requested clarification reasons. Finished sessions reject new turns. Explicit turn IDs prevent duplicate application; failed turns can be retried. Sessions support one writer, not concurrent or durable storage.

## Response bank

Use only buyer responses from an authoritative training-dialogue allowlist. Never include validation/test responses. The process is **prepare → human review → approve → build**:

```powershell
.\.venv\Scripts\python.exe -m examples.retrieval_contract --export-sources artifacts/bank-work
.\.venv\Scripts\python.exe -m buyer_agent.bank prepare artifacts/bank-work/sources.jsonl artifacts/bank-work/candidates.jsonl --train-ids artifacts/bank-work/train-ids.json
# Inspect candidates and save only selected rows as reviewed.jsonl.
.\.venv\Scripts\python.exe -m buyer_agent.bank approve artifacts/bank-work/reviewed.jsonl artifacts/bank-work/approved.jsonl --reviewer "Reviewer name"
.\.venv\Scripts\python.exe -m buyer_agent.bank build artifacts/bank-work/approved.jsonl artifacts/bank-work/bank.jsonl --train-ids artifacts/bank-work/train-ids.json
```

The exported sources above are authored examples. Corpus sources need stable `template_id`, `dialogue_id`, `turn_id`, `origin="corpus"`, `split="train"`, `speaker="BUYER"`, explicit buyer `action`, preceding `history`, seller `message`, buyer `response`, and explicit false `mixed_action`/`unresolved_reference` flags. Priced actions need one validated `{start, end, value, validated: true}` price span; reject needs none. Offsets use Python characters with an exclusive end. The value must match the bare decimal substring exactly.

Only that span becomes `{price}`. The builder excludes multiple prices, unsupported terms, and mixed actions. These checks do not prove meaning: a person must review real templates. Approval records that decision; it does not perform review. Loading checks training IDs, source consistency, approvals, unique IDs, and context window. Corrupt banks fail explicitly. Use the same `--history-window` in preparation, building, and generation.

## Replay

Replay runs recorded seller turns as independent test cases. Each restores its own pre-turn history/state. Generated replies never become input to the next recorded case.

See `examples/standalone_cases.jsonl` for the input shape. Each row needs nonblank `case_id`, `conversation_id`, `turn_id`, `origin`, `reconstruction_mode`, and `message`; a `snapshot` with public `history` and tracker `state`; and a `fixture` with scripted results. The CLI needs `fixture`; a real `replay_cases(cases, pipeline_factory)` caller can supply its own components.

Gate modes are `off`, `reference_only`, and `full`; threshold defaults to `0.7`. The scripted full gate checks unresolved references, state conflicts, and low confidence. Real factories must keep unrelated settings fixed and isolate mutable state between cases.

Output has one JSONL record per case/mode, including failures. Traces contain component identities, probabilities, policy, state before/after, generation/provenance, timing, and errors. Exit codes: **0** all cases succeeded; **1** at least one failed; **2** invalid command/input-file setup. Malformed rows report errors without stopping valid rows. A generated candidate in a failed trace was not committed. Fixtures are not gold evaluation data.

## Remaining team handoffs

| Owner | Needed for full integration |
|---|---|
| Weijie | Training splits, buyer roles, and buyer-action labels for the response bank |
| Edric | Price extraction, validated source spans, and offer-reference resolution |
| Ercong | Policy, gate, tracker, offer ownership, and agreement/termination rules |
| Barry | Gold cases and an agreed method to reconstruct pre-turn state |

Real corpus templates also need human review. These handoffs remain pending. Local planning tickets stay in ignored `.scratch/`.

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
