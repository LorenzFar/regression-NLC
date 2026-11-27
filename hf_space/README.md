---
title: Buyer Negotiation Demo
emoji: 🚲
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12.12"
app_file: app.py
short_description: Negotiate prices with a trained buyer agent
---

You play the seller. Choose the city road bike, iPhone 13, oak coffee table or apartment-sized couch. Edit the listing price, buyer target, walk-away price, concession fraction, confidence threshold or gate mode, then click **Apply settings / start fresh**. Draft edits do not affect the active negotiation.

Click any buyer reply, including the opening, to see why it was chosen. **Raw turn details** shows the model scores, price reference, gate, buyer action, offer ledger and agreement at that point. It also shows the original sentence and source for retrieved wording, and what would have happened without the gate on the same seller turn.

### Gradio API inputs

- `/reset(product="bike", listing_price=300, target=240, walk_away=280, fraction=0.5, threshold=0.7, gate_mode="full")` starts a new conversation. `product` is `bike`, `phone`, `table` or `couch`; `gate_mode` is `full`, `reference_only` or `off`. The five numeric arguments are JSON numbers. Prices are nonnegative with up to two decimals and `target` cannot exceed `walk_away`. Fraction is 0.05–1 and threshold is 0–1. The API defaults are for the bike; send all three prices when selecting another preset via the API.
- `/send(message: str)` accepts 1–4000 characters after trimming. Call `/reset` first and reuse the same Gradio client/session for subsequent turns. The visitor state is not a public API argument. Historical inspection is a UI event, not a public endpoint.

### Stage the Space

Copy only these files. Rename the three `hf_space/` files to the Space root; keep all other paths as written. Copy the checkpoint from the local checkout (it is not tracked in Git).

- `hf_space/README.md` → `README.md`
- `hf_space/app.py` → `app.py`
- `hf_space/requirements.txt` → `requirements.txt`
- `demo_app/{__init__,server}.py` (original product presets and wording lookup; no static files needed)
- `nlu.py`
- `retrieval_pipeline/{__init__,agent,bank,contracts,generator,negotiation,nlu_adapter,pipeline}.py`
- `price_reference/{__init__,evidence,extractor,gate,integration,ledger,observer,resolver,types}.py`
- `examples/craigslist_bank.jsonl`
- `models/nlu_k3_s1.pt`
- `LICENSE`

Do not stage the main checkout, other checkpoints, environments, credentials or generated artifacts. Choose Gradio and ZeroGPU hardware in the Space settings. The classifier runs on CPU; no user action requests GPU time. Keep the Space on its free ZeroGPU hosting tier rather than selecting the paid-plan CPU Basic tier. An unused decorated registration function satisfies ZeroGPU startup requirements. Startup needs the public DistilBERT config and tokenizer, not the base model weights.

For a local check, install this requirements file plus `gradio==6.32.0` and `spaces==0.53.0` in a separate Python environment, then run `python hf_space/app.py` from the source checkout root (or `python app.py` from the staged folder). Off ZeroGPU this uses the real checkpoint on CPU.
