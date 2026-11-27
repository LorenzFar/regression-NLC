"""Gradio wrapper for the real buyer pipeline."""
import spaces  # Import before torch (via NLUAdapter) for ZeroGPU CUDA setup.

from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from functools import partial
from html import escape
import os
from pathlib import Path
import sys
from uuid import uuid4

import gradio as gr

# In the Space, app.py is at the root. In the source checkout it is one level down.
ROOT = Path(__file__).resolve().parent
if not (ROOT / "retrieval_pipeline").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from demo_app.server import DemoApp, SCENARIOS
from retrieval_pipeline.agent import build_agent
from retrieval_pipeline.contracts import Turn
from retrieval_pipeline.generator import RetrievalGenerator
from retrieval_pipeline.nlu_adapter import NLUAdapter
from retrieval_pipeline.pipeline import Session, Snapshot, TurnResult

CHECKPOINT = ROOT / "models" / "nlu_k3_s1.pt"
PRODUCTS = {scenario["id"]: scenario for scenario in SCENARIOS}
DEFAULT = SCENARIOS[0]
GATE_MODES = {"full": "Full", "reference_only": "Reference only", "off": "Off"}
ACTION_BADGES = {
    "propose": ("Opening offer", "blue"),
    "counter": ("Counteroffer", "blue"),
    "accept": ("Accepted", "green"),
    "confirm": ("Deal confirmed", "green"),
    "clarify": ("Clarification", "amber"),
    "reject": ("Offer rejected", "red"),
    "answer": ("Answer", "neutral"),
}
CSS = """
.buyer-lab-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: .85rem;
    font-weight: 650;
    --buyer-lab-badge-bg: #e3eaf0;
    --buyer-lab-badge-fg: #405a70;
    background-color: var(--buyer-lab-badge-bg);
}
/* Gradio Markdown sets text colors on nested elements in both themes. */
.buyer-lab-badge,
.buyer-lab-badge * { color: var(--buyer-lab-badge-fg) !important; }
.buyer-lab-badge--blue { --buyer-lab-badge-bg: #e5eefb; --buyer-lab-badge-fg: #1f4f8a; }
.buyer-lab-badge--green { --buyer-lab-badge-bg: #e3f5ea; --buyer-lab-badge-fg: #17663a; }
.buyer-lab-badge--amber { --buyer-lab-badge-bg: #fff4dc; --buyer-lab-badge-fg: #9a5b00; }
.buyer-lab-badge--red { --buyer-lab-badge-bg: #fde8e8; --buyer-lab-badge-fg: #9b2c2c; }
.dark .buyer-lab-badge { --buyer-lab-badge-bg: #263747; --buyer-lab-badge-fg: #d7e6f2; }
.dark .buyer-lab-badge--blue { --buyer-lab-badge-bg: #19375a; --buyer-lab-badge-fg: #c7dfff; }
.dark .buyer-lab-badge--green { --buyer-lab-badge-bg: #173e2b; --buyer-lab-badge-fg: #b7f4ce; }
.dark .buyer-lab-badge--amber { --buyer-lab-badge-bg: #493311; --buyer-lab-badge-fg: #ffe1a1; }
.dark .buyer-lab-badge--red { --buyer-lab-badge-bg: #50282e; --buyer-lab-badge-fg: #ffccd0; }
.buyer-lab-note { display: inline-block; margin-inline-start: .45rem; font-size: .85rem; }
"""
# Keep the free ZeroGPU hosting tier, but run this small classifier on CPU.
NLU = NLUAdapter(CHECKPOINT, device="cpu")
print("Negotiation NLU loaded on CPU; seller turns do not request GPU time.", flush=True)


@spaces.GPU(duration=1)
def _unused_gpu_registration():
    """Register the ZeroGPU runtime; no application callback calls this function."""
    pass
GENERATOR = RetrievalGenerator(ROOT / "examples" / "craigslist_bank.jsonl", allow_provisional=True)


def price_input(value, label):
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        raise gr.Error(f"Enter a valid {label.lower()}.") from None
    if not price.is_finite() or price < 0 or price.as_tuple().exponent < -2:
        raise gr.Error(f"{label} must be nonnegative with at most two decimal places.")
    return format(price.normalize(), "f")


def action_badge(details, *, selected=False):
    policy = details["policy"]
    action = policy["action"]
    label, tone = ACTION_BADGES[action]
    if action in {"propose", "counter", "accept", "confirm"}:
        price = format(Decimal(policy["price"]).normalize(), "f")
        label += f" · SGD {price}"
    badge = f'<span class="buyer-lab-badge buyer-lab-badge--{tone}">{escape(label)}</span>'
    if details["finished"]:
        note = "Negotiation finished"
    elif details["agreement"] == "pending":
        note = "Awaiting seller confirmation"
    elif action == "reject" or selected:
        note = "Negotiation still open"
    else:
        return badge
    return badge + f' <span class="buyer-lab-note">{note}</span>'


def explain(details, *, styled=False):
    policy = details["policy"]
    if details["opening"]:
        reason = f"The buyer opened at its target of SGD {policy['price']}."
    else:
        scores = details["probabilities"]
        act = max(scores, key=scores.get)
        price = details["price_reference"].get("price")
        price = f"SGD {price}" if price is not None else "no resolved price"
        gate = details["gate"]
        reasons = ", ".join(reason.replace("_", " ") for reason in gate["reasons"])
        gate_result = reasons or ("off" if gate["mode"] == "off" else "passed")
        action_price = f" at SGD {policy['price']}" if policy["price"] is not None else ""
        reason = (f"**Seller act:** {act} ({scores[act]:.0%}) · **Price:** {price} · **Gate:** {gate_result}."
                  f"\n\n**Buyer decision:** {policy['action']}{action_price}.")
    wording = "retrieved from the response bank" if details["generation"]["source"] == "retrieval" else "a fixed reply"
    badge = f"\n{action_badge(details, selected=True)}\n\n" if styled else ""
    summary = f"### Buyer reply {details['reply_number']}\n{badge}{reason} Wording: {wording}."
    comparison = details.get("counterfactual")
    if comparison:
        if "error" in comparison:
            summary += "\n\n**Without the gate:** comparison unavailable; see details."
        elif comparison["reply"] == details["generation"]["text"] and comparison["action"] == policy["action"]:
            summary += "\n\n**Without the gate:** same reply."
        else:
            price = f" at SGD {comparison['price']}" if comparison['price'] is not None else ""
            summary += f"\n\n**Without the gate:** {comparison['action']}{price}. See details for the reply."
    return summary


def record_turn(visitor, result: TurnResult, pipeline, *, counterfactual=None):
    snapshot, trace = result.snapshot, result.trace
    last_index = len(snapshot.history) - 1
    details = {key: trace[key] for key in
               ("probabilities", "price_reference", "gate", "policy", "generation", "timing_ms", "follow_up_applied")
               if key in trace}
    opening = last_index == 0
    if opening:
        config = visitor["config"]
        details["gate"] = {"mode": config["gate_mode"], "threshold": config["threshold"], "reasons": []}
    details.update(DemoApp.wording(pipeline, result.generation), config=visitor["config"], opening=opening,
                   reply_number=last_index // 2 + 1, offers=snapshot.state["offers"],
                   agreement=snapshot.state["agreement"], finished=snapshot.finished)
    if counterfactual is not None:
        details["counterfactual"] = counterfactual
    # Publish both snapshot and traces explicitly; the ZeroGPU worker cannot mutate Gradio state in place.
    updated = {**visitor, "snapshot": asdict(snapshot),
               "traces": {**visitor["traces"], str(last_index): details}}
    return updated, details


def display(visitor, details, *, styled):
    chat = []
    # The badge is part of the same Chatbot message, preserving selection indexes.
    # Gradio sanitizes HTML but may still render safe tags with Markdown disabled.
    # Escape both parties' text for the browser, never for the snapshot or public API.
    for index, turn in enumerate(visitor["snapshot"]["history"]):
        content = escape(turn["text"], quote=False) if styled else turn["text"]
        is_buyer = turn["speaker"] == "buyer"
        if styled and is_buyer:
            content = [content, gr.HTML(action_badge(visitor["traces"][str(index)]), render=False)]
        chat.append({"role": "assistant" if is_buyer else "user", "content": content})
    return visitor, chat, "", explain(details, styled=styled), details


def reset(product, listing_price, target, walk_away, fraction, threshold, gate_mode, *, styled=False):
    """Apply settings and start fresh, replacing the previous chat and traces."""
    if product not in PRODUCTS or gate_mode not in GATE_MODES:
        raise gr.Error("Choose a listed product and gate mode.")
    if fraction is None or not .05 <= fraction <= 1:
        raise gr.Error("Concession fraction must be between 0.05 and 1.")
    if threshold is None or not 0 <= threshold <= 1:
        raise gr.Error("Confidence threshold must be between 0 and 1.")
    config = dict(listing_price=price_input(listing_price, "Listing price"),
                  target=price_input(target, "Buyer target"), walk_away=price_input(walk_away, "Walk-away"),
                  fraction=str(fraction), threshold=threshold, gate_mode=gate_mode)
    if Decimal(config["target"]) > Decimal(config["walk_away"]):
        raise gr.Error("Buyer target cannot be higher than walk-away.")
    scenario = PRODUCTS[product]
    pipeline = build_agent(NLU, generator=GENERATOR, **config)
    session = Session(uuid4().hex, state={"listing_price": config["listing_price"], "offers": [], "agreement": "none"})
    result = pipeline.open(session, context=f"{scenario['title']} {scenario['description']}")
    visitor = {"session_id": session.session_id, "product": product, "config": config, "traces": {}}
    active = (f"**Active: {scenario['title']}** — listing {config['listing_price']}, target {config['target']}, "
              f"walk-away {config['walk_away']} SGD · fraction {fraction:g} · "
              f"gate {GATE_MODES[gate_mode]} · threshold {threshold:g}")
    return (*display(*record_turn(visitor, result, pipeline), styled=styled), active)


def seller_turn(message, visitor):
    message = (message or "").strip()
    if not 1 <= len(message) <= 4000:
        raise gr.Error("Enter a seller message of 1–4000 characters.")
    if visitor is None:
        raise gr.Error("Start a new negotiation first.")
    saved = visitor["snapshot"]
    before = Snapshot(tuple(Turn(**turn) for turn in saved["history"]), saved["state"],
                      saved["finished"], tuple(saved["applied_turn_ids"]))
    if before.finished:
        raise gr.Error("This negotiation is finished. Apply settings / start fresh to continue.")
    session_id, config = visitor["session_id"], visitor["config"]
    pipeline = build_agent(NLU, generator=GENERATOR, **config)
    result = pipeline.turn(Session.from_snapshot(session_id, before), message, case_id=session_id)
    comparison = result
    try:
        if config["gate_mode"] != "off":
            # Share the per-callback NLU cache, but compare against a separate pre-turn ledger.
            without_gate = build_agent(pipeline.nlu, generator=GENERATOR, **{**config, "gate_mode": "off"})
            comparison = without_gate.turn(Session.from_snapshot("counterfactual", before), message, case_id=session_id)
        reply = comparison.generation
        counterfactual = dict(reply=reply.text, action=reply.action, price=reply.price,
                              finished=comparison.snapshot.finished)
    except Exception as exc:
        counterfactual = {"error": str(exc), "reply": None, "action": None}
    return record_turn(visitor, result, pipeline, counterfactual=counterfactual)


def send(message: str, visitor: dict | None, *, styled=False):
    """Send a seller message using this visitor's active settings."""
    # Build Gradio components in the web process, outside the ZeroGPU worker.
    return display(*seller_turn(message, visitor), styled=styled)


def inspect_reply(visitor, event: gr.SelectData):
    details = visitor and visitor["traces"].get(str(event.index))
    return (explain(details, styled=True), details) if details else (gr.skip(), gr.skip())


def preset(product):
    scenario = PRODUCTS[product]
    return (scenario["description"], float(scenario["listing_price"]),
            float(scenario["target"]), float(scenario["walk_away"]))


with gr.Blocks(title="Buyer negotiation demo", analytics_enabled=False) as demo:
    gr.Markdown("## Buyer Lab · Negotiation demo\nYou play the seller. The buyer negotiates in SGD for self-collection.")
    visitor = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### The listing")
            product = gr.Dropdown([(s["title"], s["id"]) for s in SCENARIOS], value=DEFAULT["id"], label="Product")
            description = gr.Markdown(DEFAULT["description"])
            listing = gr.Number(float(DEFAULT["listing_price"]), label="Listing price", minimum=0)
            target = gr.Number(float(DEFAULT["target"]), label="Buyer target", minimum=0)
            walk = gr.Number(float(DEFAULT["walk_away"]), label="Walk-away", minimum=0)
            with gr.Accordion("Gate and policy settings", open=False):
                fraction = gr.Number(.5, label="Concession fraction", minimum=.05, maximum=1, step=.05)
                threshold = gr.Slider(0, 1, value=.7, step=.05, label="Confidence threshold")
                gate = gr.Dropdown([(label, key) for key, label in GATE_MODES.items()], value="full", label="Gate mode")
            gr.Markdown("Settings here are drafts. Applying them starts a new conversation.")
            reset_button = gr.Button("Apply settings / start fresh", variant="primary")
        with gr.Column(scale=2):
            active = gr.Markdown()
            chat = gr.Chatbot(label="Negotiation · click a buyer reply to inspect", height=420, render_markdown=False)
            message = gr.Textbox(label="Seller message", placeholder="How about 270?", lines=1, max_length=4000)
            send_button = gr.Button("Send", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### Why did the buyer say this?\nClick any buyer reply to see its diagnostics.")
            explanation = gr.Markdown()
            with gr.Accordion("Raw turn details", open=False):
                diagnostics = gr.JSON(label="Turn details")

    settings = [product, listing, target, walk, fraction, threshold, gate]
    outputs = [visitor, chat, message, explanation, diagnostics]
    # Share a queue group so sends, resets and selections cannot race one another.
    queued = dict(concurrency_id="negotiation")
    product.input(preset, product, [description, listing, target, walk], api_visibility="private", **queued)
    reset_outputs = [*outputs, active]
    reset_ui = partial(reset, styled=True)
    demo.load(reset_ui, settings, reset_outputs, api_name="initialize", api_visibility="private", **queued)
    reset_button.click(reset_ui, settings, reset_outputs, api_visibility="private", **queued)
    gr.on([send_button.click, message.submit], partial(send, styled=True), [message, visitor], outputs,
          api_visibility="private", **queued)
    # Expose the same component-typed API without the browser-only badge markup.
    gr.on([], reset, settings, reset_outputs, api_name="reset", **queued)
    gr.on([], send, [message, visitor], outputs, api_name="send", **queued)
    chat.select(inspect_reply, visitor, [explanation, diagnostics], api_visibility="private", **queued)


if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Default(primary_hue="teal"))
