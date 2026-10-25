"""Run actual supplied-checkpoint inference, not fixtures or an accuracy evaluation.

From repository root: python -m scripts.nlu_smoke models/nlu_k2_s0.pt
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


CASES = [
    {"id": "notebook_accept", "history": [("BUYER", "Can you do $40?"),
                                            ("SELLER", "I can do 45")],
     "message": "ok that works"},
    {"id": "notebook_reject", "history": [("BUYER", "Can you do $40?"),
                                            ("SELLER", "I can do 45")],
     "message": "no way, that's too much"},
    {"id": "notebook_counter", "history": [("BUYER", "Can you do $40?"),
                                             ("SELLER", "I can do 45")],
     "message": "how about 42?"},
    {"id": "notebook_inquire", "history": [("BUYER", "Can you do $40?"),
                                             ("SELLER", "I can do 45")],
     "message": "does it come with the charger?"},
    {"id": "notebook_no_offer", "history": [], "message": "ok that works"},
    {"id": "notebook_uncertain", "history": [("BUYER", "Can you do $40?"),
                                               ("SELLER", "I can do 45")],
     "message": "hmm let me think about it"},
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true", help="Use only cached tokenizer/config resources")
    parser.add_argument("--output", type=Path, default=Path("artifacts/nlu-smoke.json"))
    args = parser.parse_args(argv)
    report = {
        "verification": "actual_checkpoint_inference", "fixture": False,
        "scope": "hand-authored smoke messages; not accuracy, F1, or calibration measurement",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_checkpoint": str(args.checkpoint.resolve()),
        "requested_model_name": args.model_name, "requested_device": args.device,
        "local_files_only": args.offline,
        "status": "error", "cases": [],
    }
    start = time.perf_counter()
    try:
        from retrieval_pipeline.nlu_adapter import NLUAdapter

        adapter = NLUAdapter(args.checkpoint, model_name=args.model_name, device=args.device,
                             local_files_only=args.offline)
        report["load_seconds"] = time.perf_counter() - start
        report["resources"] = adapter.resource_metadata
        for case in CASES:
            turn_start = time.perf_counter()
            probabilities = adapter.predict(case["history"], case["message"])
            top_act = max(probabilities, key=probabilities.get)
            report["cases"].append({**case, "probabilities": probabilities,
                                    "top_act": top_act, "top_probability": probabilities[top_act],
                                    "inference_seconds": time.perf_counter() - turn_start})
        report["status"] = "ok"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["total_seconds"] = time.perf_counter() - start
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
