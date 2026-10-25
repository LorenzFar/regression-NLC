"""Matched-context replay through the same Pipeline.turn seam as the demo."""
from copy import deepcopy
import json

from .pipeline import Session, Snapshot
from .contracts import Turn

MODES = ("off", "reference_only", "full")


def _snapshot(case):
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    if "input_error" in case:
        raise ValueError(case["input_error"])
    for key in ("case_id", "conversation_id", "turn_id", "origin", "reconstruction_mode", "message"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"{key} must be an explicit nonblank string")
    json.dumps(case, allow_nan=False)
    raw = case["snapshot"]
    if not isinstance(raw, dict) or not isinstance(raw.get("history"), list) or not isinstance(raw.get("state"), dict):
        raise ValueError("snapshot requires recorded history list and state object")
    for turn in raw["history"]:
        if (not isinstance(turn, dict) or set(turn) != {"speaker", "text"}
                or turn["speaker"] not in ("buyer", "seller") or not isinstance(turn["text"], str)):
            raise ValueError("history requires buyer/seller public turns")
    if type(raw.get("finished", False)) is not bool:
        raise ValueError("snapshot finished must be boolean")
    ids = raw.get("applied_turn_ids", [])
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("applied_turn_ids must be a list of nonblank strings")
    return Snapshot(tuple(Turn(**t) for t in raw["history"]), deepcopy(raw["state"]),
                    raw.get("finished", False), tuple(ids))


def replay_cases(cases, pipeline_factory, *, modes=MODES, threshold=0.7):
    """Return strict-JSON records, one per input case/mode, including failures.

    Factory(mode, threshold) must hold unrelated settings fixed and construct
    independent adapters (or reuse stateless inference resources).
    """
    import math

    if (not isinstance(modes, (list, tuple)) or not modes
            or any(not isinstance(mode, str) or mode not in MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError("modes must be a nonempty unique list/tuple of off, reference_only, full")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between zero and one")
    records = []
    seen = set()
    for index, case in enumerate(cases, 1):
        identity = case if isinstance(case, dict) else {}
        case_id = identity.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            case_id = f"invalid-line-{index}"
        duplicate = case_id in seen
        seen.add(case_id)
        for mode in modes:
            base = {"case_id": case_id, "input_index": index,
                    "gate": {"mode": mode, "threshold": threshold, "reasons": []},
                    "error": None}
            for key in ("conversation_id", "turn_id", "origin", "reconstruction_mode"):
                value = identity.get(key)
                base[key] = value if isinstance(value, str) else None
            base["session_id"] = json.dumps([base["conversation_id"], case_id, mode], ensure_ascii=False)
            try:
                if duplicate:
                    raise ValueError("duplicate case_id")
                before = _snapshot(case)
                session = Session.from_snapshot(base["session_id"], before)
                trace = pipeline_factory(mode, threshold).turn(
                    session, case["message"], case_id=case_id, turn_id=case["turn_id"]).trace
                record = {**base, **trace,
                          "case_provenance": deepcopy(case.get("provenance", {})),
                          "resources": deepcopy(getattr(pipeline_factory, "resource_metadata", {}))}
            except Exception as exc:
                record = {**base, **getattr(exc, "trace", {}),
                          "error": {"type": type(exc).__name__, "message": str(exc)}}
            try:
                record = json.loads(json.dumps(record, allow_nan=False))
            except (TypeError, ValueError) as exc:
                record = {**base, "error": {"type": "SerializationError", "message": str(exc)}}
            records.append(record)
    return records


def fixture_pipeline_factory(cases, *, bank_path=None, nlu=None):
    """Build a labelled fixture factory; a supplied real NLU replaces only NLU."""
    from .contracts import PolicyDecision
    from .fixtures import FixtureCase, ScriptedComponents
    from .generator import RetrievalGenerator
    from .pipeline import Pipeline

    scripted = {}
    for case in deepcopy(cases):
        try:
            _snapshot(case)
            raw = dict(case["fixture"])
            if nlu is None:
                probabilities = raw["probabilities"]
                if (not isinstance(probabilities, dict) or not probabilities
                        or any(not isinstance(act, str) or type(value) not in (int, float)
                               or not 0 <= value <= 1 for act, value in probabilities.items())):
                    raise ValueError("fixture probabilities require string acts and finite numbers in [0, 1]")
            raw["decision"] = PolicyDecision(**raw["decision"])
            scripted.setdefault(case["case_id"], FixtureCase(**raw, message=case["message"]))
        except (KeyError, TypeError, ValueError):
            # Replay reports invalid rows; exclude them from shared fixture lookups.
            continue
    generator = None
    pipelines = {}

    def build(mode, threshold):
        nonlocal generator
        key = (mode, threshold)
        if key not in pipelines:
            if generator is None:
                generator = RetrievalGenerator(bank_path=bank_path)
            # Fixture adapters are stateless lookups; replay restores a fresh Session.
            adapters = ScriptedComponents(scripted, mode=mode, threshold=threshold).adapters(nlu=nlu)
            pipelines[key] = Pipeline(**adapters, generator=generator)
        return pipelines[key]

    build.resource_metadata = {
        "nlu": deepcopy(getattr(nlu, "resource_metadata", {"fixture": nlu is None})),
        "generator": {"bank_path": str(bank_path) if bank_path is not None else None,
                      "history_window": 2},
    }
    return build


def main(argv=None):
    """Read independent case JSONL and write only diagnostic JSONL to output."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="JSONL cases with explicit pre-turn snapshots")
    parser.add_argument("--output", default="-", help="JSONL path, or - for stdout")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--bank", help="approved response-bank JSONL; otherwise fixed fallbacks")
    parser.add_argument("--checkpoint", help="supplied NLU checkpoint; other adapters remain fixtures")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output != "-" and Path(args.cases).resolve() == Path(args.output).resolve():
        parser.error("cases and output must be different files")
    try:
        lines = Path(args.cases).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    cases = []
    for index, line in enumerate(lines, 1):
        try:
            def reject_constant(value):
                raise ValueError(f"nonfinite JSON number: {value}")
            cases.append(json.loads(line, parse_constant=reject_constant))
        except ValueError as exc:
            cases.append({"case_id": f"invalid-line-{index}", "input_error": str(exc)})
    if not cases:
        print("case file is empty", file=sys.stderr)
        return 2
    try:
        nlu = None
        if args.checkpoint:
            from .nlu_adapter import NLUAdapter
            nlu = NLUAdapter(args.checkpoint, model_name=args.model_name, device=args.device)
        factory = fixture_pipeline_factory(cases, bank_path=args.bank, nlu=nlu)
    except Exception as exc:
        # Resource failures are diagnostics for every requested case/mode too.
        setup_error = exc
        def factory(mode, threshold):
            raise setup_error
    try:
        records = replay_cases(cases, factory, modes=args.modes, threshold=args.threshold)
    except ValueError as exc:
        parser.error(str(exc))
    text = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return int(any(r["error"] is not None for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
