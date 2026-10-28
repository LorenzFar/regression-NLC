"""JSONL reading, gold-set validation and JSON output shared by the evaluation CLIs.

Gold row schema (one JSON object per hand-labelled seller utterance)::

    {"case_id": "g001",
     "history": [{"speaker": "buyer", "text": "Can you do 40?"}, ...],   # oldest first
     "message": "ok that works",
     "act": "accept",                       # propose|counter|accept|reject|inquire|vague
     "price": "45",                         # optional
     "referenced_offer": "o1",              # optional
     "needs_clarification": false,          # used by RQ2
     "deal_agreed": true,                   # used by RQ2
     "challenge": ["ambiguous_acceptance"]} # optional tags, used for breakdowns
"""
from __future__ import annotations

import json
from pathlib import Path

from .nlu_metrics import ACTS


def read_jsonl(path) -> list:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def write_json(path, payload) -> None:
    text = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    if path in (None, "-"):
        print(text, end="")
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def _history(row):
    history = row.get("history", [])
    if not isinstance(history, list) or any(
        not isinstance(t, dict) or t.get("speaker") not in ("buyer", "seller")
        or not isinstance(t.get("text"), str) for t in history
    ):
        raise ValueError(f"{row['case_id']}: history must be a list of buyer/seller turns")


def load_gold(path, *, require=("act",)) -> list:
    """Load and validate gold rows; ``require`` names the fields the caller needs."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: gold file is empty")
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or not row["case_id"].strip():
            raise ValueError(f"{path}: row {index} needs a nonblank case_id")
        if row["case_id"] in seen:
            raise ValueError(f"{path}: duplicate case_id {row['case_id']!r}")
        seen.add(row["case_id"])
        if "act" in require and row.get("act") not in ACTS:
            raise ValueError(f"{row['case_id']}: act must be one of {list(ACTS)}")
        if "message" in require:
            if not isinstance(row.get("message"), str) or not row["message"].strip():
                raise ValueError(f"{row['case_id']}: message must be a nonblank string")
            _history(row)
        for key in ("needs_clarification", "deal_agreed"):
            if key in require and type(row.get(key)) is not bool:
                raise ValueError(f"{row['case_id']}: {key} must be true or false")
        tags = row.get("challenge", [])
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            raise ValueError(f"{row['case_id']}: challenge must be a list of strings")
    return rows
