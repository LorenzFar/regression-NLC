"""Prepare a HUMAN REVIEW queue, never an approved CraigslistBargain bank.

Usage (repository root, standard library only):
  python scripts/prepare_craigslist_candidates.py artifacts/craigslist/train.json \
      --parsed artifacts/craigslist/parsed.json --output-dir artifacts/craigslist/review \
      --train-ids path/to/team-train-ids.json

Without --train-ids, official training membership is PROVISIONAL until team
alignment. Inputs must be the official train JSON, not dev/test or a mixed file.
Local filenames cannot establish the authenticity of a split; stdout records
file hashes for auditing. Raw data and outputs should stay under ignored artifacts/.

sources.jsonl is NOT ready for bank.prepare: action and semantic flags are unset,
spans are unvalidated. A person must select rows, resolve action/flags/spans and
training membership, and set validated=true only after inspecting each span.
Then use retrieval_pipeline.bank prepare -> review -> approve -> build with the
team's authoritative IDs. Never approve this queue wholesale. Numeric spans and
parser intents are suggestions, not labels, price verification, or agreement state.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
import re
import sys
from pathlib import Path

# Allow direct invocation from the repository root without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_pipeline.bank import prepare_candidate
from retrieval_pipeline.contracts import ContractError


NUMBER = re.compile(r"[0-9]+(?:\.[0-9]{1,2})?")


def index_dialogues(dialogues, label):
    """Validate whole corpus identity before any filtering or output."""
    if not isinstance(dialogues, list):
        raise ValueError(f"{label} must be a JSON list")
    result = {}
    for dialogue in dialogues:
        if not isinstance(dialogue, dict):
            raise ValueError(f"{label}: dialogue must be an object")
        uuid = dialogue.get("uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise ValueError(f"{label}: nonblank UUID required")
        if uuid in result:
            raise ValueError(f"{label}: duplicate UUID {uuid}")
        if not isinstance(dialogue.get("events"), list):
            raise ValueError(f"{label}: events list required for {uuid}")
        result[uuid] = dialogue
    return result


def checked_parser_index(raw_index, parsed):
    """Join by UUID, but trust annotations only after all events align exactly."""
    if parsed is None:
        return {}
    parsed_index = index_dialogues(parsed, "parsed")
    if raw_index.keys() != parsed_index.keys():
        raise ValueError("parsed/raw UUID sets differ; require the same full training split")
    for uuid, raw in raw_index.items():
        annotated = parsed_index[uuid]["events"]
        if len(raw["events"]) != len(annotated):
            raise ValueError(f"parsed event alignment length mismatch: {uuid}")
        for index, (event, other) in enumerate(zip(raw["events"], annotated)):
            if (not isinstance(event, dict) or not isinstance(other, dict)
                    or any(key not in event or key not in other
                           or type(event[key]) is not type(other[key]) or event[key] != other[key]
                           for key in ("agent", "action", "data"))):
                raise ValueError(f"parsed event alignment mismatch: {uuid}:{index}")
    return parsed_index


def make_candidates(raw, parsed=None, train_ids=None):
    """Return conservative, explicitly unreviewed source rows and counts.

    The probe deep-copies proposed flags/spans, calls bank.prepare_candidate,
    and discards its result. A passing probe is not human semantic approval.
    Only immediate raw-event seller-message -> buyer-message pairs qualify;
    earlier history contains public message events only, oldest first.
    """
    raw_index = index_dialogues(raw, "raw")
    parsed_index = checked_parser_index(raw_index, parsed)
    if train_ids is not None and (not isinstance(train_ids, list)
            or any(not isinstance(i, str) or not i.strip() for i in train_ids)
            or len(set(train_ids)) != len(train_ids)):
        raise ValueError("team train IDs must be a JSON list of unique nonblank strings")
    eligible_ids = set(raw_index) if train_ids is None else set(raw_index).intersection(train_ids)
    ids = sorted(eligible_ids)
    split_status = "provisional_official_train" if train_ids is None else "team_allowlist_intersection"
    counts = Counter(dialogues=len(raw), eligible_dialogues=len(ids),
                     excluded_by_team_allowlist=len(raw) - len(ids),
                     team_ids_outside_official_train=0 if train_ids is None else len(set(train_ids) - set(raw_index)))
    rows = []
    for dialogue in raw:
        try:
            roles = [kb["personal"]["Role"].upper() for kb in dialogue["scenario"]["kbs"]]
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid personal.Role: {dialogue['uuid']}") from exc
        if sorted(roles) != ["BUYER", "SELLER"]:
            raise ValueError(f"exactly one buyer and seller required: {dialogue['uuid']}")
        for event in dialogue["events"]:
            if (not isinstance(event, dict) or type(event.get("agent")) is not int
                    or event["agent"] not in (0, 1) or not isinstance(event.get("action"), str)
                    or (event["action"] == "message" and not isinstance(event.get("data"), str))):
                raise ValueError(f"invalid raw event: {dialogue['uuid']}")
        if dialogue["uuid"] not in eligible_ids:
            continue
        history = []
        events = dialogue["events"]
        for index, event in enumerate(events):
            if event["action"] != "message":
                continue
            role = roles[event["agent"]]
            text = event["data"]
            if role == "BUYER" and index and events[index - 1]["action"] == "message" and roles[events[index - 1]["agent"]] == "SELLER":
                counts["buyer_after_seller"] += 1
                metadata = ((parsed_index[dialogue["uuid"]]["events"][index].get("metadata") or {})
                            if parsed_index else {})
                if not isinstance(metadata, dict):
                    raise ValueError(f"invalid parser metadata: {dialogue['uuid']}:{index}")
                intent = metadata.get("intent")
                if intent is not None and not isinstance(intent, str):
                    raise ValueError(f"invalid parser intent: {dialogue['uuid']}:{index}")
                if intent in {"agree", "accept", "confirm"}:
                    counts["skipped_agreement_intent"] += 1
                    history.append(dict(speaker=role, text=text))
                    continue
                proposed_action = {"init-price": "propose", "counter-price": "counter"}.get(intent)
                spans = [dict(start=m.start(), end=m.end(), value=m.group(), validated=False)
                         for m in NUMBER.finditer(text)]
                if len(spans) != 1:
                    counts["skipped_span_count"] += 1
                else:
                    row = dict(template_id=f"craigslist:{dialogue['uuid']}:{index}",
                               dialogue_id=dialogue["uuid"], turn_id=str(index), origin="corpus",
                               split="train", speaker="BUYER", action=None,
                               proposed_action=proposed_action,
                               action_candidates=[proposed_action] if proposed_action else ["propose", "counter"],
                               parser_intent=intent,
                               mixed_action=None, unresolved_reference=None,
                               proposed_flags=dict(mixed_action=False, unresolved_reference=False),
                               price_spans=spans, history=copy.deepcopy(history[:-1]),
                               message=events[index - 1]["data"], response=text,
                               source_event_index=index, seller_event_index=index - 1,
                               review_status="needs_human_review", approved=False, reviewer=None,
                               review_reasons=["Numeric substring may not be a price; verify span and currency.",
                                               "Choose buyer action and review mixed actions/references.",
                                               ("Training membership provisional until team alignment."
                                                if train_ids is None else "Team allowlist intersected with supplied official train."),
                                               (f"Parser intent {intent!r} suggests {proposed_action}; verify meaning."
                                                if proposed_action else "No mapped parser intent; action alternatives are review prompts only.")],
                               split_status=split_status)
                    probe = copy.deepcopy(row)
                    probe.update(action=proposed_action or "propose", mixed_action=False, unresolved_reference=False)
                    for span in probe["price_spans"]:
                        span["validated"] = True
                    try:
                        prepare_candidate(probe, ids)
                    except ContractError as exc:
                        counts["skipped_bank_probe"] += 1
                        counts[f"bank_reason:{exc}"] += 1
                    else:
                        row["eligibility_probe"] = "passes_only_under_provisional_mapping_and_spans"
                        rows.append(row)
            history.append(dict(speaker=role, text=text))
    counts["queued"] = len(rows)
    return rows, ids, dict(counts)


OFFICIAL_RAW_URL = "https://worksheets.codalab.org/rest/bundles/0xda2bae7241044dbaa4e8ebb02c280d8f/contents/blob/"
OFFICIAL_PARSED_URL = "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json"


def read_json(path):
    data = path.read_bytes()
    def reject_constant(value):
        raise ValueError(f"non-JSON constant {value}")
    content = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    return content, dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path, help="local official raw train JSON (complete list)")
    parser.add_argument("--parsed", type=Path, help="optional official parsed train JSON; exact event alignment required")
    parser.add_argument("--output-dir", type=Path, required=True, help="prefer ignored artifacts/craigslist/review")
    parser.add_argument("--train-ids", type=Path, help="team-authoritative JSON ID list; intersect with official train")
    args = parser.parse_args(argv)
    try:
        raw, raw_meta = read_json(args.train)
        parsed, parsed_meta = read_json(args.parsed) if args.parsed else (None, None)
        if args.parsed and not isinstance(parsed, list):
            raise ValueError("parsed must be a JSON list")
        team_ids, team_meta = read_json(args.train_ids) if args.train_ids else (None, None)
        if args.train_ids and team_ids is None:
            raise ValueError("team train IDs must be a JSON list, not null")
        rows, ids, counts = make_candidates(raw, parsed=parsed, train_ids=team_ids)
        sources_path = args.output_dir / "sources.jsonl"
        ids_path = args.output_dir / "train-ids.json"
        inputs = {p.resolve() for p in (args.train, args.parsed, args.train_ids) if p is not None}
        if any(path.resolve() in inputs for path in (sources_path, ids_path)):
            raise ValueError("output paths must not overwrite an input")
        # All inputs and joins are checked before creating or replacing outputs.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        sources_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        ids_path.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
        raw_meta.update(dialogues=len(raw), expected_official_train_url=OFFICIAL_RAW_URL)
        if parsed_meta is not None:
            parsed_meta.update(dialogues=len(parsed), expected_official_train_url=OFFICIAL_PARSED_URL,
                               alignment="all UUIDs and agent/action/data events checked")
        summary = dict(schema="craigslist-candidate-review-v1", review_status="needs_human_review",
                       split_status="provisional_official_train" if team_ids is None else "team_allowlist_intersection",
                       warning=("PROVISIONAL: official training split only; team split alignment required."
                                if team_ids is None else "Team IDs intersected; semantic human review still required."),
                       sources=dict(raw=raw_meta, parsed=parsed_meta, team_train_ids=team_meta),
                       counts=counts, outputs=dict(sources=str(sources_path), train_ids=str(ids_path)))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"Candidate import failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
