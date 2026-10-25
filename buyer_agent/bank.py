"""Conservative candidate preparation; semantic approval remains a human task."""
import argparse
import copy
import json
import re
from pathlib import Path

from .contracts import ContractError, Turn

PRICE_ACTIONS = {"propose", "counter", "accept", "confirm"}
ACTIONS = PRICE_ACTIONS | {"reject"}
# Deliberately conservative, not a semantic classifier or buyer-act mapper.
UNSAFE = re.compile(
    r"\b(if|unless|provided|delivery|deliver|shipping|ship|pickup|collection|collect|"
    r"warranty|cash|deposit|installment|trade|swap|plus|including|condition|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"dollars?|cents?|bucks?|grand|quid|sgd|usd|aud|gbp|eur|yen|yuan|ringgit|"
    r"billion|dozen|half|quarter|free|or|but)\b", re.I)


def _nonblank(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonblank string")


def retrieval_context(history, message, history_window=2):
    """Shared bank/query features: preceding public turns then current seller.

    Price-like digit runs are masked independently of the NLU checkpoint mask.
    Speaker labels are omitted so unrelated text cannot match on role alone.
    """
    if type(history_window) is not int or history_window < 0:
        raise ContractError("history_window must be a non-negative integer")
    if not isinstance(history, list) or any(
        not isinstance(t, Turn) or not isinstance(t.speaker, str)
        or t.speaker.lower() not in {"buyer", "seller"}
        or not isinstance(t.text, str) for t in history
    ) or not isinstance(message, str):
        raise ContractError("invalid preceding public context")
    turns = history[-history_window:] if history_window else []
    return re.sub(r"\d+(?:[.,]\d+)*", " price ", "\n".join(
        [t.text for t in turns] + [message])).lower()


def prepare_candidate(source, train_dialogue_ids, history_window=2):
    """Prepare one explicit buyer-action mapping or raise; never infer labels.

    Spans are supplied/validated upstream, with exclusive end offsets into the
    original response. This initial contract accepts a bare decimal span only.
    """
    if not isinstance(source, dict):
        raise ContractError("candidate source must be an object")
    row = copy.deepcopy(source)
    for name in ("template_id", "dialogue_id", "turn_id"):
        _nonblank(row.get(name), name)
    if not isinstance(row.get("origin"), str) or row["origin"] not in {"authored_fixture", "corpus"}:
        raise ContractError("explicit authored_fixture or corpus origin required")
    if row.get("split") != "train" or row["dialogue_id"] not in train_dialogue_ids:
        raise ContractError("source must belong to supplied training dialogue IDs")
    if (row.get("speaker") != "BUYER" or not isinstance(row.get("action"), str)
            or row["action"] not in ACTIONS):
        raise ContractError("explicit buyer role and supported buyer action required")
    if row.get("mixed_action") is not False or row.get("unresolved_reference") is not False:
        raise ContractError("mixed actions or unresolved references are ineligible")
    response = row.get("response")
    _nonblank(response, "response")
    if re.search(r"[{}]", response):
        raise ContractError("source response must not contain template slots")
    spans = row.get("price_spans")
    if not isinstance(spans, list) or len(spans) != (1 if row["action"] in PRICE_ACTIONS else 0):
        raise ContractError("exactly one validated price span required for priced actions; none for reject")
    template = response
    for span in spans:
        if not isinstance(span, dict):
            raise ContractError("price span must be an object")
        start, end, value = span.get("start"), span.get("end"), span.get("value")
        if (span.get("validated") is not True or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(response) or not isinstance(value, str)
                or re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", value) is None
                or response[start:end] != value):
            raise ContractError("invalid validated price span")
        template = response[:start] + "{price}" + response[end:]
    remainder = template.replace("{price}", "")
    if re.search(r"[\d{}$€£¥%+\-]", remainder) or UNSAFE.search(remainder):
        raise ContractError("unsupported numbers, slots, word prices or conditions")
    raw_history = row.get("history")
    if not isinstance(raw_history, list) or any(
        not isinstance(t, dict) or set(t) != {"speaker", "text"} for t in raw_history
    ):
        raise ContractError("history must contain preceding public turns")
    context = retrieval_context([Turn(**t) for t in raw_history], row.get("message"), history_window)
    row.update(template=template, context=context, history_window=history_window,
               approved=False, reviewer=None)
    return row


def approve_candidate(candidate, reviewer):
    """Record a human's explicit semantic review; does not perform that review."""
    _nonblank(reviewer, "reviewer")
    if len(reviewer.strip()) < 3 or reviewer.strip().lower() in {"yes", "true", "none", "n/a"}:
        raise ContractError("reviewer must meaningfully identify the human reviewer")
    result = copy.deepcopy(candidate)
    result.update(approved=True, reviewer=reviewer.strip())
    return result


def _validate_rows(rows, train_ids, window):
    if not isinstance(rows, list):
        raise ContractError("bank rows must be a list")
    seen = set()
    for row in rows:
        prepared = prepare_candidate(row, train_ids, window)
        if row.get("approved") is not True:
            raise ContractError("explicit approval required")
        approve_candidate(row, row.get("reviewer"))
        if any(row.get(key) != prepared[key] for key in ("template", "context", "history_window")):
            raise ContractError("bank template/context does not match validated source")
        if row["template_id"] in seen:
            raise ContractError("duplicate template_id")
        seen.add(row["template_id"])
    return rows


def build_bank(candidates, path, train_dialogue_ids, history_window=2):
    """Write approved JSONL, retaining every source (no lossy deduplication)."""
    rows = list(candidates)
    # Check window even for an empty bank.
    retrieval_context([], "", history_window)
    ids = sorted(train_dialogue_ids)
    if not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ContractError("nonempty training dialogue ID allowlist required")
    _validate_rows(rows, ids, history_window)
    manifest = dict(schema="buyer-response-bank-v1", train_dialogue_ids=ids, history_window=history_window)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [manifest, *rows]) + "\n", encoding="utf-8")


def load_bank(path, history_window=None):
    """Fail closed on corrupt, unapproved, or source-inconsistent artifacts."""
    try:
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        manifest, *rows = records
        if not isinstance(manifest, dict) or manifest.get("schema") != "buyer-response-bank-v1":
            raise ContractError("unsupported bank schema")
        ids, window = manifest["train_dialogue_ids"], manifest["history_window"]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ContractError("invalid training dialogue IDs")
        retrieval_context([], "", window)
        if history_window is not None and window != history_window:
            raise ContractError("bank/query history_window mismatch; rebuild the bank")
        return _validate_rows(rows, ids, window)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ContractError(f"invalid response bank: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "approve", "build"):
        stage = stages.add_parser(name)
        stage.add_argument("input", help="source or candidate JSONL")
        stage.add_argument("output")
        if name == "approve":
            stage.add_argument("--reviewer", required=True, help="human approving this reviewed file")
        else:
            stage.add_argument("--train-ids", required=True, help="JSON list from the training split owner")
            stage.add_argument("--history-window", type=int, default=2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    if args.stage == "approve":
        # The human must inspect/select the candidate input BEFORE invoking this.
        rows = [approve_candidate(row, args.reviewer) for row in rows]
    else:
        ids = json.loads(Path(args.train_ids).read_text(encoding="utf-8"))
        if args.stage == "build":
            build_bank(rows, args.output, ids, args.history_window)
            print(f"Built {len(load_bank(args.output))} approved templates")
            return
        eligible = []
        for row in rows:
            try:
                eligible.append(prepare_candidate(row, ids, args.history_window))
            except ContractError as exc:
                print(f"Excluded candidate: {exc}")
        rows = eligible
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"{args.stage}: wrote {len(rows)} candidates")


if __name__ == "__main__":
    main()
