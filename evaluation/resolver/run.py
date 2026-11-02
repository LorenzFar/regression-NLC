"""Run resolver conditions A-D on explicitly labelled JSONL.

Example: python -m evaluation.resolver.run --gold review.jsonl --split validation
Use --nlu package.module:factory for an actual NLU. Without it B/C require
recorded predicted_act or nlu_probabilities; gold acts are NEVER substituted.
"""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path

from price_reference import CachingNLU, build_ledger, resolve
from price_reference.evidence import collect_mentions
from price_reference.ledger import _turn
from price_reference.observer import act_from_probabilities
from .data import gold_ledger, gold_mentions, load_gold, verify_freeze
from .metrics import exact_mcnemar, paired, score, summarize

CONDITIONS = {"A": ("gold", "gold", "gold"), "B": ("gold", "predicted", "gold"),
              "C": ("extractor", "predicted", "predicted"), "D": ("gold", "gold", "predicted")}


def evaluate(rows, *, nlu=None, conditions=("A", "B", "C", "D"), include_silver=False, headline_min_n=100):
    if type(headline_min_n) is not int or headline_min_n < 1:
        raise ValueError("headline_min_n must be positive")
    report = {"exposure": "development-exposed", "conditions": {}, "paired": {}, "paired_tests": {}, "pairwise_subsets": {},
              "excluded": {"uncertain": 0, "blank_unconfirmed": 0, "pending_review": 0, "silver": 0}, "skipped": {},
              "silver_policy": "Included SILVER: non-independent lexical-seed diagnostics, not reviewed accuracy"
              if include_silver else "Silver excluded in all conditions (use --include-silver for diagnostics)",
              "headline_min_n": headline_min_n}
    eligible = []
    for row in rows:
        if row.get("uncertain") or str(row.get("gold_note", "")).lower().startswith("uncertain:"):
            report["excluded"]["uncertain"] += 1
        elif row.get("review_required"):
            report["excluded"]["pending_review"] += 1
        elif row.get("gold_price") == "" and not row.get("confirmed_blank") and "mentions" not in row:
            report["excluded"]["blank_unconfirmed"] += 1
        elif row.get("label_source") == "seeded_rule" and not include_silver:
            report["excluded"]["silver"] += 1
        else:
            eligible.append(row)
    predicted_acts = {}
    for row in eligible:
        if nlu is not None:
            history = [(speaker.upper(), text) for speaker, text in map(_turn, row["history"])]
            predicted_acts[row["case_id"]] = act_from_probabilities(nlu.predict(history, row["message"]))
        elif row.get("nlu_probabilities") is not None:
            predicted_acts[row["case_id"]] = act_from_probabilities(row["nlu_probabilities"])
        elif row.get("predicted_act") is not None:
            predicted_acts[row["case_id"]] = row["predicted_act"]
    for name in conditions:
        if name not in CONDITIONS:
            raise ValueError(f"unknown condition {name}")
        mention_source, act_source, ledger_source = CONDITIONS[name]
        used_rows, results, skipped = [], [], []
        for row in eligible:
            case_id = row["case_id"]
            if act_source == "predicted" and case_id not in predicted_acts:
                skipped.append({"case_id": case_id, "reason": "missing_predicted_act"})
                continue
            if ledger_source == "gold" and "offers" not in row:
                skipped.append({"case_id": case_id, "reason": "missing_gold_ledger"})
                continue
            if mention_source == "gold" and "mentions" not in row and "gold_price" not in row:
                skipped.append({"case_id": case_id, "reason": "missing_gold_mentions"})
                continue
            if act_source == "gold" and row.get("act") is None:
                skipped.append({"case_id": case_id, "reason": "missing_gold_act"})
                continue
            listing = row.get("listing_price")
            mentions = gold_mentions(row) if mention_source == "gold" else collect_mentions(row["message"], listing)
            ledger = gold_ledger(row) if ledger_source == "gold" else build_ledger(row["history"], row.get("state"), listing)
            act = row["act"] if act_source == "gold" else predicted_acts[case_id]
            result = resolve(row["message"], speaker=row.get("speaker", "seller"), act=act,
                             mentions=mentions, ledger=ledger)
            used_rows.append(row)
            results.append(result)
        metrics = score(used_rows, results)
        metrics["aggregate_label"] = "Diagnostic aggregate across included sources; not headline accuracy"
        metrics["evaluation_status"] = "evaluated" if used_rows else "no_evaluable_cases"
        metrics["sources"] = {"mentions": mention_source, "act": act_source, "ledger": ledger_source}
        metrics["predictions"] = [{"case_id": row["case_id"], **asdict(result)}
                                  for row, result in zip(used_rows, results)]
        report["conditions"][name] = metrics
        report["skipped"][name] = skipped
    for left, right in (("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")):
        if left in report["conditions"] and right in report["conditions"]:
            left_cases, right_cases = report["conditions"][left]["cases"], report["conditions"][right]["cases"]
            report["paired"][f"{left}:{right}"] = paired(left_cases, right_cases)
            shared_ids = {case["case_id"] for case in left_cases} & {case["case_id"] for case in right_cases}
            sources = sorted({case["label_source"] for case in left_cases} & {case["label_source"] for case in right_cases})
            report["pairwise_subsets"][f"{left}:{right}"] = {
                "n": len(shared_ids), "case_ids": sorted(shared_ids),
                "conditions": {name: {"by_label_source": {
                    source: summarize([case for case in cases if case["case_id"] in shared_ids
                                       and case["label_source"] == source]) for source in sources}}
                    for name, cases in ((left, left_cases), (right, right_cases))}}
            report["paired_tests"][f"{left}:{right}"] = {
                "scope": "pairwise shared cases, per label source; exploratory",
                "by_label_source": {source: {
                    endpoint: exact_mcnemar([case for case in left_cases if case["label_source"] == source],
                                            [case for case in right_cases if case["label_source"] == source], endpoint)
                    for endpoint in ("status_correct", "price_correct")} for source in sources}}
    # Headline explicitly limited to independently human-reviewed hard cases.
    hard_ids = {row["case_id"] for row in eligible if row["label_source"] == "human_reviewed" and row.get("subset") == "hard"}
    report["headline"] = {}
    for name, result in report["conditions"].items():
        cases = [case for case in result["cases"] if case["case_id"] in hard_ids]
        publishable = len(cases) >= headline_min_n
        report["headline"][name] = {"n": len(cases), "minimum_n": headline_min_n,
                                    "status": "reportable" if publishable else "insufficient_n",
                                    "publishable": publishable,
                                    "reason": None if publishable else "insufficient_human_reviewed_hard_cases",
                                    "metrics": summarize(cases) if publishable else None}
    # These figures have identical case IDs/denominators across selected conditions.
    case_sets = [{case["case_id"] for case in result["cases"]} for result in report["conditions"].values()]
    common = set.intersection(*case_sets) if case_sets else set()
    report["common_subset"] = {"n": len(common), "case_ids": sorted(common), "conditions": {}}
    for name, result in report["conditions"].items():
        selected = [case for case in result["cases"] if case["case_id"] in common]
        report["common_subset"]["conditions"][name] = {
            "by_label_source": {source: summarize([case for case in selected if case["label_source"] == source])
                                for source in sorted({case["label_source"] for case in selected})}}
    report["eligibility"] = {"input_n": len(rows), "eligible_n": len(eligible),
                             "status": "eligible_cases_present" if eligible else "no_eligible_cases",
                             "message": None if eligible else "No eligible evaluation rows remain after exclusions."}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "holdout"], default="validation")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--nlu", help="module:factory returning an NLU object; no model is downloaded implicitly")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--include-silver", action="store_true", help="include non-independent lexical-seed diagnostics")
    parser.add_argument("--headline-min-n", type=int, default=100, help="reporting threshold, not proof of representativeness")
    args = parser.parse_args(argv)
    if args.headline_min_n < 1:
        parser.error("--headline-min-n must be positive")
    rows = load_gold(args.gold)
    rows = [row for row in rows if row.get("split", args.split) == args.split]
    if not rows:
        parser.error("no rows for selected split")
    manifest = None
    if args.split == "holdout":
        if args.freeze_manifest is None:
            parser.error("holdout requires --freeze-manifest")
        manifest = verify_freeze(args.gold, args.freeze_manifest, rows)
    nlu = None
    if args.nlu:
        module, name = args.nlu.split(":", 1)
        nlu = CachingNLU(getattr(importlib.import_module(module), name)())
    report = evaluate(rows, nlu=nlu, conditions=args.conditions, include_silver=args.include_silver,
                      headline_min_n=args.headline_min_n)
    report["split"] = args.split
    if manifest:
        report["exposure"] = manifest["exposure"]
        report["freeze_manifest"] = manifest
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Detailed predictions live in the optional JSON report, not a noisy terminal dump.
    print(json.dumps({"split": args.split, "exposure": report["exposure"], "excluded": report["excluded"],
                      "silver_policy": report["silver_policy"],
                      "eligibility": report["eligibility"],
                      "condition_evaluation_status": {k: {"status": v["evaluation_status"], "n": v["overall"]["n"]}
                                                      for k, v in report["conditions"].items()},
                      "conditions_by_label_source": {k: v["slices"].get("label_source", {}) for k, v in report["conditions"].items()},
                      "headline_human_reviewed_hard": report["headline"],
                      "common_subset": report["common_subset"],
                      "skipped": {k: len(v) for k, v in report["skipped"].items()},
                      "paired": report["paired"], "paired_tests": report["paired_tests"],
                      "pairwise_subsets": report["pairwise_subsets"]}, indent=2))


if __name__ == "__main__":
    main()
