"""End-to-end CLI behaviour for the RQ1/RQ2 evaluators, using tiny synthetic files."""
import json
from pathlib import Path

import pytest

from evaluation import gate_eval, nlu_eval
from evaluation.data import load_gold

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def gold_rows():
    history = [{"speaker": "buyer", "text": "Can you do 40?"}, {"speaker": "seller", "text": "45"}]
    acts = ["accept", "reject", "counter", "inquire", "accept", "vague"]
    return [{"case_id": f"g{i}", "history": history, "message": f"m{i}", "act": a,
             "needs_clarification": a == "vague", "deal_agreed": a == "accept",
             "challenge": ["ambiguous_acceptance"] if a == "accept" else []}
            for i, a in enumerate(acts)]


def test_template_gold_file_is_valid():
    rows = load_gold(ROOT / "examples" / "gold_template.jsonl",
                     require=("act", "message", "needs_clarification", "deal_agreed"))
    assert len(rows) >= 3


def test_nlu_eval_compares_systems_and_pools_seeds(tmp_path, capsys):
    gold = gold_rows()
    write_jsonl(tmp_path / "gold.jsonl", gold)
    perfect = [{"case_id": g["case_id"], "act": g["act"]} for g in gold]
    weak = [{"case_id": g["case_id"], "act": "accept"} for g in gold]
    for name, rows in (("parser", weak), ("k2_s0", perfect), ("k2_s1", perfect)):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    out = tmp_path / "rq1.json"
    code = nlu_eval.main(["--gold", str(tmp_path / "gold.jsonl"),
                          "--predictions", f"parser={tmp_path / 'parser.jsonl'}",
                          "--predictions", f"k2@0={tmp_path / 'k2_s0.jsonl'}",
                          "--predictions", f"k2@1={tmp_path / 'k2_s1.jsonl'}",
                          "--resamples", "50", "--output", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["baseline"] == "parser"
    assert report["systems"]["k2@0"]["accuracy"] == 1.0
    assert report["systems"]["parser"]["accuracy"] == pytest.approx(2 / 6)
    assert report["vs_baseline"]["k2@0"]["mcnemar"]["a_only_correct"] == 4
    assert report["pooled"]["k2"]["runs"] == 2 and report["pooled"]["k2"]["macro_f1_std"] == 0
    assert report["systems"]["k2@0"]["by_challenge_tag"]["ambiguous_acceptance"]["n"] == 2
    assert "calibration" not in report["systems"]["k2@0"]  # no probabilities supplied


def test_nlu_eval_uses_probabilities_for_calibration(tmp_path):
    gold = gold_rows()
    write_jsonl(tmp_path / "gold.jsonl", gold)
    acts = list(nlu_eval.m.ACTS)
    rows = [{"case_id": g["case_id"],
             "probs": {a: (0.7 if a == g["act"] else 0.06) for a in acts}} for g in gold]
    write_jsonl(tmp_path / "p.jsonl", rows)
    out = tmp_path / "r.json"
    assert nlu_eval.main(["--gold", str(tmp_path / "gold.jsonl"), "--predictions",
                          f"m={tmp_path / 'p.jsonl'}", "--resamples", "20", "--output", str(out)]) == 0
    cal = json.loads(out.read_text(encoding="utf-8"))["systems"]["m"]["calibration"]
    assert cal["ece"] == pytest.approx(0.3) and cal["aurc"] == 0.0


def test_nlu_eval_rejects_incomplete_predictions_and_bad_arguments(tmp_path, capsys):
    gold = gold_rows()
    write_jsonl(tmp_path / "gold.jsonl", gold)
    write_jsonl(tmp_path / "short.jsonl", [{"case_id": "g0", "act": "accept"}])
    code = nlu_eval.main(["--gold", str(tmp_path / "gold.jsonl"),
                          "--predictions", f"s={tmp_path / 'short.jsonl'}"])
    assert code == 2 and "no prediction" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        nlu_eval.main(["--gold", str(tmp_path / "gold.jsonl"), "--predictions", "no-equals-sign"])


def test_gate_eval_cli_runs_on_replay_style_records(tmp_path):
    from tests.test_eval_gate import GOLD, OFF, record

    full = [record("agreed", "full", "confirm"), record("ambiguous", "full", "clarify"),
            record("unsure", "full", "clarify"), record("clear_no", "full", "clarify")]
    write_jsonl(tmp_path / "replay.jsonl", OFF + full)
    write_jsonl(tmp_path / "gold.jsonl", [{"case_id": k, **v} for k, v in GOLD.items()])
    out = tmp_path / "rq2.json"
    assert gate_eval.main(["--replay", str(tmp_path / "replay.jsonl"), "--gold",
                           str(tmp_path / "gold.jsonl"), "--thresholds", "0.5", "0.9",
                           "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["by_mode"]["full"]["incorrect_confirmations"] == 0
    assert [r.get("tau") for r in report["sweep"]["rows"]] == [None, None, 0.5, 0.9]


def test_gold_validation_reports_which_row_is_wrong(tmp_path):
    write_jsonl(tmp_path / "bad.jsonl", [{"case_id": "x", "act": "maybe"}])
    with pytest.raises(ValueError, match="x: act must be one of"):
        load_gold(tmp_path / "bad.jsonl")
    write_jsonl(tmp_path / "dup.jsonl", [{"case_id": "x", "act": "accept"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(tmp_path / "dup.jsonl")
