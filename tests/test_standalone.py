"""Standalone CLI seam: actual sklearn retrieval, explicitly fixture NLU."""
import json
from pathlib import Path
import subprocess
import sys


def test_standalone_cli_exercises_retrieval_fallback_and_clarification(tmp_path):
    output = tmp_path / "demo.jsonl"
    completed = subprocess.run(
        [sys.executable, "-m", "examples.standalone_contract", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert all(record["error"] is None for record in records)
    assert [r["generation"]["source"] for r in records] == [
        "retrieval", "retrieval", "fallback", "clarification"]
    assert records[0]["generation"]["text"] == "For the camera, I offer 80.50."
    assert records[1]["generation"]["text"] == "For the bicycle, I offer 80.50."
    assert records[2]["generation"]["fallback_reason"] == "zero_similarity"
    assert records[3]["gate"]["reasons"] == ["unresolved_reference"]
    for record in records:
        assert record["origin"] == "authored_fixture"
        assert record["resources"]["nlu"]["fixture"] is True
    assert records[3]["state_after"]["state"] == records[3]["state_before"]["state"]
