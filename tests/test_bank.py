"""Authored candidates exercise the approved preparation → bank seam."""
import json

import pytest

from buyer_agent import ContractError
from buyer_agent.bank import prepare_candidate, approve_candidate, build_bank, load_bank


def source(**changes):
    row = dict(template_id="fixture.counter.camera", dialogue_id="fixture-camera", turn_id="b2",
               origin="authored_fixture", split="train", speaker="BUYER", action="counter",
               history=[dict(speaker="BUYER", text="Camera offer 70")],
               message="The camera is pristine at 100", response="For the camera, I can offer 80.",
               price_spans=[dict(start=28, end=30, value="80", validated=True)],
               mixed_action=False, unresolved_reference=False)
    row.update(changes)
    return row


@pytest.mark.parametrize("speaker", ["system", ""])
def test_preparation_rejects_nonpublic_history(speaker):
    with pytest.raises(ContractError):
        prepare_candidate(source(history=[dict(speaker=speaker, text="Camera")]), {"fixture-camera"})


def test_preparation_needs_explicit_approval_and_preserves_source(tmp_path):
    candidate = prepare_candidate(source(), {"fixture-camera"})
    assert candidate["template"] == "For the camera, I can offer {price}."
    assert candidate["approved"] is False
    assert "For the camera" not in candidate["context"]
    path = tmp_path / "bank.jsonl"
    with pytest.raises(ContractError, match="approval"):
        build_bank([candidate], path, {"fixture-camera"})
    approved = approve_candidate(candidate, "Jiong — fixture review")
    build_bank([approved], path, {"fixture-camera"})
    saved = load_bank(path)
    assert saved[0]["origin"] == "authored_fixture"
    assert saved[0]["dialogue_id"] == "fixture-camera"
    assert saved[0]["turn_id"] == "b2"
    assert saved[0]["reviewer"] == "Jiong — fixture review"
    assert candidate["approved"] is False


@pytest.mark.parametrize("response", ["I offer -80.", "I offer +80.", "I offer 80%."])
def test_validated_span_cannot_leave_a_sign_or_percentage_term(response):
    start = response.index("80")
    with pytest.raises(ContractError):
        prepare_candidate(source(response=response, price_spans=[
            dict(start=start, end=start + 2, value="80", validated=True)]), {"fixture-camera"})

@pytest.mark.parametrize("changes", [
    {"speaker": "SELLER"}, {"split": "validation"}, {"dialogue_id": "held-out"},
    {"mixed_action": True}, {"unresolved_reference": True},
    {"response": "For the camera, I can offer 80 if delivered."},
    {"response": "For the camera, I can offer 80 or ninety."},
    {"response": "For the camera, I can offer 80 plus 5."},
    {"price_spans": []}, {"price_spans": [dict(start=28, end=30, value="80", validated=False)]},
])
def test_ineligible_sources_never_become_candidates(changes):
    with pytest.raises(ContractError):
        prepare_candidate(source(**changes), {"fixture-camera"})


def test_load_revalidates_source_and_rejects_malformed_banks(tmp_path):
    path = tmp_path / "bank.jsonl"
    approved = approve_candidate(prepare_candidate(source(), {"fixture-camera"}), "Jiong")
    build_bank([approved], path, {"fixture-camera"})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["template"] = "I accept {price} and delivery."
    path.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(ContractError):
        load_bank(path)
    path.write_text("null\n")
    with pytest.raises(ContractError):
        load_bank(path)
    records[1]["action"] = ["counter"]
    path.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(ContractError):
        load_bank(path)


def test_literal_price_slot_cannot_be_smuggled_in_source():
    with pytest.raises(ContractError):
        prepare_candidate(source(response="{price} For the camera, I can offer 80.",
                                 price_spans=[dict(start=36, end=38, value="80", validated=True)]),
                          {"fixture-camera"})


def test_malformed_source_fields_and_missing_reviewer_fail_at_public_builder(tmp_path):
    for changes in ({"action": ["counter"]}, {"origin": []}):
        with pytest.raises(ContractError):
            prepare_candidate(source(**changes), {"fixture-camera"})
    candidate = prepare_candidate(source(), {"fixture-camera"})
    with pytest.raises(ContractError):
        approve_candidate(candidate, "yes")
    candidate.update(approved="true", reviewer="Jiong")
    with pytest.raises(ContractError, match="approval"):
        build_bank([candidate], tmp_path / "bank.jsonl", {"fixture-camera"})
