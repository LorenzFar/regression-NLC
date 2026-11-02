"""Offline tagger integration tests: no pretrained model or dataset is downloaded."""
import json
import sys
from types import SimpleNamespace

import pytest

from price_tagger.prepare_data import build


def test_prepared_test_data_never_carries_training_labels():
    row = dict(split="test", dialogue="1", turn="2", text="Offer $500", role="buyer",
               category="car", listing_price="600", gold_price="500.00", gold_spans="[[6,10]]")
    assert "gold" not in build(row, labelled=False)
    assert "pos" not in build(row, labelled=False)
    labelled = build(row, labelled=True)
    assert labelled["gold"] == ["500.00"] and labelled["pos"] == [[6, 10]]
    assert labelled["train_ok"]
    row.update(text="Offer $5OO", gold_spans="")
    assert not build(row, labelled=True)["train_ok"]


def test_ensemble_module_averages_probabilities_and_decodes_original_spans(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from price_tagger import ensemble

    class Encoding(dict):
        def sequence_ids(self):
            return [None, 1, 1, None]

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, *args, **kwargs):
            return Encoding(input_ids=[0, 1, 2, 3], attention_mask=[1, 1, 1, 1],
                            offset_mapping=[(0, 0), (0, 5), (6, 10), (0, 0)])

    class Model:
        def __init__(self, path):
            self.path = path

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            logits = torch.full((*input_ids.shape, 3), -10.0)
            logits[:, :, 0] = 10.0
            # First model favors O; second strongly favors B. Averaging the token
            # probabilities favors B, so this catches using only the first model.
            logits[:, 2, :] = torch.tensor([2.0, 0.0, 0.0] if self.path == "first" else [0.0, 5.0, 0.0])
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(ensemble.AutoTokenizer, "from_pretrained", lambda path: Tokenizer())
    monkeypatch.setattr(ensemble.AutoModelForTokenClassification, "from_pretrained", Model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = tmp_path / "data"
    data.mkdir()
    (data / "validation.jsonl").write_text(json.dumps(dict(
        id="validation:1:2", text="Offer $500", role="buyer", category="car", listing=600)) + "\n", encoding="utf-8")
    output = tmp_path / "ensemble"
    monkeypatch.setattr(sys, "argv", ["ensemble", "--data", str(data), "--models", "first", "second",
                                       "--split", "validation", "--out", str(output)])
    ensemble.main()
    result = json.loads((output / "validation_predictions.jsonl").read_text(encoding="utf-8"))
    assert result == {"id": "validation:1:2", "spans": [[6, 10]], "values": ["500.00"], "unreadable": 0}
