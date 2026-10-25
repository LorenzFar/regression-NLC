"""Public adapter behavior using a tiny local model, NOT checkpoint accuracy."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from retrieval_pipeline.nlu_adapter import NLUAdapter, NLULoadError

ACTS = ["propose", "counter", "accept", "reject", "inquire", "vague"]


@pytest.fixture
def resources(tmp_path):
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[', ']', 'seller',
             'buyer', '<', '>', 'price', 'ok', 'no', 'can', 'you', 'do', '?',
             'hello', 'about', 'works', 'that', 'maybe', 'one', '50', '$']
    (tmp_path / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
    tokenizer = DistilBertTokenizerFast(vocab={token: index for index, token in enumerate(vocab)},
                                       do_lower_case=True)
    assert tokenizer.convert_tokens_to_ids('buyer') == vocab.index('buyer')
    assert tokenizer.convert_tokens_to_ids('seller') != tokenizer.unk_token_id
    tokenizer.save_pretrained(tmp_path)
    config = DistilBertConfig(vocab_size=len(vocab), n_layers=1, n_heads=2,
                             dim=16, hidden_dim=24, num_labels=6, max_position_embeddings=192, dropout=0.0,
                             seq_classif_dropout=0.0, attention_dropout=0.0)
    config.save_pretrained(tmp_path)
    torch.manual_seed(7)
    model = DistilBertForSequenceClassification(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[1] = 2.0
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'state': model.state_dict(), 'T': 2.0, 'k': 2, 'acts': ACTS}, checkpoint)
    return tmp_path, checkpoint


def test_predict_returns_ordered_unrounded_temperature_scaled_probabilities(resources):
    directory, checkpoint = resources
    adapter = NLUAdapter(checkpoint, model_name=str(directory))
    probabilities = adapter.predict([], 'ok')
    assert list(probabilities) == ACTS
    assert probabilities['counter'] == pytest.approx(0.3521874283517515, abs=1e-7)
    assert probabilities['propose'] == pytest.approx(0.1295625143296497, abs=1e-7)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(type(value) is float for value in probabilities.values())
    assert probabilities['counter'] != round(probabilities['counter'], 4)


@pytest.mark.parametrize('history,message', [
    ([('customer', 'ok')], 'ok'),
    ([('BUYER', None)], 'ok'),
    ([('BUYER', 'ok', 'extra')], 'ok'),
    ('BUYER: ok', 'ok'),
    ([], None),
    ([], '   '),
])
def test_invalid_public_turns_fail_explicitly(resources, history, message):
    directory, checkpoint = resources
    adapter = NLUAdapter(checkpoint, model_name=str(directory))
    with pytest.raises(ValueError):
        adapter.predict(history, message)


@pytest.mark.parametrize('field,value', [
    ('T', 0.0), ('T', -1.0), ('T', float('nan')), ('T', float('inf')),
    ('T', True), ('T', '1.0'), ('k', -1), ('k', 1.5), ('k', True),
    ('acts', list(reversed(ACTS))), ('state', {}),
])
def test_incompatible_checkpoint_fails_without_predictions(resources, field, value):
    directory, checkpoint = resources
    content = torch.load(checkpoint, weights_only=True)
    content[field] = value
    torch.save(content, checkpoint)
    with pytest.raises(NLULoadError):
        NLUAdapter(checkpoint, model_name=str(directory))


def test_loaded_adapter_keeps_resources_and_reports_provenance(resources):
    directory, checkpoint = resources
    adapter = NLUAdapter(checkpoint, model_name=str(directory))
    before = adapter.predict([], 'ok')
    metadata = adapter.resource_metadata
    assert metadata['history_window'] == 2
    assert metadata['temperature'] == 2.0
    assert metadata['max_length'] == 192
    assert metadata['labels'] == ACTS
    assert len(metadata['checkpoint_sha256']) == 64
    assert metadata['checkpoint_bytes'] > 0
    assert metadata['weights_only'] is True
    metadata['labels'].clear()
    assert adapter.resource_metadata['labels'] == ACTS
    checkpoint.unlink()
    for file in directory.iterdir():
        file.unlink()
    assert adapter.predict([], 'ok') == before


def test_nonfinite_model_output_is_not_a_prediction(resources):
    directory, checkpoint = resources
    content = torch.load(checkpoint, weights_only=True)
    content['state']['classifier.bias'][0] = float('nan')
    torch.save(content, checkpoint)
    adapter = NLUAdapter(checkpoint, model_name=str(directory))
    with pytest.raises(RuntimeError, match='nonfinite'):
        adapter.predict([], 'ok')


def test_missing_or_unsafe_checkpoint_cannot_fall_back(resources):
    directory, checkpoint = resources
    with pytest.raises(NLULoadError):
        NLUAdapter(directory / 'absent.pt', model_name=str(directory))
    content = torch.load(checkpoint, weights_only=True)
    content['unsupported_metadata'] = directory
    torch.save(content, checkpoint)
    with pytest.raises(NLULoadError, match='Weights only load failed'):
        NLUAdapter(checkpoint, model_name=str(directory))


@pytest.fixture
def contextual_adapter(resources):
    directory, checkpoint = resources
    config = DistilBertConfig.from_pretrained(directory)
    torch.manual_seed(13)
    model = DistilBertForSequenceClassification(config)
    torch.save({'state': model.state_dict(), 'T': 1.5, 'k': 2, 'acts': ACTS}, checkpoint)
    return NLUAdapter(checkpoint, model_name=str(directory))


def test_prices_are_masked_in_current_message_and_history(contextual_adapter):
    adapter = contextual_adapter
    baseline = adapter.predict([('BUYER', 'can you do $1,200.50?'),
                                ('SELLER', 'about fifty five')], 'can you do 50 bucks?')
    assert adapter.predict([('BUYER', 'can you do twenty?'),
                            ('SELLER', 'about 450')], 'can you do 1k?') == baseline
    assert adapter.predict([('BUYER', 'can you do <PRICE>?'),
                            ('SELLER', 'about <PRICE>')], 'can you do <PRICE>?') == baseline


def test_only_latest_two_turns_are_used_without_changing_history(contextual_adapter):
    history = [('BUYER', 'can you do 50?'), ('SELLER', 'maybe')]
    copy = list(history)
    baseline = contextual_adapter.predict(history, 'ok')
    assert contextual_adapter.predict([('SELLER', 'no')] + history, 'ok') == baseline
    assert history == copy
    assert contextual_adapter.predict([], 'ok') != baseline
    assert contextual_adapter.predict(list(reversed(history)), 'ok') != baseline
    assert contextual_adapter.predict([('SELLER', 'can you do 50?'), ('BUYER', 'maybe')], 'ok') != baseline


def test_zero_history_window_ignores_all_prior_turns(resources):
    directory, checkpoint = resources
    torch.manual_seed(13)
    model = DistilBertForSequenceClassification(DistilBertConfig.from_pretrained(directory))
    torch.save({'state': model.state_dict(), 'T': 1.5, 'k': 0, 'acts': ACTS}, checkpoint)
    adapter = NLUAdapter(checkpoint, model_name=str(directory))
    assert adapter.predict([('BUYER', 'can you do 50?')], 'ok') == adapter.predict([], 'ok')


def test_long_message_and_history_fit_the_training_token_limit(contextual_adapter):
    probabilities = contextual_adapter.predict(
        [('BUYER', 'hello ' * 400), ('SELLER', 'maybe ' * 400)], 'ok ' * 400)
    assert list(probabilities) == ACTS
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_smoke_records_loading_failure_without_synthetic_predictions(tmp_path):
    import json
    from scripts.nlu_smoke import main

    output = tmp_path / 'failed-smoke.json'
    code = main([str(tmp_path / 'absent.pt'), '--output', str(output)])
    report = json.loads(output.read_text(encoding='utf-8'))
    assert code == 1
    assert report['status'] == 'error'
    assert report['cases'] == []
    assert report['error']['type'] == 'NLULoadError'


def test_root_api_preserves_history_first_calls_and_rounding(resources, monkeypatch):
    import nlu

    directory, checkpoint = resources
    # The complete checkpoint must never trigger a base-model weight download.
    def unused_base_weights(*args, **kwargs):
        pytest.fail("base model weights must not be loaded")
    monkeypatch.setattr(nlu.AutoModelForSequenceClassification, "from_pretrained", unused_base_weights)
    model = nlu.NLU(checkpoint, "cpu", model_name=str(directory), local_files_only=True)
    monkeypatch.setattr(nlu, "_default", model)
    history = [("buyer", "can you do 50?"), ("seller", "maybe")]
    full = model._predict_probabilities(history, "ok")
    rounded = {act: round(value, 4) for act, value in full.items()}
    assert nlu.predict(history, "ok") == rounded
    assert nlu.predict_top(history, "ok") == {
        "act": "counter", "conf": rounded["counter"], "probs": rounded}
    assert model.predict_batch([(history, "ok"), (None, "ok")]) == [rounded, rounded]
    assert nlu.mask_prices("fourteen thirteen 42 dollars") == "fourteen thirteen <PRICE>"


def test_root_and_adapter_share_preprocessing_and_full_precision(contextual_adapter):
    adapter = contextual_adapter
    history = [("BUYER", "can you do 50?"), ("SELLER", "maybe")]
    full = adapter.predict(history, "ok")
    assert full == adapter._nlu._predict_probabilities(history, "ok")
    assert adapter._nlu.predict(history, "ok") == {act: round(value, 4) for act, value in full.items()}
    # Assert the actual paired input, not tiny random-model sensitivity.
    encoded = adapter._nlu._encode(history, "ok")["input_ids"]
    reversed_encoded = adapter._nlu._encode(list(reversed(history)), "ok")["input_ids"]
    assert not torch.equal(encoded, reversed_encoded)
