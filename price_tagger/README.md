# Price tagger (ML alternative to the rule-based extractor)

A token classifier (DistilBERT by default) that tags the characters of a message that form a money amount.
`price_reference.normalise_amount` then reads each tagged span, exactly as for the rules, so the model finds
the amount and never writes a number.

## Labels

`python -m scripts.gold_spans` adds a `gold_spans` column (character offsets of each gold price) to
`train_turns.csv` and `validation_turns.csv`. `test_turns.csv` is never given spans and is not used to train.

`python -m price_tagger.prepare_data` writes `price_tagger/data/{train,validation}.jsonl`. Per message:

| field | meaning |
|---|---|
| `pos` | spans of the gold prices, tagged B/I |
| `ignore` | money the rules keep but the gold does not mark (budgets, deltas, market prices) |
| `train_ok` | false for the few rows whose gold amount is a typo not written in the text |

Two ways to treat `ignore` (`train.py --ignore-mode`):

* `mask` (default): left out of the loss. The model tags all money, as an extractor should.
* `negative`: tagged O. The model learns the gold's own sense of "price", which is partly the resolver's job.

Everything else, including numbers the rules reject ("6 months", "5:30"), is tagged O. Those negatives are the
rules' judgement, not ground truth.

## Training

`train.py` needs only PyTorch + transformers. It was run on a GPU Jupyter node: copy `train.py`,
`evaluation/price_extractor/metrics.py` (as `metrics.py`), `price_reference/` and the `data/` folder to one
directory, then `python train.py --data data --out runs/NAME [--model-name roberta-base] [--ignore-mode negative]`.
Model selection uses validation only. Test is predicted once, at the end:
`python train.py --data data --out runs/NAME --predict test --model runs/NAME/model`.

## RoBERTa ensemble from the repository root

Install the existing `nlu` extra for PyTorch/transformers and `data` for CSV
preparation from Parquet: `python -m pip install -e ".[nlu,data]"`.
The scripts accept RoBERTa via `--model-name roberta-base`; the default single
model remains DistilBERT. Use the preprocessing and metadata settings used by
your trained checkpoints when reproducing a run.

```powershell
python -m price_tagger.train --data price_tagger/data --out runs/roberta-seed0 --model-name roberta-base --seed 0
python -m price_tagger.train --data price_tagger/data --out runs/roberta-seed1 --model-name roberta-base --seed 1
python -m price_tagger.ensemble --data price_tagger/data --models runs/roberta-seed0/model runs/roberta-seed1/model --split validation --out runs/roberta-ensemble
```

The ensemble averages token probabilities before decoding money spans. Its models
must use the same tokenizer, O/B/I label order, and metadata setting. Pass
`--no-meta` to inference if the models were trained that way. Prepared data,
training runs, predictions and checkpoint weights stay outside Git. Supply the
trained model directories through `--models`; this repository does not contain
the trained RoBERTa weights or establish their reported accuracy.

## Scoring

```
python -m evaluation.price_extractor.run --split validation --tagger NAME=validation_predictions.jsonl
```

scores the tagger with the same harness as the original parser and the rules.
