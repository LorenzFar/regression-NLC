# Price tagger (ML alternative to the rule-based extractor)

A token classifier (DistilBERT by default) that tags the characters of a message that form a money amount.
`price_reference.normalise_amount` then reads each tagged span, exactly as for the rules, so the model finds
the amount and never writes a number.

## Labels: one price per message

`gold_price` is the ONE price the speaker puts forward (proposes, asks, counter-offers, accepts), or `none`. A message
whose speaker really proposes several prices (a range, "50 pickup / 55 delivered") or whose price is typed wrongly and
cannot be read from the text has an empty `gold_price` and a `gold_note` starting `uncertain:`: left out of training and
scoring. `gold_spans` (train and validation only) marks where that price is in the text; `[]` = none. Test never gets
spans. The original multi-price `gold_price` / `gold_note` are in `artifacts/gold_original/`; the commands

```
# The submission includes the archived originals. Keep them when rebuilding.
python -m scripts.gold_offer propose     # score the several-price rows
python -m scripts.gold_offer apply       # rebuild gold_price, gold_note, gold_spans from the archive + decisions.txt
```

rebuild everything, so the labels are reproducible. Rules: a refused ("1700 is too much"), listed, reference
("I paid 600") or reduction ("350 off") price is not the price; for "asking X but I can do Y" it is Y; a buyer who
states what they want or can pay ("I was looking to spend $135", "my budget is 1500", "I have 200 cash") is
proposing that amount. Decisions for multi-price rows are in `artifacts/gold_offer/decisions.txt` (AI-annotated, not
human-adjudicated).

`python -m price_tagger.prepare_data --out price_tagger/data_offer` writes training and validation JSONL from the CSVs
alone. Per message: `pos` (the span of the price, tagged B/I), `ignore` (other money the rules find: reductions,
reference, refused or listed prices), `train_ok` (false for excluded rows) and `excluded`.

`train.py --ignore-mode` chooses what `ignore` spans become: `mask` (left out of the loss), `negative` (plain O) or
`other` (their own "other money" label, so the model learns money-that-is-not-the-price apart from not-money; only the
price label is returned). `--context` adds the previous message as input. `--single` / `ensemble.py --single` return
at most one price per message: the best-scoring span, extended to the whole number it sits in.

## Training

`train.py` needs only PyTorch + transformers. It was run on a GPU Jupyter node: copy `train.py`,
`evaluation/price_extractor/metrics.py` (as `metrics.py`), `price_reference/` and the `data/` folder to one
directory, then `python train.py --data data --out runs/NAME [--model-name roberta-base] [--ignore-mode negative]`.
Model selection uses validation only. Test is predicted once, at the end:
`python train.py --data data --out runs/NAME --predict test --model runs/NAME/model`.

## Reproduce the report (Section 5.5)

Use the feature/price-reference branch and run from the repository root. The
CraigslistBargain turn CSVs in `artifacts/{train,validation,test}_turns.csv`
contain the price annotations. Source dialogues: [Stanford NLP CraigslistBargain](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains).
The archive in `artifacts/gold_original/` and the recorded AI relabelling decisions in
`artifacts/gold_offer/decisions.txt` let you reconstruct these CSVs. Generate the
scorer proposals before applying the decisions.

```powershell
python -m scripts.gold_offer propose
python -m scripts.gold_offer apply
python -m evaluation.price_extractor.run --split validation
python -m evaluation.price_extractor.run --split validation --tagger v3=artifacts/price_tagger/v3_base3/validation_predictions.jsonl
python -m evaluation.price_extractor.run --split test --final --tagger v3=artifacts/price_tagger/v3_base3/test_predictions.jsonl
python -m pytest tests/test_price_eval.py tests/test_resolver_handoff.py
```

The scores above use the published predictions. For training, install the project
with its NLU and test extras (`python -m pip install -e ".[nlu,test]"`) in an
environment with CUDA-enabled PyTorch and a GPU suitable for RoBERTa-large.
The first training run needs the pretrained backbone and tokenizer. The three
fine-tuned checkpoints are generated locally.

```powershell
python -m price_tagger.prepare_data --splits train validation test --out price_tagger/data_v3
foreach ($seed in 0,1,2) {
    python -m price_tagger.train --data price_tagger/data_v3 --out "runs/s$seed" --model-name roberta-large --lr 2e-5 --ignore-mode negative --single --seed $seed
}
foreach ($split in "validation","test") {
    python -m price_tagger.ensemble --data price_tagger/data_v3 --models runs/s0/model runs/s1/model runs/s2/model --split $split --single --o-scale 2.0 --out runs/ensemble
}
python -m evaluation.price_extractor.run --split validation --tagger v3=runs/ensemble/validation_predictions.jsonl
python -m evaluation.price_extractor.run --split test --final --tagger v3=runs/ensemble/test_predictions.jsonl
```

The final run used BIO token classification, role/category/listing metadata, 256
tokens, no previous utterance, and other money labelled O. Each seed used AdamW,
batch size 32, four epochs, weight decay 0.01, 6% warm-up, bf16, and best
validation amount F1 for checkpoint selection. At inference the ensemble averages
token probabilities, multiplies the O score by 2 before argmax, and normalises
the best surviving span. See `artifacts/price_eval_test_v3.json` (new_ensemble).
