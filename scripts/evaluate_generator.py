"""Compare buyer wording strategies on held-out CraigslistBargain turns.

    python scripts/evaluate_generator.py artifacts/craigslist/heldout_cases.jsonl \
        --bank examples/craigslist_bank.jsonl --output artifacts/generator-eval.json

Each held-out case gives the human buyer's action and price as the policy
decision, so every system must express the same terms. We then compare the
output with what the human buyer actually wrote (numbers masked):

  * token_f1     unigram overlap with the human reply (higher = closer wording)
  * distinct_2   share of unique word bigrams across outputs (higher = more varied)
  * price_kept   output states the authorised price and no other number
  * retrieved    share of outputs taken from the bank instead of a fallback

Systems: fixed fallback sentences, a random template of the right action, and
TF-IDF context retrieval (unigram, and unigram+bigram). Held-out dialogues never
enter the bank (scripts/build_craigslist_bank.py splits by dialogue UUID).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval_pipeline.contracts import GenerationResult, PolicyDecision, Turn
from retrieval_pipeline.generator import RetrievalGenerator, _display

WORD = re.compile(r"[a-z']+|#")


def tokens(text):
    return WORD.findall(re.sub(r"\d+(?:\.\d+)?", "#", text.lower()))


def token_f1(predicted, reference):
    p, r = Counter(tokens(predicted)), Counter(tokens(reference))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def distinct_2(texts):
    bigrams = [tuple(t[i:i + 2]) for text in texts for t in [tokens(text)] for i in range(len(t) - 1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def price_kept(text, price):
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return numbers == ([] if price is None else [_display(price)]) or (
        price is not None and numbers == [] and "{price}" not in text)


def decision_for(case):
    price = case["price_spans"][0]["value"] if case["price_spans"] else None
    if case["action"] in {"accept", "confirm"}:
        return PolicyDecision(case["action"], price or "100", "seller-1" if case["action"] == "accept" else "buyer-1")
    return PolicyDecision(case["action"], price)


class RandomTemplate:
    """Same action pool as retrieval, but ignores the conversation."""

    def __init__(self, generator, seed=0):
        self.generator, self.random = generator, random.Random(seed)

    def generate(self, decision, history, message):
        pool = {"accept", "confirm"} if decision.action in {"accept", "confirm"} else {decision.action}
        rows = [row for row in self.generator.bank if row["action"] in pool]
        row = self.random.choice(rows)
        return GenerationResult(row["template"].format(price=_display(decision.price)), decision.action,
                                decision.price, row["template_id"], "retrieval")


def with_vectorizer(generator, **options):
    generator.vectorizer = TfidfVectorizer(**options)
    generator.features = generator.vectorizer.fit_transform([row["context"] for row in generator.bank])
    return generator


def evaluate(system, cases):
    outputs = []
    for case in cases:
        history = [Turn(t["speaker"], t["text"]) for t in case["history"]]
        decision = decision_for(case)
        result = system.generate(decision, history, case["message"])
        outputs.append((case, decision, result))
    texts = [result.text for _, _, result in outputs]
    return {
        "token_f1": round(sum(token_f1(r.text, c["response"]) for c, _, r in outputs) / len(outputs), 4),
        "distinct_2": round(distinct_2(texts), 4),
        "price_kept": round(sum(price_kept(r.text, d.price) for _, d, r in outputs) / len(outputs), 4),
        "retrieved": round(sum(r.source == "retrieval" for _, _, r in outputs) / len(outputs), 4),
        "examples": [{"seller": c["message"], "human": c["response"], "system": r.text}
                     for c, _, r in outputs[:5]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases")
    parser.add_argument("--bank", default="examples/craigslist_bank.jsonl")
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]

    def bank():
        return RetrievalGenerator(args.bank, allow_provisional=True)

    systems = {
        "fixed_fallback": RetrievalGenerator(),
        "random_template": RandomTemplate(bank()),
        "tfidf_unigram": bank(),
        "tfidf_bigram": with_vectorizer(bank(), ngram_range=(1, 2), sublinear_tf=True),
    }
    report = {"cases": len(cases), "by_action": dict(Counter(c["action"] for c in cases)),
              "systems": {name: evaluate(system, cases) for name, system in systems.items()}}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        for name, result in report["systems"].items():
            print(f"{name:16} token_f1={result['token_f1']:.3f} distinct_2={result['distinct_2']:.3f} "
                  f"price_kept={result['price_kept']:.2f} retrieved={result['retrieved']:.2f}")


if __name__ == "__main__":
    main()
