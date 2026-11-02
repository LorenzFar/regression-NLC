"""Set-based scoring of price outputs against gold_price. Pure functions, no I/O."""
from collections import defaultdict
from decimal import Decimal


def to_set(value):
    """gold_price / parser_price cell -> frozenset of "N.NN" strings. "none", empty and None -> empty set."""
    if value is None:
        return frozenset()
    text = str(value).strip()
    if text in ("", "none"):
        return frozenset()
    return frozenset(f"{Decimal(part.strip()):.2f}" for part in text.split("|"))


class Tally:
    """Counts for one group of (gold, predicted) pairs."""

    def __init__(self):
        self.n = self.exact = self.tp = self.fp = self.fn = 0
        self.has_tp = self.has_fp = self.has_fn = self.has_tn = 0

    def add(self, gold, pred):
        self.n += 1
        self.exact += gold == pred
        self.tp += len(gold & pred)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)
        if gold and pred:
            self.has_tp += 1
        elif pred:
            self.has_fp += 1
        elif gold:
            self.has_fn += 1
        else:
            self.has_tn += 1

    def summary(self):
        ratio = lambda a, b: a / b if b else 0.0
        precision, recall = ratio(self.tp, self.tp + self.fp), ratio(self.tp, self.tp + self.fn)
        presence_p = ratio(self.has_tp, self.has_tp + self.has_fp)
        presence_r = ratio(self.has_tp, self.has_tp + self.has_fn)
        return {
            "n": self.n,
            "exact_match": ratio(self.exact, self.n),
            "precision": precision, "recall": recall, "f1": ratio(2 * precision * recall, precision + recall),
            "presence_precision": presence_p, "presence_recall": presence_r,
            "presence_f1": ratio(2 * presence_p * presence_r, presence_p + presence_r),
            "confusion": {"price_found": self.has_tp, "false_price": self.has_fp,
                          "missed_price": self.has_fn, "correct_none": self.has_tn},
        }


def score(rows, predict, group_by=("role", "category")):
    """rows: dicts with "gold" (frozenset) plus the group columns. predict(row) -> frozenset.
    Returns {"overall": summary, "role": {value: summary}, "category": {...}}."""
    overall = Tally()
    groups = {g: defaultdict(Tally) for g in group_by}
    for row in rows:
        pred = predict(row)
        overall.add(row["gold"], pred)
        for g in group_by:
            groups[g][row[g]].add(row["gold"], pred)
    result = {"overall": overall.summary()}
    for g, tallies in groups.items():
        result[g] = {value: t.summary() for value, t in sorted(tallies.items())}
    return result
