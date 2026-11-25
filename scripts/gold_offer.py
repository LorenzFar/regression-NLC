"""Reduce the gold to the ONE price a message means, in the same columns (gold_price, gold_note, gold_spans).

A message's price is the one its speaker proposes, asks for, counter-offers or accepts. Prices it refuses
("1700 is too much"), quotes for reference ("I paid 600") or wishes for ("I was hoping for 1500") are not it.

  python -m scripts.gold_offer archive    copy the original gold_price / gold_note to artifacts/gold_original/ (once)
  python -m scripts.gold_offer propose    score the several-price rows (needs the archive)
  python -m scripts.gold_offer apply      write gold_price (one price), gold_note, and gold_spans (train/validation)
  python -m scripts.gold_offer status

apply always starts from the archive, so it is repeatable and never loses the original labels.
Decisions file lines:  split:dialogue:turn ;; <price|multi|none> [;; note]
  multi = the speaker really proposes several prices (a range, "50 if pickup, otherwise 55"): gold_price is left
  empty with an "uncertain:" note, so the row is excluded from training and scoring like any other uncertain row.
gold_spans = JSON [start, end] character offsets of the price in the text; "[]" when the price is none; empty when
the price is not written in the text (a typo the label corrects). test_turns.csv never gets spans.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from price_reference import explain, extract
from scripts.gold_csv import read, write

COLS = ("gold_offer", "gold_offer_spans", "gold_offer_note")
SPLITS = ("train", "validation", "test")
WORK = Path("artifacts/gold_offer")

# Wording that introduces the speaker's own proposal (looked for just before the price).
OFFER_BEFORE = re.compile(
    r"(?:how\s+(?:about|bout)|what\s+(?:about|if)|would\s+(?:you|u)\s+(?:be\s+(?:willing|able)\s+to\s+)?(?:take|accept|consider|do|go|sell|settle)|"
    r"(?:can|could|will)\s+(?:you|u)\s+(?:do|go|take|accept|settle|sell|come|drop|lower|bring)|(?:would|will)\s+\w+\s+(?:work|do)|"
    r"i\s*(?:'d|would|will|'ll|can|could|may|might|am\s+willing\s+to|am\s+able\s+to|'m\s+willing\s+to)\s+(?:even\s+|only\s+|still\s+|also\s+|just\s+|probably\s+|maybe\s+)?"
    r"(?:do|take|accept|offer|pay|go|settle|sell|give|drop|bring|come|meet|let\s+it\s+go|part\s+with|agree|say|make\s+it)|"
    r"(?:best|lowest|final|last|max|highest)\s+(?:offer\s+|price\s+)?(?:i\s+(?:can|could|will|would)\s+(?:do|go|offer)|is|would\s+be|i\s+can\s+do)|"
    r"(?:settle|meet|go|come|drop|bring|lower|down|up)\s+(?:down\s+|up\s+)?(?:for|at|to|too|in\s+the\s+middle\s+at)|make\s+it|offer(?:ing)?|"
    r"(?:sell|sold|give|let\s+it\s+go|part\s+with)\s+(?:it\s+)?(?:to\s+you\s+)?(?:for|at)\s+(?:just|only)?|"
    r"let'?s\s+(?:say|do|make\s+it|call\s+it)|(?:i\s+)?(?:will|'ll)\s+take|deal\s+at|become(?:s)?(?:\s+\w+)?)\s*[^.?!;]{0,12}$", re.I)
# Wording that marks a price as refused, quoted for comparison, or wished for.
OTHER_BEFORE = re.compile(
    r"(?:(?:can't|cannot|can\s+not|won't|not\s+able\s+to|unable\s+to|no\s+way|don't\s+have|do\s+not\s+have|not\s+for)\s+(?:\w+\s+){0,3}$|"
    r"(?:paid|worth|retail\w*|original(?:ly)?|msrp|blue\s*book|kbb|listed(?:\s+for)?|listing(?:\s+(?:is|price))?|new\s+(?:for|at|is)|"
    r"hoping\s+for|wanted|want|wish|expecting|looking\s+(?:for|to\s+(?:spend|pay))|budget(?:\s+is|\s+of)?|"
    r"you\s+(?:want|said|offered|asked|are\s+asking|were\s+asking)|your|their|his|her|offering|asking|go\s+for|going\s+for|goes\s+for|sell\s+for|sells\s+for|"
    r"cost(?:s)?(?:\s+me)?|over|above|more\s+than|than|from)\s*[^.?!;]{0,10}$)", re.I)
OTHER_AFTER = re.compile(
    r"^\s*(?:\w+\s+){0,3}?(?:is|are|was|would\s+be|seems?|sounds?)?\s*(?:just\s+|way\s+|much\s+|far\s+|a\s+bit\s+|so\s+|too\s+|really\s+|still\s+|kind\s+of\s+|a\s+little\s+)*"
    r"(?:too\s+)?(?:low|high|much|steep|little|expensive|cheap|out\s+of|not\s+enough|more\s+than)|"
    r"^\s*(?:\w+\s+){0,2}?(?:new|brand\s+new|retail|originally|at\s+(?:the\s+)?store|on\s+amazon|online|elsewhere|each|per)\b", re.I)
CONCESSION = re.compile(r"\b(?:but|however|though|although|instead|rather\s+than|yet)\b", re.I)
RANGE_OR_ALT =re.compile(r"\b(?:otherwise|or\s+else|if\s+(?:you|i|we)\b.*\bif\s+(?:you|i|we)\b|and\s+if|but\s+if|or\b.*\bor\b)", re.I)


def clause_bounds(text, start, end):
    a = max((m.end() for m in re.finditer(r"[.?!;]\s", text[:start])), default=0)
    m = re.search(r"[.?!;](?:\s|$)", text[end:])
    return a, end + (m.start() + 1 if m else len(text) - end)


def score(text, start, end):
    a, b = clause_bounds(text, start, end)
    before, after = text[a:start].lower()[-70:], text[end:b].lower()[:50]
    s = 0.0
    if OFFER_BEFORE.search(before):
        s += 2.0
    if OTHER_BEFORE.search(before):
        s -= 2.0
    if OTHER_AFTER.search(after):
        s -= 2.0
    if re.match(r"\s*(?:\w+\s+){0,2}?(?:works?|sounds?\s+(?:good|fair)|is\s+(?:fine|ok|good|my\s+(?:best|final|lowest|limit|max)))", after):
        s += 1.0
    if re.match(r"\s*(?:is|are)\s+(?:the\s+|my\s+)?(?:absolute\s+|very\s+)?(?:highest|lowest|most|least|max(?:imum)?|best|final|bottom)\b", after):
        s += 2.0   # "2300 is the absolute highest I can go"
    if re.search(r"\?\s*$", text[b - 1:b + 1]) and b == len(text):
        s += 0.3   # a closing "...75?" is usually the ask
    return s


def propose(row):
    """(price, confidence, why) for a multi-price row; price None when it cannot be told."""
    text = row["text"]
    vals = sorted({p.strip() for p in row["gold_price"].split("|")})
    listing = float(row["listing_price"]) if row["listing_price"] else None
    cands = explain(text, listing)
    kinds = {(c.start, c.end): c.mention.kind for c in cands if c.mention is not None}
    scored = []
    for v in vals:
        spans = [(c.start, c.end) for c in cands if c.value == v]
        if not spans:
            continue
        best = max(spans, key=lambda s: (score(text, *s), s[0]))
        sc = score(text, *best)
        if kinds.get(best, "absolute") == "delta":
            sc -= 3.0   # "350 off", "$20 less": a reduction, not a price
        scored.append([sc + best[0] * 1e-4, v, best])   # later position breaks ties
    # A later price introduced by "but / however / though ..." replaces an earlier statement.
    for item in scored:
        for other in scored:
            if other[2][0] > item[2][1] and CONCESSION.search(text[item[2][1]:other[2][0]]):
                item[0] -= 1.5
                break
    scored = [tuple(x) for x in scored]
    if len(scored) < 2:
        return None, 0.0, "span lookup failed"
    scored.sort(reverse=True)
    margin = scored[0][0] - scored[1][0]
    flag = " (alternatives/conditions)" if RANGE_OR_ALT.search(text) else ""
    return scored[0][1], margin, f"top {scored[0][1]} ({scored[0][0]:.1f}) vs next {scored[1][1]} ({scored[1][0]:.1f}){flag}"


# A buyer who says what they want to pay ("I was looking to spend about $135", "my budget is 1500") is
# proposing that amount. The gold counts these as the price about 80% of the time and as none 20%; the rule
# makes the 20% consistent. It applies to buyer messages with exactly one price, not refused or listed ones.
BUDGET_CUE = re.compile(
    r"(?:(?:looking|hoping|planning|trying|wanting|want|like)\s+to\s+(?:spend|pay|get\s+it\s+for)|\bbudget\b|"
    r"\bstay\s+(?:under|below)|\bno\s+more\s+than|\bnot\s+(?:be\s+able\s+to\s+)?(?:pay|spend)\s+more\s+than|"
    r"\bmost\s+i\s+can\s+(?:pay|spend|do)|\bcloser\s+to|\bmore\s+like|\bsomewhere\s+(?:around|near))", re.I)
BUDGET_NOT = re.compile(r"(?:\bcan'?t\b|\bcannot\b|\bnot\s+in\s+my\b|\bout\s+of\b|\bover\s+my\b|\bunable\s+to\b|\bnot\s+be\s+able\b|\bdon'?t\b|\bdo\s+not\b|\blisted\b|\bpaid\b|\bworth\b)", re.I)


# "I have $200 cash on me", "I can afford 1500", "stay within 2000": what the buyer can pay, said as their offer.
CAPACITY_BEFORE = re.compile(r"(?:\bhave\s+(?:about\s+|around\s+|only\s+|just\s+)?|\bafford\b(?:(?!more\s+than)[^.?!;]){0,30}?|\bbudgeted\s+(?:about\s+|around\s+)?|"
                             r"\blimit\b[^.?!;]{0,40}?\bwas\s+|\bwithin\s+(?:about\s+|around\s+)?|"
                             r"\b(?:stay|keep\s+it)\s+(?:within|under|below)\s+(?:about\s+|around\s+)?)$", re.I)
CAPACITY_AFTER = re.compile(r"\s*(?:dollars\s+)?(?:in\s+)?(?:cash|on\s+me|to\s+spend|in\s+my\s+(?:pocket|wallet))|"
                            r"\s*(?:dollars?\s+)?(?:\w+\s+)?budget\b", re.I)


def budget_price(row):
    """The amount (and its span) when a buyer's none row is really a stated target price, else None."""
    if row["role"] != "buyer":
        return None
    listing = float(row["listing_price"]) if row["listing_price"] else None
    ms = [m for m in extract(row["text"], listing) if m.kind == "absolute" and m.plausible is not False]
    if len(ms) != 1:
        return None
    m = ms[0]
    a = max((x.end() for x in re.finditer(r"[.?!;]\s", row["text"][:m.start])), default=0)
    before = row["text"][a:m.start]
    after = row["text"][m.end:m.end + 30]
    cue = (BUDGET_CUE.search(before[-60:]) or CAPACITY_BEFORE.search(before) or CAPACITY_AFTER.match(after))
    if cue and not BUDGET_NOT.search(before[-40:]):
        return m.value, (m.start, m.end)
    return None


def value_spans(row, price):
    """JSON list of the [start, end] spans in the text that read as `price` ("[]" for none)."""
    if price == "none":
        return "[]"
    listing = float(row["listing_price"]) if row["listing_price"] else None
    return json.dumps([list(sp) for sp in sorted({(c.start, c.end) for c in explain(row["text"], listing) if c.value == price})])


def multi_rows(rows):
    for r in rows:
        g = r["gold_price"].strip()
        if g and g != "none" and len({p.strip() for p in g.split("|")}) > 1:
            yield r


def key(split, r):
    return f"{split}:{r['dialogue']}:{r['turn']}"


ORIGINAL = Path("artifacts/gold_original")   # the gold_price / gold_note you validated, before any one-price change


def cmd_archive(args):
    """Copy the original gold_price and gold_note of every split to artifacts/gold_original/. Never overwrites."""
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        target = ORIGINAL / f"{split}.csv"
        if target.exists():
            print(split, "archive already exists, kept:", target)
            continue
        _, rows = read(Path(args.dir) / f"{split}_turns.csv")
        with target.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["split", "dialogue", "turn", "gold_price", "gold_note"])
            w.writeheader()
            w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
        print(split, "archived", len(rows), "rows ->", target)


def originals(split, rows):
    """Put the archived original gold_price and gold_note back onto the rows (in memory)."""
    target = ORIGINAL / f"{split}.csv"
    if not target.exists():
        sys.exit(f"no archive for {split}; run: python -m scripts.gold_offer archive (before the first apply)")
    _, archived = read(target)
    by = {(a["dialogue"], a["turn"]): a for a in archived}
    for r in rows:
        a = by[(r["dialogue"], r["turn"])]
        r["gold_price"], r["gold_note"] = a["gold_price"], a["gold_note"]
    return rows


def cmd_propose(args):
    WORK.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        _, rows = read(Path(args.dir) / f"{split}_turns.csv")
        out = []
        for r in multi_rows(originals(split, rows)):
            price, margin, why = propose(r)
            out.append({"id": key(split, r), "role": r["role"], "gold": r["gold_price"], "text": r["text"],
                        "proposal": price, "margin": round(margin, 2), "why": why,
                        "alt": bool(RANGE_OR_ALT.search(r["text"]))})
        (WORK / f"{split}_proposals.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        sure = sum(1 for o in out if o["proposal"] and o["margin"] >= args.sure and not o["alt"])
        print(f"{split}: {len(out)} multi-price rows; {sure} confident (margin >= {args.sure}, no alternatives wording)")


def load_decisions(path):
    decisions = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split(";;")]
            if len(parts) >= 2 and parts[0] and not parts[0].startswith("#"):
                decisions[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return decisions


def located(row, price):
    """gold_spans cell: "[]" for none, the JSON spans of the price, "" when the price is not written in the text."""
    if price == "none":
        return "[]"
    spans = value_spans(row, price)
    return "" if spans == "[]" else spans


def cmd_apply(args):
    """Write the final one-price labels: gold_price (one price, none, or empty+'uncertain' note when the speaker
    proposes several), gold_note, and for train/validation gold_spans. Everything is derived from the archive."""
    decisions = load_decisions(args.decisions)
    for split in args.splits:
        path = Path(args.dir) / f"{split}_turns.csv"
        fields, rows = read(path)
        originals(split, rows)
        proposals = {}
        pf = WORK / f"{split}_proposals.json"
        if pf.exists():
            proposals = {o["id"]: o for o in json.loads(pf.read_text(encoding="utf-8"))}
        counts = {"copied (one price or none)": 0, "chose one of several": 0, "chosen by scorer": 0,
                  "excluded (several proposed)": 0, "budget rule: none -> price": 0, "undecided": 0}
        for r in rows:
            g, note = r["gold_price"].strip(), r["gold_note"]
            for old in ("gold_offer", "gold_offer_spans", "gold_offer_note"):
                r.pop(old, None)                               # columns of the earlier layout
            r["gold_spans"] = ""
            if not g:
                continue                                       # unread, or already uncertain: unchanged
            if g == "none" or len({p.strip() for p in g.split("|")}) == 1:
                hit = budget_price(r) if (g == "none" and args.budget_rule) else None
                if hit:
                    r["gold_price"] = hit[0]
                    r["gold_note"] = "buyer's stated amount counted as their proposal (rule); was none"
                    r["gold_spans"] = json.dumps([list(hit[1])])
                    counts["budget rule: none -> price"] += 1
                else:
                    r["gold_price"] = g.split("|")[0].strip()
                    r["gold_spans"] = located(r, r["gold_price"])
                    counts["copied (one price or none)"] += 1
                continue
            k = key(split, r)
            if k in decisions:
                price, why = decisions[k]
                how = "chose one of several"
            elif k in proposals and proposals[k]["proposal"] and proposals[k]["margin"] >= args.sure and not proposals[k]["alt"]:
                price, why, how = proposals[k]["proposal"], "scorer", "chosen by scorer"
            else:
                r["gold_price"], r["gold_note"] = "", f"uncertain: several prices ({g}); not decided"
                counts["undecided"] += 1
                continue
            if price == "multi":
                r["gold_price"], r["gold_note"] = "", f"uncertain: speaker proposes several prices ({g}) {why}".rstrip()
                counts["excluded (several proposed)"] += 1
                continue
            r["gold_price"] = price
            r["gold_note"] = f"one price: {price} chosen from {g}" + (f" ({why})" if why else "")
            r["gold_spans"] = located(r, price)
            counts[how] += 1
        for r in rows:
            # A typed amount that is not what the speaker meant cannot be read from the text: leave the row out of
            # training and scoring (empty gold_price + "uncertain:" note), keeping the intended price in the note.
            typo_none = r["gold_price"] == "none" and "typo" in r["gold_note"].lower()
            unlocated = r["gold_price"] not in ("", "none") and not r["gold_spans"]
            if typo_none or unlocated:
                intended = "" if typo_none else f"; intended {r['gold_price']}"
                r["gold_note"] = f"uncertain: amount typed wrongly, not readable from the text{intended}. {r['gold_note']}".strip()
                r["gold_price"], r["gold_spans"] = "", ""
                counts["excluded (typo)"] = counts.get("excluded (typo)", 0) + 1
        fields = [c for c in fields if c not in ("gold_offer", "gold_offer_spans", "gold_offer_note", "gold_spans")]
        if split != "test":
            fields.append("gold_spans")     # where the price is in the text, for training; test never gets spans
        else:
            for r in rows:
                r.pop("gold_spans", None)
        out_dir = Path(args.out or args.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write(out_dir / f"{split}_turns.csv", fields, rows)
        print(split, counts)


def cmd_status(args):
    for split in args.splits:
        _, rows = read(Path(args.dir) / f"{split}_turns.csv")
        labelled = [r for r in rows if r["gold_price"].strip()]
        excluded = [r for r in rows if not r["gold_price"].strip() and r["gold_note"].startswith("uncertain")]
        out = {"labelled": len(labelled), "none": sum(r["gold_price"] == "none" for r in labelled),
               "excluded (uncertain)": len(excluded)}
        if "gold_spans" in rows[0]:
            out["price not located in text"] = sum(r["gold_price"] not in ("", "none") and not r["gold_spans"] for r in rows)
        print(split, out)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["archive", "propose", "apply", "status"])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--out", help="write the converted CSVs here instead of over --dir (e.g. while a file is open in Excel)")
    ap.add_argument("--decisions", default=str(WORK / "decisions.txt"))
    ap.add_argument("--no-budget-rule", dest="budget_rule", action="store_false",
                    help="keep buyer budget/target statements labelled none where the gold says none")
    ap.add_argument("--sure", type=float, default=2.0, help="minimum score margin to accept a proposal without review")
    args = ap.parse_args(argv)
    {"archive": cmd_archive, "propose": cmd_propose, "apply": cmd_apply, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
