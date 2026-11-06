"""Rule-based price extractor: every money amount mentioned in one message.

Act-independent and history-free. Input is the raw message and the listing price;
deciding which mention is the offer, and what it refers to, belongs to the resolver.

Rules, in order:
1. Find digit amounts ("$1,200", "9.99", "10k", "12 hundred", "13 000") and
   number-word amounts ("fifty five", "twelve hundred", "five fifty").
2. Reject numbers that are not money: attached to letters ("s7", "1080p", "2br"),
   times ("5:30", "5 pm"), percentages, dimensions ("6x4"), phone numbers,
   counted units ("6 months", "2 bedrooms", "50,000 miles"), model numbers
   ("iPhone 7"), and calendar years ("a 2012 Honda").
3. Number words need a price cue, because most are not prices ("one of", "two years").
4. Mark each kept amount as absolute or a delta ("50 less"), and whether it is
   plausible for the listing price.
"""
from decimal import Decimal, InvalidOperation
import re

from .types import Candidate, PriceMention

UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
SCALES = {"hundred": 100, "thousand": 1000, "grand": 1000}

# Words directly after a number that make it a count, not money.
NONPRICE_UNITS = {
    "year", "years", "yr", "yrs", "month", "months", "mo", "mos", "week", "weeks",
    "day", "days", "hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins",
    "second", "seconds", "am", "pm", "a.m", "p.m", "o'clock",
    "mile", "miles", "mi", "km", "mph", "mpg", "kilometers",
    "inch", "inches", "ft", "feet", "foot", "sq", "sqft", "square", "cm", "mm", "meters",
    "lb", "lbs", "pound", "pounds", "kg", "oz",
    "gb", "tb", "mb", "ghz", "mhz", "hz", "mp", "megapixel", "megapixels", "hp", "cc",
    "watt", "watts", "w", "v", "volt", "volts",
    "bedroom", "bedrooms", "bed", "beds", "bd", "br", "bath", "baths", "bathroom",
    "bathrooms", "ba", "story", "stories", "floor", "floors", "car", "cars", "garage",
    "people", "person", "persons", "seater", "seats", "seat", "door", "doors",
    "speed", "speeds", "gear", "gears", "cylinder", "cylinders", "owner", "owners",
    "time", "times", "piece", "pieces", "pcs", "pc", "item", "items", "chair", "chairs",
    "drawer", "drawers", "shelf", "shelves", "set", "sets", "pair", "pairs",
    "%", "percent", "degree", "degrees", "star", "stars", "kids", "children", "of",
    "gig", "gigs", "gigabyte", "gigabytes", "d", "h", "l",
}
# "14k miles" is a distance, though a "k" suffix otherwise means money.
DISTANCE_UNITS = {"mile", "miles", "mi", "km", "kilometers", "kilometres"}
# Words directly before a number that make it a model or identifier.
MODEL_WORDS = {
    "iphone", "galaxy", "note", "pixel", "ipad", "macbook", "xbox", "ps", "playstation",
    "windows", "model", "version", "gen", "generation", "series", "number", "no", "size",
    "room", "apt", "apartment", "unit", "suite", "route", "highway", "hwy", "interstate",
    "level", "grade", "class", "core", "chapter", "page", "step", "phase", "zone", "area", "wd",
}
# Words that may run straight into a price when the space is missing ("how about9000").
GLUED_PRICE_CUES = {"about", "around", "for", "than", "is", "at", "do", "to", "take", "be", "only", "just"}
# Two words before a bare 4-digit number that show it is a price ("the price of 2000").
YEAR_PRICE_WORDS = {
    "price", "offer", "accept", "firm", "lowest", "final", "take", "rent", "asking", "pay", "best",
}
# A bare 4-digit year after these words is a date ("bought in 2015"), not money.
YEAR_PREV = {
    "in", "since", "from", "of", "model", "year", "circa", "bought", "built", "made",
    "a", "an", "the", "my", "this", "that", "our", "his", "her", "its", "late", "early",
    "mid", "until", "till", "before", "after", "by",
}
STREET_WORDS = {
    "st", "street", "ave", "avenue", "rd", "road", "dr", "drive", "blvd", "boulevard",
    "ln", "lane", "way", "ct", "court", "pl", "place", "pkwy", "parkway", "cir", "circle",
    "terrace", "ter", "hwy", "highway",
}
CURRENCY_AFTER = {"dollar", "dollars", "buck", "bucks", "usd"}
DELTA_AFTER = {"off", "less", "more", "lower", "higher", "cheaper", "discount", "extra", "down", "up"}
DELTA_BEFORE = {"another", "extra", "additional"}
# A number word needs one of these just before it to count as a price ("do fifty").
PRICE_CUES = {
    "for", "do", "take", "offer", "pay", "at", "accept", "around", "about", "you", "is",
    "be", "to", "only", "say", "like", "than", "go", "get", "me", "sell", "under", "over",
    "least", "max", "maybe", "just", "was", "asking", "firm", "it", "give", "price",
}

# A reduction ("take 50 off", "reduce the rent by 200") is a delta, not a price.
_DELTA_BEFORE_RE = re.compile(
    r"\b(?:reduc\w*|lower\w*|knock\w*|cut\w*|drop\w*|discount\w*|deduct\w*|take\s+off|took\s+off|"
    r"come\s+down|go\s+down)\b(?:\s+(?:it|that|this|the\s+(?:price|rent)))?(?:\s+\w+){0,3}?\s+(?:by|off)\s*$|"
    r"\b(?:lower|reduce|knock|cut|drop)\s+(?:it|that|the\s+price|the\s+rent)\s*$", re.I)
# A price quoted for comparison (what it cost, retails for, is worth) is not an asking or offered price.
# Refused prices ("1700 is too much") and budgets ("I only have 140") stay absolute: the gold labels count them.
_ROLE_CUES = (
    # a price quoted for comparison: what it cost, retails for, is worth
    ("reference",
     re.compile(r"(?:\b(?:i|we|they|you)\s+(?:also\s+)?paid|\bpaid|\bcost(?:s)?\s+(?:me|us|you|new)|\bworth|\bvalued?\s+(?:at|of)|\bretail(?:s|ed)?\s+(?:for|at)|"
                r"\bmsrp|\bblue\s?book|\bkbb|\boriginally|\binvested|\bspent)(?:\s+(?:almost|about|around|over|nearly|like|is|says|at|for|of|value|price))*\s*$|\balmost\s+$", re.I),
     re.compile(r"\s*(?:\w+\s+)?(?:new|brand\s+new|when\s+new|originally|at\s+(?:the\s+)?store|retail|at\s+walmart|at\s+amazon|online|elsewhere|msrp)\b", re.I)),
)
_CLAUSE_END_RE = re.compile(r"[.?!;](?:\s|$)")

_DIGITS_RE = re.compile(
    r"(?P<cur>\$\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s?k(?![a-z])|\s*(?:hundred|thousand|grand)\b)?", re.I)
_NUMBER_WORD = "|".join(sorted([*UNITS, *TENS, *SCALES], key=len, reverse=True))
_WORDS_RE = re.compile(
    rf"\b(?:a\s+(?=(?:hundred|thousand|grand)\b))?(?:{_NUMBER_WORD})"
    rf"(?:(?:\s+|-)(?:and\s+)?(?:{_NUMBER_WORD}))*\b", re.I)
_PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_RANGE_TAIL_RE = re.compile(r"\s?(?:-|to)\s?\$?\d[\d,.]*\s?k?(?![\d])", re.I)
_NEXT_WORD_RE = re.compile(r"\+?\s*(?P<w>[ap]\.m\b|o'clock|[a-z%][a-z'%]*)", re.I)
_PREV_WORD_RE = re.compile(r"(?P<w>[a-z#]+)\.?\s*$", re.I)


def extract(text: str, listing_price: float | None = None) -> list[PriceMention]:
    """Return every money amount in ``text``, in order of position."""
    return [c.mention for c in explain(text, listing_price) if c.mention is not None]


def explain(text: str, listing_price: float | None = None) -> list[Candidate]:
    """Return every number-like span with the value read and why it was rejected, if it was."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    listing = Decimal(str(listing_price)) if listing_price is not None and listing_price > 0 else None
    phones = [m.span() for m in _PHONE_RE.finditer(text)]
    spans = _merge_space_thousands(text, list(_digit_spans(text)), listing)
    taken = [(s["start"], s["end"]) for s in spans]
    spans += [s for s in _word_spans(text) if not _overlaps(s, taken)]
    spans.sort(key=lambda s: s["start"])
    return [_judge(text, s, listing, phones) for s in spans]


def normalise_amount(surface: str) -> str | None:
    """Exact text of ONE money amount -> canonical decimal string, or None if unreadable.

    Always two decimal places: "$1.2k" -> "1200.00", "13 000" -> "13000.00",
    "fifty five" -> "55.00", "9.5" -> "9.50". Shared by the extractor and the gold labels
    so both agree. Idempotent: normalise_amount("55.00") == "55.00".
    Only reads the amount; whether it is a price in context is decided elsewhere.
    """
    if not isinstance(surface, str):
        raise TypeError("surface must be a string")
    s = surface.strip().strip(".,;:!?").strip()
    s = re.sub(r"\s*\b(?:dollars?|bucks?|usd)$", "", s, flags=re.I).strip()
    if re.fullmatch(r"\d[\d,.]*\$", s):   # "10$": dollar sign after the number
        s = s[:-1]
    if re.fullmatch(r"\$?\s?\d{1,3}(?: \d{3})+(?:\.\d+)?", s):   # "13 000"
        return _format(Decimal(re.sub(r"[\s$]", "", s)))
    # Spellings seen in gold labels only: the extractor does not look for these in free text.
    if re.fullmatch(r"hundo|a\s+cool\s+hundred", s, re.I):   # "hundo bucks", "a cool hundred"
        return _format(Decimal(100))
    large = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)\s+large", s, re.I)   # "9 large" = 9000
    if large:
        return _format(Decimal(large[1]) * 1000)
    if re.fullmatch(r"\$\s?\d{1,3}, \d{3}", s):   # "$10, 000": stray space after the comma
        return _format(Decimal(re.sub(r"\D", "", s)))
    for spans in (list(_digit_spans(s)), list(_word_spans(s))):
        if len(spans) == 1 and spans[0]["start"] == 0 and spans[0]["end"] == len(s):
            return _format(spans[0]["value"])
    return None


def _digit_spans(text):
    for m in _DIGITS_RE.finditer(text):
        num, suffix = m["num"], (m["suffix"] or "").strip().lower()
        try:
            value = Decimal(num.replace(",", ""))
        except InvalidOperation:
            continue
        mult = {"k": 1000, "hundred": 100, "thousand": 1000, "grand": 1000}.get(suffix, 1)
        yield {"start": m.start(), "end": m.end(), "value": value * mult, "mult": mult, "digits": num,
               "currency": bool(m["cur"]) or suffix in ("k", "grand"), "words": False,
               "scaled": bool(suffix), "cues_needed": False}


def _merge_space_thousands(text, spans, listing):
    """Join "13 000" into 13000 when only the joined amount fits the listing price."""
    if listing is None:
        return spans
    out, i = [], 0
    while i < len(spans):
        a = spans[i]
        b = spans[i + 1] if i + 1 < len(spans) else None
        if (b and text[a["end"]:b["start"]] == " " and re.fullmatch(r"\d{1,3}", a["digits"])
                and re.fullmatch(r"\d{3}", b["digits"]) and not a["scaled"]
                and text[b["start"]] != "$"):
            joined = Decimal(a["digits"] + b["digits"]) * b["mult"]
            if _plausible(joined, listing, "absolute") and not _plausible(a["value"], listing, "absolute"):
                out.append({**a, "end": b["end"], "value": joined, "digits": a["digits"] + b["digits"],
                            "currency": a["currency"] or b["currency"], "scaled": b["scaled"]})
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def _word_spans(text):
    for m in _WORDS_RE.finditer(text):
        tokens = [(t.group().lower(), m.start() + t.start(), m.start() + t.end())
                  for t in re.finditer(r"[a-z]+", m.group(), re.I)]
        while tokens:
            end, value = _longest_number(tokens)
            if end == 0:
                tokens = tokens[1:]
                continue
            used = tokens[:end]
            words = [t[0] for t in used]
            yield {"start": used[0][1], "end": used[-1][2], "value": Decimal(value),
                   "digits": None, "currency": "grand" in words, "words": True,
                   "scaled": any(w in SCALES for w in words),
                   "cues_needed": len([w for w in words if w not in ("a", "and")]) == 1}
            tokens = tokens[end:]


def _longest_number(tokens):
    for end in range(len(tokens), 0, -1):
        value = _words_value([t[0] for t in tokens[:end]])
        if value is not None:
            return end, value
    return 0, None


def _words_value(words):
    """Parse "fifty five", "a hundred and fifty", "twelve hundred", "five fifty" (=550)."""
    if not words or words[-1] in ("and", "a"):
        return None
    total, current, prev = 0, 0, None
    for w in words:
        if w == "and":
            if prev != "scale":
                return None
        elif w == "a":
            if prev is not None:
                return None
            current = 1
        elif w in SCALES:
            if prev == "scale" and SCALES[w] == 100:
                return None
            if SCALES[w] == 100:
                current = (current or 1) * 100
            else:
                total, current = total + (current or 1) * SCALES[w], 0
            w = "scale"
        elif w in TENS:
            if prev in (None, "scale", "and", "a") and current % 100 == 0:
                current += TENS[w]
            elif prev == "unit" and current < 100:   # "five fifty" spoken price
                current = current * 100 + TENS[w]
            else:
                return None
            w = "tens"
        else:
            v = UNITS[w]
            if prev in (None, "scale", "and", "a") and current % 100 == 0:
                current += v
            elif prev == "tens" and 1 <= v <= 9:
                current += v
            elif prev == "unit" and current < 100 and v >= 10:   # "nine fifteen"
                current = current * 100 + v
            else:
                return None
            w = "unit"
        prev = w
    return total + current


def _judge(text, span, listing, phones):
    start, end, value = span["start"], span["end"], span["value"]
    surface = text[start:end]
    shown = _format(value)
    reason = _reject_reason(text, span, phones)
    if reason is None and shown is None:
        reason = "more than two decimal places"
    if reason is not None:
        return Candidate(surface, start, end, shown, reason)

    after = _word_after(text, end)
    before = _word_before(text, start)
    currency = span["currency"] or after in CURRENCY_AFTER
    if span["words"] and span["cues_needed"] and not currency and not (
            before in PRICE_CUES and value >= 5):
        return Candidate(surface, start, end, shown, "number word without a price cue")
    after_currency = _word_after(text, end, skip=CURRENCY_AFTER)
    clause_before = _CLAUSE_END_RE.split(text[:start])[-1]
    if after_currency in DELTA_AFTER or before in DELTA_BEFORE or _DELTA_BEFORE_RE.search(clause_before):
        kind = "delta"
    else:
        kind = "absolute"
    plausible = None if listing is None else _plausible(value, listing, kind)
    if kind == "absolute":
        kind = _role(clause_before, _CLAUSE_END_RE.split(text[end:], 1)[0])
    mention = PriceMention(shown, start, end, surface, kind, currency, plausible)
    return Candidate(surface, start, end, shown, None, mention)


def _role(clause_before, clause_after):
    """"reference" when the words around the amount show it is a comparison price, else "absolute"."""
    for name, before_re, after_re in _ROLE_CUES:
        if before_re.search(clause_before) or after_re.match(clause_after):
            return name
    return "absolute"


def _reject_reason(text, span, phones):
    start, end = span["start"], span["end"]
    if any(a <= start < b for a, b in phones):
        return "phone number"
    prev_char = text[start - 1] if start else ""
    next_char = text[end] if end < len(text) else ""
    if text[start] == "$":
        prev_char = ""   # "accept$1500": a letter before "$" does not make it a code
    if not span["words"]:
        # Missing spaces still read as money: "how about9000", "$35and", "60bucks".
        glued_word = re.search(r"[a-z]+$", text[:start], re.I)
        cue_before = bool(glued_word) and glued_word[0].lower() in GLUED_PRICE_CUES
        money_after = text[start] == "$" or re.match(r"(?:bucks?|dollars?)\b", text[end:], re.I)
        if ((prev_char.isalpha() and not cue_before) or (prev_char and prev_char in "#@")
                or (next_char.isalpha() and not money_after)):
            return "attached to letters (model, ordinal or code)"
        if prev_char == ":" or (next_char == ":" and text[end + 1:end + 2].isdigit()):
            return "time of day"
        if re.match(r"\s?[x×*]\s?\d", text[end:], re.I) or re.search(r"\d\s?[x×*]\s?$", text[:start], re.I):
            return "dimension"
        if next_char in ('"', "”", "″") or (next_char == "'" and not re.match(r"'s\b", text[end:])):
            return "inches or feet mark"   # 27", 70'
        if prev_char == "'" and text[start - 2:start - 1].isdigit():
            return "height"   # 5'7"
        if prev_char == "/" and text[start - 2:start - 1].isdigit():
            return "fraction or date"
        if re.match(r"'s\b", text[end:]):
            return "decade"
        if next_char == "/" and text[end + 1:end + 2].isdigit():
            return "fraction or date"
        if re.match(r"\s\d+/\d+", text[end:]):
            return "whole part of a fraction"
    if span["currency"] and text[start] != "$" and _word_after(text, end) in DISTANCE_UNITS:
        return "counted unit (distance)"
    if span["currency"] or _word_after(text, end) in CURRENCY_AFTER:
        return None   # "$5", "10k", "50 bucks": money whatever follows
    tail = _RANGE_TAIL_RE.match(text, end)
    unit = _word_after(text, tail.end() if tail else end, raw=True)
    if unit in NONPRICE_UNITS:
        return f"counted unit ({unit})"
    before = _word_before(text, start)
    # "No 2100 is my best" opens a sentence; "item no 5" is an identifier.
    sentence_no = before == "no" and re.search(r"(?:^|[.!?]\s*)no\b[.,]?\s*$", text[:start], re.I)
    if before in MODEL_WORDS and not sentence_no:
        return f"model or identifier ({before})"
    # "2487 Graystone Drive": a capitalised name then a street word within three words.
    address = re.match(r"\s+(?:[A-Z][a-z]+\s+){1,3}(?P<w>[A-Za-z]+)\b", text[end:])
    # "3185 Garrity Way in": the street word itself may be capitalised.
    named = re.match(r"\s+(?:[A-Z][a-z]+\s+){1,3}", text[end:])
    if (address and address["w"].lower() in STREET_WORDS) or (
            named and any(w.lower() in STREET_WORDS for w in named.group().split()[1:])):
        return "street address"
    if (span["digits"] and re.fullmatch(r"(19|20)\d\d", span["digits"]) and not span["currency"]
            and not span["scaled"]):
        nxt = _NEXT_WORD_RE.match(text, end)
        # "I" opens a clause ("offer 2000 I can't"); it does not name a car.
        capital = nxt and nxt["w"][0].isupper() and nxt["w"] != "I"
        price_context = _word_before(text, start - len(before or "") - 1) in YEAR_PRICE_WORDS
        # "the year for the car is 2004", but not "1 year lease ... is 1950 a month".
        year_asked = re.search(r"\byear\b(?!\s+(?:lease|contract|term))[^,]*\b(?:is|was|of)\s*$",
                               _CLAUSE_END_RE.split(text[:start])[-1], re.I)
        if (before in YEAR_PREV or capital or year_asked) and not (price_context and not capital):
            return "calendar year"
    return None


def _word_after(text, pos, *, skip=(), raw=False):
    m = _NEXT_WORD_RE.match(text, pos)
    while m:
        word = m["w"].lower().rstrip(".'") if not raw else m["w"].lower().rstrip(".")
        if word.endswith("'s"):
            word = word[:-2]
        if word not in skip:
            return word
        m = _NEXT_WORD_RE.match(text, m.end())
    return None


def _word_before(text, pos):
    m = _PREV_WORD_RE.search(text[max(0, pos - 30):pos].rstrip("$ "))
    return m["w"].lower().lstrip("#") if m else None


def _plausible(value, listing, kind):
    if kind == "delta":
        return 0 < value < listing
    return listing * Decimal("0.1") <= value <= listing * 3


def _format(value):
    """Canonical amount: always two decimal places ("5.00", "9.50"); None if it needs more."""
    try:
        cents = value.quantize(Decimal("0.01"))
    except InvalidOperation:   # absurdly large
        return None
    return format(cents, "f") if cents == value else None


def _overlaps(span, taken):
    return any(span["start"] < b and a < span["end"] for a, b in taken)
