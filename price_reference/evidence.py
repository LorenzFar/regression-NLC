"""Shared local text evidence; neither the ledger nor these helpers call resolve."""
import re

from .extractor import extract
from .types import PriceCandidate, ResolverMention, mention_from_price_mention

OFFER_WORDING = re.compile(
    r"\b(?:how about|what about|(?:would|will|could|can|do) you (?:take|accept|consider|do|go)|"
    r"i (?:can|could|will|would) (?:do|go|pay|offer|take|accept)|i['’]ll (?:do|go|pay|take|accept)|"
    r"meet in the middle|i offer|my (?:price|offer) is|(?:my )?lowest|asking|firm|"
    r"sell(?:ing)? (?:it )?for|listed for|take it for|make it|instead|it['’]s free|it is free|"
    r"free to a good home|is my max(?:imum)?|works for me|(?:would|could)\s+\$?\d[\d,.]*\s+work|"
    r"is\s+\$?\d[\d,.]*\s+(?:ok|okay|acceptable))\b", re.I)
POINTER = re.compile(r"\b(?:same (?:as before|price)|that (?:price|works|offer)|"
                     r"your (?:last |previous )?offer|my (?:last |previous )?offer|"
                     r"ok(?:ay)?[, ]+deal|it(?:['’]s| is) a deal|we have a deal|you have a deal|"
                     r"no deal)\b|^\s*deal[!.]?\s*(?:if\b|$)", re.I)
ACCEPT_WORDING = re.compile(r"\b(?:accept|agree|that works|works for me|sold)\b|"
                            r"\b(?:ok(?:ay)?[, ]+deal|it(?:['’]s| is) a deal)\b|^\s*deal\b", re.I)
BUDGET = re.compile(r"\b(?:i (?:only )?have|my budget|can (?:only )?afford|"
                    r"(?:my )?budget (?:is|of)|looking to spend)\b", re.I)
REFUSAL_BEFORE = re.compile(r"\b(?:can(?:not|'t|’t) (?:do|pay|take|accept|afford)|"
                            r"won['’]t (?:do|pay|take|accept)|will not (?:do|pay|take|accept)|"
                            r"(?:don['’]t|do not) (?:accept|want|pay|take|do)|not interested (?:at|in)|"
                            r"not (?:accepting|taking|paying)|no deal(?: (?:at|for|on))?|"
                            r"no way(?: i can (?:do|pay|take|accept))?|refuse|reject)\b", re.I)
REFUSAL_AFTER = re.compile(r"^\s*(?:is |would be |seems )?(?:too (?:much|low|high)|"
                           r"(?:is )?not (?:enough|acceptable))\b", re.I)
CONDITION = re.compile(r"\b(?:only if|as long as|provided(?: that)?|if|when|unless)\b", re.I)
REVISION = re.compile(r"\b(?:oops|i meant|i ment|correction)\b", re.I)
_BREAK = re.compile(r"[!?;,]|\.(?=\s|$)|\b(?:but|however|whereas)\b", re.I)
CLARIFICATION = re.compile(
    r"\b(?:could|can|would) you clarify\b|\bwhat (?:do )?you mean\b|"
    r"\b(?:are you|which offer are you) referring to\b|\bdid you say\b|\bis that\s+\$?\d", re.I)
NEW_PROPOSAL = re.compile(
    r"\b(?:how about|what about|(?:would|will|could|can|do) you (?:take|accept|consider|do|go)|"
    r"i (?:(?:can|could|will|would) )?(?:offer|do|go|pay|take|accept)|i['’]ll (?:do|go|pay|take|accept)|my offer is|is my max(?:imum)?)\b", re.I)
PRICE_QUOTE = re.compile(
    r"\b(?:is (?:that|your|the) (?:asking )?price|you (?:said|want|offered)|"
    r"you(?:['’]re| are) asking|your (?:asking )?price)\b", re.I)
FLOOR = re.compile(
    r"\b(?:(?:can(?:not|['’]t)|won['’]t|will not|don['’]t|do not|no way(?: i)? can)\s+"
    r"(?:go|do|take|accept|sell|let it go for)(?: (?:it|anything|any(?: amount)?))?\s+"
    r"(?:below|under|less than|(?:any )?lower than)|"
    r"not (?:taking|accepting|going|selling)(?: (?:it|anything|any(?: amount)?))?\s+"
    r"(?:below|under|less than|(?:any )?lower than)|"
    r"(?:no|nothing) less than)\s*\$?\s*$", re.I)
COMPARISON_BOUND = re.compile(
    r"\b(?:(?:any )?(?:higher|better|more|lower|less|cheaper) than|above|over|below|under|"
    r"(?:up|down) from)\s*\$?\s*$", re.I)


def collect_mentions(text, listing_price=None):
    mentions = [mention_from_price_mention(m) for m in extract(text, listing_price)]
    for match in re.finditer(r"\bfree\b", text, re.I):
        before, after = text[:match.start()], text[match.end():]
        # Require item-price wording and clause-final free (or an explicit good-home offer).
        # "free of scratches", free shipping/parking/etc. must not manufacture a zero.
        home = re.match(r"\s+to a good home\b", after, re.I)
        terminal = re.match(r"\s*(?:$|[,.!?;]|(?:if|when|provided|as long as)\b)", after, re.I)
        item = re.search(r"(?:\bit(?:['’]s| is)|\b(?:give|take|have)(?: it)?\s+for|"
                         r"\b(?:sell|selling|offer)(?: it)?\s+for)\s+$", before, re.I)
        service = re.search(r"\b(?:deliver|delivery|ship|shipping|parking|pickup|deposit|fee)\b[^.!?;,]*$", before, re.I)
        if (home or terminal and (item or not before.strip())) and not service:
            mentions.append(ResolverMention("0.00", surface=match[0], start=match.start(), end=match.end()))
    return sorted(mentions, key=lambda m: m.start)


def clause(text, mention):
    if mention.start is None:
        return "", "", text
    before_breaks = list(_BREAK.finditer(text[:mention.start]))
    lo = before_breaks[-1].end() if before_breaks else 0
    following = _BREAK.search(text, mention.end)
    hi = following.start() if following else len(text)
    return text[lo:mention.start], text[mention.end:hi], text[lo:hi]


def floor_price(text, mention, speaker="seller"):
    return speaker.lower() == "seller" and mention.start is not None and bool(FLOOR.search(clause(text, mention)[0]))


def later_cancellation(text, mention):
    """Explicit retraction/replacement in the immediately following sentence.

    A leading bare 'No' is not enough to cancel an amount; it may reject someone
    else's price. Only a referential negation or explicit replacement is handled.
    """
    boundary = re.search(r"[!?;]|\.(?=\s|$)", text[mention.end:])
    if boundary is None:
        return None
    reply = text[mention.end + boundary.end():]
    if re.match(r"\s*(?:actually[, ]+)?(?:no[, ]+)?(?:it|that)\s+"
                r"(?:(?:isn['’]t|is not)(?: (?:ok|okay|acceptable|right|fair|a deal))?|"
                r"(?:won['’]t|will not|doesn['’]t|does not) work(?: for me)?)\s*(?:[,.!?;]|$)", reply, re.I):
        return "withdrawn"
    if re.match(r"\s*(?:actually[, ]+)?no[, ]+(?:make it|how about|instead|"
                r"i (?:can|could|will) (?:do|go|pay)|i['’]ll (?:do|go|pay)|\$?\d[\d,.]*)\b", reply, re.I):
        return "superseded"
    return None


def demotion(text, mention, speaker="seller"):
    if mention.kind == "reference":
        return "reference"
    if mention.start is None:
        return None  # no fabricated position; the resolver restricts unlocated inputs
    before, after, local = clause(text, mention)
    cancelled = later_cancellation(text, mention)
    if cancelled:
        return cancelled
    # This buyer question checks a seller's conditional price; it is not an offer.
    if speaker.lower() == "buyer" and re.fullmatch(
        r"\s*Would you accept \$?\d[\d,.]* (?:without that condition|if I collect on another day)\s*", local, re.I
    ):
        return "quoted_price"
    if (CLARIFICATION.search(local) or PRICE_QUOTE.search(local)) and not NEW_PROPOSAL.search(local):
        return "quoted_price"
    floor = floor_price(text, mention, speaker)
    if not floor and COMPARISON_BOUND.search(before):
        return "comparison_bound"
    if not floor and (REFUSAL_BEFORE.search(before) or REFUSAL_AFTER.search(after)):
        return "refusal"
    if BUDGET.search(before):
        # Bare "I have" is a buyer budget, but seller "I have it listed"/"I have 500
        # firm" is an asking price. Strong budget wording remains context for either side.
        strong = re.search(r"\b(?:i (?:only|just) have|my budget|afford|looking to spend)\b", before, re.I)
        asking = re.search(r"\b(?:listed|asking)\b", before, re.I) or re.match(r"\s*firm\b", after, re.I)
        if strong or speaker.lower() == "buyer" and not asking:
            return "budget"
    # Fees and deposits remain context, including a clearly labelled amount after one.
    if (re.search(r"\b(?:deposit|fee|shipping|delivery charge)\s*(?:(?:is|of|at)\s*)?$", before, re.I)
            and not re.search(r"\bfree shipping at\s*$", before, re.I)):
        return "fee"
    if re.match(r"\s*(?:(?:security |pet |delivery |shipping )?(?:deposit|fee)|for (?:shipping|delivery))\b", after, re.I):
        return "fee"
    return None


def _material_condition(condition):
    return not (re.match(r"(?:if|when)\s+(?:you(?:['’]re| are)?\s+interested|interested|new\b)", condition, re.I)
                or re.match(r"when\s+(?:i |we |it was|new\b)", condition, re.I))


def conditions(text, mention=None):
    """Price-attached conditions, not conversational 'if interested' or past-tense when."""
    local = clause(text, mention)[2] if mention is not None else text
    found = []
    for match in re.finditer(r"\b(?:only if|as long as|provided(?: that)?|if|when|unless)\b[^,;.!?]*", local, re.I):
        condition = match[0].strip()
        if not _material_condition(condition):
            continue
        # In explicit offers, conditions after the amount are attached to that price.
        # Leading conditions are retained only when a price is in the same clause.
        found.append(condition)
    if mention is not None and mention.start is not None:
        # A comma separates a leading condition from its consequence, not its meaning.
        leading = re.search(
            r"(?:^|[.!?;]\s*)((?:only if|as long as|provided(?: that)?|if|when|unless)\b[^,;.!?]+),\s*[^,;.!?]*$",
            text[:mention.start], re.I)
        trailing = re.match(
            r"\s*,\s*((?:only if|as long as|provided(?: that)?|if|when|unless)\b[^,;.!?]+)",
            text[mention.end:], re.I)
        for match in (leading, trailing):
            if match and _material_condition(match[1].strip()):
                found.append(match[1].strip())
    return tuple(dict.fromkeys(found))


def candidate(text, mention):
    span = None if mention.start is None else (mention.start, mention.end)
    local = clause(text, mention)[2]
    attached = conditions(text, mention)
    tail = attached[0] if attached else None
    role = "conditional" if tail else "primary"
    if re.search(r"\b(?:otherwise|or else|else)\b", local, re.I):
        role = "alternative"
    result = PriceCandidate(mention.value, span, role, tail)
    result.validate(text)
    return result


def multiple_reason(text):
    if REVISION.search(text):
        return "revision"
    if CONDITION.search(text) or re.search(r"\b(?:otherwise|or|else)\b", text, re.I):
        return "alternatives"
    return "multiple_prices"
