"""Rule-based extractor on phrasings seen in CraigslistBargain train, NOT accuracy evidence."""
import pytest

from price_reference import explain, extract, normalise_amount


def values(text, listing=None):
    return [m.value for m in extract(text, listing)]


@pytest.mark.parametrize("text, listing, expected", [
    ("Can you sell it to me for $5?", 10, ["5.00"]),
    ("I think the lowest I would want to go is 8.", 10, ["8.00"]),
    ("7, and we have a deal.", 10, ["7.00"]),
    ("im on a budget so i could do 1850", 2000, ["1850.00"]),
    ("How about $1,900 and I'll wave the deposit.", 2100, ["1900.00"]),
    ("I could even meet you in the middle at 70.00.", 75, ["70.00"]),
    ("It is $9.99 at the store", 10, ["9.99"]),
    ("10k is a bit low", 14800, ["10000.00"]),
    ("I could do 1.2k", 1500, ["1200.00"]),
    ("We could meet in the middle and say 13 000?", 14800, ["13000.00"]),
    ("2 grand is too much", 2500, ["2000.00"]),
    ("I'd take 12 hundred", 1500, ["1200.00"]),
    ("I can do 45 bucks", 50, ["45.00"]),
    ("Rent is $1500/month", 1500, ["1500.00"]),
    ("I can do 50 in cash", 60, ["50.00"]),
    ("would you accept$1500 for one without fireplace?", 2102, ["1500.00"]),
    ("We could probably get you in around 1950", 2250, ["1950.00"]),
])
def test_digit_amounts(text, listing, expected):
    assert values(text, listing) == expected


@pytest.mark.parametrize("text, expected", [
    ("I can do fifty five", ["55.00"]),
    ("would you take five fifty?", ["550.00"]),
    ("I'd pay twelve hundred", ["1200.00"]),
    ("how about a hundred and fifty", ["150.00"]),
    ("fifty dollars is my limit", ["50.00"]),
])
def test_number_words(text, expected):
    assert values(text) == expected


@pytest.mark.parametrize("text, listing", [
    ("about 6 month....bought a new one", 13),        # parser took 6 as a price
    ("It's only 2 years old", 100),
    ("It has 2 bedrooms, 1 bath", 2100),
    ("50,000 miles on it", 9000),
    ("it's an iPhone 7 with 64gb", 600),
    ("battery is at 50%", 100),
    ("it is 6x4 feet", 100),
    ("pick up at 5:30 or 6pm", 100),
    ("call 555-123-4567", 600),
    ("it's the 2015 Honda", 9000),
    ("bought in 2012", 2500),
    ("1080p screen, 2nd owner", 300),
    ("Do you have one in a different color?", 13),
    ("one of the best", 13),
    ("2-3 days for pickup", 60),
    ("I don't want to move to 2487 Graystone Drive, in Tracy", 2349),
    ("27 1/2 inches wide", 20),
    ("it has 14k miles on it", 5000),                       # "k" is thousands of miles, not money
    ("I am 5'7\", would this fit?", 60),
    ("a table 27\" wide and 19\" tall", 150),
    ("it is 70'x24'x17'", 30),
    ("the case fits iPhone 6/6+", 8),
    ("It's on 3185 Garrity Way in Richmond", 2330),         # a capitalised street word
    ("it has 32 Gigs of memory", 300),
    ("a 2004 Honda, the year for the car is 2004", 8990),
])
def test_non_prices_are_rejected(text, listing):
    assert values(text, listing) == []


def test_every_rejection_has_a_reason():
    rejected = [c for c in explain("2 years old, iPhone 7, call 555-123-4567", 600) if c.mention is None]
    assert rejected and all(c.reject_reason for c in rejected)


def test_several_mentions_keep_order_and_exact_spans():
    text = "I see the list price is 585. I can give you $400 for it"
    mentions = extract(text, 585)
    assert [m.value for m in mentions] == ["585.00", "400.00"]
    assert [text[m.start:m.end] for m in mentions] == ["585", "$400"]
    assert all(m.surface == text[m.start:m.end] for m in mentions)


def test_delta_and_currency_flags():
    first, = extract("$20 less than that", 100)
    assert first.kind == "delta" and first.has_currency
    plain, = extract("how about 42?", 50)
    assert plain.kind == "absolute" and not plain.has_currency


@pytest.mark.parametrize("text, listing, kind", [
    ("I can reduce the rent by 200 if you sign", 1595, "delta"),
    ("I am willing to lower it ten dollars", 100, "delta"),
    ("could you take another $50 off", 1900, "delta"),
    ("I can lower it to 150", 200, "absolute"),              # "to" names the new price
    ("I paid $600 for the set two years ago", 300, "reference"),
    ("it retails for 150 but I need it gone", 140, "reference"),
    ("the kelly blue book value is $6000", 5500, "reference"),
    ("I'd pay 500 for it", 600, "absolute"),
    ("$750 is too low", 900, "absolute"),                    # a refused price is still a stated price
    ("my budget is around 245", 280, "absolute"),            # so is a budget: the gold labels count both
])
def test_mention_kind(text, listing, kind):
    assert [m.kind for m in extract(text, listing)][0] == kind


@pytest.mark.parametrize("text, listing, expected", [
    ("No 2100 is the best I can offer you", 2311, ["2100.00"]),     # sentence-opening "No", not "No. 5"
    ("I will accept the price of 1900.", 2450, ["1900.00"]),         # a year-shaped price after "price of"
    ("My last offer would be 2000 I can't lower more", 2698, ["2000.00"]),
    ("I paid $600 and now ask $300", 400, ["300.00"]),
    ("how about9000", 9900, ["9000.00"]),                            # missing space after a price cue
    ("are you willing to knock 60bucks off?", 700, []),             # money, but a delta
    ("$35and you have a deal.", 50, ["35.00"]),
    ("1 year lease and no deposit is 1950 a month", 2000, ["1950.00"]),
    ("the year for the car is 2004", 8990, []),
])
def test_offers_among_other_numbers(text, listing, expected):
    assert [m.value for m in extract(text, listing) if m.kind == "absolute"] == expected


def test_plausibility_uses_listing_price_and_is_unknown_without_it():
    assert [m.plausible for m in extract("I can do 6 or 600", 13)] == [True, False]
    assert extract("I can do 600")[0].plausible is None


def test_year_shaped_amount_is_kept_when_it_is_an_offer():
    assert values("I can do 2000 for it", 3000) == ["2000.00"]


def test_more_than_two_decimals_is_not_money():
    assert values("it weighs 1.125 kg", 50) == []
    assert values("it is 3.14159 wide", 50) == []


@pytest.mark.parametrize("surface, expected", [
    ("$5", "5.00"), ("5", "5.00"), ("$1,900", "1900.00"), ("9.99", "9.99"), ("70.00", "70.00"), ("9.50", "9.50"),
    ("9.5", "9.50"), ("55.00", "55.00"),
    ("10k", "10000.00"), ("$1.2k", "1200.00"), ("10K", "10000.00"), ("13 000", "13000.00"), ("2 grand", "2000.00"),
    ("12 hundred", "1200.00"), ("45 bucks", "45.00"), ("50 dollars", "50.00"), ("$ 5", "5.00"), ("$20.", "20.00"),
    ("10$", "10.00"), ("50$", "50.00"),
    ("fifty five", "55.00"), ("Thirteen hundred", "1300.00"), ("five fifty", "550.00"),
    ("a hundred and fifty", "150.00"), ("twenty-five", "25.00"),
    ("hundo bucks", "100.00"), ("a cool hundred", "100.00"), ("9 large", "9000.00"), ("$10, 000", "10000.00"),
    ("a thousand", "1000.00"),
    ("10, 000", None), ("2 60", None), ("large", None), ("$10, 0000", None),
    ("six months", None), ("hello", None), ("", None), ("$5 or $6", None), ("1.125", None),
])
def test_normalise_amount(surface, expected):
    assert normalise_amount(surface) == expected


def test_normaliser_agrees_with_extractor_values():
    texts = ["I could do $1.2k or 13 000, maybe fifty five", "twelve hundred is it, 70.00 even",
             "Can you do 2 grand? or $1,900", "how about five fifty"]
    for text in texts:
        for m in extract(text, 5000):
            assert normalise_amount(m.surface) == m.value


def test_text_must_be_a_string():
    with pytest.raises(TypeError):
        extract(None)
