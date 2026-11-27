# Labelling guide: seller acts and prices

The finished labels are in `data/craigslist_test/labels_barry.csv` (235 rows; reviewed in `labels_review.xlsx`). The blank sheet is `data/craigslist_test/labelling_sheet.csv`.

Label every row of the blank sheet. Each row is one seller message from the CraigslistBargain **test** split, which the NLU never saw in training. Read `context` (the last three turns) before you label. For more context, look up the dialogue in `transcripts.md`.

Label without looking at any model output. Write your name in `annotator`. Use `notes` for anything uncertain.

## `gold_act`: one of six acts

| Act | Use when the seller... | Examples |
|---|---|---|
| `propose` | states a price for the first time, with no buyer offer on the table | "I'm asking 100 for it" |
| `counter` | gives a new price in reply to a buyer offer, or repeats their own price | "how about 60?", "I can't go below 90" |
| `accept` | agrees to a price that is on the table | "ok, 80 works", "deal" (right after an offer) |
| `reject` | declines an offer **without** giving a new price | "no, that's way too low" |
| `inquire` | asks a question | "when can you pick it up?" |
| `vague` | makes no clear negotiation move: answers, small talk, greetings | "Original tires", "hi there" |

Decision rules, applied in order:

1. **A new price takes priority over a rejection.** "Too low, how about 60?" is `counter`.
2. **Agreement is judged from the full context, not the wording.** Ask: is the seller saying yes to a price here? If so it is `accept`, however short ("Sure", "Great!", "deal"). This includes the seller confirming a deal the buyer just agreed to ("We have a deal" → "Great!"). With no price agreed or on the table, "ok" or "deal" is `vague`, and so is "Sure!" in reply to "thanks".
3. **Conditional acceptance is still `accept`** ("I can do 35 if you pick it up"). The condition is recorded for RQ2 in `needs_clarification` / `deal_agreed`, not in the act.
4. **A question that also offers a price** ("would you do 60?") is `counter` or `propose`, not `inquire`.
5. **Answer plus question** in one message ("Yes, are you interested?") is `inquire`.
6. **Answers to the buyer's questions** about the item are `vague`, even when they're informative.
7. **Defending the price without a new number** is `reject` when the buyer made a specific offer, and `vague` when the buyer only asked to "go lower".
8. **`propose` and `counter` always carry a price** (a `new_offer`, or a `reference` when the seller restates an offer). Without any price, use `reject`, `inquire` or `vague`. Discounts ("200 off") are not prices.
9. **Hidden offers.** Some seller messages react to an offer you can't see ("Your offer is too low"). This version of the dataset drops the offer-button turns. Label from the text, and write `hidden offer` in `notes`.

Rules 1, 4 and 6 match how the NLU's training labels were built. Rule 2 deliberately differs: training relabelled agreement as `vague` whenever the previous turn had no price, so expect the NLU to miss some confirmations.

Excluded: `cb-test-0497-t5` (the source message is truncated to "Tha").

## `gold_prices`: every price in the message

Write each price as `value:role`, separated by `;`. Leave the cell blank if the message has no price.

```
60:new_offer
200:not_offer; 150:new_offer
```

- **value** is the plain number meant: `$1.5k` → `1500`, `fifty` → `50`, `2,600` → `2600`.
- **role**:
  - `new_offer`: a price the seller offers now ("how about 60")
  - `reference`: repeats an offer already made ("you said 40, that's too low")
  - `not_offer`: a money amount that isn't an offer ("I paid 200 for it", "it lists for 300 new")
- **Not prices at all** (years, sizes, model numbers, counts like "2 bedrooms") are not written down. This is how false positives are caught.
- For a range ("60-65"), write both values with the same role.

## Agreement check

Before labelling everything, label about 30 rows with a teammate independently and compare them with Cohen's kappa (`evaluation.nlu_metrics.cohen_kappa`, matching rows by `case_id`). If kappa is below about 0.7, discuss the disagreements, clarify this guide, and relabel those rows. After that, have the teammate double-label about 20% of the sheet.
