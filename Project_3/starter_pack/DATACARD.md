# Data card: Judge the Judge (paired edits)

## Task
Each case is a **pair**: a good reply to a customer question (`original`) and an `edited` copy of it with one small
change. Some edits are **harmless** (the verdict must not change); some are **harmful** (the reply now violates the
criterion, so a good judge flips from `pass` to `fail`). Build a method that improves the baseline judge.

Judge 1 criterion: `JUDGE1_CRITERION.md` (does the reply over-promise?). Policy: `policy.md`.
Near the end a **surprise criterion** is revealed: your recipe has to build a good judge for it too, unchanged. Only
the inputs change, not your code: you get a new criterion file and a few labelled example pairs in the same format as
`dev_pairs.jsonl`, and run your judge with `--criterion` set to the new file and `--dev` set to the new examples.

**Your recipe is code.** It is run with `run_judge.py` on cases you have not seen. Your judge sees one reply at a
time (question, reply, language, criterion text, policy text), never the pair or the edit type.

## What the unseen cases contain that dev does not
- Source replies that are not in dev (split by source, so no dev reply or its translation appears there).
- **Edit types that are not in dev**, both harmless and harmful. Tuning to the dev edit types will not carry over.
- Filler sentences and openers that never occur in dev.

## Files
| File | What |
|---|---|
| `dev_pairs.jsonl` | 226 labelled dev cases for Judge 1. Use it for final numbers. |
| `dev_pairs_fast.jsonl` | A stratified subset (3 per edit type and language). Iterate on this; use `dev_pairs.jsonl` for final numbers. |
| `baseline_dev_verdicts.csv`, `baseline_report.txt` | The baseline's verdicts and report card on `dev_pairs.jsonl`, precomputed so you can start from its failures. |
| `diagnose.py` | Shows where a judge fails, with example replies: `python3 diagnose.py baseline_dev_verdicts.csv dev_pairs.jsonl` |
| `my_judge.py` | A starting point for your judge, laid out as the four recipe steps. Copy it, fill the steps in. |
| `RECIPE.md` | The one-page recipe you hand in. Fill it in as you go. |
| `JUDGE1_CRITERION.md` | The criterion text, passed to your judge as a string |
| `policy.md` | The policy, passed to your judge as a string |
| `baseline_judge.py` | The baseline: one LLM call per reply, any OpenAI-compatible endpoint. Set `JUDGE_LLM_BASE_URL`, `JUDGE_LLM_API_KEY` and `JUDGE_LLM_MODEL` (see its docstring). |
| `run_judge.py` | Runs a judge file on a pairs file, writes `case_id,original,edited` verdicts |
| `score.py` | Scores verdicts: `python3 score.py verdicts.csv dev_pairs.jsonl [--baseline baseline_dev_verdicts.csv]` |

## Schema (`dev_pairs.jsonl`)
`case_id`, `source_id` (same id = same source reply; use it for grouped cross-validation), `source_kind`
(synthetic | bitext), `intent`, `lang` (language of the edited reply), `orig_lang` (language of the original; differs
only for `translation`), `question_original`, `question`, `original`, `edited`, `edit_type`, `edit_family`
(harmless | harmful), `expected_verdict_change`, `original_label` (always `pass`), `edited_label`, `edit_note`,
`edited_unpadded` (the edited reply before length padding: the minimal edit itself), `length_padding` (the harmless
length change added to a harmful edit to keep its length close to the original's, or `null`; see below).

## Dev edit types
- harmless: `filler_padding`, `polite_opener`, `padding_removal`, `sentence_reorder`, `translation`
- harmful: `hedge_removal`, `condition_removal`, `wrong_number`, `wrong_policy_detail`

`wrong_number` here always means a number more generous than the policy (an over-promise).

## Sizes
- dev by language: de: 47, en: 38, es: 47, fr: 47, it: 47
- by family: harmful: 110, harmless: 116; by source kind: synthetic: 226
- source replies in dev: 9 (synthetic: 9)

| | en | de | fr | es | it | total |
|---|---|---|---|---|---|---|
| filler_padding | 2 | 3 | 1 | 3 | 6 | 15 |
| polite_opener | 7 | 6 | 8 | 6 | 3 | 30 |
| padding_removal | 1 | 1 | 1 | 1 | 1 | 5 |
| sentence_reorder | 6 | 6 | 6 | 6 | 6 | 30 |
| translation | 0 | 9 | 9 | 9 | 9 | 36 |
| hedge_removal | 4 | 4 | 4 | 4 | 4 | 20 |
| condition_removal | 6 | 6 | 6 | 6 | 6 | 30 |
| wrong_number | 7 | 7 | 7 | 7 | 7 | 35 |
| wrong_policy_detail | 5 | 5 | 5 | 5 | 5 | 25 |
| **total** | 38 | 47 | 47 | 47 | 47 | 226 |

## Scoring
- **catch**: share of harmful cases where the judge passes the original and fails the edited reply.
- **keep**: share of harmless cases where the judge passes both.
- **combined** (the score): harmonic mean of catch and keep, each averaged over edit types first. A judge that
  rejects everything, or accepts everything, scores 0.
- Also reported: raw stability (verdict unchanged on harmless edits), Cohen's kappa against the construction labels,
  per-language and per-edit-type results, and the improvement over the baseline.
- **Consistency**: run your judge twice on the same pairs and pass the second run with `--repeat run2.csv`. It reports
  how often the judge gave the same verdict both times. Consistent is not the same as correct: read it next to the
  score.

## Where the data comes from
- **Synthetic part**: hand-written replies to car-rental questions in EN/DE/FR/ES/IT (written by us, not machine
  translated), each with marked spans (numbers, conditions, hedges, scopes, quantifiers, policy details) so an edit
  changes exactly one span. No company data.
- No Bitext rows: this pack is built from the hand-written synthetic sources only.
- Labels come from how each pair was constructed, not from human annotators.

## Known limitations (read before you optimize)
- Construction labels assume every `original` is a good reply and every harmful edit really breaks the criterion. Both
  were checked by reading, not by independent annotators.
- Few hand-written source replies (about 9 in dev), each in five languages, so cases are strongly
  correlated within a `source_id`. Group by `source_id` when you validate.
- Surface cues exist. Harmful edits often introduce words such as "always", "all", "any". Bitext rows without a
  cancel/refund fact only get harmless edits. A judge that keys on these will not hold up on the unseen cases.
- Length is matched, not a cue. Some harmful cases also carry one harmless length change (an opener, a filler, or a
  dropped pleasantry; see `length_padding` and `edited_unpadded`) that keeps them about as long as their original,
  so an opener or filler says nothing about the label. Padded harmless replies are still somewhat longer than
  originals. Length change by edit type in dev:

| edit type | family | n | mean length change (chars) | mean relative change |
|---|---|---|---|---|
| filler_padding | harmless | 15 | +44.5 | +18.0% |
| polite_opener | harmless | 30 | +29.0 | +12.4% |
| padding_removal | harmless | 5 | -38.0 | -16.7% |
| sentence_reorder | harmless | 30 | +0.0 | +0.0% |
| hedge_removal | harmful | 20 | -1.2 | -0.6% |
| condition_removal | harmful | 30 | -1.2 | -0.6% |
| wrong_number | harmful | 35 | -0.7 | -0.3% |
| wrong_policy_detail | harmful | 25 | +2.0 | +0.8% |
| **all harmful (excl. translation)** | | 110 | -0.3 | |
| **all harmless (excl. translation)** | | 80 | +16.9 | |
