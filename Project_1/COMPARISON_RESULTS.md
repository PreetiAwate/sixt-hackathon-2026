# Project 1: Saved Comparison Results

**Run status:** completed from saved outputs on 2026-11-13. No public-test or hidden-test labels were used. This is a development-only diagnostic, not a validated submission score.

## Run manifest

- Dataset: `data/unified/dev.jsonl`; 40 sampled documents, seed 7.
- Scored split: 21 ExtractBench documents and 19 CUAD contracts, 1,769 fields.
- Common-value training split: the other 500 dev documents; scored document IDs were excluded.
- Shared extractor: `gpt-6.1-sol` through `https://llm.orange.sixt.com/` chat completions.
- Full/slim verifier: `microsoft-decision-1` through the gateway's typed SystemOne interface.
- Full challenger and rereader: `gpt-6.1-sol`; slim uses the rereader and bounded recovery without the challenger.
- Vision setting: `PG_VISION=auto`; the extractor's image probe succeeded. Full traces mark 15/21 ExtractBench documents as scans. Slim traces omit document scan/evidence metadata.
- Diagnostic scoring cutoff: 0.95. This was not a calibrated or declared submission policy.
- Full manifest: `runs/compare_decision1_dev40_20261113/manifest.json`.

The saved starter predictions were generated before the post-run parser hardening in `baseline_llm_compat.py`; the starter results describe those saved predictions, not the hardened parser. Raw starter model responses were not preserved, so malformed-response frequency in that run cannot be reconstructed from the prediction file alone.

## Candidate extraction: official scorer at threshold 0

At threshold 0, all non-null predictions are counted as candidates. Candidate precision is correct divided by all non-null candidates, including hallucinations in the denominator.

| Version | Source | Fields | Non-null candidates | Correct | Wrong | Hallucinated | Candidate precision | Official coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Common | All | 1,769 | 634 | 227 | 361 | 46 | 35.8% | 69.1% |
| Common | CUAD | 779 | 38 | 8 | 30 | 0 | 21.1% | 15.2% |
| Common | ExtractBench | 990 | 596 | 219 | 331 | 46 | 36.7% | 91.5% |
| Starter | All | 1,769 | 773 | 578 | 134 | 61 | 74.8% | 83.7% |
| Starter | CUAD | 779 | 277 | 132 | 88 | 57 | 47.7% | 88.0% |
| Starter | ExtractBench | 990 | 496 | 446 | 46 | 4 | 89.9% | 81.9% |
| Full | All | 1,769 | 334 | 188 | 94 | 52 | 56.3% | 33.1% |
| Full | CUAD | 779 | 277 | 146 | 79 | 52 | 52.7% | 90.0% |
| Full | ExtractBench | 990 | 57 | 42 | 15 | 0 | 73.7% | 9.5% |
| Slim | All | 1,769 | 331 | 184 | 95 | 52 | 55.6% | 32.8% |
| Slim | CUAD | 779 | 275 | 143 | 80 | 52 | 52.0% | 89.2% |
| Slim | ExtractBench | 990 | 56 | 41 | 15 | 0 | 73.2% | 9.3% |

If every non-null candidate were automatically booked, official net saving would be negative for all four: common -22.6%, starter -10.6%, full -8.1%, slim -8.1%. Threshold 0 is an extraction diagnostic, not a recommended policy.

## Diagnostic cutoff: 0.95

This separate table uses the official scorer at a fixed 0.95 cutoff. It is not a frozen/calibrated acceptance policy. Official net saving uses review cost 1 and wrong-value cost 100.

| Version | Source | Fields | Accepted | Correct | Wrong | Hallucinated | Precision | Coverage | Net saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Common | All | 1,769 | 129 | 107 | 21 | 1 | 82.9% | 15.0% | -117.1% |
| Common | CUAD | 779 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |
| Common | ExtractBench | 990 | 129 | 107 | 21 | 1 | 82.9% | 21.3% | -209.2% |
| Starter | All | 1,769 | 751 | 571 | 125 | 55 | 76.0% | 81.8% | -975.1% |
| Starter | CUAD | 779 | 270 | 132 | 87 | 51 | 48.9% | 87.6% | -1,736.8% |
| Starter | ExtractBench | 990 | 481 | 439 | 38 | 4 | 91.3% | 79.4% | -375.7% |
| Full | All | 1,769 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |
| Full | CUAD | 779 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |
| Full | ExtractBench | 990 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |
| Slim | All | 1,769 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |
| Slim | CUAD | 779 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |
| Slim | ExtractBench | 990 | 0 | 0 | 0 | 0 | n/a | 0.0% | 0.0% |

Full/slim's uncalibrated `heuristic_p()` explicitly caps confidence at 0.9, so they cannot pass a 0.95 cutoff without a fitted calibrator. In this run the runtime threshold is the safe all-review fallback, 1.0001.

### Starter hallucination-count reconciliation

The starter has 61 hallucinated non-null candidates at threshold 0 (all candidates). At threshold 0.95, 55 hallucinations remain accepted. The other six have confidence below 0.95 and are excluded by that cutoff. Both counts are correct for different thresholds.

## Decision-1 verifier audit

The manifest and per-field traces record `microsoft-decision-1` through `api=systemone`. This is not inferred from a model name alone: successful candidate rows are labeled `verifier=systemone` and contain `f1.v_p` support probabilities. The earlier fictional `pg.py doctor` probe also returned `P(supported)=0.99977`, `P(complete)=0.99850`, and `P(right_kind)=0.99909`.

| Version | Candidate fields | Attempted | Successful probabilities | Failed responses | Not attempted |
|---|---:|---:|---:|---:|---:|
| Full | 334 | 303 | 303 | 0 | 31 |
| Slim | 331 | 300 | 300 | 0 | 31 |

The 31 unattempted candidates in each version consist of 25 hard-stop candidates and 6 manual JSON fields. They are not failed requests. Of 1,769 fields, the rest had no candidate to verify. Among attempted calls, all had `systemone` labels and available probabilities; no missing or failed result was found. Empty error logs alone would not prove this—the per-field signals and counters are the evidence.

Verifier probabilities vary and rank correct candidates higher on average, but overlap substantially:

| Version | Correct candidates: n / mean P(supported) / median | Incorrect or hallucinated: n / mean / median | Incorrect candidates with P >= 0.95 | Incorrect candidates with P >= 0.99 |
|---|---|---|---:|---:|
| Full | 181 / 0.774 / 0.967 | 122 / 0.571 / 0.679 | 34 | 16 |
| Slim | 178 / 0.785 / 0.969 | 122 / 0.548 / 0.622 | 39 | 18 |

These are verifier signals, not calibrated end-to-end correctness probabilities: some incorrect candidates receive very high support scores.

## Recovery audit

Slim saves `original_value`, so transitions can be checked with the official field normalizers. Of 343 original candidate values:

| Transition | Count |
|---|---:|
| Incorrect to correct | 0 |
| Correct to incorrect | 2 |
| Candidate to null | 12 |
| Unchanged under official normalization | 314 |
| Correct to different but still correct | 1 |
| Incorrect to different but still incorrect | 14 |
| Null to candidate | 0 |

Full traces do not save `original_value`, so a transition audit is unavailable for full. In the full pipeline code, rereading contributes agreement signals but does not replace the original candidate.

## Offline OOF diagnostic and leakage

Offline, document-folded OOF diagnostics were generated from the saved 40 development traces only and stored under `runs/compare_decision1_dev40_20261113/offline_calibration/`:

| Pipeline | OOF-selected threshold | OOF accepted candidates | OOF precision | Point net saving | Reported lower-quantile saving |
|---|---:|---:|---:|---:|---:|
| Full | 0.9393 | 37 | 100% | 2.09% | 1.53% |
| Slim | 0.96276 | 23 | 100% | 1.30% | 0.90% |

These are exploratory development diagnostics, not validated or submission thresholds. Calibration uses five document folds and a document-seeded bootstrap, but the upstream retriever weights were learned on the full dev set; the 40 scored dev documents may therefore have influenced retrieval. The retriever file has no training-document manifest to prove otherwise. Threshold selection and reported OOF performance use the same small OOF set. Public-test labels were not used.

The calibration CLI default is `--quantile 0.1`; the report field `p20_saving_pct` is misleading because these runs use the 10th percentile, not the 20th. Do not install these exploratory files as `calibrators/llm.json`.

## Runtime and evidence

| Stage | Runtime | Tokens per document | Logged role calls, including cache hits | Cache hits |
|---|---:|---:|---:|---:|
| Common-value baseline | 0.31 s | local | — | — |
| Starter LLM | 1,205.94 s | not instrumented | not instrumented | not instrumented |
| Full | 2,520.48 s | 21,693 | 681 | 40 |
| Slim | 1,515.39 s | 13,920 | 708 | 306 |

Provider prices were not configured. Usage JSON reports USD 0.00 as a default placeholder; actual API cost is unknown, not free.

Full traces contain 46 evidence quotes, all from ExtractBench and none from CUAD. Slim contains the same 46 ExtractBench evidence quotes but does not retain per-document scan/evidence metadata. The run used `PG_VISION=auto`; the doctor image probe succeeded. The GUI distinguishes Suggested, Needs review, and Human approved; benchmark-calibrated auto-acceptance is not enabled for arbitrary uploads.

## Reproduction from saved files

No model calls are needed to reproduce scorer tables:

```bash
cd ~/sixt-hackathon-2026/Project_1
for version in common starter full slim; do
  .venv/bin/python 5_score.py --gold runs/compare_decision1_dev40_20261113/gold.jsonl \
    --pred runs/compare_decision1_dev40_20261113/${version}.jsonl --threshold 0
done
for version in common starter full slim; do
  .venv/bin/python 5_score.py --gold runs/compare_decision1_dev40_20261113/gold.jsonl \
    --pred runs/compare_decision1_dev40_20261113/${version}.jsonl --threshold 0.95
done
```

The original live invocation was:

```bash
.venv/bin/python compare_versions.py --data data/unified/dev.jsonl --sample 40 \
  --seed 7 --outdir runs/compare_decision1_dev40_20261113 --threshold 0.95
```

That output directory already exists; a new live run requires a new output directory and incurs model calls. `test_public` was not run. `submission_config.json` still has a null threshold and `calibrators/llm.json` is missing, so preflight and hidden-test submission are not ready. Do not invent or install a live calibrator to pass preflight.