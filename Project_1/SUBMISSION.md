# Submission: ProofGuard

**Status: development-only evaluation completed; not ready for automated hidden-test submission.**
Team and author names were not supplied in the workspace.

## Approach
ProofGuard extracts candidate values, checks evidence/schema consistency, verifies candidates with a typed decision model, and sends uncertain fields for re-reading or human review. The assisted-review UI requires a human to approve suggested values. Uploaded arbitrary documents are always review-only because benchmark calibration has not been validated for their domains or schemas.

The saved comparison used the existing full and slim pipelines; no new model layer was added. Full uses the configured challenger on recheck cases. Slim uses bounded rereading and re-verification, without the challenger. See [COMPARISON_RESULTS.md](COMPARISON_RESULTS.md) for aggregate tables and audits.

## Models and why
| Role | Model/interface used | Run evidence |
|---|---|---|
| Extractor | `gpt-6.1-sol`, OpenAI-compatible `/chat/completions` on `llm.orange.sixt.com` | Live doctor passed text, JSON, and image probes; manifest records model and host |
| Verifier | `microsoft-decision-1` via the gateway's typed `/v1/systemone` interface | 303/303 full and 300/300 slim attempted results returned `systemone` probabilities |
| Challenger | `gpt-6.1-sol`, chat endpoint | Used by full only on recheck cases |
| Second reader | `gpt-6.1-sol`, chat endpoint | Used selectively by full and by bounded slim recovery |

Run settings included `PG_VISION=auto`, `PG_WORKERS=1`, `PG_MAX_CONTEXT_CHARS=40000`, `PG_MAX_PAGES=8`, `PG_FIELDS_PER_CALL=18`, and `PG_VERIFY_MAX_PER_DOC=60`. The run manifest is [here](runs/compare_decision1_dev40_20261113/manifest.json). Credentials are not included.

## Reproduce
Score the saved candidate files without making model calls:

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

Offline OOF diagnostics can be reproduced without model calls:

```bash
PG_PIPELINE=full .venv/bin/python pg.py calibrate --traces runs/compare_decision1_dev40_20261113/full.trace.jsonl \
	--gold runs/compare_decision1_dev40_20261113/gold.jsonl \
	--out runs/compare_decision1_dev40_20261113/offline_calibration/full_dev_oof.json --quantile 0.1
PG_PIPELINE=slim .venv/bin/python pg.py calibrate --traces runs/compare_decision1_dev40_20261113/slim.trace.jsonl \
	--gold runs/compare_decision1_dev40_20261113/gold.jsonl \
	--out runs/compare_decision1_dev40_20261113/offline_calibration/slim_dev_oof.json --quantile 0.1
```

The original live invocation was:

```bash
.venv/bin/python compare_versions.py --data data/unified/dev.jsonl --sample 40 \
	--seed 7 --outdir runs/compare_decision1_dev40_20261113 --threshold 0.95
```

That output directory already exists; a new live run requires a new output directory and makes paid model calls. The saved comparison ran before the strict malformed/partial-output parser hardening in `baseline_llm_compat.py`; its starter scores therefore describe the saved run, not the hardened parser. The saved starter log contains no errors, but raw responses were not preserved for retrospective parser validation.

## Threshold and results

`submission_config.json` currently has a null threshold and points to a missing `calibrators/llm.json`. The runtime safe fallback is 1.0001 (accept none). The 0.95 cutoff below is a diagnostic comparison threshold, not a declared policy. Break-even for review cost 1 and wrong-value cost 100 is p=0.99; positive expected saving requires p>0.99.

Candidate quality at threshold 0 and accepted quality at the fixed 0.95 diagnostic threshold are separate:

| System | Documents | Candidates | Correct candidates | Wrong candidates | Hallucinated candidates | Candidate precision (t=0) | Accepted (t=.95) | Precision (t=.95) | Net saving (t=.95) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Common-value baseline | 40 | 634 | 227 | 361 | 46 | 35.8% | 129 | 82.9% | -117.1% |
| LLM starter | 40 | 773 | 578 | 134 | 61 | 74.8% | 751 | 76.0% | -975.1% |
| ProofGuard full | 40 | 334 | 188 | 94 | 52 | 56.3% | 0 | n/a | 0.0% |
| ProofGuard slim | 40 | 331 | 184 | 95 | 52 | 55.6% | 0 | n/a | 0.0% |

No version reached the official 99% precision target at any threshold in this sample. There is no positive-savings automated policy supported by this run. Per-source rows, verifier audits, and recovery results are in [COMPARISON_RESULTS.md](COMPARISON_RESULTS.md). Provider pricing was not configured; reported USD 0.00 means unknown, not free.

## Evaluation limits
The saved OOF diagnostics use five document folds, but all 40 scored documents are dev documents and the retriever weights were learned on the full dev set. The 40 documents are therefore not an independent validation set for the whole pipeline. No independent public-test result exists; public-test data and labels were not used.

The saved OOF-selected thresholds (full 0.9393, slim 0.96276) and reported OOF 100% precision are exploratory only. They are selected and scored from the same small OOF set and are not validation results. No live calibrator was installed. `submission_config.json` still declares `threshold: null`; preflight and hidden-test submission are not ready. Do not copy the OOF artifacts into `calibrators/llm.json` or fabricate a threshold.

The calibration CLI default is `--quantile 0.1`; the report field `p20_saving_pct` is mislabeled and represents the 10th percentile in these runs. The code uses 5 document folds, L2=2, 300 bootstraps and seed 7. Retrieval weights lack training-document provenance. The 40-document sample is small and shows no threshold reaching 99% precision; it does not demonstrate zero errors or production reliability. Uploaded arbitrary documents remain human-reviewed.

## Deliverables
Code + this SUBMISSION.md + EDA.md + a 1-minute demo or up to 5 slides are required by the hackathon. Current preflight fails because the live calibrator is missing and the declared threshold is null. Do not submit the exploratory calibrator files as a live policy.

For a demo, show three fictional/sample fields: a source-supported candidate, a truly missing field, and a low-confidence/uncertain field. Show the evidence and human approve/review actions. Do not imply benchmark-calibrated automation for arbitrary uploads. Keep `.env`, downloaded datasets, caches, and raw traces out of Git; preserve the official scorer and teammates' existing changes.

## Credits
CUAD — Hendrycks et al., 2021, CC BY 4.0 labels. ExtractBench — Zhang et al., 2026, Apache-2.0. Preserve original dataset notices. Mention H2O only if actually used.
