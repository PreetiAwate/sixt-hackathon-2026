# ProofGuard — Project 1 submission draft

**Status: patched implementation; no live benchmark results measured for this revision.**
Team: FILL. Authors: FILL.

## Approach
Shared extraction pipeline for the GUI and JSONL runner: document reading, candidate extraction, deterministic evidence checks, model verification, selective second reading and empirical risk calibration. Evidence checks do not prove correctness. The GUI retains all candidates for human review; benchmark output contains only fields accepted by the frozen policy.

All model roles currently fall back to the extractor unless explicitly configured. H2O support is implemented but do not claim H2O was used until verified in the live run. Text-derived verifier scores are heuristic features, not calibrated probabilities.

## Models and why
| Role | Actual model / endpoint | Open or closed | Reason | Live validation |
|---|---|---|---|---|
| Extractor | FILL | FILL | FILL | FILL |
| Verifier | FILL | FILL | FILL | FILL |
| Challenger / rereader | FILL | FILL | FILL | FILL |

## Reproduce
Follow START_HERE.md. Install with `bash setup_app.sh`, configure the existing `.env`, run `python pg.py doctor`, then `bash workflow.sh smoke`. Calibration uses development data only: `bash workflow.sh calibrate 40` (increase the sample for the real experiment). Freeze settings and run `bash workflow.sh evaluate`.
Hidden test: `bash run_hidden.sh INPUT.jsonl OUTPUT.jsonl`. Raw data supplied under data/raw. The runner checks the calibrator signature and declared threshold before any prediction.

Required environment: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Optional role variables and runtime settings are documented in .env.example. Use PG_VISION only after checking the gateway's image path. Ubuntu PDF/OCR tools: poppler-utils, tesseract-ocr (OCR optional).

## Threshold and results
Declared threshold: FILL from submission_config.json after calibration. A threshold of 1.0001 means all-review and zero automation. Break-even at review cost 1 / error cost 100 is p=0.99; positive expected saving requires p>0.99. Net saving=(accepted - 100*wrong)/total fields.

| System | Evaluation documents | Accepted | Wrong | Precision | Coverage (official) | Net saving |
|---|---:|---:|---:|---:|---:|---:|
| Official baseline | FILL | FILL | FILL | FILL | FILL | FILL |
| Frozen starter LLM | FILL | FILL | FILL | FILL | FILL | FILL |
| ProofGuard | FILL | FILL | FILL | FILL | FILL | FILL |

Attach aggregate official_score.txt and baseline_score.txt. Pick each baseline threshold on dev, never select it from the test sweep. workflow.sh's baseline report at the ProofGuard threshold is only a same-threshold reference. Report provider errors and failed documents as well as performance. API cost is unknown until actual role prices are configured. Do not report zero default cost as free.

## Evaluation limits
Existing retrieval weights were learned from dev; cross-validation of the calibrator alone does not validate the whole pipeline independently. Public test must remain untouched by tuning. Group related pages when making further splits. Small samples and threshold selection increase uncertainty; bootstrap selection scores are not deployment guarantees. Uploaded arbitrary documents have no validated calibration and require human review.

## Deliverables
Code + this completed SUBMISSION.md + EDA.md + submission/slides.pdf (max 5 pages) OR submission/demo.mp4 (under 50 MB). Regenerate and review slides after obtaining results. Existing slides in the old project are stale. No secrets, downloaded source data, raw traces or model caches in git.

## Credits
CUAD — Hendrycks et al., 2021, CC BY 4.0 labels. ExtractBench — Zhang et al., 2026, Apache-2.0. Preserve original dataset notices. Mention H2O only if actually used.
