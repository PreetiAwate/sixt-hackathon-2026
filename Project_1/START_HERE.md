# Start here — repaired ProofGuard hackathon workflow

## What changed

- GUI now selects **ProofGuard** (default) or the original **Baseline visual extractor**.
- ProofGuard GUI and CLI share `ProofGuard.run_record`. Uploaded arbitrary documents require human review, because the dataset calibration is not validated for their schema/domain.
- Missing or incompatible calibration means all-review, never the previous 0.88 fallback. The hidden runner refuses missing/stale calibration and mismatched thresholds before starting.
- Pipeline signatures include model roles, endpoints, relevant settings and source/retriever hashes, without API keys. Recalibrate after code/model/settings changes.
- Calibration uses full candidate values, not truncated 300-character clauses. Trace provenance is checked; resumed runs require matching input/config manifests.
- Correct break-even: p > 0.99 under review 1 / wrong 100. Official scorer is unchanged.
- Doctor probes use larger token budgets and a valid red image. A failed vision probe is not labeled definitive proof of unsupported vision. Shared client omits temperature, enforces >=16 tokens, limits retries to three, and redacts the API key from provider errors.
- API errors are not counted as successful extractions when every field fails. GUI uses temporary upload/cache storage for ProofGuard requests; benchmark cache remains disk-based and ignored by git.

## Install this overlay

Extract the archive **from the repository root** so files land inside Project_1, not Project_1/Project_1. This package contains code/config templates only. It does not contain `.env`, data, caches, generated runs, or an already-fitted live calibrator. Preserve your local data and .env. Back up source before overlaying (command in the response).

The existing `submission/slides.pdf`, report.html and offline calibrators on your machine are not deleted by an overlay; they are old artifacts and must not be represented as new measured results.

## 1. Setup (no dataset download)

```bash
cd ~/sixt-hackathon-2026/Project_1
bash setup_app.sh
source .venv/bin/activate
```

For PDF text/rendering and optional OCR on Ubuntu:

```bash
sudo apt install poppler-utils tesseract-ocr
```

## 2. Configuration

Edit **.env**, not .env.example. Keep the working LLM_BASE_URL, LLM_API_KEY and LLM_MODEL. Do not overwrite your existing key. Optional role settings stay commented until those services exist. Initial active settings: PG_WORKERS=1, PG_MAX_PAGES=8, PG_VISION=auto. `PG_VISION=1` attaches images for all form field groups; `0` disables extractor images, which may reduce quality. Vision support still needs validation on your gateway.

The real process environment takes precedence over .env. If you exported old settings, use a new terminal or reload your trusted file:

```bash
set -a
source .env
set +a
python -u pg.py doctor
```

No logprobs means heuristic verifier signals, not calibrated probabilities. Prices default to zero but must be labeled unknown unless configured with real rates.

## 3. Launch the app

```bash
bash start_app.sh
```

Open http://127.0.0.1:8000. For a remote Ubuntu machine forward port 8000 via your editor/SSH. Keep the app bound to localhost. Try sample mode, then a TXT upload with a small schema. Choose ProofGuard for the enhanced pipeline. Baseline uses the original app extractor; it is distinct from the organizer's text starter.

The upload page reports candidate evidence and review status. It does not report accuracy or net saving without reference labels. It keeps user edits in the browser until JSON export. No login or persistent multiuser database is implemented.

## 4. Validate locally

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s document_app/tests -v
python test_score.py
bash workflow.sh smoke
```

The automated model-server tests are simulations. The smoke command makes real paid calls on two dev documents and gathers all signals. It is a plumbing test, not a calibration result. Look for field_errors/error in runs/smoke.trace.jsonl. No live API calls were made while preparing this repair.

## 5. Calibrate and declare

```bash
bash workflow.sh calibrate 40
```

This runs 40 randomly selected dev documents, fits the risk model and writes the declared threshold into submission_config.json. Start with 40 to inspect runtime; use a larger representative dev sample for final estimates. It is not a guarantee of 99% accuracy. Related pages should be grouped for stronger calibration studies. Existing retrieval weights were learned on dev; out-of-fold risk calibration does not independently validate the entire upstream retrieval pipeline.

A threshold >1 means the calibration found no profitable acceptance policy: zero automation, not a software failure. Improve reading/checks using dev and rerun. Do not lower the threshold merely to manufacture coverage.

## 6. Freeze and evaluate

```bash
bash workflow.sh evaluate
```

This checks matching calibration, runs public predictions, and writes the authoritative scorer outputs to submission/official_score.txt and submission/baseline_score.txt. Do not tune on public-test results. The baseline uses the same threshold for a simple reference; for a fair tuned-baseline comparison select its own threshold on development predictions first. Keep the existing 4_extract_example.py and saved starter predictions as B1. Do not confuse GUI visual baseline with that text starter.

## 7. Hidden-test compatibility

```bash
bash run_hidden.sh INPUT.jsonl OUTPUT.jsonl
```

Input may omit gold/gold_evidence. Raw documents must be in data/raw. Non-calibration processing strips reference labels. Global configuration errors fail explicitly; per-document failures produce null fields with confidence 0 and logs. The runner uses submission_config.json and the exact matching live calibrator. Fresh predictions overwrite output rather than mixing previous runs.

## 8. Submission by 16:45

- Fill SUBMISSION.md: team, actual models, threshold, official measured results, installation/run instructions.
- Keep EDA.md, but verify its numbers against current data.
- Regenerate max 5 slides or record a <=1 minute video (<50 MB). Existing slides are stale.
- If using make_report.py/make_submission.py, inspect generated text; they are draft helpers, not a substitute for validating every claim. Do not claim H2O or independent-model comparisons unless actually run.
- Commit code, selected aggregate scores, matching calibrators/llm.json and submission_config.json. Exclude data, .env, raw traces/predictions and caches. Preserve LICENSE and organizer files.

Do not rerun old setup.sh to repair app dependencies: setup_app.sh avoids another download attempt on your Zscaler-affected machine.
