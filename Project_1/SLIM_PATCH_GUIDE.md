# Controlled comparison patch

This is an experimental code patch, not a validated accuracy improvement.

## Install
Unzip the patch in a separate folder. Run `python install_patch.py /home/awate/sixt-hackathon-2026/Project_1` from that folder. It checks the reviewed versions of files before changing them and backs up replaced files. It does not modify .env or datasets. Do not force installation if it detects newer local changes.

## Tests and comparison
From Project_1 with .venv active:

    python -m unittest discover -s tests -p test_slim.py -v
    python compare_versions.py --data runs/smoke.gold.jsonl --outdir runs/compare_gpt_v3

The output directory must be new. This makes paid API calls. It runs the starter, full ProofGuard in signal-collection mode, and slim ProofGuard on exactly the same documents. The gateway and all roles are fixed to the configured extractor model with no temperature or extra-body options. It keeps their existing prompts/context strategies because those are algorithm differences. Workers=1. Each ProofGuard variant has a separate initially empty cache. Full mode collects extra signals, so this runtime is calibration-mode cost, NOT production latency. The starter uses at most 60000 text characters; ProofGuard uses the configured budget and retrieval. Same source documents do not imply identical prompt text.

Inspect each .log for failures. Summaries at .95 describe uncalibrated candidate ranking only; they are not final acceptance policies. Strict outputs stay all-review without a matching calibrator. No improvement has been measured by the patch author. Extend experiments to held-out development contracts AND forms before choosing an approach.

## Slim mode

    export PG_PIPELINE=slim
    export PG_RECOVERY_MAX_FIELDS=6

Full remains the default when PG_PIPELINE is unset. Slim uses schema/evidence checks, a verifier, and at most one second extraction per selected field. Selection uses a fixed uncalibrated heuristic in [0.2,0.9), not fitted acceptance scores, so fitting acceptance does not silently alter the recovery distribution. Recovery does not cover null candidates, hard failures or nested JSON in this first version. A successful reread replaces the candidate, including withdrawing it when null. Every replacement undergoes document-wide proof checks and verification again. The challenger is omitted. Verification of unchanged candidates may hit the response cache after cross-field checks. Unavailable verification blocks automatic acceptance. The six-field cap is schema-order based, not an optimized value-of-information policy.

Calibration traces contain FINAL candidate features and labels. Slim acceptance uses stage 1. No automatic acceptance is enabled without calibration. Existing calibrators are invalid after this patch because code/pipeline fingerprints change. Use disjoint dev tuning/calibration/validation documents; do not fit from the two smoke documents.

For a selected slim configuration:

    PG_PIPELINE=slim python pg.py predict --data YOUR_CALIBRATION_SUBSET.jsonl --mode llm --calibrating --out runs/slim_cal.jsonl --trace runs/slim_cal.trace.jsonl
    PG_PIPELINE=slim python pg.py calibrate --traces runs/slim_cal.trace.jsonl --gold YOUR_CALIBRATION_SUBSET.jsonl --out calibrators/llm.json
    PG_PIPELINE=slim python declare_submission.py
    PG_PIPELINE=slim python preflight.py

YOUR_CALIBRATION_SUBSET.jsonl is a placeholder: create a suitable development split first. Keep identical model/env settings for prediction, calibration, declaration and evaluation. Microsoft must not replace GPT without new calibration.

The GUI can use slim when launched with PG_PIPELINE=slim, but arbitrary uploads intentionally remain human-reviewed. The GUI baseline is still a different extractor from the terminal benchmark. Neither limitation is hidden by this patch. Sanitized .env.example restored for fresh setup; existing .env is untouched.

## Microsoft Decision-1

For a Microsoft Foundry deployment, configure the optional verifier in `.env`:

    VERIFIER_API=foundry
    VERIFIER_BASE_URL=<resource endpoint from Foundry>
    VERIFIER_MODEL=<deployment name from Foundry>
    VERIFIER_API_KEY=<Foundry API key>

The adapter calls `<resource endpoint>/providers/microsoft/v1/systemone`, sends the deployment name in `model`, and uses the `api-key` header. It sends text evidence only; Foundry Decision-1 is not treated as a vision model. `python3 pg.py doctor` probes the configured verifier. The mock test validates request plumbing, not live access, calibration, or extraction accuracy.

If the provided model gateway exposes `/v1/systemone` and uses Bearer authentication, configure `VERIFIER_API=systemone`, `VERIFIER_BASE_URL=<gateway base URL>`, `VERIFIER_MODEL=microsoft-decision-1` (or its deployment name), and `VERIFIER_API_KEY=<gateway key>`. The adapter includes `model` in the request and accepts either `probability` or `noul` values in typed answers. The gateway variables can be mapped to these existing `VERIFIER_*` settings; do not put the key in code.

The separate probe below is only for a chat-compatible provider route such as OpenRouter. It does not configure the active verifier.

    python probe_decision.py

It calls `/chat/completions` with `microsoft-decision-1` and three fictional examples. It does not call Microsoft Foundry's typed decision endpoint or change the active verifier. The probe deliberately stops on errors rather than retrying repeatedly.

Temperature acceptance can be tested separately:

    python probe_decision.py --model gpt-6.1-sol --temperature 0

This does NOT enable temperature globally. The shared client's temperature argument remains omitted; only explicitly configured per-role extra_body can supply it. Comparison deliberately clears extra_body. Do not equate temperature support with correctness or determinism.

## Validation performed
19 scorer tests, 17 existing backend/usage tests, 5 new recovery tests, and 7 app tests passed. Four dataset-dependent tests skipped (data absent). Python compilation passed. No live gateway, browser, calibration, accuracy or latency validation performed. Mocked recovery tests verify replacement re-verification, null withdrawal, invalid replacement rejection, failed-reread retention and separate pipeline signatures.
