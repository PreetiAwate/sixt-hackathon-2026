#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ ! -f .venv/bin/activate ] || source .venv/bin/activate
mkdir -p runs calibrators submission
case "${1:-help}" in
 smoke)
  python pg.py predict --data data/unified/dev.jsonl --sample 2 --seed 7 --mode llm --workers 1 --calibrating --out runs/smoke.jsonl --full-out runs/smoke.full.jsonl --trace runs/smoke.trace.jsonl
  ;;
 calibrate)
  # Development data ONLY. Start small for plumbing, enlarge before reporting.
  python pg.py predict --data data/unified/dev.jsonl --sample "${2:-40}" --seed 7 --mode llm --calibrating --out runs/dev_llm.jsonl --trace runs/dev_llm.trace.jsonl
  python pg.py calibrate --traces runs/dev_llm.trace.jsonl --gold data/unified/dev.jsonl --out calibrators/llm.json
  python declare_submission.py
  python preflight.py
  ;;
 evaluate)
  python preflight.py
  bash run_hidden.sh data/unified/test_public.jsonl runs/pub_llm.jsonl
  T=$(python -c 'import json; print(json.load(open("submission_config.json"))["threshold"])')
  python 5_score.py --gold data/unified/test_public.jsonl --pred runs/pub_llm.jsonl --threshold "$T" > submission/official_score.txt
  python pg.py evaluate --gold data/unified/test_public.jsonl --pred runs/pub_llm.jsonl --threshold "$T" --out runs/pub_llm.results.json --label 'ProofGuard live frozen policy'
  python 3_baseline.py --train data/unified/dev.jsonl --target data/unified/test_public.jsonl --out runs/official_baseline.jsonl
  python 5_score.py --gold data/unified/test_public.jsonl --pred runs/official_baseline.jsonl --threshold "$T" > submission/baseline_score.txt
  cat submission/official_score.txt
  ;;
 *) echo 'Usage: bash workflow.sh smoke | calibrate [DEV_DOC_COUNT] | evaluate'; exit 2;;
esac
