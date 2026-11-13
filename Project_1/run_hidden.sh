#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then echo 'Usage: bash run_hidden.sh INPUT.jsonl OUTPUT.jsonl' >&2; exit 2; fi
# Resolve caller-relative paths before changing directory.
INPUT=$(realpath "$1")
OUTPUT=$(realpath -m "$2")
cd "$(dirname "${BASH_SOURCE[0]}")"
[ ! -f .venv/bin/activate ] || source .venv/bin/activate
PY="${PYTHON:-python3}"
"$PY" preflight.py
CAL=$("$PY" -c 'from proofguard.submission_config import load_submission; print(load_submission()[2])')
"$PY" pg.py predict --data "$INPUT" --out "$OUTPUT" --mode llm --calibrator "$CAL" --workers "${PG_WORKERS:-1}"
"$PY" pg.py fill-missing --data "$INPUT" --out "$OUTPUT"
