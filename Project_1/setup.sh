#!/usr/bin/env bash
# Project 1 quick setup (about 10 min, mostly the 60 MB download). Usage: bash setup.sh [--accept-licenses]
# Python is our starting point, not a requirement: use any language you like.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python3}"
"$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || { echo "Need Python 3.11+ (try: PYTHON=python3.x bash setup.sh)"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip && pip install -q -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "Created .env: add your LLM key later (never commit it)."; }
[ -f data/unified/dev.jsonl ] || { bash 1_download.sh "$@"; python 2_prepare.py; }
python 3_baseline.py --train data/unified/dev.jsonl --target data/unified/test_public.jsonl --out preds.jsonl
python 5_score.py --gold data/unified/test_public.jsonl --pred preds.jsonl --threshold 0.95 | tail -8
echo; echo "Ready. Activate with: . .venv/bin/activate. Next: read 4_extract_example.py and build on it."
