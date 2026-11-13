#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -r document_app/requirements.txt 'jsonschema>=4,<5' 'pypdf>=5,<8' 'numpy>=1.26,<3' 'matplotlib>=3.8,<4'
if [ ! -f .env ]; then cp .env.example .env; echo 'Created .env; configure the three LLM fields locally.'; fi
printf '\nReady. Existing data and .env were preserved. Run: bash start_app.sh\n'
