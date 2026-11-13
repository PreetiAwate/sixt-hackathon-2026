#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ ! -f .venv/bin/activate ] || source .venv/bin/activate
exec bash document_app/run.sh
