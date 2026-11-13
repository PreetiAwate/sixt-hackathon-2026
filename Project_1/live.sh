#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo 'Use explicit stages: bash workflow.sh smoke; bash workflow.sh calibrate 40; bash workflow.sh evaluate'
echo 'No paid calls started. Read START_HERE.md before launching the full run.'
