#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
MODEL_ROOT="$REPO_ROOT/research/local_runs/herg_v10_1_expanded_platform"

cd "$REPO_ROOT"

if [[ ! -f "$MODEL_ROOT/manifest.json" ]]; then
  print -u2 "ERROR: V10.1 has not been built at $MODEL_ROOT"
  exit 2
fi

.venv/bin/python pipeline/scripts/run_local_herg_v10_1_expanded_platform.py validate \
  --output-root "$MODEL_ROOT"

print "Opening hERG V10.1 at http://127.0.0.1:8788"
exec caffeinate -dimsu .venv/bin/python \
  pipeline/scripts/run_local_herg_v10_1_expanded_platform.py serve \
  --model-root "$MODEL_ROOT" \
  --host 127.0.0.1 \
  --port 8788
