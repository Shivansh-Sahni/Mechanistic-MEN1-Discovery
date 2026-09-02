#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="$REPO_ROOT/research/local_runs/herg_v10_2_coherent_platform"

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "ERROR: project Python is missing: $PYTHON_BIN"
  exit 2
fi

cd "$REPO_ROOT"
if [[ ! -f "$OUTPUT_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" pipeline/scripts/run_local_herg_v10_2_coherent_platform.py build \
    --repo-root "$REPO_ROOT" \
    --output-root "$OUTPUT_ROOT"
fi

exec caffeinate -dimsu "$PYTHON_BIN" \
  pipeline/scripts/run_local_herg_v10_2_coherent_platform.py serve \
  --repo-root "$REPO_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --host 127.0.0.1 \
  --port 8790
