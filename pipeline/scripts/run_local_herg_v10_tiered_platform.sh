#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
OUTPUT_ROOT="${REPO_ROOT}/research/local_runs/herg_v10_tiered_platform"

if [[ ! -x "${PYTHON}" ]]; then
  print -u2 "ERROR: missing project Python: ${PYTHON}"
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec caffeinate -dimsu "${PYTHON}" \
  "${REPO_ROOT}/pipeline/scripts/run_local_herg_v10_tiered_platform.py" launch \
  --repo-root "${REPO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --workers 6 \
  --host 127.0.0.1 \
  --port 8787
