#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
OUTPUT_ROOT="$REPO_ROOT/research/local_runs/herg_comprehensive_optimization_v11"
SOURCE_ROOT="$REPO_ROOT/research/local_runs/herg_domain_mixture_campaign_v9"
LOG_PATH="$OUTPUT_ROOT/campaign.log"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  print -u2 "ERROR: expected project Python at $REPO_ROOT/.venv/bin/python"
  exit 2
fi

FREE_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')
if (( FREE_KIB < 15728640 )); then
  print -u2 "ERROR: at least 15 GiB of free disk is required."
  exit 2
fi

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

print "Starting or resuming hERG V11 comprehensive optimization"
print "Output: $OUTPUT_ROOT"
print "Six compute threads; fixed nested scaffold evaluation; validation/test labels sealed."
print "The identical command resumes from validated unit boundaries."

exec caffeinate -dimsu "$REPO_ROOT/.venv/bin/python" \
  "$REPO_ROOT/pipeline/scripts/run_local_herg_comprehensive_optimization_v11.py" run \
  --repo-root "$REPO_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --workers 6 \
  --screen-maximum 12 \
  --finalist-maximum 6 \
  --bootstrap-replicates 10000 \
  --max-active-hours 12 \
  > >(tee -a "$LOG_PATH") 2>&1
