#!/bin/zsh
set -euo pipefail

if (( $# != 0 )); then
  print -u2 "ERROR: this governed launcher accepts no arguments"
  exit 2
fi

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON="$REPO_ROOT/.venv/bin/python"
V11_ROOT="$REPO_ROOT/research/local_runs/herg_comprehensive_optimization_v11_1"
V101_ROOT="$REPO_ROOT/research/local_runs/herg_v10_1_expanded_platform"
V12_ROOT="$REPO_ROOT/research/local_runs/herg_endpoint_receptor_campaign_v12_1"
LOG_PATH="$V12_ROOT/v11_v12_complete.log"

cd "$REPO_ROOT"
mkdir -p "$V11_ROOT" "$V12_ROOT"

[[ -x "$PYTHON" ]] || { print -u2 "ERROR: missing $PYTHON"; exit 2; }
command -v caffeinate >/dev/null || { print -u2 "ERROR: caffeinate is required"; exit 2; }

FREE_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')
if (( FREE_KIB < 15728640 )); then
  print -u2 "ERROR: at least 15 GiB of free disk is required"
  exit 2
fi

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

print "Starting or resuming governed hERG V11 + V12"
print "V11: nested-scaffold IC50 optimization on 18,801 structures"
print "V12: coherent direct IC10/IC30 gaps and six-state receptor contract"
print "True docking remains fail-closed until receptor preparation and docking-engine gates pass"
print "Identical command resumes validated outputs; hard V11 active ceiling is 40 hours"

caffeinate -dimsu "$PYTHON" \
  "$REPO_ROOT/pipeline/scripts/run_local_herg_comprehensive_optimization_v11.py" run \
  --repo-root "$REPO_ROOT" \
  --source-root "$REPO_ROOT/research/local_runs/herg_domain_mixture_campaign_v9" \
  --output-root "$V11_ROOT" \
  --workers 6 \
  --screen-maximum 12 \
  --finalist-maximum 6 \
  --bootstrap-replicates 10000 \
  --max-active-hours 40 \
  2>&1 | tee -a "$LOG_PATH"

"$PYTHON" "$REPO_ROOT/pipeline/scripts/run_local_herg_comprehensive_optimization_v11.py" validate \
  --repo-root "$REPO_ROOT" \
  --source-root "$REPO_ROOT/research/local_runs/herg_domain_mixture_campaign_v9" \
  --output-root "$V11_ROOT" \
  --workers 6 2>&1 | tee -a "$LOG_PATH"

caffeinate -dimsu "$PYTHON" \
  "$REPO_ROOT/pipeline/scripts/run_local_herg_endpoint_receptor_campaign_v12.py" run \
  --repo-root "$REPO_ROOT" \
  --v11-root "$V11_ROOT" \
  --v101-root "$V101_ROOT" \
  --output-root "$V12_ROOT" \
  --workers 6 2>&1 | tee -a "$LOG_PATH"

"$PYTHON" "$REPO_ROOT/pipeline/scripts/run_local_herg_endpoint_receptor_campaign_v12.py" validate \
  --repo-root "$REPO_ROOT" \
  --v11-root "$V11_ROOT" \
  --v101-root "$V101_ROOT" \
  --output-root "$V12_ROOT" \
  --workers 6 2>&1 | tee -a "$LOG_PATH"

print "V11 + V12 COMPLETE"
print "Results: $V12_ROOT"
