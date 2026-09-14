#!/usr/bin/env bash
# Reruns LEQ on D4RL medium (hopper/halfcheetah/walker2d), one GPU each, in parallel.
#
# Detaches the same way run_example/bash_scr/mult_seed/combo_dbg_all_estimators_sparse.sh
# does in OfflineRL-Kit2: re-exec itself under setsid+nohup so closing the terminal/SSH
# session can't SIGHUP it. Output goes to a log file instead of your terminal.
set -euo pipefail

# Absolute, resolved before the cd below -- the detach re-exec needs it.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate LEQ2

REPO="/home/brian/repos/LEQ2"
cd "$REPO"
export PYTHONPATH="$REPO"

# name | env_name | expectile | extra_args | gpu
# expectile: task-specific tau from LEQ paper Table 10 (arXiv 2407.00699 appendix).
TASKS=(
  "hopper|hopper-medium-v2|0.1||5"
  "halfcheetah|halfcheetah-medium-v2|0.3|--seed 1|4"
  "walker2d|walker2d-medium-v2|0.3||7"
)

if [ "${DETACH:-1}" = 1 ] && [ -z "${LEQ_DETACHED:-}" ]; then
  mkdir -p "$REPO/tmp/_runs"
  LOGFILE="$REPO/tmp/_runs/leq_medium_$(date +%m%d-%H%M%S).log"
  LEQ_DETACHED=1 setsid nohup bash "$SELF" "$@" >"$LOGFILE" 2>&1 </dev/null &
  echo "detached: pid $!"
  echo "  tail -f $LOGFILE"
  exit 0
fi

mkdir -p "$REPO/tmp/_runs"
pids=()
for row in "${TASKS[@]}"; do
  IFS='|' read -r name env_name expectile extra gpu <<< "$row"
  TASKLOG="$REPO/tmp/_runs/${name}_$(date +%m%d-%H%M%S).log"
  echo ">>> $name: env=$env_name expectile=$expectile gpu=$gpu -> $TASKLOG"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH='.' python train/train_LEQ.py \
    --env_name="$env_name" --expectile "$expectile" $extra \
    >"$TASKLOG" 2>&1 &
  pids+=($!)
done

echo "launched pids: ${pids[*]}"
wait
echo "All LEQ medium runs completed."
