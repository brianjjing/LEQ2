#!/usr/bin/env bash
# One-time prereq for run_leq_medium.sh in LEQ2: trains the 3 dynamics ensembles
# LEQ2's train_LEQ.py expects at ../OfflineRL-Kit2/models/dynamics-ensemble/<seed>/<task>
# (hopper/walker2d at seed 42, halfcheetah at seed 1 -- matching that script's seeds).
# Only dynamics.train()+save() runs; run_dynamics.py builds a COMBOPolicy but never
# calls its trainer, so this is just the world-model step, not full COMBO training.
#
# Detaches the same way run_example/bash_scr/mult_seed/combo_dbg_all_estimators_sparse.sh
# does: re-exec itself under setsid+nohup so closing the terminal/SSH session can't
# SIGHUP it. Output goes to a log file instead of your terminal.
set -euo pipefail

# Absolute, resolved before the cd below -- the detach re-exec needs it.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate LEQ2

LEQ2_DIR="$(cd "$(dirname "$SELF")/.." && pwd)"
REPO="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
cd "$REPO"
export PYTHONPATH="$REPO"

# name | task | seed | gpu
TASKS=(
  "hopper|hopper-medium-v2|42|5"
  "halfcheetah|halfcheetah-medium-v2|1|4"
  "walker2d|walker2d-medium-v2|42|7"
)

if [ "${DETACH:-1}" = 1 ] && [ -z "${DYN_DETACHED:-}" ]; then
  mkdir -p "$REPO/log/_runs"
  LOGFILE="$REPO/log/_runs/pretrain_leq_medium_dynamics_$(date +%m%d-%H%M%S).log"
  DYN_DETACHED=1 setsid nohup bash "$SELF" "$@" >"$LOGFILE" 2>&1 </dev/null &
  echo "detached: pid $!"
  echo "  tail -f $LOGFILE"
  exit 0
fi

mkdir -p "$REPO/log/_runs"
pids=()
for row in "${TASKS[@]}"; do
  IFS='|' read -r name task seed gpu <<< "$row"
  TASKLOG="$REPO/log/_runs/dynamics_${name}_$(date +%m%d-%H%M%S).log"
  echo ">>> $name: task=$task seed=$seed gpu=$gpu -> $TASKLOG"
  CUDA_VISIBLE_DEVICES="$gpu" python run_example/run_dynamics.py \
    --task "$task" --seed "$seed" --device cuda \
    >"$TASKLOG" 2>&1 &
  pids+=($!)
done

echo "launched pids: ${pids[*]}"
wait
echo "All dynamics pretraining completed. Models under $REPO/models/dynamics-ensemble/<seed>/<task>/"
echo "Starting LEQ training on LEQ2..."
# DETACH=0: we're already inside the setsid-detached process from above, no need to re-detach.
DETACH=0 bash "$LEQ2_DIR/bash_scr/run_leq_medium.sh"
