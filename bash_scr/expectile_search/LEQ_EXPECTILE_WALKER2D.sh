#!/bin/bash
set -e
# Expectile hyperparameter search for LEQ on sparse Walker2d-Medium-Expert,
# WITHOUT a density-based guardian (guardian_penalty_coef=0, no guardian
# model loaded).
#
# Uses the sparse offline pickle for LEQ:
#   /public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl
#
# Pipeline:
#   1. Verify a PRE-TRAINED transition-dynamics ensemble already exists at
#      DYN_DIR (this script does NOT train dynamics -- it must already be
#      placed there, e.g. copied in from elsewhere).
#   2. Reuse that saved dynamics model to run LEQ once per expectile value
#      in EXPECTILES, ALL IN PARALLEL, one job per GPU.
#
# GPU layout (fixed, so this script never collides with itself):
#   GPUs 0-3, one per expectile value. NOTE: LEQ_EXPECTILE_HOPPER.sh also
#   uses GPUs 0-3 -- do not run both scripts at the same time unless you
#   override GPUS below (LEQ_EXPECTILE_HALFCHEETAH.sh uses GPUs 4-7 and is
#   safe to run alongside either one).
#
# Usage (from LEQ2 root):
#   bash bash_scr/expectile_search/LEQ_EXPECTILE_WALKER2D.sh
#
# Env overrides:
#   DATASET_PATH, SEED, EXPECTILES, GPUS, DYN_BASE_DIR, LEQ2_DIR, LEQ2_ENV

TASK="${TASK:-walker2d-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
DYN_TAG="${DYN_TAG:-walker2d-medium-expert-v2_sparse_73}"

SEED="${SEED:-42}"

if [ -z "${EXPECTILES+x}" ] || [ -z "$EXPECTILES" ]; then
    EXPECTILES=(0.1 0.3 0.4 0.5)
else
    # shellcheck disable=SC2206
    EXPECTILES=($EXPECTILES)
fi

if [ -z "${GPUS+x}" ] || [ -z "$GPUS" ]; then
    GPUS=(0 1 2 3)
else
    # shellcheck disable=SC2206
    GPUS=($GPUS)
fi

if [ "${#GPUS[@]}" -lt "${#EXPECTILES[@]}" ]; then
    echo "ERROR: need at least ${#EXPECTILES[@]} GPUs in GPUS, got ${#GPUS[@]}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ2_DIR="${LEQ2_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DYN_BASE_DIR="${DYN_BASE_DIR:-/public/gormpo/models/dynamics-ensemble}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"
# mujoco-py needs this; conda run's non-interactive shell doesn't source ~/.bashrc.
MUJOCO_LD_PATH="${MUJOCO_LD_PATH:-$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia}"

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: sparse dataset not found: $DATASET_PATH"
    exit 1
fi

DYN_DIR="$DYN_BASE_DIR/${SEED}/${DYN_TAG}"
SAVE_DIR="./tmp/EP_sparse_expectile_search/${DYN_TAG}"
LOG_DIR="$SAVE_DIR/logs"
RESULTS_FILE="$SAVE_DIR/results_${DYN_TAG}.csv"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "LEQ expectile search on sparse Walker2d (no guardian)"
echo "  Env task:      $TASK"
echo "  Dataset:       $DATASET_PATH"
echo "  Dynamics tag:  $DYN_TAG"
echo "  Seed:          $SEED"
echo "  Expectiles:    ${EXPECTILES[*]}"
echo "  GPUs:          ${GPUS[*]}"
echo "  Dynamics base: $DYN_BASE_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "============================================"
echo ""

echo "Step 1/2: Verify pre-trained dynamics ensemble -> $DYN_DIR"
if [ -f "$DYN_DIR/dynamics.pth" ] && [ -f "$DYN_DIR/mu.npy" ] && [ -f "$DYN_DIR/std.npy" ]; then
    echo "  Found dynamics.pth, mu.npy, std.npy. Not training -- reusing as-is."
else
    echo "ERROR: pre-trained dynamics ensemble not found at $DYN_DIR"
    echo "  Expected files: dynamics.pth, mu.npy, std.npy"
    echo "  This script does not train dynamics -- copy a pre-trained ensemble there first."
    exit 1
fi
echo ""

echo "Step 2/2: LEQ expectile sweep (parallel, no guardian)"
declare -A PIDS
declare -A GPU_FOR_EXPECTILE
for i in "${!EXPECTILES[@]}"; do
    expectile="${EXPECTILES[$i]}"
    gpu="${GPUS[$i]}"
    GPU_FOR_EXPECTILE[$expectile]="$gpu"
    logfile="$LOG_DIR/expectile_${expectile}.log"
    echo "  Launching expectile=$expectile on GPU $gpu -> $logfile"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "export LD_LIBRARY_PATH=\"$MUJOCO_LD_PATH:\$LD_LIBRARY_PATH\" && \
            cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$gpu' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$SEED' \
            --expectile '$expectile' \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --guardian_penalty_coef 0.0 \
            --eval_episodes 10 \
            --save_dir '$SAVE_DIR/' \
            --debug" > "$logfile" 2>&1 &
    PIDS[$expectile]=$!
done

FAILED_EXPECTILES=()
for expectile in "${EXPECTILES[@]}"; do
    if wait "${PIDS[$expectile]}"; then
        echo "  expectile=$expectile finished (GPU ${GPU_FOR_EXPECTILE[$expectile]})"
    else
        status=$?
        echo "  WARNING: expectile=$expectile FAILED (exit $status, GPU ${GPU_FOR_EXPECTILE[$expectile]}) -- see $LOG_DIR/expectile_${expectile}.log"
        FAILED_EXPECTILES+=("$expectile")
    fi
done
echo ""

echo "Collecting results -> $RESULTS_FILE"
echo "expectile,final_score,final_length" > "$RESULTS_FILE"
for expectile in "${EXPECTILES[@]}"; do
    logfile="$LOG_DIR/expectile_${expectile}.log"
    line=$(grep "Final score:" "$logfile" | tail -n 1)
    if [ -n "$line" ]; then
        score=$(echo "$line" | sed -n 's/.*Final score: \([^ ]*\) Final length:.*/\1/p')
        length=$(echo "$line" | sed -n 's/.*Final length: \(.*\)/\1/p')
        echo "$expectile,$score,$length" >> "$RESULTS_FILE"
    else
        echo "$expectile,NA,NA" >> "$RESULTS_FILE"
        echo "  WARNING: no 'Final score:' line found in $logfile"
    fi
done

BEST_LINE=$(tail -n +2 "$RESULTS_FILE" | grep -v ',NA,NA$' | sort -t, -k2 -g -r | head -n 1)
if [ -n "$BEST_LINE" ]; then
    BEST_EXPECTILE=$(echo "$BEST_LINE" | cut -d, -f1)
    BEST_SCORE=$(echo "$BEST_LINE" | cut -d, -f2)
else
    BEST_EXPECTILE="N/A"
    BEST_SCORE="N/A"
fi

echo ""
echo "============================================"
echo "Done. Results (each score = train_LEQ.py's final eval, averaged over"
echo "--eval_episodes=10 episodes):"
column -s, -t "$RESULTS_FILE"
echo ""
echo ">>> BEST expectile for $DYN_TAG: $BEST_EXPECTILE (final_score=$BEST_SCORE) <<<"
if [ "${#FAILED_EXPECTILES[@]}" -gt 0 ]; then
    echo ""
    echo "!!! FAILED expectiles (excluded from results above): ${FAILED_EXPECTILES[*]}"
fi
echo ""
echo "  Results CSV: $RESULTS_FILE"
echo "  Per-run logs: $LOG_DIR/"
echo "  Dynamics:     $DYN_DIR/"
echo "============================================"
