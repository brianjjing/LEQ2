#!/bin/bash
set -e
# Train LEQ on sparse Walker2d-Medium-Expert with the expectile chosen by the
# expectile hyperparameter search (bash_scr/expectile_search), WITHOUT a
# density-based guardian (guardian_penalty_coef=0, no guardian model loaded).
#
# Uses the sparse offline pickle for LEQ:
#   /public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl
#
# Pipeline:
#   1. Verify the PRE-TRAINED transition-dynamics ensemble already exists at
#      DYN_DIR (this script does NOT train dynamics). That ensemble was only
#      ever trained for seed 42 (during the expectile search), so it is
#      reused as-is for every training seed below -- the dynamics model does
#      NOT get retrained per seed.
#   2. Run LEQ once per seed in SEEDS with the fixed EXPECTILE, ALL IN
#      PARALLEL, one job per GPU.
#
# GPU layout (fixed, so this script never collides with itself):
#   GPUs 2-3, one per seed. NOTE: LEQ_SPARSE_HOPPER.sh uses GPUs 0-1 and
#   LEQ_SPARSE_HALFCHEETAH.sh uses GPUs 4-5, so all three can run together.
#
# Usage (from LEQ2 root):
#   bash bash_scr/leq_sparse_train/LEQ_SPARSE_WALKER2D.sh
#
# Env overrides:
#   DATASET_PATH, EXPECTILE, DYN_SEED, SEEDS, GPUS, DYN_BASE_DIR, LEQ2_DIR, LEQ2_ENV

TASK="${TASK:-walker2d-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
DYN_TAG="${DYN_TAG:-walker2d-medium-expert-v2_sparse_73}"

# Expectile chosen by the expectile search for Walker2d.
EXPECTILE="${EXPECTILE:-0.4}"

# Seed the pre-trained dynamics ensemble was trained with (do not change --
# no ensemble exists for other seeds, and it does not need to be retrained).
DYN_SEED="${DYN_SEED:-42}"

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(123 456)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

if [ -z "${GPUS+x}" ] || [ -z "$GPUS" ]; then
    GPUS=(2 3)
else
    # shellcheck disable=SC2206
    GPUS=($GPUS)
fi

if [ "${#GPUS[@]}" -lt "${#SEEDS[@]}" ]; then
    echo "ERROR: need at least ${#SEEDS[@]} GPUs in GPUS, got ${#GPUS[@]}"
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

DYN_DIR="$DYN_BASE_DIR/${DYN_SEED}/${DYN_TAG}"
SAVE_DIR="./tmp/EP_sparse/${DYN_TAG}"
LOG_DIR="$SAVE_DIR/logs"
RESULTS_FILE="$SAVE_DIR/results_${DYN_TAG}.csv"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "LEQ training on sparse Walker2d (no guardian)"
echo "  Env task:      $TASK"
echo "  Dataset:       $DATASET_PATH"
echo "  Dynamics tag:  $DYN_TAG"
echo "  Dynamics seed: $DYN_SEED (reused for all training seeds below)"
echo "  Expectile:     $EXPECTILE"
echo "  Seeds:         ${SEEDS[*]}"
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

echo "Step 2/2: LEQ training sweep over seeds (parallel, no guardian)"
declare -A PIDS
declare -A GPU_FOR_SEED
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    GPU_FOR_SEED[$seed]="$gpu"
    logfile="$LOG_DIR/seed_${seed}.log"
    echo "  Launching seed=$seed on GPU $gpu -> $logfile"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "export LD_LIBRARY_PATH=\"$MUJOCO_LD_PATH:\$LD_LIBRARY_PATH\" && \
            cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$gpu' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile '$EXPECTILE' \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --guardian_penalty_coef 0.0 \
            --eval_episodes 10 \
            --save_dir '$SAVE_DIR/' \
            --debug" > "$logfile" 2>&1 &
    PIDS[$seed]=$!
done

FAILED_SEEDS=()
for seed in "${SEEDS[@]}"; do
    if wait "${PIDS[$seed]}"; then
        echo "  seed=$seed finished (GPU ${GPU_FOR_SEED[$seed]})"
    else
        status=$?
        echo "  WARNING: seed=$seed FAILED (exit $status, GPU ${GPU_FOR_SEED[$seed]}) -- see $LOG_DIR/seed_${seed}.log"
        FAILED_SEEDS+=("$seed")
    fi
done
echo ""

echo "Collecting results -> $RESULTS_FILE"
echo "seed,expectile,final_score,final_length" > "$RESULTS_FILE"
for seed in "${SEEDS[@]}"; do
    logfile="$LOG_DIR/seed_${seed}.log"
    line=$(grep "Final score:" "$logfile" | tail -n 1)
    if [ -n "$line" ]; then
        score=$(echo "$line" | sed -n 's/.*Final score: \([^ ]*\) Final length:.*/\1/p')
        length=$(echo "$line" | sed -n 's/.*Final length: \(.*\)/\1/p')
        echo "$seed,$EXPECTILE,$score,$length" >> "$RESULTS_FILE"
    else
        echo "$seed,$EXPECTILE,NA,NA" >> "$RESULTS_FILE"
        echo "  WARNING: no 'Final score:' line found in $logfile"
    fi
done

echo ""
echo "============================================"
echo "Done. Results (each score = train_LEQ.py's final eval, averaged over"
echo "--eval_episodes=10 episodes):"
column -s, -t "$RESULTS_FILE"
if [ "${#FAILED_SEEDS[@]}" -gt 0 ]; then
    echo ""
    echo "!!! FAILED seeds (excluded from results above): ${FAILED_SEEDS[*]}"
fi
echo ""
echo "  Results CSV: $RESULTS_FILE"
echo "  Per-run logs: $LOG_DIR/"
echo "  Dynamics:     $DYN_DIR/"
echo "============================================"
