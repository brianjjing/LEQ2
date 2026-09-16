#!/bin/bash
set -e
# reward_penalty_coef hyperparameter search for LEQ + DBG (NeuralODE) on
# sparse Walker2d (73% sparse), SEED 42 ONLY.
#
# Uses the sparse offline pickle for LEQ:
#   /public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl
#
# Pipeline:
#   1. Verify a PRE-TRAINED transition-dynamics ensemble already exists at
#      DYN_DIR (this script does NOT train dynamics).
#   2. Verify a PRE-TRAINED NeuralODE guardian already exists at
#      GUARDIAN_PATH for seed 42 (this script does NOT train the guardian --
#      the guardian is a fixed density model independent of
#      reward_penalty_coef, so the SAME guardian is reused for every
#      coefficient below).
#   3. Run LEQ once per value in COEFS (the reward_penalty_coef / DBG penalty
#      coefficient, i.e. train_LEQ.py's --guardian_penalty_coef), ALL IN
#      PARALLEL, one job per GPU, seed 42 only.
#
# GPU layout (fixed): GPUs 0-3 -- NOTE: the hopper search scripts also default to GPUs
#   0-3, so don't run a hopper and a walker2d search at the same time unless
#   you override GPUS below (the halfcheetah search scripts use GPUs 4-7 and
#   are safe to run alongside either one).
#
# Usage (from LEQ2 root):
#   bash bash_scr/leq_dbg_penalty_search/LEQ_DBG_SEARCH_WALKER2D_NEURALODE.sh
#
# Env overrides:
#   DATASET_PATH, SEED, COEFS, GPUS, DYN_BASE_DIR, GUARDIAN_ROOT, LEQ2_DIR, LEQ2_ENV

TASK="${TASK:-walker2d-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
DYN_TAG="${DYN_TAG:-walker2d-medium-expert-v2_sparse_73}"
GUARDIAN_ROOT="${GUARDIAN_ROOT:-/public/gormpo/models/walker2d_medium_expert_sparse_3}"
DBG="neuralode"

# reward_penalty_coef hyperparameter search -- seed 42 only.
SEED="${SEED:-42}"

if [ -z "${COEFS+x}" ] || [ -z "$COEFS" ]; then
    COEFS=(0.1 0.3 0.5 0.7)
else
    # shellcheck disable=SC2206
    COEFS=($COEFS)
fi

if [ -z "${GPUS+x}" ] || [ -z "$GPUS" ]; then
    GPUS=(0 1 2 3)
else
    # shellcheck disable=SC2206
    GPUS=($GPUS)
fi

if [ "${#GPUS[@]}" -lt "${#COEFS[@]}" ]; then
    echo "ERROR: need at least ${#COEFS[@]} GPUs in GPUS, got ${#GPUS[@]}"
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
GUARDIAN_PATH="$GUARDIAN_ROOT/neuralODE"
SAVE_DIR="./tmp/EP_dbg_penalty_search/${DBG}/${DYN_TAG}"
LOG_DIR="$SAVE_DIR/logs"
RESULTS_FILE="$SAVE_DIR/results_${DYN_TAG}_${DBG}.csv"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "LEQ + DBG (NeuralODE) reward_penalty_coef search on sparse Walker2d"
echo "  Env task:      $TASK"
echo "  Dataset:       $DATASET_PATH"
echo "  Dynamics tag:  $DYN_TAG"
echo "  Seed:          $SEED"
echo "  Guardian:      $GUARDIAN_PATH"
echo "  Coefs:         ${COEFS[*]}"
echo "  GPUs:          ${GPUS[*]}"
echo "  Dynamics base: $DYN_BASE_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "============================================"
echo ""

echo "Step 1/3: Verify pre-trained dynamics ensemble -> $DYN_DIR"
if [ -f "$DYN_DIR/dynamics.pth" ] && [ -f "$DYN_DIR/mu.npy" ] && [ -f "$DYN_DIR/std.npy" ]; then
    echo "  Found dynamics.pth, mu.npy, std.npy. Not training -- reusing as-is."
else
    echo "ERROR: pre-trained dynamics ensemble not found at $DYN_DIR"
    echo "  Expected files: dynamics.pth, mu.npy, std.npy"
    echo "  This script does not train dynamics -- copy a pre-trained ensemble there first."
    exit 1
fi
echo ""

echo "Step 2/3: Verify pre-trained NeuralODE guardian -> $GUARDIAN_PATH"
p="$GUARDIAN_PATH"
if [ -f "${p}/model.pt" ] && [ -f "${p}/metadata.pkl" ]; then
    echo "  Found guardian. Not training -- reusing as-is."
else
    echo "ERROR: pre-trained NeuralODE guardian not found at $GUARDIAN_PATH"
    echo "  Expected files: <path>/model.pt, <path>/metadata.pkl"
    echo "  This script does not train the guardian -- train it first (e.g. via bash_scr/leq_dbg_new)."
    exit 1
fi
echo ""

echo "Step 3/3: LEQ reward_penalty_coef sweep (parallel, seed=$SEED, guardian_type=$DBG)"
declare -A PIDS
declare -A GPU_FOR_COEF
for i in "${!COEFS[@]}"; do
    coef="${COEFS[$i]}"
    gpu="${GPUS[$i]}"
    GPU_FOR_COEF[$coef]="$gpu"
    logfile="$LOG_DIR/coef_${coef}.log"
    echo "  Launching reward_penalty_coef=$coef on GPU $gpu -> $logfile"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "export LD_LIBRARY_PATH=\"$MUJOCO_LD_PATH:\$LD_LIBRARY_PATH\" && \
            cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$gpu' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$SEED' \
            --expectile 0.5 \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --save_dir '$SAVE_DIR/' \
            --guardian_model_name '$GUARDIAN_PATH' \
            --guardian_type '$DBG' \
            --guardian_penalty_coef '$coef' \
            --eval_episodes 10 \
            --debug" > "$logfile" 2>&1 &
    PIDS[$coef]=$!
done

FAILED_COEFS=()
for coef in "${COEFS[@]}"; do
    if wait "${PIDS[$coef]}"; then
        echo "  reward_penalty_coef=$coef finished (GPU ${GPU_FOR_COEF[$coef]})"
    else
        status=$?
        echo "  WARNING: reward_penalty_coef=$coef FAILED (exit $status, GPU ${GPU_FOR_COEF[$coef]}) -- see $LOG_DIR/coef_${coef}.log"
        FAILED_COEFS+=("$coef")
    fi
done
echo ""

echo "Collecting results -> $RESULTS_FILE"
echo "reward_penalty_coef,final_score,final_length" > "$RESULTS_FILE"
for coef in "${COEFS[@]}"; do
    logfile="$LOG_DIR/coef_${coef}.log"
    line=$(grep "Final score:" "$logfile" | tail -n 1)
    if [ -n "$line" ]; then
        score=$(echo "$line" | sed -n 's/.*Final score: \([^ ]*\) Final length:.*/\1/p')
        length=$(echo "$line" | sed -n 's/.*Final length: \(.*\)/\1/p')
        echo "$coef,$score,$length" >> "$RESULTS_FILE"
    else
        echo "$coef,NA,NA" >> "$RESULTS_FILE"
        echo "  WARNING: no 'Final score:' line found in $logfile"
    fi
done

BEST_LINE=$(tail -n +2 "$RESULTS_FILE" | grep -v ',NA,NA$' | sort -t, -k2 -g -r | head -n 1)
if [ -n "$BEST_LINE" ]; then
    BEST_COEF=$(echo "$BEST_LINE" | cut -d, -f1)
    BEST_SCORE=$(echo "$BEST_LINE" | cut -d, -f2)
else
    BEST_COEF="N/A"
    BEST_SCORE="N/A"
fi

echo ""
echo "============================================"
echo "Done. Results (each score = train_LEQ.py's final eval, averaged over"
echo "--eval_episodes=10 episodes):"
column -s, -t "$RESULTS_FILE"
echo ""
echo ">>> BEST reward_penalty_coef for $DYN_TAG ($DBG): $BEST_COEF (final_score=$BEST_SCORE) <<<"
if [ "${#FAILED_COEFS[@]}" -gt 0 ]; then
    echo ""
    echo "!!! FAILED coefs (excluded from results above): ${FAILED_COEFS[*]}"
fi
echo ""
echo "  Results CSV: $RESULTS_FILE"
echo "  Per-run logs: $LOG_DIR/"
echo "  Dynamics:     $DYN_DIR/"
echo "  Guardian:     $GUARDIAN_PATH"
echo "============================================"
