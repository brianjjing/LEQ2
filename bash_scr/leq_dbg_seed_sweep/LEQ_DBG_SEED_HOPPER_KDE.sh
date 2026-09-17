#!/bin/bash
set -e
# LEQ + DBG (KDE) with a FIXED reward_penalty_coef, across SEEDS 42 123 456,
# on sparse Hopper (78% sparse).
#
# Uses the sparse offline pickle for LEQ:
#   /public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl
#
# Pipeline:
#   1. Verify a PRE-TRAINED transition-dynamics ensemble already exists at
#      DYN_DIR (this script does NOT train dynamics). Dynamics are only
#      trained for seed 42 -- that SAME ensemble is reused for every LEQ
#      seed below; only the LEQ agent's own --seed varies.
#   2. Verify a PRE-TRAINED KDE guardian exists per seed at
#      GUARDIAN_ROOT/kde_<seed> (this script does NOT train guardians).
#   3. Run LEQ once per seed in SEEDS, ALL IN PARALLEL, one job per GPU,
#      fixed reward_penalty_coef (from LAMBDA_TABLE["hopper-m-e-sparse"]["KDE"]).
#
# GPU layout (default): GPUs 0-2 -- NOTE: the walker2d seed-sweep scripts
#   also default to GPUs 0-2, so don't run a hopper and a walker2d seed
#   sweep at the same time unless you override GPUS below (the halfcheetah
#   seed-sweep scripts default to GPUs 1-3 and mostly overlap too -- check
#   GPU availability before running multiple sweeps concurrently).
#
# Usage (from LEQ2 root):
#   bash bash_scr/leq_dbg_seed_sweep/LEQ_DBG_SEED_HOPPER_KDE.sh
#
# Env overrides:
#   DATASET_PATH, SEEDS, COEF, GPUS, DYN_SEED, DYN_BASE_DIR, GUARDIAN_ROOT, LEQ2_DIR, LEQ2_ENV

TASK="${TASK:-hopper-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl}"
DYN_TAG="${DYN_TAG:-hopper-medium-expert-v2_sparse_78}"
GUARDIAN_ROOT="${GUARDIAN_ROOT:-/public/gormpo/models/hopper_medium_expert_sparse_3}"
DBG="kde"

# Fixed reward_penalty_coef -- LAMBDA_TABLE["hopper-m-e-sparse"]["KDE"].
COEF="${COEF:-0.05}"

# Dynamics ensemble is only trained for seed 42; reused for every LEQ seed below.
DYN_SEED="${DYN_SEED:-42}"

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(42 123 456)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

if [ -z "${GPUS+x}" ] || [ -z "$GPUS" ]; then
    GPUS=(0 1 2)
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
SAVE_DIR="./tmp/EP_dbg_seed_sweep/${DBG}/${DYN_TAG}"
LOG_DIR="$SAVE_DIR/logs"
RESULTS_FILE="$SAVE_DIR/results_${DYN_TAG}_${DBG}_coef${COEF}.csv"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "LEQ + DBG (KDE) seed sweep on sparse Hopper"
echo "  Env task:      $TASK"
echo "  Dataset:       $DATASET_PATH"
echo "  Dynamics tag:  $DYN_TAG"
echo "  Dynamics seed: $DYN_SEED (fixed, reused for all LEQ seeds)"
echo "  Fixed coef:    $COEF"
echo "  Seeds:         ${SEEDS[*]}"
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

echo "Step 2/3: Verify pre-trained KDE guardians (one per seed)"
declare -A GUARDIAN_FOR_SEED
for seed in "${SEEDS[@]}"; do
    p="$GUARDIAN_ROOT/kde_${seed}"
    if [ -f "${p}.faiss" ] && [ -f "${p}_metadata.pkl" ]; then
        echo "  seed=$seed -> $p (found)"
        GUARDIAN_FOR_SEED[$seed]="$p"
    else
        echo "ERROR: pre-trained KDE guardian not found at $p"
        echo "  Expected files: <path>.faiss, <path>_metadata.pkl"
        echo "  This script does not train the guardian -- train it first (e.g. via bash_scr/leq_dbg_new)."
        exit 1
    fi
done
echo ""

echo "Step 3/3: LEQ seed sweep (parallel, fixed reward_penalty_coef=$COEF, guardian_type=$DBG)"
declare -A PIDS
declare -A GPU_FOR_SEED
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    GPU_FOR_SEED[$seed]="$gpu"
    guardian_path="${GUARDIAN_FOR_SEED[$seed]}"
    logfile="$LOG_DIR/seed_${seed}.log"
    echo "  Launching seed=$seed on GPU $gpu -> $logfile"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "export LD_LIBRARY_PATH=\"$MUJOCO_LD_PATH:\$LD_LIBRARY_PATH\" && \
            export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
            cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$gpu' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile 0.3 \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --save_dir '$SAVE_DIR/' \
            --guardian_model_name '$guardian_path' \
            --guardian_type '$DBG' \
            --guardian_penalty_coef '$COEF' \
            --eval_episodes 10 \
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
echo "seed,final_score,final_length" > "$RESULTS_FILE"
for seed in "${SEEDS[@]}"; do
    logfile="$LOG_DIR/seed_${seed}.log"
    line=$(grep "Final score:" "$logfile" | tail -n 1)
    if [ -n "$line" ]; then
        score=$(echo "$line" | sed -n 's/.*Final score: \([^ ]*\) Final length:.*/\1/p')
        length=$(echo "$line" | sed -n 's/.*Final length: \(.*\)/\1/p')
        echo "$seed,$score,$length" >> "$RESULTS_FILE"
    else
        echo "$seed,NA,NA" >> "$RESULTS_FILE"
        echo "  WARNING: no 'Final score:' line found in $logfile"
    fi
done

MEAN_STD=$(tail -n +2 "$RESULTS_FILE" | awk -F, '$2!="NA"{sum+=$2; sumsq+=$2*$2; n++} END{if(n>0){mean=sum/n; std=(n>1)?sqrt((sumsq-sum*sum/n)/(n-1)):0; printf "%.4f ± %.4f (n=%d)", mean, std, n} else print "N/A"}')

echo ""
echo "============================================"
echo "Done. Results (each score = train_LEQ.py's final eval, averaged over"
echo "--eval_episodes=10 episodes) for fixed reward_penalty_coef=$COEF:"
column -s, -t "$RESULTS_FILE"
echo ""
echo ">>> $DYN_TAG ($DBG, coef=$COEF) final_score across seeds: $MEAN_STD <<<"
if [ "${#FAILED_SEEDS[@]}" -gt 0 ]; then
    echo ""
    echo "!!! FAILED seeds (excluded from results above): ${FAILED_SEEDS[*]}"
fi
echo ""
echo "  Results CSV: $RESULTS_FILE"
echo "  Per-run logs: $LOG_DIR/"
echo "  Dynamics:     $DYN_DIR/"
echo "============================================"
