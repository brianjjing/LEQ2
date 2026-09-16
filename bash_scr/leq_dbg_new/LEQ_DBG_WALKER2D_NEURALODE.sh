#!/bin/bash
set -e

# LEQ + DBG (NEURALODE) on sparse Walker2d (73% sparse), 3 seeds launched IN PARALLEL
# (one self-detaching background job per seed, one GPU each).
#
# Split out of bash_scr/leq_dbg/LEQ_DBG_WALKER2D.sh, which loops over all 5
# estimators sequentially per seed. This script is scoped to one estimator so
# all 3 seeds can run concurrently instead -- same idea as the ad hoc parallel
# MOBILE sweep script (per-seed GPU map, launch all seeds, wait), applied to
# LEQ2's own train/train_LEQ.py rather than a different trainer.
#
# reward_penalty_coef is GORMPO's own tuned value for this (dataset,
# estimator) pair -- same number as bash_scr/leq_dbg/LEQ_DBG_WALKER2D.sh's
# REWARD_PENALTY_COEF[neuralode], sourced from
# GORMPO/configs/neuralODE/gormpo_walker2d_medium_expert_sparse_3.yaml.
#
# Pipeline (per seed, in parallel):
#   1. Verify a PRE-TRAINED dynamics ensemble already exists (this script
#      does NOT train dynamics)
#   2. Train/reuse the density-model guardian           (GORMPO)
#   3. Train LEQ with that guardian's OOD penalty        (LEQ2)
#
# Usage (from LEQ2 root):
#   bash bash_scr/leq_dbg_new/LEQ_DBG_WALKER2D_NEURALODE.sh
#   GPU_IDS="0 1 2" bash bash_scr/leq_dbg_new/LEQ_DBG_WALKER2D_NEURALODE.sh
#   DETACH=0 bash bash_scr/leq_dbg_new/LEQ_DBG_WALKER2D_NEURALODE.sh   # foreground, all 3 seeds interleaved
#
# Env overrides:
#   SEEDS, GPU_IDS, LEQ2_ENV, DYN_BASE_DIR, LEQ2_DIR, GORMPO_ROOT,
#   DATASET_PATH, GUARDIAN_ROOT, REWARD_PENALTY_COEF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ2_DIR="${LEQ2_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

TASK="walker2d-medium-expert-v2"
DBG="neuralode"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
DIFFUSION_NPZ="${DIFFUSION_NPZ:-/public/d4rl/sparse_datasets/diffusion_processed/walker2d_medium_expert_sparse_73_train.npz}"
GUARDIAN_ROOT="${GUARDIAN_ROOT:-/public/gormpo/models/walker2d_medium_expert_sparse_3}"
DYN_TAG="walker2d-medium-expert-v2_sparse_73"
CONFIG_TAG="walker2d_medium_expert_sparse_3"
REWARD_PENALTY_COEF="${REWARD_PENALTY_COEF:-0.05}"  # GORMPO's tuned value, see header

GORMPO_ROOT="${GORMPO_ROOT:-$LEQ2_DIR/../GORMPO}"
DYN_BASE_DIR="${DYN_BASE_DIR:-/public/gormpo/models/dynamics-ensemble}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(42)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi
if [ -z "${GPU_IDS+x}" ] || [ -z "$GPU_IDS" ]; then
    GPU_IDS=(0 1 2)
else
    # shellcheck disable=SC2206
    GPU_IDS=($GPU_IDS)
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: sparse dataset not found: $DATASET_PATH"
    exit 1
fi

# --- Guardian save path for seed $1 ---
guardian_path() {
    local seed="$1"
    echo "$GUARDIAN_ROOT/neuralODE"
}

# --- 0 (true) if the guardian at path $1 already exists ---
guardian_exists() {
    local path="$1"
    [ -f "${path}/model.pt" ] && [ -f "${path}/metadata.pkl" ]
}

# --- Train the guardian at path $1 for seed $2 ---
train_guardian() {
    local path="$1" seed="$2"
    mkdir -p "$(dirname "$path")"
    (cd "$GORMPO_ROOT" && python neuralODE/neural_ode_density.py \
        --config "configs/neuralODE/${CONFIG_TAG}_train.yaml" \
        --seed "$seed" --out "$path")
}

# --- Full pipeline for one seed: dynamics -> guardian -> LEQ ---
run_seed() {
    local seed="$1"
    echo "=========================================="
    echo ">>> seed = $seed (dbg=$DBG, gpu=$CUDA_VISIBLE_DEVICES)"
    echo "=========================================="

    # Resume: skip entirely if LEQ already finished (same check as LEQ_DBG_WALKER2D.sh).
    LEQ_DONE="$LEQ2_DIR/tmp/EP_dbg/${DBG}/models/$TASK/$seed/0.5/${seed}_${LEQ_MAX_STEPS:-1000000}.pkl"
    if [ -f "$LEQ_DONE" ]; then
        echo "  LEQ already trained ($LEQ_DONE) -- skipping."
        return 0
    fi

    # --- Dynamics ensemble: PRE-TRAINED, reused as-is (never trained here) ---
    DYN_DIR="$DYN_BASE_DIR/${seed}/${DYN_TAG}"
    if [ -f "$DYN_DIR/dynamics.pth" ] && [ -f "$DYN_DIR/mu.npy" ] && [ -f "$DYN_DIR/std.npy" ]; then
        echo "  Found pre-trained dynamics -> $DYN_DIR"
    else
        echo "  ERROR: pre-trained dynamics ensemble not found at $DYN_DIR"
        echo "    Expected files: dynamics.pth, mu.npy, std.npy"
        echo "    This script does not train dynamics -- copy a pre-trained ensemble there first."
        exit 1
    fi

    # --- Guardian ---
    # ponytail: same flock reasoning -- for this estimator specifically the
    # guardian path may be shared across seeds (neuralode is), so the 3
    # parallel seeds launched by *this* script would otherwise race on it.
    path="$(guardian_path "$seed")"
    GUARDIAN_LOCK="$LEQ2_DIR/tmp/_locks/guardian_${DBG}_${TASK}_$(basename "$path").lock"
    mkdir -p "$(dirname "$GUARDIAN_LOCK")"
    (
        flock -x 9
        if guardian_exists "$path"; then
            echo "  Guardian already exists -> $path"
        else
            echo "  Training guardian ($DBG) -> $path"
            train_guardian "$path" "$seed"
            echo "  Guardian training complete"
        fi
    ) 9>"$GUARDIAN_LOCK"

    # --- LEQ ---
    echo "  LEQ + $DBG guardian (penalty_coef=$REWARD_PENALTY_COEF)"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile 0.5 \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --save_dir './tmp/EP_dbg/${DBG}/' \
            --guardian_model_name '$path' \
            --guardian_type '$DBG' \
            --guardian_penalty_coef '$REWARD_PENALTY_COEF' \
            --debug"
    echo "  LEQ training complete (seed $seed)"
}

if [ -n "${SEED:-}" ]; then
    # Worker mode: one seed, pinned to one GPU, self-detaching.
    if [ "${DETACH:-1}" = 1 ] && [ -z "${LEQ_DETACHED:-}" ]; then
        mkdir -p "$LEQ2_DIR/tmp/_runs"
        LOGFILE="$LEQ2_DIR/tmp/_runs/WALKER2D_neuralode_seed${SEED}_$(date +%m%d-%H%M%S).log"
        LEQ_DETACHED=1 setsid nohup bash "$0" >"$LOGFILE" 2>&1 </dev/null &
        echo "seed $SEED detached: pid $!  (log: $LOGFILE)"
        exit 0
    fi
    run_seed "$SEED"
    exit 0
fi

# Launcher mode: fan out one self-detaching worker per seed, then exit.
echo "============================================"
echo "LEQ + DBG (neuralode): $TASK"
echo "  GORMPO:        $GORMPO_ROOT"
echo "  Dynamics base: $DYN_BASE_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "  Seeds:         ${SEEDS[*]}"
echo "  GPU_IDS:       ${GPU_IDS[*]}"
echo "  penalty_coef:  $REWARD_PENALTY_COEF"
echo "============================================"

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPU_IDS[$i]:-0}"
    SEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" DETACH="${DETACH:-1}" bash "$0" &
done
wait
echo "All ${#SEEDS[@]} seeds launched (each self-detached; tail logs under $LEQ2_DIR/tmp/_runs/)."
