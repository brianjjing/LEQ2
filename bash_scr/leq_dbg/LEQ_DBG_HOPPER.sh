#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ2_DIR="${LEQ2_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Detach from the terminal/SSH session so training survives it closing --
# same pattern as bash_scr/run_leq_medium.sh. Re-execs itself once under
# setsid+nohup with stdout/stderr going to a logfile; set DETACH=0 to run
# in the foreground instead.
if [ "${DETACH:-1}" = 1 ] && [ -z "${LEQ_DETACHED:-}" ]; then
    mkdir -p "$LEQ2_DIR/tmp/_runs"
    LOGFILE="$LEQ2_DIR/tmp/_runs/hopper_dbg_$(date +%m%d-%H%M%S).log"
    LEQ_DETACHED=1 setsid nohup bash "$0" "$@" >"$LOGFILE" 2>&1 </dev/null &
    echo "detached: pid $!"
    echo "  tail -f $LOGFILE"
    exit 0
fi
# LEQ + DBG (density-based guardian) on sparse Hopper-Medium-Expert
# (78% sparse), 3 seeds x 5 guardians.
#
# One of 5 density estimators backs the guardian each run: KDE, VAE, DDPM,
# RealNVP, NeuralODE. ("KDE", not "KAE" -- no KAE estimator exists anywhere
# in GORMPO/GORMPO_abiomed; GORMPO implements exactly these 5.)
#
# reward_penalty_coef below is GORMPO's own tuned coefficient per estimator
# for this dataset, read directly from
# GORMPO/configs/<estimator>/gormpo_hopper_medium_expert_sparse_3.yaml
#
# Pipeline (per seed):
#   1. Verify a PRE-TRAINED dynamics ensemble already exists (this script
#      does NOT train dynamics)
#   2. For each of the 5 guardians:
#        a. Train/reuse the density-model guardian      (GORMPO)
#        b. Train LEQ with that guardian's OOD penalty   (LEQ2)
#
# Usage (from LEQ2 root):
#   bash bash_scr/leq_dbg/LEQ_DBG_HOPPER.sh
#   CUDA_VISIBLE_DEVICES=2 DBG_TYPES="kde ddpm" bash bash_scr/leq_dbg/LEQ_DBG_HOPPER.sh
#
# Env overrides:
#   SEEDS, DBG_TYPES, DEVID_KDE, DEVID_DYN, LEQ2_ENV,
#   DYN_BASE_DIR, LEQ2_DIR, GORMPO_ROOT, DATASET_PATH, GUARDIAN_ROOT

TASK="${TASK:-hopper-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl}"
DIFFUSION_NPZ="${DIFFUSION_NPZ:-/public/d4rl/sparse_datasets/diffusion_processed/hopper_medium_expert_sparse_78_train.npz}"
GUARDIAN_ROOT="${GUARDIAN_ROOT:-/public/gormpo/models/hopper_medium_expert_sparse_3}"
DYN_TAG="${DYN_TAG:-hopper-medium-expert-v2_sparse_78}"
CONFIG_TAG="hopper_medium_expert_sparse_3"  # GORMPO's config filename stem for this dataset

# estimator -> reward_penalty_coef (GORMPO's tuned value for this dataset)
declare -A REWARD_PENALTY_COEF=(
    [kde]=0.05
    [vae]=0.3
    [realnvp]=0.8
    [neuralode]=0.5
    [ddpm]=0.05
)

DEVID_KDE="${DEVID_KDE:-0}"
DEVID_DYN="${DEVID_DYN:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVID_DYN"
fi

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(42)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

if [ -z "${DBG_TYPES+x}" ] || [ -z "$DBG_TYPES" ]; then
    DBG_TYPES=(kde vae realnvp neuralode ddpm)
else
    # shellcheck disable=SC2206
    DBG_TYPES=($DBG_TYPES)
fi

GORMPO_ROOT="${GORMPO_ROOT:-$LEQ2_DIR/../GORMPO}"
DYN_BASE_DIR="${DYN_BASE_DIR:-/public/gormpo/models/dynamics-ensemble}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: sparse dataset not found: $DATASET_PATH"
    exit 1
fi

echo "============================================"
echo "LEQ + DBG: $TASK (sparse Hopper)"
echo "  GORMPO:        $GORMPO_ROOT"
echo "  Dynamics base: $DYN_BASE_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "  Seeds:         ${SEEDS[*]}"
echo "  DBG types:     ${DBG_TYPES[*]}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

# --- Guardian save path for a given (dbg type, seed) ---
guardian_path() {
    local dbg="$1" seed="$2"
    case "$dbg" in
        kde)       echo "$GUARDIAN_ROOT/kde_${seed}" ;;
        vae)       echo "$GUARDIAN_ROOT/vae_${seed}" ;;
        realnvp)   echo "$GUARDIAN_ROOT/realnvp_${seed}" ;;
        neuralode) echo "$GUARDIAN_ROOT/neuralODE" ;;  # one shared model, not trained per-seed
        ddpm)      echo "$GUARDIAN_ROOT/diffusion_${seed}" ;;
    esac
}

# --- 0 (true) if the guardian at $2 (path) for type $1 already exists ---
guardian_exists() {
    local dbg="$1" path="$2"
    case "$dbg" in
        kde)         [ -f "${path}.faiss" ] && [ -f "${path}_metadata.pkl" ] ;;
        vae|realnvp) [ -f "${path}_model.pth" ] && [ -f "${path}_meta_data.pkl" ] ;;
        neuralode)   [ -f "${path}/model.pt" ] && [ -f "${path}/metadata.pkl" ] ;;
        ddpm)        [ -f "${path}/checkpoint.pt" ] ;;
    esac
}

# --- Train the guardian of type $1 at path $2 for seed $3 ---
train_guardian() {
    local dbg="$1" path="$2" seed="$3"
    mkdir -p "$(dirname "$path")"
    case "$dbg" in
        kde)
            (cd "$GORMPO_ROOT" && python kde_module/kde.py \
                --config "configs/kde/${CONFIG_TAG}.yaml" \
                --seed "$seed" --save_path "$path" --devid "$DEVID_KDE")
            ;;
        vae)
            (cd "$GORMPO_ROOT" && python vae_module/vae.py \
                --config "configs/vae/${CONFIG_TAG}.yaml" \
                --seed "$seed" --model_save_path "$path" --device "cuda:$DEVID_KDE")
            ;;
        realnvp)
            (cd "$GORMPO_ROOT" && python realnvp_module/realnvp.py \
                --config "configs/realnvp/${CONFIG_TAG}.yaml" \
                --seed "$seed" --model_save_path "$path" --device "cuda:$DEVID_KDE")
            ;;
        neuralode)
            (cd "$GORMPO_ROOT" && python neuralODE/neural_ode_density.py \
                --config "configs/neuralODE/${CONFIG_TAG}_train.yaml" \
                --seed "$seed" --out "$path")
            ;;
        ddpm)
            (cd "$GORMPO_ROOT/diffusion" && python ddim_training.py \
                --npz "$DIFFUSION_NPZ" --out "$path" --seed "$seed")
            ;;
    esac
}

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    # --- Dynamics ensemble: PRE-TRAINED, reused as-is (never trained here) ---
    DYN_DIR="$DYN_BASE_DIR/${seed}/${DYN_TAG}"
    echo "Dynamics ensemble -> $DYN_DIR"
    if [ -f "$DYN_DIR/dynamics.pth" ] && [ -f "$DYN_DIR/mu.npy" ] && [ -f "$DYN_DIR/std.npy" ]; then
        echo "  Found pre-trained dynamics. Not training -- reusing as-is."
    else
        echo "  ERROR: pre-trained dynamics ensemble not found at $DYN_DIR"
        echo "    Expected files: dynamics.pth, mu.npy, std.npy"
        echo "    This script does not train dynamics -- copy a pre-trained ensemble there first."
        exit 1
    fi
    echo ""

    for dbg in "${DBG_TYPES[@]}"; do
        coef="${REWARD_PENALTY_COEF[$dbg]}"
        path="$(guardian_path "$dbg" "$seed")"

        echo "------ dbg = $dbg (seed $seed, penalty_coef=$coef) ------"

        # Resume: skip this (seed, dbg) cell entirely if LEQ already finished.
        LEQ_DONE="$LEQ2_DIR/tmp/EP_dbg/${dbg}/models/$TASK/$seed/0.5/${seed}_${LEQ_MAX_STEPS:-1000000}.pkl"
        if [ -f "$LEQ_DONE" ]; then
            echo "  LEQ already trained ($LEQ_DONE) -- skipping."
            echo ""
            continue
        fi

        echo "Guardian ($dbg) -> $path"
        if guardian_exists "$dbg" "$path"; then
            echo "  Guardian already exists, skipping."
        else
            train_guardian "$dbg" "$path" "$seed"
            echo "  Guardian training complete"
        fi
        echo ""

        echo "LEQ + $dbg guardian (penalty_coef=$coef)"
        conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
            "cd '$LEQ2_DIR' && \
                CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
                PYTHONPATH='.' python train/train_LEQ.py \
                --env_name '$TASK' \
                --seed '$seed' \
                --expectile 0.5 \
                --dataset_path '$DATASET_PATH' \
                --load_dir '$DYN_DIR' \
                --save_dir './tmp/EP_dbg/${dbg}/' \
                --guardian_model_name '$path' \
                --guardian_type '$dbg' \
                --guardian_penalty_coef '$coef' \
                --debug"
        echo "  LEQ training complete (seed $seed, dbg $dbg)"
        echo ""
    done
done

echo "============================================"
echo "Done. Checkpoints in LEQ2/tmp/EP_dbg/<dbg>/models/${TASK}/"
echo "Guardians in ${GUARDIAN_ROOT}/"
echo "============================================"
