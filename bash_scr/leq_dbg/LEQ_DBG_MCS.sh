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
    LOGFILE="$LEQ2_DIR/tmp/_runs/mcs_dbg_$(date +%m%d-%H%M%S).log"
    LEQ_DETACHED=1 setsid nohup bash "$0" "$@" >"$LOGFILE" 2>&1 </dev/null &
    echo "detached: pid $!"
    echo "  tail -f $LOGFILE"
    exit 0
fi
# LEQ + DBG (density-based guardian) on MCS (Abiomed), 1 seed x 5 guardians.
#
# One of 5 density estimators backs the guardian each run: KDE, VAE, DDPM,
# RealNVP, NeuralODE. ("KDE", not "KAE" -- no KAE estimator exists anywhere
# in GORMPO/GORMPO_abiomed; GORMPO implements exactly these 5.)
#
# reward_penalty_coef below is GORMPO_abiomed's own tuned coefficient per
# estimator for MCS, read directly from
# GORMPO_abiomed/cormpo/config/real/mbpo_<estimator>.yaml
#
# NOTE -- deliberate real/synthetic data split (interim, see below):
#   - The 5 guardians are trained on REAL clinical MCS data
#     (/public/gormpo/10min_1hr_all_data.pkl, via GORMPO_abiomed/cormpo's
#     config/<estimator>/real.yaml configs). Guardians are density models, so
#     they only need real state/action samples, not reward labels -- the raw
#     pkl works directly.
#   - LEQ itself (dynamics + policy) still trains on the existing SYNTHETIC
#     SAC-rollout dataset (synthetic_data/SAC_5000eps_stochastic.npz).
#     The real pkl is NOT an RL-tuple dataset: it's raw sensor windows
#     (train/val/test tensors, shape [N, 12, 13]), with no rewards or
#     terminals. Turning it into one means either driving AbiomedRLEnv's
#     WorldModel-predicted rollouts (the same mechanism the synthetic data
#     already uses) or hand-building reward labels over real transitions via
#     compute_reward_smooth/AbiomedRLEnv._compute_reward -- both are
#     meaningfully new pipelines, not done here. This script's guardians are
#     real; its LEQ policy tuples are synthetic. Revisit if/when a real
#     RL-ready MCS dataset exists.
#
# Pipeline (1 seed):
#   1. Train dynamics ensemble on the synthetic offline data  (OfflineRL-Kit2)
#   2. For each of the 5 guardians:
#        a. Train/reuse the density-model guardian on REAL data (GORMPO_abiomed/cormpo)
#        b. Train LEQ (on synthetic data) with that guardian's OOD penalty (LEQ2)
#
# Usage (from LEQ2 root):
#   bash bash_scr/leq_dbg/LEQ_DBG_MCS.sh
#   CUDA_VISIBLE_DEVICES=2 DBG_TYPES="kde ddpm" bash bash_scr/leq_dbg/LEQ_DBG_MCS.sh
#
# Env overrides:
#   DATASET_PATH, SEEDS, DBG_TYPES, DEVID_KDE, DEVID_DYN, LEQ2_ENV,
#   OFFLINERLKIT_DIR, LEQ2_DIR, GORMPO_ABIOMED_DIR, GUARDIAN_BASE

TASK="${TASK:-abiomed-v0}"

# estimator -> reward_penalty_coef (GORMPO_abiomed's tuned value for MCS)
declare -A REWARD_PENALTY_COEF=(
    [kde]=0.2
    [vae]=0.1
    [realnvp]=0.2
    [neuralode]=0.2
    [ddpm]=0.4
)

DEVID_KDE="${DEVID_KDE:-0}"
DEVID_DYN="${DEVID_DYN:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVID_DYN"
fi

# MCS only ever runs 1 seed; override with SEEDS="1 2 3" if you want more
# (only seed 42's guardians are pretrained for every estimator -- others
# fall back to Step 1 training).
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

GORMPO_ABIOMED_DIR="${GORMPO_ABIOMED_DIR:-$LEQ2_DIR/../GORMPO_abiomed}"
GORMPO_CORMPO="$GORMPO_ABIOMED_DIR/cormpo"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"
GUARDIAN_BASE="${GUARDIAN_BASE:-/public/gormpo/models/abiomed}"

# LEQ policy step: synthetic data (see NOTE above -- the real pkl has no
# actions/rewards/terminals).
DATASET_PATH="${DATASET_PATH:-$GORMPO_ABIOMED_DIR/synthetic_data/SAC_5000eps_stochastic.npz}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: MCS dataset not found: $DATASET_PATH"
    echo "Set DATASET_PATH to an abiomed offline .npz"
    exit 1
fi

echo "============================================"
echo "LEQ + DBG: $TASK (MCS / Abiomed)"
echo "  Guardians: REAL data. LEQ policy tuples: SYNTHETIC data (see NOTE in script)."
echo "  GORMPO_abiomed: $GORMPO_ABIOMED_DIR"
echo "  OfflineRL-Kit:  $OFFLINERLKIT_DIR"
echo "  LEQ2:           $LEQ2_DIR"
echo "  Dataset:        $DATASET_PATH"
echo "  Seeds:          ${SEEDS[*]}"
echo "  DBG types:      ${DBG_TYPES[*]}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

# --- Guardian save path for a given (dbg type, seed) ---
guardian_path() {
    local dbg="$1" seed="$2"
    case "$dbg" in
        kde)       echo "$GUARDIAN_BASE/trained_kde_${seed}/trained_kde_1" ;;
        vae)       echo "$GUARDIAN_BASE/trained_vae_${seed}/trained_vae_1" ;;
        realnvp)   echo "$GUARDIAN_BASE/trained_realnvp_${seed}/trained_realnvp_1" ;;
        neuralode) echo "$GUARDIAN_BASE/neuralODE/" ;;  # one shared model; trailing slash required
        ddpm)      echo "$GUARDIAN_BASE/trained_diffusion_${seed}" ;;
    esac
}

# --- 0 (true) if the guardian at $2 (path) for type $1 already exists ---
guardian_exists() {
    local dbg="$1" path="$2"
    case "$dbg" in
        kde)         [ -f "${path}.faiss" ] && [ -f "${path}_metadata.pkl" ] ;;
        vae|realnvp) [ -f "${path}_model.pth" ] && [ -f "${path}_meta_data.pkl" ] ;;
        neuralode)   [ -f "${path}_model.pt" ] && [ -f "${path}_metadata.pkl" ] ;;
        ddpm)        [ -f "${path}/checkpoint.pt" ] ;;
    esac
}

# --- Train the guardian of type $1 at path $2 for seed $3, on REAL data ---
train_guardian() {
    local dbg="$1" path="$2" seed="$3"
    mkdir -p "$(dirname "$path")"
    case "$dbg" in
        kde)
            (cd "$GORMPO_CORMPO" && python mbpo_kde/kde.py \
                --config config/kde/real.yaml \
                --seed "$seed" --save_path "$path" --devid "$DEVID_KDE")
            ;;
        vae)
            (cd "$GORMPO_CORMPO" && python vae_module/vae.py \
                --config config/vae/real.yaml \
                --seed "$seed" --model_save_path "$path" --device "cuda:$DEVID_KDE")
            ;;
        realnvp)
            (cd "$GORMPO_CORMPO" && python realnvp_module/realnvp.py \
                --config config/realnvp/real.yaml \
                --seed "$seed" --model_save_path "$path" --device "cuda:$DEVID_KDE")
            ;;
        neuralode)
            # config/neuralode/real.yaml already embeds model_save_path.
            (cd "$GORMPO_CORMPO" && python neuralode_module/train_neuralode.py \
                --config config/neuralode/real.yaml --seed "$seed")
            ;;
        ddpm)
            # config/diffusion/real.yaml already embeds model_save_path.
            (cd "$GORMPO_CORMPO" && python diffusion_module/train_diffusion.py \
                --config config/diffusion/real.yaml --seed "$seed")
            ;;
    esac
}

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    # --- Dynamics ensemble (synthetic data): once per seed, shared across all 5 guardians ---
    DYN_DIR="$OFFLINERLKIT_DIR/models/dynamics-ensemble/${seed}/${TASK}"
    echo "Dynamics ensemble (synthetic data) -> $DYN_DIR"
    if [ -d "$DYN_DIR" ] && [ -n "$(ls -A "$DYN_DIR" 2>/dev/null)" ]; then
        echo "  Dynamics already exist, skipping."
    else
        conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
            "cd '$OFFLINERLKIT_DIR' && \
                CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
                PYTHONPATH='$OFFLINERLKIT_DIR' \
                python run_example/run_dynamics.py \
                    --task '$TASK' --seed '$seed' \
                    --dataset-path '$DATASET_PATH'"
        echo "  Dynamics training complete"
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

        echo "Guardian ($dbg, REAL data) -> $path"
        if guardian_exists "$dbg" "$path"; then
            echo "  Guardian already exists, skipping."
        else
            train_guardian "$dbg" "$path" "$seed"
            echo "  Guardian training complete"
        fi
        echo ""

        echo "LEQ + $dbg guardian (penalty_coef=$coef, SYNTHETIC policy data)"
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
echo "Guardians (real data) in ${GUARDIAN_BASE}/"
echo "============================================"
