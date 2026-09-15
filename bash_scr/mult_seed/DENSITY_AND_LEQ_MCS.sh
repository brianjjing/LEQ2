#!/bin/bash
set -e
# LEQ + KDE density guardian on MCS (Abiomed).
#
# Pipeline (per seed):
#   1. Train KDE guardian                          (GORMPO_abiomed mbpo_kde)
#   2. Train dynamics ensemble on MCS .npz       (OfflineRL-Kit2)
#   3. Train LEQ with guardian OOD penalty         (LEQ2)
#
# Usage (from LEQ2 root):
#   bash bash_scr/mult_seed/DENSITY_AND_LEQ_MCS.sh
#
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/DENSITY_AND_LEQ_MCS.sh
#
# Env overrides:
#   DATASET_PATH, SEEDS, GUARDIAN_PENALTY_COEF, DEVID_KDE, DEVID_DYN,
#   LEQ2_ENV, OFFLINERLKIT_DIR, LEQ2_DIR, GORMPO_ABIOMED_DIR, GUARDIAN_BASE

TASK="${TASK:-abiomed-v0}"
KDE_CONFIG="${KDE_CONFIG:-config/kde/real.yaml}"
GUARDIAN_BASE="${GUARDIAN_BASE:-/public/gormpo/models/abiomed}"
GUARDIAN_PENALTY_COEF="${GUARDIAN_PENALTY_COEF:-0.5}"

DEVID_KDE="${DEVID_KDE:-0}"
DEVID_DYN="${DEVID_DYN:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVID_DYN"
fi

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(42 123 456)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ2_DIR="${LEQ2_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GORMPO_ABIOMED_DIR="${GORMPO_ABIOMED_DIR:-$LEQ2_DIR/../GORMPO_abiomed}"
GORMPO_CORMPO="${GORMPO_ABIOMED_DIR}/cormpo"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

DATASET_PATH="${DATASET_PATH:-$GORMPO_ABIOMED_DIR/synthetic_data/SAC_5000eps_stochastic.npz}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: MCS dataset not found: $DATASET_PATH"
    echo "Set DATASET_PATH to an abiomed offline .npz"
    exit 1
fi

echo "============================================"
echo "LEQ + KDE guardian: $TASK (MCS / Abiomed)"
echo "  GORMPO_abiomed: $GORMPO_ABIOMED_DIR"
echo "  OfflineRL-Kit:  $OFFLINERLKIT_DIR"
echo "  LEQ2:           $LEQ2_DIR"
echo "  Dataset:        $DATASET_PATH"
echo "  Seeds:          ${SEEDS[*]}"
echo "  Guardian λ:     $GUARDIAN_PENALTY_COEF"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    # Matches existing layout: .../trained_kde_<seed>/trained_kde_1.{faiss,_metadata.pkl}
    GUARDIAN_DIR="${GUARDIAN_BASE}/trained_kde_${seed}"
    GUARDIAN_PATH="${GUARDIAN_DIR}/trained_kde_1"

    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    # Resume: skip the whole seed if LEQ already finished (final checkpoint present).
    LEQ_DONE="$LEQ2_DIR/tmp/EP/models/$TASK/$seed/0.5/${seed}_${LEQ_MAX_STEPS:-1000000}.pkl"
    if [ -f "$LEQ_DONE" ]; then
        echo "  ✓ LEQ already trained ($LEQ_DONE) — skipping seed."
        echo ""
        continue
    fi

    # --- Step 1: KDE density guardian ---
    echo "Step 1/3: KDE guardian → $GUARDIAN_PATH"
    if [ -f "${GUARDIAN_PATH}_metadata.pkl" ] && [ -f "${GUARDIAN_PATH}.faiss" ]; then
        echo "  Guardian already exists, skipping."
    else
        mkdir -p "$GUARDIAN_DIR"
        (cd "$GORMPO_CORMPO" && python mbpo_kde/kde.py \
            --config "$KDE_CONFIG" \
            --seed "$seed" \
            --save_path "$GUARDIAN_DIR" \
            --devid "$DEVID_KDE")
        echo "  ✓ KDE training complete"
    fi
    echo ""

    # --- Step 2: Dynamics ensemble ---
    DYN_DIR="$OFFLINERLKIT_DIR/models/dynamics-ensemble/${seed}/${TASK}"
    echo "Step 2/3: Dynamics ensemble → $DYN_DIR"
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
        echo "  ✓ Dynamics training complete"
    fi
    echo ""

    # --- Step 3: LEQ + guardian ---
    echo "Step 3/3: LEQ + guardian (λ=$GUARDIAN_PENALTY_COEF)"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile 0.5 \
            --dataset_path '$DATASET_PATH' \
            --guardian_model_name '$GUARDIAN_PATH' \
            --guardian_penalty_coef '$GUARDIAN_PENALTY_COEF' \
            --debug"
    echo "  ✓ LEQ training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "Done. Checkpoints in LEQ2/tmp/EP/models/${TASK}/"
echo "Guardians in ${GUARDIAN_BASE}/trained_kde_<seed>/trained_kde_1.*"
echo "============================================"
