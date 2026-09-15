#!/bin/bash
set -e
# LEQ on MCS (Abiomed) — NO density guardian.
#
# Same as DENSITY_AND_LEQ_MCS.sh but drops the KDE guardian entirely:
# no guardian training step, and LEQ runs with no OOD rollout penalty.
#
# Pipeline (per seed):
#   1. Train dynamics ensemble on MCS .npz       (OfflineRL-Kit2)
#   2. Train LEQ (plain, no guardian penalty)      (LEQ2)
#
# Usage (from LEQ2 root):
#   bash bash_scr/mult_seed/LEQ_MCS.sh
#
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/LEQ_MCS.sh
#
# Env overrides:
#   DATASET_PATH, SEEDS, DEVID_DYN,
#   LEQ2_ENV, OFFLINERLKIT_DIR, LEQ2_DIR, GORMPO_ABIOMED_DIR

TASK="${TASK:-abiomed-v0}"

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
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

DATASET_PATH="${DATASET_PATH:-$GORMPO_ABIOMED_DIR/synthetic_data/SAC_5000eps_stochastic.npz}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: MCS dataset not found: $DATASET_PATH"
    echo "Set DATASET_PATH to an abiomed offline .npz"
    exit 1
fi

echo "============================================"
echo "LEQ (no guardian): $TASK (MCS / Abiomed)"
echo "  GORMPO_abiomed: $GORMPO_ABIOMED_DIR"
echo "  OfflineRL-Kit:  $OFFLINERLKIT_DIR"
echo "  LEQ2:           $LEQ2_DIR"
echo "  Dataset:        $DATASET_PATH"
echo "  Seeds:          ${SEEDS[*]}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    # --- Step 1: Dynamics ensemble ---
    DYN_DIR="$OFFLINERLKIT_DIR/models/dynamics-ensemble/${seed}/${TASK}"
    echo "Step 1/2: Dynamics ensemble → $DYN_DIR"
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

    # --- Step 2: LEQ (no guardian) ---
    echo "Step 2/2: LEQ (plain, no guardian penalty)"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile 0.5 \
            --dataset_path '$DATASET_PATH' \
            --debug"
    echo "  ✓ LEQ training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "Done. Checkpoints in LEQ2/tmp/EP/models/${TASK}/"
echo "============================================"
