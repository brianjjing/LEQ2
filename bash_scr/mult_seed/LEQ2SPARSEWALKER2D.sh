#!/bin/bash
set -e
# LEQ on sparse Walker2d-Medium-Expert (no density guardian).
#
# Uses the sparse offline pickle for BOTH dynamics and LEQ:
#   /public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl
#
# Pipeline (per seed):
#   1. Train dynamics ensemble on sparse data   (OfflineRL-Kit2)
#   2. Train LEQ on sparse data                 (LEQ2)
#
# Dynamics are saved under a separate tag so they do not collide with
# full-D4RL models from DENSITY_AND_LEQ_WALKER2D.sh.
#
# Usage (from LEQ2 root):
#   bash bash_scr/mult_seed/LEQ2SPARSEWALKER2D.sh
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/LEQ2SPARSEWALKER2D.sh
#
# Env overrides:
#   DATASET_PATH, SEEDS, DEVID_DYN, LEQ2_ENV, OFFLINERLKIT_DIR, LEQ2_DIR

TASK="${TASK:-walker2d-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
DYN_TAG="${DYN_TAG:-walker2d-medium-expert-v2_sparse_73}"

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
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: sparse dataset not found: $DATASET_PATH"
    exit 1
fi

echo "============================================"
echo "LEQ on sparse Walker2d"
echo "  Env task:      $TASK"
echo "  Dataset:       $DATASET_PATH"
echo "  Dynamics tag:  $DYN_TAG"
echo "  OfflineRL-Kit: $OFFLINERLKIT_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "  Seeds:         ${SEEDS[*]}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    DYN_DIR="$OFFLINERLKIT_DIR/models/dynamics-ensemble/${seed}/${DYN_TAG}"

    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    echo "Step 1/2: Dynamics ensemble → $DYN_DIR"
    if [ -f "$DYN_DIR/dynamics.pth" ]; then
        echo "  Dynamics already exist, skipping."
    else
        conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
            "cd '$OFFLINERLKIT_DIR' && \
                CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
                PYTHONPATH='$OFFLINERLKIT_DIR' \
                python run_example/run_dynamics.py \
                    --task '$TASK' --seed '$seed' \
                    --dataset-path '$DATASET_PATH' \
                    --model-tag '$DYN_TAG'"
        echo "  ✓ Dynamics training complete"
    fi
    echo ""

    echo "Step 2/2: LEQ on sparse dataset"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile 0.5 \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --save_dir './tmp/EP_sparse/' \
            --debug"
    echo "  ✓ LEQ training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "Done."
echo "  LEQ ckpts:  LEQ2/tmp/EP_sparse/models/${TASK}/"
echo "  Dynamics:   OfflineRL-Kit2/models/dynamics-ensemble/<seed>/${DYN_TAG}/"
echo "  Dataset:    $DATASET_PATH"
echo "============================================"
