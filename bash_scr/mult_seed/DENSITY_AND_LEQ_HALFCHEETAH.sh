#!/bin/bash
set -e
# LEQ + KDE density guardian on HalfCheetah-Medium-Expert (sparse 72.5%).
#
# Pipeline (per seed):
#   1. Train KDE guardian on sparse offline data   (GORMPO kde_module)
#   2. Train dynamics ensemble                     (OfflineRL-Kit2)
#   3. Train LEQ with guardian OOD penalty         (LEQ2)
#
# Usage (from LEQ2 root):
#   bash bash_scr/mult_seed/DENSITY_AND_LEQ_HALFCHEETAH.sh
#
# Env overrides:
#   SEEDS, GUARDIAN_PENALTY_COEF, DEVID_KDE, DEVID_DYN, LEQ2_ENV,
#   OFFLINERLKIT_DIR, LEQ2_DIR, GORMPO_ROOT

TASK="${TASK:-halfcheetah-medium-expert-v2}"
KDE_CONFIG="${KDE_CONFIG:-configs/kde/halfcheetah_medium_expert_sparse_3.yaml}"
GUARDIAN_ROOT="${GUARDIAN_ROOT:-/public/gormpo/models/halfcheetah_medium_expert_sparse_3}"
GUARDIAN_PENALTY_COEF="${GUARDIAN_PENALTY_COEF:-0.5}"
DEVID_KDE="${DEVID_KDE:-0}"
DEVID_DYN="${DEVID_DYN:-0}"

# Default seeds from the draft script; override with SEEDS="42 123 456" if needed
if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(67 68 69)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ2_DIR="${LEQ2_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GORMPO_ROOT="${GORMPO_ROOT:-$LEQ2_DIR/../GORMPO}"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

echo "============================================"
echo "LEQ + KDE guardian: $TASK (sparse HalfCheetah)"
echo "  GORMPO:        $GORMPO_ROOT"
echo "  OfflineRL-Kit: $OFFLINERLKIT_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "  Seeds:         ${SEEDS[*]}"
echo "  Guardian λ:    $GUARDIAN_PENALTY_COEF"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    GUARDIAN_PATH="${GUARDIAN_ROOT}/kde_${seed}"

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
        (cd "$GORMPO_ROOT" && python kde_module/kde.py \
            --config "$KDE_CONFIG" \
            --seed "$seed" \
            --save_path "$GUARDIAN_PATH" \
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
        conda run -n "$LEQ2_ENV" bash -c \
            "cd '$OFFLINERLKIT_DIR' && python run_example/run_dynamics.py \
                --task '$TASK' --seed '$seed' --device cuda:${DEVID_DYN}"
        echo "  ✓ Dynamics training complete"
    fi
    echo ""

    # --- Step 3: LEQ + guardian ---
    echo "Step 3/3: LEQ + guardian (λ=$GUARDIAN_PENALTY_COEF)"
    conda run -n "$LEQ2_ENV" bash -c \
        "cd '$LEQ2_DIR' && PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$seed' \
            --expectile 0.5 \
            --guardian_model_name '$GUARDIAN_PATH' \
            --guardian_penalty_coef '$GUARDIAN_PENALTY_COEF' \
            --debug"
    echo "  ✓ LEQ training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "Done. Checkpoints in LEQ2/tmp/EP/models/${TASK}/"
echo "Guardians in ${GUARDIAN_ROOT}/kde_<seed>.*"
echo "============================================"
