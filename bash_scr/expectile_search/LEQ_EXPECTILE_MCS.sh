#!/bin/bash
set -e
# Expectile hyperparameter search for LEQ + DBG (density-based guardian) on
# MCS (Abiomed) -- same pattern as LEQ_EXPECTILE_{HALFCHEETAH,HOPPER,WALKER2D}.sh,
# applied to the MCS/DBG pipeline from bash_scr/leq_dbg/LEQ_DBG_MCS.sh instead
# of the guardian-free sparse-D4RL pipeline (guardian_penalty_coef=0 there;
# here a real guardian is always loaded, since LEQ_DBG_MCS has no no-guardian mode).
#
# LEQ itself trains on the REAL clinical dataset (unlike LEQ_DBG_MCS.sh's
# default, which is still the synthetic SAC-rollout data):
#   $GORMPO_ABIOMED_DIR/synthetic_data/real_train_val.npz  (built by build_real_mcs_dataset.py)
# DYN_DIR must match -- it needs a dynamics ensemble trained on that same real
# dataset (default dynamics-ensemble-real/, not LEQ_DBG_MCS.sh's dynamics-ensemble/).
#
# Pipeline:
#   1. Verify a PRE-TRAINED dynamics ensemble already exists at DYN_DIR
#      (this script does NOT train dynamics, same as the sparse variants).
#   2. Verify a PRE-TRAINED guardian (GUARDIAN_TYPE, default realnvp) already
#      exists (this script does NOT train the guardian either -- run
#      bash_scr/leq_dbg/LEQ_DBG_MCS.sh once first if one is missing).
#   3. Reuse both to run LEQ once per expectile value in EXPECTILES, ALL IN
#      PARALLEL, one job per GPU, with that guardian's OOD penalty applied.
#
# GPU layout (fixed): GPUs 0-3, one per expectile value. NOTE: this overlaps
# with LEQ_EXPECTILE_HOPPER.sh / LEQ_EXPECTILE_WALKER2D.sh -- don't run those
# at the same time as this unless you override GPUS below
# (LEQ_EXPECTILE_HALFCHEETAH.sh uses GPUs 4-7 and is safe alongside this).
#
# Usage (from LEQ2 root):
#   bash bash_scr/expectile_search/LEQ_EXPECTILE_MCS.sh
#   GUARDIAN_TYPE=kde bash bash_scr/expectile_search/LEQ_EXPECTILE_MCS.sh
#
# Env overrides:
#   DATASET_PATH, SEED, EXPECTILES, GPUS, GUARDIAN_TYPE, GUARDIAN_PENALTY_COEF,
#   GUARDIAN_BASE, DYN_BASE_DIR, GORMPO_ABIOMED_DIR, OFFLINERLKIT_DIR, LEQ2_DIR, LEQ2_ENV

TASK="${TASK:-abiomed-v0}"
SEED="${SEED:-42}"  # MCS is single-seed; only seed 42 has pretrained guardians for every type

# Which density estimator backs the guardian, and its GORMPO_abiomed-tuned
# penalty coefficient -- same table as bash_scr/leq_dbg/LEQ_DBG_MCS.sh.
GUARDIAN_TYPE="${GUARDIAN_TYPE:-realnvp}"
declare -A REWARD_PENALTY_COEF=(
    [kde]=0.2
    [vae]=0.1
    [realnvp]=0.2
    [neuralode]=0.2
    [ddpm]=0.4
)
GUARDIAN_PENALTY_COEF="${GUARDIAN_PENALTY_COEF:-${REWARD_PENALTY_COEF[$GUARDIAN_TYPE]:-}}"
if [ -z "$GUARDIAN_PENALTY_COEF" ]; then
    echo "ERROR: unknown GUARDIAN_TYPE '$GUARDIAN_TYPE' (must be one of: ${!REWARD_PENALTY_COEF[*]})"
    exit 1
fi

if [ -z "${EXPECTILES+x}" ] || [ -z "$EXPECTILES" ]; then
    EXPECTILES=(0.1 0.3 0.4 0.5)
else
    # shellcheck disable=SC2206
    EXPECTILES=($EXPECTILES)
fi

if [ -z "${GPUS+x}" ] || [ -z "$GPUS" ]; then
    GPUS=(0 1 2 3)
else
    # shellcheck disable=SC2206
    GPUS=($GPUS)
fi

if [ "${#GPUS[@]}" -lt "${#EXPECTILES[@]}" ]; then
    echo "ERROR: need at least ${#EXPECTILES[@]} GPUs in GPUS, got ${#GPUS[@]}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEQ2_DIR="${LEQ2_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GORMPO_ABIOMED_DIR="${GORMPO_ABIOMED_DIR:-$LEQ2_DIR/../GORMPO_abiomed}"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$LEQ2_DIR/../OfflineRL-Kit2}"
GUARDIAN_BASE="${GUARDIAN_BASE:-/public/gormpo/models/abiomed}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

DATASET_PATH="${DATASET_PATH:-$GORMPO_ABIOMED_DIR/synthetic_data/real_train_val.npz}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: MCS dataset not found: $DATASET_PATH"
    echo "Set DATASET_PATH to an abiomed offline .npz, or build the real one:"
    echo "  python $GORMPO_ABIOMED_DIR/../LEQ2/build_real_mcs_dataset.py $DATASET_PATH"
    exit 1
fi

DYN_BASE_DIR="${DYN_BASE_DIR:-$OFFLINERLKIT_DIR/models/dynamics-ensemble-real}"
DYN_DIR="$DYN_BASE_DIR/${SEED}/${TASK}"

# Guardian checkpoint path -- same layout as guardian_path() in bash_scr/leq_dbg/LEQ_DBG_MCS.sh.
case "$GUARDIAN_TYPE" in
    kde)       GUARDIAN_PATH="$GUARDIAN_BASE/trained_kde_${SEED}/trained_kde_1" ;;
    vae)       GUARDIAN_PATH="$GUARDIAN_BASE/trained_vae_${SEED}/trained_vae_1" ;;
    realnvp)   GUARDIAN_PATH="$GUARDIAN_BASE/trained_realnvp_${SEED}/trained_realnvp_1" ;;
    neuralode) GUARDIAN_PATH="$GUARDIAN_BASE/neuralODE/" ;;  # one shared model; trailing slash required
    ddpm)      GUARDIAN_PATH="$GUARDIAN_BASE/trained_diffusion_${SEED}" ;;
esac

SAVE_DIR="./tmp/EP_dbg_expectile_search/${GUARDIAN_TYPE}"
LOG_DIR="$SAVE_DIR/logs"
RESULTS_FILE="$SAVE_DIR/results_${TASK}_${GUARDIAN_TYPE}.csv"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "LEQ expectile search on MCS (Abiomed) + DBG"
echo "  Env task:      $TASK"
echo "  Dataset:       $DATASET_PATH"
echo "  Seed:          $SEED"
echo "  Guardian:      $GUARDIAN_TYPE (penalty_coef=$GUARDIAN_PENALTY_COEF) -> $GUARDIAN_PATH"
echo "  Expectiles:    ${EXPECTILES[*]}"
echo "  GPUs:          ${GPUS[*]}"
echo "  Dynamics dir:  $DYN_DIR"
echo "  LEQ2:          $LEQ2_DIR"
echo "============================================"
echo ""

echo "Step 1/3: Verify pre-trained dynamics ensemble -> $DYN_DIR"
if [ -f "$DYN_DIR/dynamics.pth" ] && [ -f "$DYN_DIR/mu.npy" ] && [ -f "$DYN_DIR/std.npy" ]; then
    echo "  Found dynamics.pth, mu.npy, std.npy. Not training -- reusing as-is."
else
    echo "ERROR: pre-trained dynamics ensemble not found at $DYN_DIR"
    echo "  Expected files: dynamics.pth, mu.npy, std.npy"
    echo "  This script does not train dynamics. LEQ_DBG_MCS.sh won't produce this either --"
    echo "  its dynamics are trained on the synthetic dataset, into dynamics-ensemble/, not"
    echo "  dynamics-ensemble-real/. Train one on the REAL dataset directly instead:"
    echo "    cd $OFFLINERLKIT_DIR && python run_example/run_dynamics.py \\"
    echo "      --task $TASK --seed $SEED --dataset-path $DATASET_PATH \\"
    echo "      --model-base-dir $DYN_BASE_DIR/"
    exit 1
fi
echo ""

echo "Step 2/3: Verify pre-trained guardian -> $GUARDIAN_PATH"
guardian_ok=1
case "$GUARDIAN_TYPE" in
    kde)         [ -f "${GUARDIAN_PATH}.faiss" ] && [ -f "${GUARDIAN_PATH}_metadata.pkl" ] || guardian_ok=0 ;;
    vae|realnvp) [ -f "${GUARDIAN_PATH}_model.pth" ] && [ -f "${GUARDIAN_PATH}_meta_data.pkl" ] || guardian_ok=0 ;;
    neuralode)   [ -f "${GUARDIAN_PATH}_model.pt" ] && [ -f "${GUARDIAN_PATH}_metadata.pkl" ] || guardian_ok=0 ;;
    ddpm)        [ -f "${GUARDIAN_PATH}/checkpoint.pt" ] || guardian_ok=0 ;;
esac
if [ "$guardian_ok" = 1 ]; then
    echo "  Found. Not training -- reusing as-is."
else
    echo "ERROR: pre-trained '$GUARDIAN_TYPE' guardian not found at $GUARDIAN_PATH"
    echo "  This script does not train guardians -- run bash_scr/leq_dbg/LEQ_DBG_MCS.sh"
    echo "  once first (it trains any missing guardian for SEED=$SEED)."
    exit 1
fi
echo ""

echo "Step 3/3: LEQ expectile sweep (parallel, with $GUARDIAN_TYPE guardian)"
declare -A PIDS
declare -A GPU_FOR_EXPECTILE
for i in "${!EXPECTILES[@]}"; do
    expectile="${EXPECTILES[$i]}"
    gpu="${GPUS[$i]}"
    GPU_FOR_EXPECTILE[$expectile]="$gpu"
    logfile="$LOG_DIR/expectile_${expectile}.log"
    echo "  Launching expectile=$expectile on GPU $gpu -> $logfile"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "cd '$LEQ2_DIR' && \
            CUDA_VISIBLE_DEVICES='$gpu' \
            PYTHONPATH='.' python train/train_LEQ.py \
            --env_name '$TASK' \
            --seed '$SEED' \
            --expectile '$expectile' \
            --dataset_path '$DATASET_PATH' \
            --load_dir '$DYN_DIR' \
            --guardian_model_name '$GUARDIAN_PATH' \
            --guardian_type '$GUARDIAN_TYPE' \
            --guardian_penalty_coef '$GUARDIAN_PENALTY_COEF' \
            --eval_episodes 10 \
            --save_dir '$SAVE_DIR/' \
            --debug" > "$logfile" 2>&1 &
    PIDS[$expectile]=$!
done

FAILED_EXPECTILES=()
for expectile in "${EXPECTILES[@]}"; do
    if wait "${PIDS[$expectile]}"; then
        echo "  expectile=$expectile finished (GPU ${GPU_FOR_EXPECTILE[$expectile]})"
    else
        status=$?
        echo "  WARNING: expectile=$expectile FAILED (exit $status, GPU ${GPU_FOR_EXPECTILE[$expectile]}) -- see $LOG_DIR/expectile_${expectile}.log"
        FAILED_EXPECTILES+=("$expectile")
    fi
done
echo ""

echo "Collecting results -> $RESULTS_FILE"
echo "expectile,final_score,final_length" > "$RESULTS_FILE"
for expectile in "${EXPECTILES[@]}"; do
    logfile="$LOG_DIR/expectile_${expectile}.log"
    line=$(grep "Final score:" "$logfile" | tail -n 1)
    if [ -n "$line" ]; then
        score=$(echo "$line" | sed -n 's/.*Final score: \([^ ]*\) Final length:.*/\1/p')
        length=$(echo "$line" | sed -n 's/.*Final length: \(.*\)/\1/p')
        echo "$expectile,$score,$length" >> "$RESULTS_FILE"
    else
        echo "$expectile,NA,NA" >> "$RESULTS_FILE"
        echo "  WARNING: no 'Final score:' line found in $logfile"
    fi
done

BEST_LINE=$(tail -n +2 "$RESULTS_FILE" | grep -v ',NA,NA$' | sort -t, -k2 -g -r | head -n 1)
if [ -n "$BEST_LINE" ]; then
    BEST_EXPECTILE=$(echo "$BEST_LINE" | cut -d, -f1)
    BEST_SCORE=$(echo "$BEST_LINE" | cut -d, -f2)
else
    BEST_EXPECTILE="N/A"
    BEST_SCORE="N/A"
fi

echo ""
echo "============================================"
echo "Done. Results (each score = train_LEQ.py's final eval, averaged over"
echo "--eval_episodes=10 episodes):"
column -s, -t "$RESULTS_FILE"
echo ""
echo ">>> BEST expectile for $TASK / $GUARDIAN_TYPE: $BEST_EXPECTILE (final_score=$BEST_SCORE) <<<"
if [ "${#FAILED_EXPECTILES[@]}" -gt 0 ]; then
    echo ""
    echo "!!! FAILED expectiles (excluded from results above): ${FAILED_EXPECTILES[*]}"
fi
echo ""
echo "  Results CSV:  $RESULTS_FILE"
echo "  Per-run logs: $LOG_DIR/"
echo "  Dynamics:     $DYN_DIR/"
echo "  Guardian:     $GUARDIAN_PATH"
echo "============================================"
