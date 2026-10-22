#!/bin/bash
# Master orchestration script for the STACK cross-cell-line evaluation pipeline.
#
# Runs all pipeline stages for one or all holdout cell lines sequentially:
#   1. Data preparation
#   2. STACK training  (GPU required — runs in foreground; add cluster headers to 02_train_stack.sh)
#   3. Prediction generation  (GPU required)
#   4. Prediction formatting
#   5. Evaluation with cell-eval
#
# Usage:
#   bash run_pipeline.sh k562        # run one holdout
#   bash run_pipeline.sh all         # run all three holdouts sequentially

set -euo pipefail

HOLDOUT=${1:?"Usage: bash run_pipeline.sh <holdout>  (k562|jurkat|rpe1|all)"}

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caduceus_env
export PYTHONPATH="${STACK_DIR}/stack/src:${PYTHONPATH:-}"

run_single() {
    local CL=$1
    echo ""
    echo "############################################"
    echo "  Starting pipeline for holdout: ${CL}"
    echo "############################################"

    # ── Stage 1: Data preparation ─────────────────────────────────────────────
    echo "[${CL}] Stage 1: Preparing data..."
    python "${STACK_DIR}/01_prepare_data.py" --holdout "${CL}"
    echo "[${CL}] Stage 1 complete."

    # ── Stage 2: STACK training ───────────────────────────────────────────────
    echo "[${CL}] Stage 2: Training STACK..."
    bash "${STACK_DIR}/02_train_stack.sh" "${CL}"

    CKPT_DIR="${STACK_DIR}/output/checkpoints/${CL}_holdout"
    if [ ! -d "${CKPT_DIR}" ] || [ -z "$(ls "${CKPT_DIR}"/*.ckpt 2>/dev/null)" ]; then
        echo "ERROR: No checkpoints found in ${CKPT_DIR}. Training may have failed."
        exit 1
    fi
    echo "[${CL}] Stage 2 complete. Checkpoint dir: ${CKPT_DIR}"

    # ── Stage 3: Generate predictions ─────────────────────────────────────────
    echo "[${CL}] Stage 3: Generating predictions..."
    bash "${STACK_DIR}/03_generate_predictions.sh" "${CL}"

    PRED_DIR="${STACK_DIR}/output/predictions/${CL}_holdout"
    if [ ! -d "${PRED_DIR}" ] || [ -z "$(ls "${PRED_DIR}"/*.h5ad 2>/dev/null)" ]; then
        echo "ERROR: No prediction files found in ${PRED_DIR}. Generation may have failed."
        exit 1
    fi
    echo "[${CL}] Stage 3 complete. Predictions in: ${PRED_DIR}"

    # ── Stage 4: Format predictions ───────────────────────────────────────────
    echo "[${CL}] Stage 4: Formatting predictions..."
    python "${STACK_DIR}/04_format_predictions.py" --holdout "${CL}"
    echo "[${CL}] Stage 4 complete."

    # ── Stage 5: Evaluate ─────────────────────────────────────────────────────
    echo "[${CL}] Stage 5: Running cell-eval..."
    bash "${STACK_DIR}/05_evaluate.sh" "${CL}"
    echo "[${CL}] Stage 5 complete."

    echo ""
    echo "############################################"
    echo "  Pipeline DONE for holdout: ${CL}"
    echo "  Results: ${STACK_DIR}/output/results/${CL}/"
    echo "############################################"
}

if [ "${HOLDOUT}" = "all" ]; then
    for CL in k562 jurkat rpe1; do
        run_single "${CL}"
    done
else
    run_single "${HOLDOUT}"
fi

echo ""
echo "All done. Results summary:"
for CL in k562 jurkat rpe1; do
    AGG="${STACK_DIR}/output/results/${CL}/agg_results.csv"
    if [ -f "${AGG}" ]; then
        echo "  [${CL}] ${AGG}"
    fi
done
