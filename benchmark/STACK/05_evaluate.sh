#!/bin/bash
# Run cell-eval metrics on STACK cross-cell-line predictions.
# Produces agg_results.csv and results.csv in output/results/{holdout}/
#
# Usage: bash 05_evaluate.sh k562   (or jurkat, rpe1, all)
#
# Requires caduceus_env (which has cell-eval installed).

set -euo pipefail

HOLDOUT=${1:?"Usage: bash 05_evaluate.sh <holdout>  (k562|jurkat|rpe1|all)"}

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${STACK_DIR}/output"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caduceus_env

# Prefer the cell-eval console script; fall back to python -m cell_eval if the package is importable.
run_cell_eval() {
    if command -v cell-eval >/dev/null 2>&1; then
        cell-eval "$@"
        return
    fi
    if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('cell_eval') else 1)" 2>/dev/null; then
        python -m cell_eval "$@"
        return
    fi
    echo "ERROR: cell-eval is not available." >&2
    echo "  - No 'cell-eval' on PATH after: conda activate caduceus_env" >&2
    echo "  - Python module 'cell_eval' is not importable" >&2
    echo "  Install cell-eval in this environment or fix PATH, then retry." >&2
    exit 127
}

evaluate_one() {
    local CL=$1
    local RESULTS_DIR="${OUTPUT_DIR}/results/${CL}"
    local PRED="${RESULTS_DIR}/adata_pred.h5ad"
    local REAL="${RESULTS_DIR}/adata_real.h5ad"

    if [ ! -f "${PRED}" ] || [ ! -f "${REAL}" ]; then
        echo "ERROR: Missing prediction files for ${CL}."
        echo "  Expected: ${PRED}"
        echo "  Expected: ${REAL}"
        echo "  Run 04_format_predictions.py first."
        return 1
    fi

    echo "========================================"
    echo " Evaluating: ${CL} holdout"
    echo " Pred: ${PRED}"
    echo " Real: ${REAL}"
    echo "========================================"

    run_cell_eval run \
        -ap  "${PRED}" \
        -ar  "${REAL}" \
        --num-threads 16 \
        --profile full \
        --outdir "${RESULTS_DIR}"

    echo "Results saved to ${RESULTS_DIR}"
    if [ -f "${RESULTS_DIR}/agg_results.csv" ]; then
        echo "--- Aggregated results ---"
        cat "${RESULTS_DIR}/agg_results.csv"
    fi
}

if [ "${HOLDOUT}" = "all" ]; then
    for CL in k562 jurkat rpe1; do
        evaluate_one "${CL}"
    done
else
    evaluate_one "${HOLDOUT}"
fi
