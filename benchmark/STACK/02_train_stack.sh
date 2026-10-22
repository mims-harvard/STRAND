#!/bin/bash
# Train STACK for one holdout cell line.
# Requires 1 GPU; on a cluster add appropriate scheduler headers for your system.
#
# Usage: bash 02_train_stack.sh k562   (or jurkat, rpe1)

set -euo pipefail

HOLDOUT=${1:?"Usage: bash 02_train_stack.sh <holdout>  (k562|jurkat|rpe1)"}

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generate config with resolved paths (template uses __STACK_DIR__ placeholder)
CONFIG_TEMPLATE="${STACK_DIR}/configs/${HOLDOUT}_holdout.yaml"
CONFIG_RESOLVED="${STACK_DIR}/output/configs/${HOLDOUT}_holdout.yaml"

if [ ! -f "${CONFIG_TEMPLATE}" ]; then
    echo "ERROR: Config template not found: ${CONFIG_TEMPLATE}"
    exit 1
fi

mkdir -p "${STACK_DIR}/output/configs"
sed "s|__STACK_DIR__|${STACK_DIR}|g" "${CONFIG_TEMPLATE}" > "${CONFIG_RESOLVED}"

echo "========================================"
echo " STACK Training: ${HOLDOUT} holdout"
echo " Config: ${CONFIG_RESOLVED}"
echo "========================================"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caduceus_env

# Ensure STACK package is on PYTHONPATH (installed from stack/ directory)
export PYTHONPATH="${STACK_DIR}/stack/src:${PYTHONPATH:-}"

echo "Starting training at $(date)"
stack-train \
    --config "${CONFIG_RESOLVED}" \
    --no_localize_data

echo "Training completed at $(date)"
