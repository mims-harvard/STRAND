#!/bin/bash
# Common paths and settings for the STACK cross-cell-line pipeline.
# Source this file from other scripts: source 00_setup_paths.sh
#
# Before running: set DATA_DIR to the directory containing your cross-cell-line data.
# Expected structure:
#   DATA_DIR/
#     k562/
#       k562_train_SE_embeddings_filtered.h5ad
#       k562_test_SE_embeddings_filtered.h5ad
#       k562_test_top300_SE_embeddings_filtered.h5ad  (for top-300 eval)
#     jurkat/   (same structure)
#     rpe1/     (same structure)

export STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="/path/to/cross_cellline_data"
export OUTPUT_DIR="${STACK_DIR}/output"
export CONDA_ENV="caduceus_env"
