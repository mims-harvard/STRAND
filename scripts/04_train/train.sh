#!/usr/bin/env bash
set -euo pipefail

# Local-first launcher.
#
# Usage:
#   bash scripts/04_train/train.sh k562
#   bash scripts/04_train/train.sh combined_perturbqa_hvg2k
#
# Environment (recommended):
#   export DATA_ROOT=/path/to/data/processed
#   export OUTPUT_ROOT=/path/to/outputs
#   export DNA_REFERENCE_PATH=/path/to/hg38_fasta_dir
#   export GENE_ANNOTATION_PATH=/path/to/gencode.gtf
#   export CCRE_BED_PATH=/path/to/ccres.bed.gz   (optional)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${1:-k562}"

CONFIG_FILE=""
case "$DATASET" in
  k562) CONFIG_FILE="${REPO_ROOT}/configs/k562_config.yaml" ;;
  rpe1) CONFIG_FILE="${REPO_ROOT}/configs/rpe1_config.yaml" ;;
  jurkat) CONFIG_FILE="${REPO_ROOT}/configs/jurkat_config.yaml" ;;
  combined_perturbqa_hvg2k) CONFIG_FILE="${REPO_ROOT}/configs/combined_config.yaml" ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Expected one of: k562 rpe1 jurkat combined_perturbqa_hvg2k"
    exit 1
    ;;
esac

python "${SCRIPT_DIR}/train.py" \
  --dname "${DATASET}" \
  --config "${CONFIG_FILE}" \
  --RNA_loss htg_nll \
  --out_suffix cache_line_specific \
  --wandb_name local_run \
  --version v1 \
  --no_lr_scheduler
