#!/bin/bash
# Generate per-perturbation predictions for one holdout cell line.
# Requires 1 GPU; on a cluster add appropriate scheduler headers for your system.
#
# Usage: bash 03_generate_predictions.sh k562   (or jurkat, rpe1)
#
# Uses the best checkpoint from the training run: prefers val_loss in the
# filename when present; otherwise lowest tloss (train-only runs); else last.ckpt.
# Generates predictions for each perturbation in context.h5ad, using
# test_control.h5ad cells as the ICL query (zero-shot cross-cell-line).

set -euo pipefail

HOLDOUT=${1:?"Usage: bash 03_generate_predictions.sh <holdout>  (k562|jurkat|rpe1)"}

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${STACK_DIR}/output/data/${HOLDOUT}_holdout"
CKPT_DIR="${STACK_DIR}/output/checkpoints/${HOLDOUT}_holdout"
PRED_DIR="${STACK_DIR}/output/predictions/${HOLDOUT}_holdout"

echo "========================================"
echo " STACK Generation: ${HOLDOUT} holdout"
echo "========================================"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caduceus_env

export PYTHONPATH="${STACK_DIR}/stack/src:${PYTHONPATH:-}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: Checkpoint directory does not exist: ${CKPT_DIR}"
    exit 1
fi

# ── Find best checkpoint (val_loss in filename, else tloss, else last.ckpt) ───
BEST_CKPT=$(python3 - "${CKPT_DIR}" <<'PY'
import os
import re
import sys

def main() -> None:
    ckpt_dir = sys.argv[1]
    assert ckpt_dir, "checkpoint directory required as argv[1]"
    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: not a directory: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and f != "last.ckpt"]
    if not ckpts:
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.isfile(last):
            print(last)
            return
        print(f"ERROR: No .ckpt files in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    val_ckpts = [f for f in ckpts if re.search(r"val_loss=([0-9]+\.[0-9]+)", f)]
    if val_ckpts:

        def val_loss(fname: str) -> float:
            m = re.search(r"val_loss=([0-9]+\.[0-9]+)", fname)
            assert m is not None
            return float(m.group(1))

        best = min(val_ckpts, key=val_loss)
        print(os.path.join(ckpt_dir, best))
        return
    t_ckpts = [f for f in ckpts if re.search(r"tloss(?:_epoch)?=([0-9]+\.[0-9]+)", f)]
    if t_ckpts:

        def tloss(fname: str) -> float:
            m = re.search(r"tloss(?:_epoch)?=([0-9]+\.[0-9]+)", fname)
            assert m is not None
            return float(m.group(1))

        best = min(t_ckpts, key=tloss)
        print(os.path.join(ckpt_dir, best))
        return
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last):
        print(last)
        return
    print(f"ERROR: Could not score checkpoints in {ckpt_dir}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
PY
)

if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: No checkpoint resolved in ${CKPT_DIR}"
    exit 1
fi

echo "Using checkpoint: ${BEST_CKPT}"
mkdir -p "${PRED_DIR}"

# ── Resume: skip genes that already have a prediction file ──────────────────
mapfile -t REMAINING < <(python3 - "${DATA_DIR}/context.h5ad" "${PRED_DIR}" <<'PY'
import sys, re
from pathlib import Path
import anndata as ad

def sanitize(s):
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(s))
    return safe or "split"

context_path, pred_dir = sys.argv[1], Path(sys.argv[2])
adata = ad.read_h5ad(context_path, backed="r")
all_splits = sorted(adata.obs["gene"].astype(str).unique())
done = {f.stem for f in pred_dir.glob("*.h5ad")}
for s in all_splits:
    if sanitize(s) not in done:
        print(s)
PY
)

if [ ${#REMAINING[@]} -eq 0 ]; then
    echo "All predictions already exist in ${PRED_DIR}. Nothing to do."
    exit 0
fi
echo "Remaining: ${#REMAINING[@]} gene(s) to generate (skipping already-done)"

echo "Starting generation at $(date)"
stack-generation \
    --checkpoint   "${BEST_CKPT}" \
    --base-adata   "${DATA_DIR}/context.h5ad" \
    --test-adata   "${DATA_DIR}/test_control.h5ad" \
    --genelist     "${DATA_DIR}/hvg.pkl" \
    --split-column gene \
    --split-values "${REMAINING[@]}" \
    --output-dir   "${PRED_DIR}" \
    --num-steps    5 \
    --prompt-ratio 0.25 \
    --context-ratio 0.4 \
    --context-ratio-min 0.2 \
    --batch-size   16 \
    --num-workers  4 \
    --mode         vanilla \
    --show-progress

echo "Generation completed at $(date)"
echo "Predictions saved to: ${PRED_DIR}"
ls -lh "${PRED_DIR}"
