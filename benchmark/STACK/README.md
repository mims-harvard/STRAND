# STACK

Official repository: [ArcInstitute/stack](https://github.com/ArcInstitute/stack)

STACK is a single-cell foundation model that performs **in-context learning (ICL)** for perturbation prediction. We evaluate it on a leave-one-cell-line-out cross-cell-line zero-shot prediction task across three cell lines: K562, Jurkat, RPE1.

## Setup

### 1. Install STACK

Clone the official repository into this folder and install:

```bash
git clone https://github.com/ArcInstitute/stack stack
cd stack
pip install -e .
cd ..
# Verify:
stack-train --help
stack-generation --help
```

### 2. Conda environment

```bash
conda activate caduceus_env
```

### 3. Configure data path

Edit `00_setup_paths.sh` and set `DATA_DIR` to the directory containing the cross-cell-line h5ad files:

```bash
export DATA_DIR="/path/to/cross_cellline_data"
```

Expected data layout:

```
DATA_DIR/
  k562/
    k562_train_SE_embeddings_filtered.h5ad
    k562_test_SE_embeddings_filtered.h5ad
    k562_test_top300_SE_embeddings_filtered.h5ad   # top-300 eval only
  jurkat/   (same structure)
  rpe1/     (same structure)
```

Input files are log-normalized, 2000 HVG-filtered. Relevant `obs` columns: `gene` (perturbation name), `cell_line`. Control label: `non-targeting`.

## Procedure

### Full pipeline (all three holdouts)

```bash
source 00_setup_paths.sh
bash run_pipeline.sh all
```

### Single holdout

```bash
source 00_setup_paths.sh
bash run_pipeline.sh k562
```

### Step by step

```bash
source 00_setup_paths.sh

# 1. Prepare data (creates combined_train_dir/, context.h5ad, test_control.h5ad, hvg.pkl)
python 01_prepare_data.py --holdout k562

# 2. Train STACK (~24h on 1 GPU)
bash 02_train_stack.sh k562

# 3. Generate predictions (~12h on 1 GPU, run after training completes)
bash 03_generate_predictions.sh k562

# 4. Format predictions (creates adata_pred.h5ad and adata_real.h5ad)
python 04_format_predictions.py --holdout k562

# 5. Evaluate
bash 05_evaluate.sh k562
```

For the top-300 perturbation subset, substitute step 3 with:

```bash
bash 03_generate_predictions_top300.sh k562
```

## Script inventory

| File | Purpose |
|------|---------|
| `00_setup_paths.sh` | Set `DATA_DIR`, `STACK_DIR`, `OUTPUT_DIR`, `CONDA_ENV` — source before running |
| `01_prepare_data.py` | Merge training cell lines; produce `combined_train_dir/`, `context.h5ad`, `test_control.h5ad`, `hvg.pkl` |
| `02_train_stack.sh` | Train STACK (1 GPU); resolves `__STACK_DIR__` placeholders in configs before calling `stack-train` |
| `03_generate_predictions.sh` | Run `stack-generation` for all perturbations; resumes automatically if interrupted |
| `03_generate_predictions_top300.sh` | Same as above, restricted to the top-300 perturbation subset |
| `04_format_predictions.py` | Aggregate per-perturbation h5ads → pseudo-bulk `adata_pred.h5ad` / `adata_real.h5ad` |
| `05_evaluate.sh` | Run `cell-eval` on `adata_pred` / `adata_real`; writes `agg_results.csv` and `results.csv` |
| `run_pipeline.sh` | Orchestrate all five stages for one or all holdouts |
| `configs/{holdout}_holdout.yaml` | STACK training hyperparameters per holdout |

## How STACK makes predictions

STACK uses **in-context learning (ICL)**:

1. **Context** (`context.h5ad`): all perturbed cells from the 2 training cell lines, grouped by perturbation (`--split-column gene`). For each perturbation, STACK sees how those 2 cell lines respond.
2. **Query** (`test_control.h5ad`): control (non-targeting) cells from the held-out cell line.
3. **Output**: for each perturbation, STACK predicts the held-out cell line's expression under that perturbation — purely from the cross-cell-line ICL context.

## Modifications to upstream STACK

Three minimal patches were applied to the official STACK codebase:

### 1. `src/stack/cli/generation.py` (line 325)

Changed `model_class` from `"ICL_FinetunedModel"` to `"StateICLModel"`.

**Why:** `StateICLModel` (trained by `stack-train`) has no `cls` classification head. The default `--mode mdm` generation uses per-cell logit scores from `self.cls` to guide iterative unmasking; without `cls`, `logit_predictions` is an empty list `[]` and the subsequent boolean-array indexing raises `TypeError`. Using `--mode vanilla` (same iterative schedule, no logit guidance) avoids this. The model class change makes `stack-generation` compatible with `stack-train` checkpoints by default.

### 2. `src/stack/cli/launch_training.py`

Added conditional validation-loss checkpointing when the validation set is empty (single-file split).

**Why:** When all data goes to training (no validation split), PyTorch Lightning raises an error if checkpoint callbacks monitor `val_loss`. The patch detects an empty validation dataset at callback-configuration time, switches the checkpoint monitor to train loss (`tloss_epoch`), and disables early stopping.

### 3. `src/stack/models/core/inference.py`

Added `filter_organism=False` to both `get_latent_representation()` and `get_prediction()` calls inside `get_incontext_prediction()`.

**Why:** `TestSamplerDataset` defaults to `filter_organism=True`, which removes cells whose `organism` obs column is not exactly `"Homo sapiens"`. When applied to the in-memory `full_mixed_adata` assembled during inference via `ad.concat`, the `organism` column may be absent or have inconsistent values, raising `ValueError: No Homo sapiens cells found`. Since the data is pre-selected human cell lines, organism filtering is not needed at inference time.

## Evaluation

Results are written to `output/results/{holdout}/`:

| File | Contents |
|------|---------|
| `adata_pred.h5ad` | Pseudo-bulk predicted expression (n_perts × n_genes) |
| `adata_real.h5ad` | Pseudo-bulk true expression (same shape) |
| `agg_results.csv` | Aggregated metrics (one row per metric) |
| `results.csv` | Per-perturbation metrics |

Evaluation uses the `cell-eval` package (`cell-eval run -ap adata_pred.h5ad -ar adata_real.h5ad --profile full`).
