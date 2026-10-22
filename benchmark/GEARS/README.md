# GEARS

Benchmark wrapper for **GEARS** (SOTA baseline for perturbation response prediction). We use the official GEARS code with minimal wrappers for data format and evaluation; no over-engineering of the original method.

**Official repository:** https://github.com/snap-stanford/GEARS

---

## Setup

- **Local files:** This folder includes **`Mapping.py`** (data mapping; no optimal transport is used in this pipeline) and **`utils.py`**, so those are self-contained here.
- **GEARS repo:** Clone the official GEARS repo. Set **`GEARS_REPO_PATH`** to the root of that clone (the directory containing `gears/` and `data_utils`), or place the repo at **`benchmark/GEARS/official/GEARS`**. The script needs it for the `gears` package and `data_utils` (e.g. `get_DE_genes`, `DataSplitter`).

The script inserts the local GEARS folder first, then the repo path, so `Mapping` and `utils` are taken from this folder; `gears` and `data_utils` come from the clone. A clear error is raised at import time if the repo is not found.

---

## Procedure overview

1. **Data**: our h5ad format with columns `split_few_shot` (or equivalent split), `gene` (perturbation), and `non-targeting` as control label.
2. **Training**: Build GEARS `PertData` from the h5ad, then train the GNN via the official GEARS code.
3. **Inference**: For each test perturbation, sample 256 test control cells, run GNN forward, and save predictions and ground truths as h5ad.
4. **Evaluation**: Run cell-eval on the saved pred/real h5ad; results go to `stats_few_shot/<dname>/`.

---

## Step-by-step procedure (single-dataset demo)

1. **Prepare data**: Ensure input h5ad is in our h5ad format (e.g. `split_few_shot`, `gene`, `non-targeting`). Paths are set in the SLURM script or CLI.
2. **Train and predict**: Run `run_gears.py` with `--adata_path`, `--pert_data_path`, `--pert_dataset_name`, and output paths. For a single cell line (e.g. k562), use one `DATANAME`; the script trains and runs inference (256 control cells per perturbation).
3. **Outputs**: Script writes `predictions.h5ad` and `ground_truths.h5ad` under the chosen output directory (e.g. `predictions_few_shot/<dname>/`).
4. **Evaluate**: Run the cell-eval script on pred/real files; results go to `stats_few_shot/<dname>/`.

**Key environment / CLI**: `--perturbation_col` (e.g. `gene`), `--control_label` (e.g. `non-targeting`), `--split_column` (e.g. `split_few_shot`).

*Optional:* Combined/multi-cell-line runs use the same script with different DATANAME/paths.

---

## Script and file inventory

| Script / file | Purpose | Main inputs / outputs |
|---------------|---------|------------------------|
| `run_gears.py` | Training + inference entrypoint (calls official GEARS) | adata path, pert_data path, model_save_path, predictions_path, ground_truths_path |
| `Mapping.py` | Data wrapper (renamed from OptimalTransport.py; no OT logic); defines `OptimalTransportData(PertData)` | Used by run_gears; relies on repo for `data_utils` |
| `utils.py` | GO download, similarity network, helpers (local) | Used by run_gears and Mapping |
| Eval script | Run cell-eval on GEARS predictions | pred/real h5ad; writes to `stats_few_shot/<dname>/` |
| `data_utils` (from repo) | DE genes, DataSplitter; required by Mapping | From GEARS clone at GEARS_REPO_PATH |

---

## Key functions

- **`predict_perturbation_from_control_cells`** (or `predict_with_test_controls`): Inference entrypoint. Takes trained model, test control AnnData, perturbation gene(s), and number of samples (e.g. 256); returns predicted expression matrix by running GNN on control-cell graphs.
- **`evaluate_few_shot_by_cell_line`** (or `evaluate_256_controls_cell_line_aware`): Evaluation entrypoint. Runs inference over test set with 256 control samples per perturbation; supports multiple cell lines by processing each separately; writes predictions and ground truths to h5ad.
- **`RobustGEARS`**: Thin subclass of official `GEARS`. Only adjusts control-label handling and GO edge list (e.g. adding `importance` column if missing). No changes to core GNN or training logic.

---

## Evaluation

Metrics are computed with the **cell-eval** package. The eval script invokes `cell-eval run` on the prediction and ground-truth h5ad files. Results are written to `stats_few_shot/<dname>/`. No custom metrics beyond cell-eval.
