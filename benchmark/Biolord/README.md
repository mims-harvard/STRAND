# BioLord

Benchmark wrapper for **biolord**. We use the official biolord API with minimal wrappers for data format and evaluation; no over-engineering of the original method.

**Official repository:** https://github.com/nitzanlab/biolord

---

## Procedure overview

1. **Data**: our h5ad format (e.g. from scripts 01–04) plus optional gene list, mapping, and mask; **format to BioLord** using `format_biolord_adata.py` in this folder to produce `obs['condition']` and `obs['cell_type']`. For the full pipeline (including combined/HVG runs), provide gene list, mapping, and mask; **for GO features** (used in training), provide GO graph and gene list (e.g. `go_essential_all.csv`, `essential_all_data_pert_genes.pkl`—these can be obtained independently).
2. **Training**: Run `run_biolord_standard.py` with paths to formatted adata, optional gene/mapping/mask, and optional `--go_graph_path` / `--go_gene_path`.
3. **Inference**: The same script generates predicted expression per condition and writes pred/real h5ad.
4. **Evaluation**: Run cell-eval on the pred/real h5ad; results to the chosen output directory.

---

## Step-by-step procedure (single-dataset demo)

1. **Prepare data**: Run `format_biolord_adata.py` on your our h5ad format to produce BioLord-ready h5ad (condition, cell_type). Example: `python format_biolord_adata.py <input.h5ad> <output_formatted.h5ad>`.
2. **Train and predict**: Run `run_biolord_standard.py` with `--adata_path` pointing to the formatted h5ad, plus optional `--gene_path`, `--mapping_path`, `--mask_path`, and `--go_graph_path` / `--go_gene_path`. Use `--gen_prediction` and `--gen_groundtruth` to write pred/real files.
3. **Evaluate**: Run cell-eval on the written pred/real h5ad; results go to the specified output directory.

---

## Script and file inventory

| Script / file | Purpose | Main inputs / outputs |
|---------------|---------|------------------------|
| `run_biolord_standard.py` | Train + inference entrypoint | adata path, optional gene/mapping/mask and GO paths; pred/real h5ad |
| `format_biolord_adata.py` | Format our h5ad format to BioLord (condition, cell_type) | input h5ad, output h5ad |

External inputs (optional, obtain independently): gene list, mapping, mask; GO graph CSV and gene list (e.g. `go_essential_all.csv`, `essential_all_data_pert_genes.pkl`) for GO features in training.

---

## Key functions

- **Training / inference**: `run_biolord_standard.py` loads adata, optionally applies gene/mask and GO features, trains biolord, and writes pred/real h5ad.
- **Formatting**: `format_biolord_adata.py` maps perturbation labels to BioLord `condition` (e.g. `ctrl`, `GENE+ctrl`) and sets `cell_type`; run it before training if your h5ad is not already in BioLord format.

---

## Evaluation

Metrics are computed with the **cell-eval** package. Pred/real h5ad are passed to cell-eval; results are written to the specified output directory.
