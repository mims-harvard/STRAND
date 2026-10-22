# STATE (State Transition Model)

Benchmark wrapper for **STATE** (SOTA baseline for predicting cellular responses to perturbation). This benchmark uses the official STATE CLI via `run_state.py` with minimal wrappers for data paths and config. **Preprocessing and data generation are done by this repo's scripts 01–04** (under `scripts/`); no separate STATE data-prep scripts are shipped in this repo.

**Official repository:** https://github.com/ArcInstitute/state

---

## Data preparation (scripts 01–04)

Preprocessing and data generation are performed by the unified pipeline in `scripts/`:

- **01_curate**: PerturbQA curation and data folder layout. See [scripts/01_curate/README.md](../../scripts/01_curate/README.md).
- **02_embed_and_match**: SCimilarity embeddings and OT matching (used by the STRAND pipeline; **STATE uses its own SE model** for embeddings—see below). See [scripts/02_embed_and_match/README.md](../../scripts/02_embed_and_match/README.md).
- **03_make_splits**: OT train/val index splitting. See [scripts/03_make_splits/README.md](../../scripts/03_make_splits/README.md).
- **04_train**: STRAND training (separate from STATE; included for completeness).

Run these steps first; details and examples are in each `scripts/*/README.md`. Output layout under a single root (e.g. `DATA_ROOT` or `DATA_DIR`):

| Path | Description |
|------|-------------|
| `<DATA_ROOT>/<dname>/<dname>.h5ad` | Log-normalized counts; `obs['gene']`, `obs['split']`, `obs['curated_perturbation']`, `obs['cell_line']`. |
| `<DATA_ROOT>/<dname>/<dname>_raw_with_split.h5ad` | Raw counts, **full gene set** (union of genes before HVG selection); same obs columns. Use this as input for **STATE SE embeddings**. |
| `<DATA_ROOT>/<dname>/<dname>_se_embeddings.h5ad` |  **STATE SE model** embeddings (from `run_state_se_embeddings.py` or `state emb transform`). Derived from full-gene input. |
| `<DATA_ROOT>/<dname>/<dname>.X_scimilarity.h5` | SCimilarity embeddings from step 02 (`scripts/02_embed_and_match/`; used by the STRAND pipeline only; standalone `.h5`). |
| `<DATA_ROOT>/combined_perturbqa_hvg2k/` | Combined HVG dataset (e.g. `combined_perturbqa_hvg2k.h5ad`). **Do not use for STATE SE embeddings.** |

---

## STATE SE embeddings (middle step)

**STATE embeddings are produced by the official STATE SE model**. They are generated in a **middle step** before ST train/predict.

**Flow:** raw h5ad (`<dname>_raw_with_split.h5ad`) → `state emb transform --embed-key X_state` → **merge embeddings into filtered data** → TOML → `state tx train` → `state tx predict` → cell-eval.

- **Input for SE**: Use the **full gene set** prepared by 01–04 **before** HVG and log-normalization. That is the data in `<dname>_raw_with_split.h5ad` (raw counts, union gene set from curation). **Do not use HVG-only or log-normalized matrices** (e.g. `combined_perturbqa_hvg2k.h5ad` or `<dname>.h5ad`) as input to the SE model.
- **Embed key**: Use **`X_state`** as the default `--embed-key` for `state emb transform` and `data.kwargs.embed_key` in training. Set `EMBED_KEY=X_state` (or the key under `adata.obsm` where SE embeddings are stored) when running `run_state.py`.
- **Command** (from the [official STATE README](https://github.com/ArcInstitute/state)):

```bash
state emb transform \
  --model-folder /path/to/SE-600M \
  --checkpoint /path/to/SE-600M/se600m_epoch15.ckpt \
  --input /path/to/input.h5ad \
  --output /path/to/output.h5ad \
  --embed-key X_state
```

Use `<dname>_raw_with_split.h5ad` as `--input`. You can use the prototype wrapper: `python run_state_se_embeddings.py --model_folder ... --checkpoint ... --input ... --output ...` (default `--embed_key X_state`).

- **Merge step**: After generating embeddings, **merge them into the filtered h5ad** used for training. Then point your TOML at the merged h5ad and set `EMBED_KEY=X_state`.

---

## Alignment with run_state.py inputs

`run_state.py` reads paths from environment variables (or CLI). Use the same root as 01–04 output so paths resolve.

- **TOML_CONFIG** (required for train): Path to your STATE TOML config. The TOML must point to h5ad paths produced by 01–04 (e.g. `DATA_ROOT/k562/k562.h5ad`), or paths relative to `DATA_DIR`. See the [STATE docs](https://github.com/ArcInstitute/state) for the expected TOML schema.
- **STATE_RUN_DIR** (required): Output directory for training; same dir is used for predict.
- **DATA_DIR**: Set to the same root as 01–04 output (e.g. `DATA_ROOT`) so that paths in the TOML resolve.
- **EMBED_KEY** (optional): Key in `adata.obsm` for **STATE SE embeddings** (from `state emb transform`). Use **`X_state`**. Embeddings should be prepared using the STATE SE model on raw, full-gene-set input, then merged into the filtered h5ad.
- **EMBEDDINGS_PATH** (optional): Path to precomputed SE embeddings; merge into h5ad or set in TOML as STATE expects.

**Training overrides:** When calling `state tx train`, use overrides such as `data.kwargs.embed_key=X_state`, `data.kwargs.batch_col=batch`, `data.kwargs.pert_col=gene`, `data.kwargs.cell_type_key=cell_type`, `data.kwargs.control_pert=non-targeting`, `data.kwargs.output_space=gene`, and optionally `data.kwargs.perturbation_features_file`. Use the same (or equivalent TOML) for consistent behavior.

**Format alignment**: 01–04 produce `obs['gene']`, `obs['split']` (train/test), `obs['curated_perturbation']`, `obs['cell_line']`, and `var` index (and often `gene_name` from curation). If STATE or your verification expects `obs['batch']`, you can set it from `cell_line` or add it manually. `var['gene_name']` is set by curation where applicable.

---

## Setup and environment

Install STATE (e.g. `pip install arc-state` or `uv tool install arc-state`). Then set paths via environment variables or CLI:

- **`TOML_CONFIG`** (required for train): Path to STATE TOML config defining datasets/splits.
- **`STATE_RUN_DIR`** (required): Output directory for training; same dir is used for predict.
- **`DATA_DIR`**: Base data directory (referenced in TOML or for adata paths); use same root as 01–04 output.
- **`EMBED_KEY`** (optional): Key in adata.obsm for STATE SE embeddings. Default is **`X_state`**.
- **`EMBEDDINGS_PATH`** (optional): Path to precomputed embeddings; merge into h5ad or set in TOML.

Run the pipeline: `python run_state.py [train|predict|both]` (default: both). Override with `--toml_config`, `--run_dir`, `--embed_key`, `--checkpoint`.

Minimal example (after running 01–04, generating SE embeddings, and creating a STATE TOML):

```bash
export DATA_DIR="/path/to/data_root"   # same as 01–04 output root
export TOML_CONFIG="/path/to/state_config.toml"
export STATE_RUN_DIR="/path/to/state_run"
python run_state.py both
```

---

## Procedure overview

1. **Data**: Run scripts 01–04 to produce h5ad files under `DATA_ROOT`. Curated data includes full-gene raw files (`<dname>_raw_with_split.h5ad`) and log-normalized/HVG outputs where applicable (see scripts/01_curate).
2. **SE embeddings (middle step)**: Run `state emb transform` with `--embed-key X_state` (or `run_state_se_embeddings.py`) on **full-gene raw** input (`<dname>_raw_with_split.h5ad`) to produce STATE SE embeddings. **Merge embeddings into the filtered h5ad**.
3. **Config**: Create a STATE TOML config that points to the **merged** h5ad (or paths from step 1) and set `EMBED_KEY=X_state` (or the key where SE embeddings are stored).
4. **Training**: Run `run_state.py train` (or `both`). This calls `state tx train` with your TOML and run dir.
5. **Inference**: Run `run_state.py predict` (or `both`). This calls `state tx predict`; outputs pred/real for evaluation.
6. **Evaluation**: Run cell-eval on the prediction and ground-truth h5ad produced by STATE; results to your chosen output directory.

---

## Script and file inventory (this repo)

| Item | Purpose |
|------|---------|
| `run_state.py` | Train and/or predict via STATE CLI. Reads TOML_CONFIG, STATE_RUN_DIR from env or CLI; calls `state tx train` and `state tx predict`. |
| `run_state_se_embeddings.py` | Prototype wrapper for `state emb transform`: generates STATE SE embeddings from full-gene raw h5ad input. |
| **Scripts 01–04** | Data preparation. See `scripts/01_curate/`, `scripts/02_embed_and_match/`, `scripts/03_make_splits/`, `scripts/04_train/` and their READMEs. STATE SE embeddings use full-gene output (e.g. `<dname>_raw_with_split.h5ad`), not HVG. |
| TOML config | User-created (or from your own scripts). Must reference h5ad paths produced by 01–04. |
| Eval script | User-provided or external. Run cell-eval on STATE pred/real h5ad; results to specified output directory. |

Assume data is prepared by 01–04 and that you provide a TOML that points to that layout.

---

## Key functions

- **CLI entrypoints**: Training and inference are done via the official STATE CLI (`state tx train`, `state tx predict` / `state tx infer`). `run_state.py` only sets paths and invokes the CLI.
- **SE embeddings**: Generated by the official STATE SE model via `state emb transform` (or `run_state_se_embeddings.py`), using full-gene raw input—not SCimilarity and not HVG data.

---

## Evaluation

Metrics are computed with the **cell-eval** package. Run your eval script on the prediction and ground-truth h5ad produced by STATE; results are written to the specified output directory.
