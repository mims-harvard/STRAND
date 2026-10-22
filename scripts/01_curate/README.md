## 01_curate

This folder contains the **PerturbQA curation** step and a **bridge** step that rewrites outputs into the folder layout expected by downstream scripts.

### 1) Run curation (`curate_perturbqa.py`)

Required inputs (not included in this repository):
- **PerturbQA metadata CSVs** directory (e.g. contains `k562-de.csv`, `rpe1-de.csv`, `jurkat-de.csv`)
- **Raw h5ad** files directory (K562/RPE1/Jurkat source h5ads referenced by the script)

Recommended environment variables:

```bash
export DATA_ROOT="/path/to/data_root"
export PERTURBQA_META_PATH="/path/to/perturbqa/datasets"
export SCRPERTURB_DATA_PATH="/path/to/raw_h5ad_dir"
```

Example (process all):

```bash
python curate_perturbqa.py --mode all --output_dir "$DATA_ROOT"   # per-line files + combined HVG file
python curate_perturbqa.py --mode raw --output_dir "$DATA_ROOT"   # raw-count files for SCimilarity
```

`--mode all` already writes the combined file; `--mode concat` rebuilds only that file.

Outputs (written into `--output_dir`):
- `k562_perturbqa_reformatted.h5ad`, `rpe1_perturbqa_reformatted.h5ad`, `jurkat_perturbqa_reformatted.h5ad`
- `k562_raw.h5ad`, `rpe1_raw.h5ad`, `jurkat_raw.h5ad`
- `combined_perturbqa_hvg2k.h5ad`

### 2) Bridge to downstream layout (`make_layout.py`)

Downstream scripts expect per-dataset folders:

```bash
python make_layout.py --curation_output_dir "$DATA_ROOT" --data_root "$DATA_ROOT"
```

This creates:
- `<DATA_ROOT>/<dname>/<dname>.h5ad` with `obs['curated_perturbation']` and `obs['split']`
- `<DATA_ROOT>/<dname>/<dname>_raw_with_split.h5ad` (for SCimilarity embedding generation)

Notes:
- `obs['curated_perturbation']` uses tags `{gene}-{dname}` and `control-{dname}` (required by OT matching bootstrap).
- If the input has no `split` column, `obs['split']` is assigned per cell at random from `--seed`
  (default 42) and `--train_frac` (default 0.8).

Next step: [`02_embed_and_match`](../02_embed_and_match/).

