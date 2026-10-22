## 02_embed_and_match

This folder generates:
- **RNA embeddings** using **SCimilarity** (optional; required if you want to reproduce embeddings)
- **OT-matched perturb/control index pairs** using an external **HiRef** OT solver

### Requirements
- SCimilarity install: see `https://github.com/Genentech/scimilarity`
- HiRef OT solver code (external): provide its `src/` so that `import HR_OT` and `import rank_annealing` works.

Recommended environment variables:

```bash
export DATA_ROOT="/path/to/data_root"
export SCIMILARITY_MODEL_PATH="/path/to/scimilarity/model_dir"   # required for embedding generation
export HIREF_SRC="/path/to/HiRef/src"                            # required for OT matching
```

### 1) Generate SCimilarity embeddings (chunked)

This reads `<DATA_ROOT>/<dname>/<dname>_raw_with_split.h5ad` and writes chunk files under:
`<DATA_ROOT>/<dname>/_scim_chunks/`.

```bash
python embed_and_match.py --dataset k562 --data_root "$DATA_ROOT" --gen_embeddings --chunk_id 0
```

Run additional chunks by varying `--chunk_id` (the script uses a fixed `n_chunks=10`).

### 2) Aggregate chunked embeddings

This writes the final file:
`<DATA_ROOT>/<dname>/<dname>.X_scimilarity.h5`

```bash
python embed_and_match.py --dataset k562 --data_root "$DATA_ROOT" --agg_embeddings
```

### 3) OT matching (writes idx CSVs)

This reads:
- `<DATA_ROOT>/<dname>/<dname>.h5ad`  (must contain `obs['curated_perturbation']` and `obs['split']`)
- `<DATA_ROOT>/<dname>/<dname>.X_scimilarity.h5`

and writes:
`<DATA_ROOT>/ot_ids/<ot_ids_tag>/<dname>/<dname>_ot_match.{train|test}.idx.csv`

```bash
python embed_and_match.py --dataset k562 --data_root "$DATA_ROOT" --ot_match --ot_ids_tag line_specific
```

If your split column is not `split`, pass `--split_column your_column`.

The split column fixes the experimental setting for everything downstream: the train/test
membership is written into the OT index files, which training and
`scripts/05_evaluate/evaluate.py` read. `train.py --run_group` only labels the output. By
convention `split` holds the zero-shot split and `split_few_shot` the few-shot split.

### 4) DNA embedding cache (GPU)

Runs the frozen Flashzoi/Borzoi backbone once per perturbed gene and writes the cache that
training and evaluation read. Needs `DNA_REFERENCE_PATH`, `GENE_ANNOTATION_PATH`,
`HGNC_COMPLETE_SET_PATH` (HGNC `hgnc_complete_set.txt`) and, for the cCRE bias used in the paper,
`CCRE_BED_PATH` (ENCODE cCREs, `ENCFF234XEZ.bed.gz`).

```bash
python build_dna_cache.py --idx_dir "$DATA_ROOT/ot_ids/line_specific/k562" --dna_cache_dir "$DNA_CACHE_DIR"
```

Next step: [`03_make_splits`](../03_make_splits/).

