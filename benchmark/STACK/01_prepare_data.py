#!/usr/bin/env python
"""
Prepare cross-cell-line data for STACK training and generation.

For each holdout cell line:
  - Merges training cell lines into combined_train_dir/combined_train.h5ad
    (STACK expects a directory containing h5ad files, so we put it in a subdir)
  - Extracts gene list (hvg.pkl) from var_names (data is already 2000-HVG-filtered)
  - Saves context.h5ad for ICL generation (all training cell perturbations)
  - Saves test_control.h5ad for ICL generation (only non-targeting cells from holdout)

Usage:
    python 01_prepare_data.py --holdout k562
    python 01_prepare_data.py --holdout all

Set DATA_DIR and STACK_DIR as environment variables (see 00_setup_paths.sh).
"""
import argparse
import os
import pickle

import anndata

CELL_LINES = ["k562", "jurkat", "rpe1"]

DATA_DIR = os.environ.get("DATA_DIR", "/path/to/cross_cellline_data")
STACK_DIR = os.environ.get("STACK_DIR", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(STACK_DIR, "output")

CONTROL_LABEL = "non-targeting"
PERTURBATION_COL = "gene"


def prepare_holdout(holdout: str) -> None:
    training_cell_lines = [cl for cl in CELL_LINES if cl != holdout]
    out_dir = os.path.join(OUTPUT_DIR, "data", f"{holdout}_holdout")
    # STACK training expects a directory (it globs *.h5ad inside it)
    train_dir = os.path.join(out_dir, "combined_train_dir")
    os.makedirs(train_dir, exist_ok=True)

    # ── 1. Load and concatenate training cell lines ────────────────────────────
    print(f"\n[{holdout}] Loading training cell lines: {training_cell_lines}")
    adatas = []
    for cl in training_cell_lines:
        path = os.path.join(DATA_DIR, cl, f"{cl}_train_SE_embeddings_filtered.h5ad")
        print(f"  Reading {path} ...")
        adata = anndata.read_h5ad(path)
        # Drop SE embeddings stored in obsm – STACK only uses X
        adata.obsm.clear()
        print(f"    -> {adata.n_obs} cells, {adata.n_vars} genes, "
              f"cell_lines: {adata.obs['cell_line'].unique().tolist()}")
        adatas.append(adata)

    combined = anndata.concat(adatas, axis=0)
    print(f"  Combined: {combined.n_obs} cells, {combined.n_vars} genes")

    # ── 2. Save combined to training directory ─────────────────────────────────
    train_h5ad = os.path.join(train_dir, "combined_train.h5ad")
    print(f"  Saving combined training data -> {train_h5ad}")
    combined.write_h5ad(train_h5ad)

    # ── 3. Save HVG gene list (var_names = already 2000 HVGs) ─────────────────
    hvg_path = os.path.join(out_dir, "hvg.pkl")
    gene_list = combined.var_names.tolist()
    with open(hvg_path, "wb") as f:
        pickle.dump(gene_list, f)
    print(f"  HVG list ({len(gene_list)} genes) -> {hvg_path}")

    # ── 4. Save context.h5ad (ICL base = all training perturbations) ──────────
    context_path = os.path.join(out_dir, "context.h5ad")
    print(f"  Saving context h5ad -> {context_path}")
    combined.write_h5ad(context_path)

    # ── 5. Save test_control.h5ad (control cells from held-out cell line) ─────
    test_path = os.path.join(DATA_DIR, holdout, f"{holdout}_test_SE_embeddings_filtered.h5ad")
    print(f"  Loading held-out test data from {test_path} ...")
    test_adata = anndata.read_h5ad(test_path)
    test_adata.obsm.clear()

    control_mask = test_adata.obs[PERTURBATION_COL] == CONTROL_LABEL
    n_control = control_mask.sum()
    if n_control == 0:
        raise ValueError(
            f"No control cells found in {holdout} test set "
            f"(column '{PERTURBATION_COL}', value '{CONTROL_LABEL}'). "
            f"Available values: {test_adata.obs[PERTURBATION_COL].unique().tolist()[:10]}"
        )

    test_control = test_adata[control_mask].copy()
    control_path = os.path.join(out_dir, "test_control.h5ad")
    print(f"  Control cells ({n_control}) -> {control_path}")
    test_control.write_h5ad(control_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[{holdout}] Data preparation complete:")
    print(f"  Train dir  : {train_dir}")
    print(f"  HVG pkl    : {hvg_path}  ({len(gene_list)} genes)")
    print(f"  Context    : {context_path}  ({combined.n_obs} cells)")
    print(f"  Test ctrl  : {control_path}  ({n_control} cells)")
    print(f"  Perturbations in context: "
          f"{sorted(combined.obs[PERTURBATION_COL].unique().tolist())[:5]} ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        required=True,
        choices=CELL_LINES + ["all"],
        help="Cell line to hold out (or 'all' to run all three)",
    )
    args = parser.parse_args()

    if args.holdout == "all":
        for cl in CELL_LINES:
            prepare_holdout(cl)
    else:
        prepare_holdout(args.holdout)


if __name__ == "__main__":
    main()
