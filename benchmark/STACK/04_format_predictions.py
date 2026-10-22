#!/usr/bin/env python
"""
Convert STACK per-perturbation predictions into cell-eval format.

STACK's stack-generation outputs one h5ad per perturbation (named after the
sanitized perturbation label). This script:

  1. Loads each per-perturbation predicted h5ad from predictions/{holdout}/
  2. Averages cells per perturbation -> pseudo-bulk predicted profile
  3. Loads the true test h5ad, filters to matching perturbations, averages cells
  4. Saves adata_pred.h5ad and adata_real.h5ad to output/results/{holdout}/
     (same format consumed by cell-eval)

Usage:
    python 04_format_predictions.py --holdout k562
    python 04_format_predictions.py --holdout all

Set DATA_DIR and STACK_DIR as environment variables (see 00_setup_paths.sh).
"""
import argparse
import os
import re
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

CELL_LINES = ["k562", "jurkat", "rpe1"]

DATA_DIR = os.environ.get("DATA_DIR", "/path/to/cross_cellline_data")
STACK_DIR = os.environ.get("STACK_DIR", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(STACK_DIR, "output")

PERTURBATION_COL = "gene"
CONTROL_LABEL = "non-targeting"


def _sanitize(name: str) -> str:
    """Replicate STACK's _sanitize_split_value logic."""
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name))
    return safe or "split"


def _dense(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def load_predictions(pred_dir: str) -> dict[str, np.ndarray]:
    """
    Load per-perturbation h5ads from pred_dir.

    Returns a dict mapping perturbation_name -> mean expression array (n_genes,).
    Also returns the var_names (gene list) from the first file.
    """
    pred_path = Path(pred_dir)
    h5ad_files = sorted(pred_path.glob("*.h5ad"))

    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {pred_dir}")

    print(f"  Found {len(h5ad_files)} prediction files in {pred_dir}")

    var_names = None
    predictions: dict[str, np.ndarray] = {}

    for fpath in h5ad_files:
        pert_name = fpath.stem  # sanitized perturbation name is the filename stem
        adata = anndata.read_h5ad(fpath)

        if var_names is None:
            var_names = adata.var_names.tolist()
        elif adata.var_names.tolist() != var_names:
            warnings.warn(f"Gene order mismatch in {fpath.name}, skipping")
            continue

        # Pseudo-bulk: average across cells
        X = _dense(adata.X)
        mean_expr = X.mean(axis=0)  # (n_genes,)
        predictions[pert_name] = mean_expr

    return predictions, var_names


def reverse_sanitize_lookup(pred_keys: list[str], true_perts: list[str]) -> dict[str, str]:
    """
    Map sanitized prediction keys back to original perturbation names.

    For each sanitized key in pred_keys, find the matching original name
    from true_perts by comparing sanitized versions.
    """
    sanitized_true = {_sanitize(p): p for p in true_perts}
    mapping: dict[str, str] = {}
    unmatched = []

    for key in pred_keys:
        if key in sanitized_true:
            mapping[key] = sanitized_true[key]
        else:
            unmatched.append(key)

    if unmatched:
        print(f"  WARNING: {len(unmatched)} prediction files could not be matched "
              f"to true perturbations: {unmatched[:5]}")
    return mapping


def format_holdout(holdout: str) -> None:
    pred_dir = os.path.join(OUTPUT_DIR, "predictions", f"{holdout}_holdout")
    results_dir = os.path.join(OUTPUT_DIR, "results", holdout)
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Load predictions ────────────────────────────────────────────────────
    print(f"\n[{holdout}] Loading STACK predictions from {pred_dir}")
    predictions, var_names = load_predictions(pred_dir)
    print(f"  Loaded {len(predictions)} predicted perturbations, {len(var_names)} genes")

    # ── 2. Load true test data ─────────────────────────────────────────────────
    test_h5ad = os.path.join(DATA_DIR, holdout, f"{holdout}_test_SE_embeddings_filtered.h5ad")
    print(f"  Loading true test data from {test_h5ad}")
    test_adata = anndata.read_h5ad(test_h5ad)

    # Get available perturbations in test set (exclude control)
    true_perts_all = test_adata.obs[PERTURBATION_COL].unique().tolist()
    true_perts = [p for p in true_perts_all if p != CONTROL_LABEL]
    print(f"  True test perturbations (excl. control): {len(true_perts)}")

    # ── 3. Map sanitized keys -> original perturbation names ──────────────────
    pred_keys = list(predictions.keys())
    mapping = reverse_sanitize_lookup(pred_keys, true_perts)
    print(f"  Matched {len(mapping)} perturbations")

    if len(mapping) == 0:
        raise RuntimeError(
            "No predictions could be matched to true perturbations. "
            "Check that generation completed successfully and perturbation names align."
        )

    # ── 4. Build adata_pred (pseudo-bulk predictions) ─────────────────────────
    matched_sanitized = sorted(mapping.keys())
    matched_orig = [mapping[k] for k in matched_sanitized]

    pred_matrix = np.stack([predictions[k] for k in matched_sanitized], axis=0)
    # (n_matched_perts, n_genes)

    obs_pred = pd.DataFrame(
        {"perturbation": matched_orig},
        index=matched_orig,
    )
    var_pred = pd.DataFrame(index=var_names)

    adata_pred = anndata.AnnData(
        X=pred_matrix.astype(np.float32),
        obs=obs_pred,
        var=var_pred,
    )
    print(f"  adata_pred: {adata_pred.n_obs} obs x {adata_pred.n_vars} vars")

    # ── 5. Build adata_real (pseudo-bulk true expression) ─────────────────────
    # Align gene space to prediction gene list
    test_genes = test_adata.var_names.tolist()
    if test_genes != var_names:
        # Find intersection
        common_genes = [g for g in var_names if g in set(test_genes)]
        print(f"  Gene alignment: {len(common_genes)}/{len(var_names)} genes matched")
        test_adata = test_adata[:, common_genes].copy()
        # Reorder pred to match as well
        gene_idx = [var_names.index(g) for g in common_genes]
        pred_matrix_aligned = pred_matrix[:, gene_idx]
        adata_pred = anndata.AnnData(
            X=pred_matrix_aligned.astype(np.float32),
            obs=obs_pred,
            var=pd.DataFrame(index=common_genes),
        )
    else:
        common_genes = var_names

    real_rows = []
    for pert_orig in matched_orig:
        mask = test_adata.obs[PERTURBATION_COL] == pert_orig
        if not mask.any():
            warnings.warn(f"No cells found for perturbation '{pert_orig}' in test set")
            real_rows.append(np.zeros(len(common_genes), dtype=np.float32))
            continue
        X_pert = _dense(test_adata[mask].X)
        real_rows.append(X_pert.mean(axis=0).astype(np.float32))

    real_matrix = np.stack(real_rows, axis=0)

    obs_real = pd.DataFrame(
        {"perturbation": matched_orig},
        index=matched_orig,
    )
    adata_real = anndata.AnnData(
        X=real_matrix,
        obs=obs_real,
        var=pd.DataFrame(index=common_genes),
    )
    print(f"  adata_real: {adata_real.n_obs} obs x {adata_real.n_vars} vars")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    pred_out = os.path.join(results_dir, "adata_pred.h5ad")
    real_out = os.path.join(results_dir, "adata_real.h5ad")

    adata_pred.write_h5ad(pred_out)
    adata_real.write_h5ad(real_out)
    print(f"  Saved adata_pred -> {pred_out}")
    print(f"  Saved adata_real -> {real_out}")
    print(f"\n[{holdout}] Formatting complete. {len(matched_orig)} perturbations evaluated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        required=True,
        choices=CELL_LINES + ["all"],
        help="Cell line holdout to format (or 'all')",
    )
    args = parser.parse_args()

    if args.holdout == "all":
        for cl in CELL_LINES:
            format_holdout(cl)
    else:
        format_holdout(args.holdout)


if __name__ == "__main__":
    main()
