"""
Format our h5ad format to BioLord-ready h5ad.

Input: our h5ad format with obs['curated_perturbation'] or obs['gene'] (and optionally
obs['cell_line'] or obs['celltype']). Output: h5ad with obs['condition'] and
obs['cell_type'] as expected by run_biolord_standard.py (ctrl / GENE+ctrl logic).
Preserves split columns and adds target_gene for evaluation.
"""

import argparse
import os
import sys

import anndata


def _is_control(val: str) -> bool:
    v = str(val).strip().lower()
    return v in {"non-targeting", "non_targeting", "control", "ctrl", "nt"}


def build_condition_from_curated(pert: str) -> str:
    """Match run_biolord_standard: control -> ctrl, else GENE+ctrl or GENE1+GENE2."""
    pert = pert.rsplit("-", 2)[0].strip()
    if _is_control(pert) or pert == "control":
        return "ctrl"
    if "_" in pert:
        parts = pert.split("_")
        return parts[0] + "+" + parts[1]
    return pert + "+ctrl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format our h5ad format to BioLord condition/cell_type. Input: h5ad with curated_perturbation or gene."
    )
    parser.add_argument("input_file", help="Path to input h5ad")
    parser.add_argument("output_file", help="Path to save BioLord-formatted h5ad")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file does not exist: {args.input_file}", file=sys.stderr)
        return 1

    print(f"Reading {args.input_file}...")
    try:
        adata = anndata.read_h5ad(args.input_file)
        print(f"Loaded AnnData: {adata.n_obs} cells, {adata.n_vars} genes")
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}", file=sys.stderr)
        return 1

    # Source column for perturbation: curated_perturbation or gene
    if "curated_perturbation" in adata.obs.columns:
        src = adata.obs["curated_perturbation"].astype(str)
    elif "gene" in adata.obs.columns:
        src = adata.obs["gene"].astype(str)
    else:
        print(
            "ERROR: Need obs['curated_perturbation'] or obs['gene']. "
            f"Available: {list(adata.obs.columns)}",
            file=sys.stderr,
        )
        return 1

    # Build condition (same logic as run_biolord_standard.py)
    condition = [build_condition_from_curated(p) for p in src]
    adata.obs["condition"] = condition
    adata.obs["condition"] = adata.obs["condition"].to_numpy()

    # cell_type: from cell_line or celltype if present, else default
    if "cell_line" in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs["cell_line"].astype(str).values
    elif "celltype" in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs["celltype"].astype(str).values
    else:
        adata.obs["cell_type"] = "default"

    # target_gene for evaluation (non-targeting / gene name)
    target_gene = []
    for p in src:
        p_clean = str(p).rsplit("-", 2)[0].strip()
        if _is_control(p_clean) or p_clean == "control":
            target_gene.append("non-targeting")
        else:
            target_gene.append(p_clean.split("_")[0] if "_" in p_clean else p_clean)
    adata.obs["target_gene"] = target_gene

    adata.obs["ind"] = adata.obs.index

    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving to {args.output_file}...")
    try:
        adata.write_h5ad(args.output_file)
    except Exception as e:
        print(f"ERROR: Failed to write: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
