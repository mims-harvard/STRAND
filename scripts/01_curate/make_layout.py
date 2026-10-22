#!/usr/bin/env python3
"""
Bridge script: convert `curate_perturbqa.py` outputs into the folder/file layout
expected by the downstream embedding/OT and training scripts.

Expected downstream layout (under DATA_ROOT):
  DATA_ROOT/
    k562/
      k562.h5ad
      k562_raw_with_split.h5ad
    rpe1/
      rpe1.h5ad
      rpe1_raw_with_split.h5ad
    jurkat/
      jurkat.h5ad
      jurkat_raw_with_split.h5ad
    combined_perturbqa_hvg2k/
      combined_perturbqa_hvg2k.h5ad
      combined_perturbqa_hvg2k_raw_with_split.h5ad  (optional/fallback)

This script ensures:
  - obs['curated_perturbation'] exists
  - obs['split'] exists with values in {'train','test'}
  - obs['cell_line'] exists when available
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(_repo_root() / "data" / "processed"))).expanduser().resolve()


def _is_control(val: str) -> bool:
    v = str(val).strip().lower()
    return v in {"non-targeting", "non_targeting", "control", "ctrl", "nt"}


def ensure_curated_labels(adata, dname: str) -> None:
    """
    Populate `adata.obs['curated_perturbation']` with tags that match the assumptions in
    `scripts/02_embed_and_match/ot_utils.bootstrap_cells`:
      - perturb tags:   {gene}-{dname}
      - control tag:    control-{dname}
    """
    if "curated_perturbation" in adata.obs.columns:
        # Assume upstream already created correct labels.
        return

    # Prefer PerturbQA-style `gene` column; fall back to `perturbation` if present.
    if "gene" in adata.obs.columns:
        src = adata.obs["gene"].astype(str)
    elif "perturbation" in adata.obs.columns:
        src = adata.obs["perturbation"].astype(str)
    else:
        raise ValueError(
            f"Cannot derive curated labels for {dname}: need obs['gene'] or obs['perturbation']."
        )

    out = []
    for x in src.values:
        if _is_control(x):
            out.append(f"control-{dname}")
        else:
            out.append(f"{x}-{dname}")
    adata.obs["curated_perturbation"] = np.asarray(out, dtype=object)


def ensure_split(adata, seed: int, train_frac: float) -> None:
    if "split" in adata.obs.columns:
        return
    rng = np.random.default_rng(seed)
    mask = rng.random(adata.n_obs) < float(train_frac)
    split = np.where(mask, "train", "test")
    adata.obs["split"] = split.astype(object)


def ensure_cell_line(adata, cell_line: Optional[str]) -> None:
    if "cell_line" in adata.obs.columns:
        return
    if cell_line is None:
        return
    adata.obs["cell_line"] = str(cell_line)


def write_dataset(
    *,
    dname: str,
    src_h5ad: Path,
    dst_dir: Path,
    dst_name: str,
    seed: int,
    train_frac: float,
    cell_line: Optional[str],
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / dst_name

    adata = sc.read_h5ad(src_h5ad)
    ensure_cell_line(adata, cell_line=cell_line)
    ensure_curated_labels(adata, dname=dname)
    ensure_split(adata, seed=seed, train_frac=train_frac)
    adata.write(dst_path)
    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curation_output_dir",
        type=str,
        default=None,
        help="Directory containing outputs from curate_perturbqa.py. "
             "If omitted, uses DATA_ROOT (recommended).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Target DATA_ROOT for downstream pipeline. Defaults to env DATA_ROOT or <repo>/data/processed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument(
        "--combined_raw_fallback",
        action="store_true",
        default=True,
        help="If no true combined raw file exists, create a fallback "
             "`combined_perturbqa_hvg2k_raw_with_split.h5ad` by copying the curated combined file "
             "(note: this is NOT raw counts).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else _default_data_root()
    curation_out = (
        Path(args.curation_output_dir).expanduser().resolve()
        if args.curation_output_dir
        else data_root
    )

    # Map expected curation outputs → dataset folders.
    spec = [
        ("k562", "k562_perturbqa_reformatted.h5ad", "k562.h5ad", "k562"),
        ("rpe1", "rpe1_perturbqa_reformatted.h5ad", "rpe1.h5ad", "rpe1"),
        ("jurkat", "jurkat_perturbqa_reformatted.h5ad", "jurkat.h5ad", "jurkat"),
        ("combined_perturbqa_hvg2k", "combined_perturbqa_hvg2k.h5ad", "combined_perturbqa_hvg2k.h5ad", None),
    ]

    raw_spec = [
        ("k562", "k562_raw.h5ad", "k562_raw_with_split.h5ad", "k562"),
        ("rpe1", "rpe1_raw.h5ad", "rpe1_raw_with_split.h5ad", "rpe1"),
        ("jurkat", "jurkat_raw.h5ad", "jurkat_raw_with_split.h5ad", "jurkat"),
        # combined raw is not produced by curate_perturbqa; handled separately.
    ]

    print(f"[make_layout] data_root: {data_root}")
    print(f"[make_layout] curation_output_dir: {curation_out}")

    # Curated files
    for dname, src_name, dst_name, cell_line in spec:
        src = curation_out / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing curation output: {src}")
        dst_dir = data_root / dname
        out = write_dataset(
            dname=dname,
            src_h5ad=src,
            dst_dir=dst_dir,
            dst_name=dst_name,
            seed=args.seed,
            train_frac=args.train_frac,
            cell_line=cell_line,
        )
        print(f"[make_layout] wrote: {out}")

    # Raw-with-split files
    for dname, src_name, dst_name, cell_line in raw_spec:
        src = curation_out / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing raw curation output: {src}")
        dst_dir = data_root / dname
        out = write_dataset(
            dname=dname,
            src_h5ad=src,
            dst_dir=dst_dir,
            dst_name=dst_name,
            seed=args.seed,
            train_frac=args.train_frac,
            cell_line=cell_line,
        )
        print(f"[make_layout] wrote: {out}")

    # Combined raw fallback (optional)
    combined_dir = data_root / "combined_perturbqa_hvg2k"
    combined_curated = combined_dir / "combined_perturbqa_hvg2k.h5ad"
    combined_raw_dst = combined_dir / "combined_perturbqa_hvg2k_raw_with_split.h5ad"
    if args.combined_raw_fallback and combined_curated.exists() and (not combined_raw_dst.exists()):
        adata = sc.read_h5ad(combined_curated)
        ensure_curated_labels(adata, dname="combined_perturbqa_hvg2k")
        ensure_split(adata, seed=args.seed, train_frac=args.train_frac)
        # if cell_line exists upstream (it should), keep it; else leave absent
        adata.write(combined_raw_dst)
        print(
            "[make_layout] wrote combined raw fallback (note: not raw counts): "
            f"{combined_raw_dst}"
        )


if __name__ == "__main__":
    main()

