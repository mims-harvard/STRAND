# %%
####################################
### OT train pair splitting ###
####################################

import pandas as pd
import numpy as np
import os
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create train/val splits from OT train pairs.")
    parser.add_argument(
        "--idx_root",
        type=str,
        default=os.path.join(os.environ.get("DATA_ROOT", "./data/processed"), "ot_ids", "line_specific"),
        help="Root directory containing OT id folders: <idx_root>/<dname>/<dname>_ot_match.train.idx.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["k562", "rpe1", "jurkat", "combined_perturbqa_hvg2k"],
        help="Dataset name (dname).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.85,
        help="Fraction of unique perturbations to assign to train; remainder goes to val.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    idx_root = Path(args.idx_root).expanduser().resolve()
    dname = args.dataset
    dataset_dir = idx_root / dname
    train_path = dataset_dir / f"{dname}_ot_match.train.idx.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train OT idx file not found: {train_path}")

    data_info_train = pd.read_csv(train_path)
    if "perturbation_index" not in data_info_train.columns:
        raise ValueError(f"Expected column 'perturbation_index' in {train_path}")

    unique_pert_indices = data_info_train["perturbation_index"].unique()
    total_unique = len(unique_pert_indices)
    n_train = int(total_unique * float(args.train_frac))
    train_pert_indices = np.random.choice(unique_pert_indices, size=n_train, replace=False)
    val_pert_indices = np.array(list(set(unique_pert_indices) - set(train_pert_indices)))

    train_mask = data_info_train["perturbation_index"].isin(train_pert_indices)
    val_mask = data_info_train["perturbation_index"].isin(val_pert_indices)

    data_info_sub_train = data_info_train[train_mask]
    data_info_sub_val = data_info_train[val_mask]

    print(f"NaN in perturbation_index: {data_info_train['perturbation_index'].isna().sum()}")
    print(f"Original pairs: {len(data_info_train)}")
    print(f"Train pairs: {len(data_info_sub_train)}")
    print(f"Val pairs: {len(data_info_sub_val)}")
    print(f"Sum: {len(data_info_sub_train) + len(data_info_sub_val)}")
    print(f"Match: {len(data_info_train) == len(data_info_sub_train) + len(data_info_sub_val)}")

    out_train = dataset_dir / f"{dname}_ot_match.trainsub.idx.csv"
    out_val = dataset_dir / f"{dname}_ot_match.valsub.idx.csv"
    data_info_sub_train.to_csv(out_train, index=False)
    data_info_sub_val.to_csv(out_val, index=False)
    print(f"Wrote: {out_train}")
    print(f"Wrote: {out_val}")


if __name__ == "__main__":
    main()
