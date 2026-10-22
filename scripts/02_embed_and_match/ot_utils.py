"""
Minimal utilities for `embed_and_match.py`.

This file intentionally vendors only what the pipeline needs:
  - `bootstrap_cells`: build case-control index pairs from `adata.obs`.

It is adapted from the main Diffusion-OT repository but removes any personal
absolute-path imports.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm


def bootstrap_cells(
    adata,
    dname,
    perturb_col="curated_perturbation",
    baseline_strategy="median",
    min_cells=60,
    max_repeat_factor=3.0,
    baseline=None,
    random_state=None,
    enable_augmentation=True,
    cell_line_col="cell_line",
    match_within_cell_line=True,
):
    """
    Bootstrap cells to create case-control pairs.

    Assumptions on `adata.obs[perturb_col]` tags:
      - perturb tags:   {gene}-{dname}  OR {gene}-{dname}-batch_{batch_id}
      - control tags:   control-{dname} OR control-{dname}-batch_{batch_id}

    If `match_within_cell_line=True` and `cell_line_col` exists, matching is
    stratified by cell line (perturbations are paired only with controls from
    the same cell line).
    """
    rng = np.random.default_rng(seed=random_state)

    use_cell_line = bool(match_within_cell_line) and (cell_line_col in adata.obs.columns)
    if match_within_cell_line and not use_cell_line:
        print(
            f"Warning: match_within_cell_line=True but '{cell_line_col}' not found in adata.obs; "
            "falling back to non-stratified matching.",
            flush=True,
        )
    cell_line_values = None
    if use_cell_line:
        cell_line_values = adata.obs[cell_line_col].astype("string").to_numpy()

    print("Creating indices for each perturbation tag...", flush=True)
    grouped_by_tag = adata.obs.groupby(perturb_col, observed=True, sort=False)
    perturb_to_indices = {
        g: np.asarray(pos, dtype=np.int64)
        for g, pos in grouped_by_tag.indices.items()
        if pd.notna(g)
    }

    batch_pattern = f"-{dname}-batch_"
    data_pattern = f"-{dname}"
    perturb_tags = adata.obs[perturb_col].astype("string")

    def extract_perturb_type(tag):
        if pd.isna(tag) or not isinstance(tag, str):
            return tag
        if batch_pattern in tag:
            return tag.split(batch_pattern)[0]
        if data_pattern in tag:
            return tag.split(data_pattern)[0]
        return tag

    perturb_type = perturb_tags.apply(extract_perturb_type)
    perturb_type_counts = perturb_type.value_counts(dropna=True)

    baseline_by_cell_line = None
    if enable_augmentation:
        if use_cell_line:
            unique_cls = pd.unique(cell_line_values)
            if isinstance(baseline, dict):
                baseline_by_cell_line = {
                    str(cl): max(int(baseline.get(cl, min_cells)), int(min_cells)) for cl in unique_cls
                }
            elif baseline is not None:
                baseline_scalar = max(int(baseline), int(min_cells))
                baseline_by_cell_line = {str(cl): baseline_scalar for cl in unique_cls}
            else:
                cell_line_arr = np.asarray(cell_line_values)
                perturb_type_arr = np.asarray(perturb_type.astype(str), dtype=object)
                df_counts = pd.DataFrame({"cell_line": cell_line_arr, "perturb_type": perturb_type_arr})
                counts = df_counts.groupby(["cell_line", "perturb_type"], observed=True).size()
                baseline_by_cell_line = {}
                for cl in unique_cls:
                    if pd.isna(cl):
                        continue
                    try:
                        sub = counts.xs(cl, level="cell_line")
                    except KeyError:
                        sub = pd.Series(dtype=int)
                    non_control_counts = [
                        int(c)
                        for p_type, c in sub.items()
                        if isinstance(p_type, str) and (not p_type.startswith("control"))
                    ]
                    if baseline_strategy == "median":
                        b = int(np.median(non_control_counts)) if non_control_counts else int(min_cells)
                    elif isinstance(baseline_strategy, (int, float)):
                        b = int(baseline_strategy)
                    else:
                        b = int(min_cells)
                    baseline_by_cell_line[str(cl)] = max(int(b), int(min_cells))

            print(
                f"Baseline for {dname} (per cell_line): "
                + ", ".join([f"{cl}={baseline_by_cell_line[str(cl)]}" for cl in list(baseline_by_cell_line.keys())[:5]])
                + (" ..." if len(baseline_by_cell_line) > 5 else ""),
                flush=True,
            )
        else:
            if baseline is None:
                if baseline_strategy == "median":
                    non_control_counts = [
                        count
                        for p_type, count in perturb_type_counts.items()
                        if isinstance(p_type, str) and not p_type.startswith("control")
                    ]
                    baseline = int(np.median(non_control_counts)) if non_control_counts else int(min_cells)
                elif isinstance(baseline_strategy, (int, float)):
                    baseline = int(baseline_strategy)
                else:
                    baseline = int(min_cells)
            baseline = max(int(baseline), int(min_cells))
            print(f"Baseline for {dname}: {baseline}", flush=True)
    else:
        if baseline is None:
            baseline = int(min_cells)
        print(f"Augmentation disabled for {dname} - using actual perturbation cell counts", flush=True)

    perturb_type_to_tags = defaultdict(list)
    for ptag in perturb_to_indices:
        if isinstance(ptag, str) and (not ptag.startswith("control")) and pd.notna(ptag):
            if batch_pattern in ptag:
                p_type = ptag.split(batch_pattern)[0]
            elif data_pattern in ptag:
                p_type = ptag.split(data_pattern)[0]
            else:
                p_type = ptag
            perturb_type_to_tags[p_type].append(ptag)

    bootstrap_indices = defaultdict(lambda: defaultdict(dict))

    print("Processing perturbations and sampling...", flush=True)
    for p_type, ptags in tqdm(perturb_type_to_tags.items(), desc="Bootstrap sampling"):
        total_count = int(perturb_type_counts.get(p_type, 0))
        if total_count == 0:
            continue

        pairable_by_cell_line = defaultdict(list)
        for ptag in sorted(ptags):
            pert_idx = perturb_to_indices.get(ptag)
            if pert_idx is None or len(pert_idx) == 0:
                continue

            if batch_pattern in ptag:
                try:
                    batch_id = ptag.split(batch_pattern)[1]
                    ctag = f"control-{dname}-batch_{batch_id}"
                except IndexError:
                    continue
            elif data_pattern in ptag:
                batch_id = "no_batch"
                ctag = f"control-{dname}"
            else:
                continue

            ctrl_idx = perturb_to_indices.get(ctag)
            if ctrl_idx is None or len(ctrl_idx) == 0:
                continue

            if use_cell_line:
                pert_cls = cell_line_values[pert_idx]
                ctrl_cls = cell_line_values[ctrl_idx]
                for cl in pd.unique(pert_cls):
                    cl_str = str(cl)
                    pert_idx_cl = pert_idx[pert_cls == cl]
                    if pert_idx_cl is None or len(pert_idx_cl) == 0:
                        continue
                    ctrl_idx_cl = ctrl_idx[ctrl_cls == cl]
                    if ctrl_idx_cl is None or len(ctrl_idx_cl) == 0:
                        print(
                            f"Warning: skipping {ptag} (batch={batch_id}, cell_line={cl_str}) "
                            "because no same-cell-line controls found.",
                            flush=True,
                        )
                        continue
                    pairable_by_cell_line[cl_str].append(
                        (ptag, batch_id, cl_str, pert_idx_cl, ctrl_idx_cl, len(pert_idx_cl), len(ctrl_idx_cl))
                    )
            else:
                pairable_by_cell_line["__all__"].append(
                    (ptag, batch_id, "__all__", pert_idx, ctrl_idx, len(pert_idx), len(ctrl_idx))
                )

        if len(pairable_by_cell_line) == 0:
            continue

        for cl_key, pairable in pairable_by_cell_line.items():
            if not pairable:
                continue
            total_pairable_pert = int(sum(pc for *_, pc, _ in pairable))
            total_pairable_ctrl = int(sum(cc for *_, _, cc in pairable))
            if total_pairable_pert <= 0 or total_pairable_ctrl <= 0:
                continue

            if enable_augmentation:
                baseline_use = (
                    int(baseline_by_cell_line.get(cl_key, min_cells))
                    if (use_cell_line and baseline_by_cell_line is not None)
                    else int(baseline)
                )
                target_total_samples = max(int(total_pairable_pert), int(baseline_use), int(min_cells))
                max_allowed_global_pert = int(np.floor(total_pairable_pert * float(max_repeat_factor)))
                max_allowed_global_ctrl = int(np.floor(total_pairable_ctrl * float(max_repeat_factor)))
                max_allowed_global = min(max_allowed_global_pert, max_allowed_global_ctrl)
                max_allowed_global = max(int(total_pairable_pert), int(max_allowed_global))
                target_total_samples = min(int(target_total_samples), int(max_allowed_global))

                counts = np.array([pc for *_, pc, _ in pairable], dtype=int)
                total_counts = int(counts.sum())
                if total_counts <= 0:
                    continue

                extra = int(target_total_samples - total_counts)
                extra_alloc = np.zeros_like(counts, dtype=int)
                if extra > 0:
                    weights = counts.astype(float)
                    ideal = weights / float(weights.sum()) * extra
                    extra_alloc = np.floor(ideal).astype(int)
                    remainder = int(extra - extra_alloc.sum())
                    if remainder > 0:
                        frac = ideal - extra_alloc
                        order = np.lexsort((-weights, -frac))[::-1]
                        for idx in order[:remainder]:
                            extra_alloc[idx] += 1

                want = counts + extra_alloc
                caps = []
                for i, (*_, pert_count, ctrl_count) in enumerate(pairable):
                    cap_pert = int(np.floor(pert_count * float(max_repeat_factor)))
                    cap_ctrl = int(np.floor(ctrl_count * float(max_repeat_factor)))
                    cap = min(cap_pert, cap_ctrl)
                    cap = max(int(pert_count), int(cap))
                    caps.append(cap)
                caps = np.asarray(caps, dtype=int)
                want = np.minimum(want, caps)

                deficit = int(target_total_samples - int(want.sum()))
                if deficit > 0:
                    headroom = caps - want
                    weights = counts.astype(float)
                    rank = np.lexsort((-weights, -headroom))[::-1]
                    for i in rank:
                        if deficit == 0:
                            break
                        if headroom[i] <= 0:
                            continue
                        add = min(int(headroom[i]), int(deficit))
                        want[i] += add
                        deficit -= add

                for (ptag, batch_id, cl, pert_idx, ctrl_idx, pert_count, ctrl_count), n in zip(pairable, want):
                    if pert_count <= 0:
                        continue
                    n = int(max(n, pert_count))
                    extra_n = int(n - pert_count)
                    if extra_n > 0:
                        extra_pert = rng.choice(pert_idx, size=extra_n, replace=True).astype(np.int64)
                        perturb_sample = np.concatenate([pert_idx.astype(np.int64), extra_pert], axis=0)
                    else:
                        perturb_sample = pert_idx.astype(np.int64).copy()
                    rng.shuffle(perturb_sample)

                    repl_ctrl = (ctrl_count < n)
                    control_sample = rng.choice(ctrl_idx, size=n, replace=repl_ctrl).astype(np.int64)

                    out_tag = f"{ptag}||cell_line={cl}" if use_cell_line else ptag
                    bootstrap_indices[batch_id][out_tag] = (perturb_sample, control_sample)
            else:
                for (ptag, batch_id, cl, pert_idx, ctrl_idx, pert_count, ctrl_count) in pairable:
                    if pert_count <= 0:
                        continue
                    n = int(pert_count)
                    perturb_sample = pert_idx.astype(np.int64).copy()
                    rng.shuffle(perturb_sample)
                    repl_ctrl = (ctrl_count < n)
                    control_sample = rng.choice(ctrl_idx, size=n, replace=repl_ctrl).astype(np.int64)

                    out_tag = f"{ptag}||cell_line={cl}" if use_cell_line else ptag
                    bootstrap_indices[batch_id][out_tag] = (perturb_sample, control_sample)

    return bootstrap_indices

