"""
Utilities for unseen cell-line inference via convex mixtures over *seen* cell lines.

Design goals:
- Backward compatible: nothing here is used unless explicitly invoked by training/eval scripts.
- Occam: no tunable hyperparameters by default.
- General: supports any number of seen cell lines (C >= 1).

Core idea:
Given seen cell-line prototypes mu_c in a shared control-latent space (z_ctrl),
infer mixture weights w on the simplex:
    w* = argmin_{w >= 0, sum w = 1} || q - sum_c w_c mu_c ||^2
where q is a query embedding for an unseen cell line (typically a batch mean).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def compute_cell_line_prototypes_from_arrays(
    control_emb: np.ndarray,
    cell_line_ids: np.ndarray,
    num_cell_lines: int,
) -> np.ndarray:
    """
    Compute per-cell-line prototypes mu_c = E[z_ctrl | cell_line=c].

    Args:
        control_emb: (N, D) control embeddings (same space as batch['control'])
        cell_line_ids: (N,) integer ids in [0, num_cell_lines-1]
        num_cell_lines: number of seen cell lines C

    Returns:
        mus: (C, D) float32 array of prototypes.
    """
    if control_emb.ndim != 2:
        raise ValueError(f"control_emb must be 2D (N,D), got shape {control_emb.shape}")
    if cell_line_ids.ndim != 1:
        cell_line_ids = cell_line_ids.reshape(-1)
    if control_emb.shape[0] != cell_line_ids.shape[0]:
        raise ValueError(
            f"control_emb N={control_emb.shape[0]} != cell_line_ids N={cell_line_ids.shape[0]}"
        )
    if num_cell_lines <= 0:
        raise ValueError(f"num_cell_lines must be positive, got {num_cell_lines}")

    N, D = control_emb.shape
    mus = np.zeros((num_cell_lines, D), dtype=np.float64)
    cnt = np.zeros((num_cell_lines,), dtype=np.int64)

    ids = cell_line_ids.astype(np.int64, copy=False)
    if ids.min(initial=0) < 0 or ids.max(initial=0) >= num_cell_lines:
        raise ValueError(
            f"cell_line_ids out of range: min={ids.min()}, max={ids.max()}, num_cell_lines={num_cell_lines}"
        )

    # Accumulate sums per id (vectorized scatter-add in numpy via bincount per dimension)
    for c in range(num_cell_lines):
        m = (ids == c)
        if not np.any(m):
            continue
        mus[c] = control_emb[m].mean(axis=0, dtype=np.float64)
        cnt[c] = int(m.sum())

    # If some cell lines are absent, keep their prototype at 0 (caller can decide).
    # Convert to float32 for downstream torch usage.
    return mus.astype(np.float32, copy=False)


def _solve_equality_constrained_ls(
    q: np.ndarray,
    mu: np.ndarray,
    active: np.ndarray,
    ridge: float = 1e-12,
) -> np.ndarray:
    """
    Solve min ||mu_active^T w - q||^2 subject to sum(w)=1 (no nonnegativity here).

    mu: (C, D)
    q: (D,)
    active: boolean mask (C,)
    """
    mu_a = mu[active]  # (k, D)
    k = mu_a.shape[0]
    if k == 1:
        return np.array([1.0], dtype=np.float64)

    # H = A^T A where A = mu_a^T (D,k) => H = mu_a mu_a^T (k,k)
    H = mu_a @ mu_a.T  # (k,k)
    if ridge > 0:
        H = H + ridge * np.eye(k, dtype=np.float64)
    g = mu_a @ q  # (k,)

    # KKT system:
    # [H  1] [w ] = [g]
    # [1^T0] [nu]   [1]
    KKT = np.zeros((k + 1, k + 1), dtype=np.float64)
    KKT[:k, :k] = H
    KKT[:k, k] = 1.0
    KKT[k, :k] = 1.0
    rhs = np.zeros((k + 1,), dtype=np.float64)
    rhs[:k] = g
    rhs[k] = 1.0

    sol = np.linalg.solve(KKT, rhs)
    w = sol[:k]
    return w


def simplex_constrained_mixture_weights(
    q: np.ndarray,
    prototypes: np.ndarray,
    ridge: float = 1e-12,
) -> np.ndarray:
    """
    Active-set solver for:
        argmin_{w >= 0, sum w = 1} || q - sum_c w_c prototypes[c] ||^2

    This is hyperparameter-free (ridge is a tiny numerical stabilizer, not a tunable knob).

    Args:
        q: (D,) query embedding (float32/float64)
        prototypes: (C, D) prototypes for seen cell lines
        ridge: small diagonal added to normal equations for stability

    Returns:
        w: (C,) float32 weights on the simplex.
    """
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must be 2D (C,D), got {prototypes.shape}")
    C, D = prototypes.shape
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != D:
        raise ValueError(f"q dim {q.shape[0]} != prototypes dim {D}")
    mu = np.asarray(prototypes, dtype=np.float64)

    if C == 1:
        return np.array([1.0], dtype=np.float32)

    active = np.ones((C,), dtype=bool)
    w_full = np.zeros((C,), dtype=np.float64)

    # Active-set: solve equality LS on active set, drop negative weights, repeat.
    # Deterministic and finite: each iteration removes at least one index.
    while True:
        if active.sum() == 0:
            # Degenerate fallback: uniform
            return (np.ones((C,), dtype=np.float32) / C)

        w_a = _solve_equality_constrained_ls(q=q, mu=mu, active=active, ridge=ridge)  # (k,)

        # Place into full vector
        w_full[:] = 0.0
        w_full[active] = w_a

        # Check feasibility
        neg = (w_full < 0)
        if not np.any(neg):
            # Numerical cleanup: clamp tiny negatives to 0 and renormalize
            w = np.maximum(w_full, 0.0)
            s = w.sum()
            if s <= 0:
                w = np.ones((C,), dtype=np.float64) / C
            else:
                w = w / s
            return w.astype(np.float32, copy=False)

        # Drop the most negative weight(s)
        most_neg_idx = np.where(neg)[0][np.argmin(w_full[neg])]
        active[most_neg_idx] = False


def infer_cell_line_mix_weights_from_batch(
    z_ctrl_batch: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    level: str = "batch",
) -> torch.Tensor:
    """
    Infer mixture weights given a batch of control embeddings and prototypes.

    Args:
        z_ctrl_batch: (B, D) control embeddings (same as batch['control'])
        prototypes: (C, D) prototypes tensor
        level: "batch" (default) or "sample"

    Returns:
        W: (B, C) weights tensor on CPU (float32).
    """
    if level not in {"batch", "sample"}:
        raise ValueError(f"level must be 'batch' or 'sample', got {level}")
    if z_ctrl_batch.ndim != 2:
        raise ValueError(f"z_ctrl_batch must be (B,D), got {tuple(z_ctrl_batch.shape)}")
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must be (C,D), got {tuple(prototypes.shape)}")
    B, D = z_ctrl_batch.shape
    C, D2 = prototypes.shape
    if D != D2:
        raise ValueError(f"dim mismatch: z_ctrl D={D} prototypes D={D2}")

    z_np = z_ctrl_batch.detach().cpu().to(torch.float32).numpy()
    proto_np = prototypes.detach().cpu().to(torch.float32).numpy()

    if level == "batch":
        q = z_np.mean(axis=0)
        w = simplex_constrained_mixture_weights(q=q, prototypes=proto_np)  # (C,)
        W = np.repeat(w[None, :], repeats=B, axis=0)
        return torch.from_numpy(W.astype(np.float32, copy=False))

    # per-sample
    W = np.zeros((B, C), dtype=np.float32)
    for i in range(B):
        W[i] = simplex_constrained_mixture_weights(q=z_np[i], prototypes=proto_np)
    return torch.from_numpy(W)


