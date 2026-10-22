"""Perturbation-impact summaries used in the paper's locus scans.

``energy_distance`` is the same unbiased, unsquared energy distance as
``src.diffusion_ot.evaluation.metrics.compute_e_distance`` (pooled, no labels), kept here
without the heavier package imports so locus-scan predictions can be analysed with NumPy/SciPy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist, pdist


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """``2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` with within-set means over distinct pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    d_xy = cdist(x, y).mean()
    d_xx = pdist(x).mean() if x.shape[0] > 1 else 0.0
    d_yy = pdist(y).mean() if y.shape[0] > 1 else 0.0
    return float(2.0 * d_xy - d_xx - d_yy)


def impact_score(
    predicted_pca: np.ndarray,
    reference_pca: np.ndarray,
    n_subsamples: int = 20,
    subsample_size: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Mean and standard error of energy distance between predicted cells and a control
    reference, over ``n_subsamples`` draws of ``subsample_size`` predicted cells.

    This is the per-window "perturbation impact" in the paper's scans: 256 predicted cells,
    20 draws of 64 cells, compared with 2,000 reference control cells in a 50-component PCA
    fitted on control expression.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = predicted_pca.shape[0]
    if subsample_size > n:
        raise ValueError("subsample_size exceeds the number of predicted cells")
    eds = np.array(
        [
            energy_distance(predicted_pca[rng.choice(n, subsample_size, replace=False)], reference_pca)
            for _ in range(n_subsamples)
        ]
    )
    se = eds.std(ddof=1) / np.sqrt(len(eds)) if len(eds) > 1 else 0.0
    return float(eds.mean()), float(se)


def smooth_profile(values: np.ndarray, sigma_windows: float = 1.5) -> np.ndarray:
    """Gaussian smoothing over scan windows, as used to draw the e-HDAC6 panel of Fig. 5."""
    values = np.asarray(values, dtype=float)
    return values if values.size < 3 else gaussian_filter1d(values, sigma=sigma_windows)


__all__ = ["energy_distance", "impact_score", "smooth_profile"]
