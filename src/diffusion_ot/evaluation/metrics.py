import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, ks_2samp
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_kernels
from sklearn.metrics import pairwise_distances
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances


from scipy.stats import energy_distance


import numpy as np
from scipy.spatial.distance import cdist, pdist



def compute_e_distance(X, Y, labels=None, average='macro', return_all=False):
    """
    Energy distance (unsquared) between two empirical samples.
    If `labels` are provided as a single list, compute per-perturbation distances and aggregate.

    Parameters
    ----------
    X : array-like, shape (n, d)
    Y : array-like, shape (m, d)
    labels : None OR single list
        - length n+m: first n are for X, next m are for Y
        - length n (requires n == m): shared labels applied to both X and Y
        If None, computes pooled energy distance over all samples.
    average : {'macro','micro','pooled'}
    return_all : bool

    Returns
    -------
    float or dict
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]

    # --- helpers ---
    def _intraset_mean_unbiased(A):
        if A.shape[0] < 2:
            return 0.0
        return pdist(A, metric='euclidean').mean()

    def _energy_unbiased(A, B):
        if A.shape[0] == 0 or B.shape[0] == 0:
            return np.nan
        d_xy_mean = cdist(A, B, metric='euclidean').mean()
        d_xx_mean = _intraset_mean_unbiased(A)
        d_yy_mean = _intraset_mean_unbiased(B)
        return 2.0 * d_xy_mean - d_xx_mean - d_yy_mean

    # --- pooled (no labels) ---
    if labels is None:
        pooled = _energy_unbiased(X, Y)
        if return_all:
            return {
                'macro': pooled, 'micro': pooled, 'pooled': pooled,
                'per_group': {'ALL': pooled}, 'counts': {'ALL': (n, m)}
            }
        return pooled  # average ignored since all three equal pooled

    # --- single labels list parsing ---
    labels = list(labels)
    if len(labels) == n + m:
        lx = np.asarray(labels[:n])
        ly = np.asarray(labels[n:])
    elif len(labels) == n and n == m:
        lx = ly = np.asarray(labels)
    else:
        raise ValueError(
            "labels must be a single list of length n+m (concat) or length n when n==m (shared)."
        )

    # groups present in BOTH X and Y
    groups = np.intersect1d(np.unique(lx), np.unique(ly))
    if groups.size == 0:
        raise ValueError("No overlapping labels between X and Y.")

    per_group = {}
    counts = {}
    weights = []
    values = []

    for g in groups:
        Xg = X[lx == g]
        Yg = Y[ly == g]
        Eg = _energy_unbiased(Xg, Yg)
        per_group[str(g)] = Eg
        counts[str(g)] = (Xg.shape[0], Yg.shape[0])
        if not np.isnan(Eg):
            values.append(Eg)
            weights.append(Xg.shape[0] * Yg.shape[0])

    if len(values) == 0:
        macro = micro = np.nan
    else:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        macro = float(values.mean())
        micro = float(np.average(values, weights=weights))

    

    if return_all:
        pooled = _energy_unbiased(X, Y)
        return {
            'macro': macro,
            'micro': micro,
            'pooled': pooled,
            'per_group': per_group,
            'counts': counts
        }

    if average == 'macro':
        return macro
    elif average == 'micro':
        return micro
    elif average == 'pooled':
        pooled = _energy_unbiased(X, Y)
        return pooled
    else:
        raise ValueError("average must be one of {'macro','micro','pooled'}")
    
    
    
def compute_mmd_multi(x, y, sigmas=[10.0, 1.0, 0.1, 0.01, 0.001], unbiased=True):
    def pdist(a, b, sigma):
        sq = ((a.unsqueeze(1) - b.unsqueeze(0))**2).sum(2)
        return torch.exp(-sq / (2 * sigma**2))

    mmd_vals = []
    for sigma in sigmas:
        Kxx = pdist(x, x, sigma)
        Kyy = pdist(y, y, sigma)
        Kxy = pdist(x, y, sigma)
        m, n = x.size(0), y.size(0)
        if unbiased:
            Kxx = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
            Kyy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
        else:
            Kxx = Kxx.mean()
            Kyy = Kyy.mean()
        Kxy = Kxy.mean()
        print(f'Kxx: {Kxx}, Kyy: {Kyy}, Kxy: {Kxy}')
        mmd_vals.append(Kxx + Kyy - 2 * Kxy)

    return sum(mmd_vals) / len(mmd_vals)

