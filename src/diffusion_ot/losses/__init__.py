"""
Loss functions: Gaussian, hurdle, count, and regularization losses.

This module provides:
- Base utilities (sigma_reg, log_phi)
- Gaussian losses (MSE, NLL, mixture)
- Hurdle losses (truncated Gaussian, residual, delta)
- Count losses (negative binomial, hurdle count)
- Regularization losses (variance, covariance, shuffling)
"""

from .base import sigma_reg, log_phi
from .gaussian import (
    ZINBMSELoss,
    zinb_log_prob,
    GaussianLoss,
    MixtureOfGaussiansLoss,
)
from .hurdle import (
    HurdleGaussianMSELoss,
    HurdleGaussianLoss,
    HurdleTruncGaussLoss,
    HurdleTruncGaussResidualLoss,
    HurdleTruncGaussDeltaLoss,
    predict_unconditional_mean,
    trunc_normal_cond_mean,
    predict_unconditional_mean_residual,
    predict_unconditional_mean_delta,
)
from .count import (
    NBLoss,
    SimplifiedNBLoss,
    HurdleCountNLLLoss,
)
from .regularization import (
    SparseMSELoss,
    variance_floor_loss,
    cov_offdiag_loss,
    offdiag_cosine_stats,
    shuffling_dependency_loss,
)

__all__ = [
    # Base
    'sigma_reg',
    'log_phi',
    # Gaussian
    'ZINBMSELoss',
    'zinb_log_prob',
    'GaussianLoss',
    'MixtureOfGaussiansLoss',
    # Hurdle
    'HurdleGaussianMSELoss',
    'HurdleGaussianLoss',
    'HurdleTruncGaussLoss',
    'HurdleTruncGaussResidualLoss',
    'HurdleTruncGaussDeltaLoss',
    'predict_unconditional_mean',
    'trunc_normal_cond_mean',
    'predict_unconditional_mean_residual',
    'predict_unconditional_mean_delta',
    # Count
    'NBLoss',
    'SimplifiedNBLoss',
    'HurdleCountNLLLoss',
    # Regularization
    'SparseMSELoss',
    'variance_floor_loss',
    'cov_offdiag_loss',
    'offdiag_cosine_stats',
    'shuffling_dependency_loss',
]

