"""
Regularization losses: Variance, covariance, and shuffling dependency losses.

This module contains:
- variance_floor_loss: Variance floor regularization
- cov_offdiag_loss: Covariance off-diagonal regularization
- offdiag_cosine_stats: Off-diagonal cosine statistics
- shuffling_dependency_loss: Shuffling dependency loss
- SparseMSELoss: Sparse MSE loss
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import math

class SparseMSELoss(nn.Module):
    def __init__(self, sparsity_weight=1., epsilon=1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Standard MSE
        mse_loss = F.mse_loss(pred, target)
        
        # L1 penalty for sparsity (promotes exact zeros)
        sparsity_loss = torch.mean(torch.abs(pred))
        
        # Adaptive weighting: penalize false positives more
        false_positive_mask = (target < self.epsilon) & (pred > self.epsilon)
        fp_penalty = torch.mean(pred[false_positive_mask] ** 2)
        
        return mse_loss + self.sparsity_weight * sparsity_loss + fp_penalty





def shuffling_dependency_loss(pi: torch.Tensor,
                              mu: torch.Tensor,
                              sigma: torch.Tensor,
                              pi_shuf: torch.Tensor,
                              mu_shuf: torch.Tensor,
                              sigma_shuf: torch.Tensor,
                              delta: float = 0.1,
                              alpha_mu: float = 1.0,
                              alpha_sigma: float = 1.0,
                              reduction: str = 'mean',
                              eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a hinge loss that penalizes similarity between parametric
    outputs and their shuffled counterparts.

    Args:
      pi, mu, sigma:        (B, G)
      pi_shuf, mu_shuf, sigma_shuf: (B, G)
      delta:   positive margin – larger requires greater separation
      alpha_mu, alpha_sigma: relative weights for μ and σ terms
      reduction: 'mean' | 'sum' | 'none'
    """
    # Work in fp32 for numerical stability even under autocast
    pi_f   = pi.float();         pi_s   = pi_shuf.float()
    mu_f   = mu.float();         mu_s   = mu_shuf.float()
    sig_f  = sigma.float().clamp_min(eps)
    sig_s  = sigma_shuf.float().clamp_min(eps)

    # L1 distances per sample, averaged over feature dim
    d_pi = (pi_f - pi_s).abs().mean(dim=-1)
    d_mu = (mu_f - mu_s).abs().mean(dim=-1)
    d_ls = (sig_f.log() - sig_s.log()).abs().mean(dim=-1)

    score = d_pi + alpha_mu * d_mu + alpha_sigma * d_ls
    loss_vec = (delta - score).clamp_min(0.0)

    if reduction == 'mean':
        return loss_vec.mean().to(pi.dtype)
    if reduction == 'sum':
        return loss_vec.sum().to(pi.dtype)
    return loss_vec.to(pi.dtype)



def variance_floor_loss(emb, gamma=1.0, eps=1e-4):
    emb = emb.squeeze(1) if emb.dim() == 3 else emb
    std = torch.sqrt(emb.var(dim=0) + eps) # Correct placement of eps
    return torch.relu(gamma - std).mean()

def cov_offdiag_loss(emb):
    emb = emb.squeeze(1) if emb.dim() == 3 else emb
    N, D = emb.shape
    x = emb - emb.mean(dim=0, keepdim=True)
    C = (x.T @ x) / max(N-1, 1)
    off = C - torch.diag(torch.diag(C))
    return (off**2).sum() / D # Use .sum() and divide by dimension D



def offdiag_cosine_stats(x):
    """
    Compute mean/std of off-diagonal cosine similarities for x ∈ R^{B×D}.
    
    Args:
        x: (B, D) tensor

    Returns:
        mean_offdiag, std_offdiag (scalars)
    """
    # Normalize rows
    x_norm = F.normalize(x, p=2, dim=1)   # (B, D)
    
    # Cosine similarity matrix
    cos = x_norm @ x_norm.T               # (B, B)
    
    # Mask out diagonal
    B = x.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    offdiag_vals = cos[mask]              # (B*B - B,) entries
    
    return offdiag_vals.mean(), offdiag_vals.std()

