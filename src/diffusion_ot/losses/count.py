"""
Count-based loss functions: Negative binomial and hurdle count losses.

This module contains:
- NBLoss: Negative binomial loss
- SimplifiedNBLoss: Simplified negative binomial loss
- HurdleCountNLLLoss: Hurdle count negative log-likelihood loss
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import math


class NBLoss(nn.Module):
    """
    Negative Binomial loss for perturbation expression prediction.
    
    For each gene, predicts:
    - mu: mean expression change magnitude 
    - theta: dispersion parameter (inverse overdispersion)
    - sign_logit: logit for up/down regulation direction
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 3 * n_genes) - [mu, theta, sign_logit] for each gene
            targets: (batch_size, n_genes) - observed expression changes (can be negative)
        """
        batch_size, total_dim = predictions.shape
        n_genes = total_dim // 3
        
        # Reshape predictions to (batch_size, n_genes, 3)
        pred_reshaped = predictions.view(batch_size, n_genes, 3)
        
        # Extract parameters for each gene
        mu_raw = pred_reshaped[:, :, 0]
        theta_raw = pred_reshaped[:, :, 1] 
        sign_logit = pred_reshaped[:, :, 2]
        
        # Apply constraints
        mu = F.softplus(mu_raw) + self.eps  # magnitude (>0)
        theta = F.softplus(theta_raw) + self.eps  # dispersion (>0)
        sign_prob = torch.sigmoid(sign_logit)  # P(positive change)
        
        # Convert targets to magnitude and sign
        targets_abs = torch.abs(targets) + self.eps
        targets_sign = torch.sign(targets)  # -1, 0, or 1
        
        # For NegativeBinomial distribution
        # PyTorch uses (total_count, probs) parameterization
        total_count = theta
        probs = theta / (theta + mu)
        
        # Create NB distribution
        nb_dist = NegativeBinomial(total_count=total_count, probs=probs)
        
        # Log probability of magnitude
        magnitude_logprob = nb_dist.log_prob(targets_abs)
        
        # Log probability of sign
        # For zero targets, sign doesn't matter
        zero_mask = (targets == 0)
        pos_mask = (targets > 0)
        neg_mask = (targets < 0)
        
        sign_logprob = torch.zeros_like(targets)
        sign_logprob[pos_mask] = torch.log(sign_prob[pos_mask] + self.eps)
        sign_logprob[neg_mask] = torch.log(1 - sign_prob[neg_mask] + self.eps)
        # Zero targets contribute 0 to sign loss
        
        # Total log probability
        total_logprob = magnitude_logprob + sign_logprob
        
        # Return negative log likelihood
        return -total_logprob.sum(dim=1).mean()


class SimplifiedNBLoss(nn.Module):
    """
    Simplified NB loss that directly models signed expression changes.
    Uses a shifted/scaled NB to handle negative values.
    """
    
    def __init__(self, eps=1e-8, shift=10.0):
        super().__init__()
        self.eps = eps
        self.shift = shift  # Shift to make all values positive
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 2 * n_genes) - [mu, theta] for each gene
            targets: (batch_size, n_genes) - observed expression changes
        """
        batch_size, total_dim = predictions.shape
        n_genes = total_dim // 2
        
        # Reshape predictions
        pred_reshaped = predictions.view(batch_size, n_genes, 2)
        
        # Extract parameters
        mu = pred_reshaped[:, :, 0]  # can be negative
        theta = F.softplus(pred_reshaped[:, :, 1]) + self.eps  # dispersion (>0)
        
        # Shift both predictions and targets to positive domain
        mu_shifted = mu + self.shift
        targets_shifted = targets + self.shift
        
        # Ensure positive values
        mu_shifted = torch.clamp(mu_shifted, min=self.eps)
        targets_shifted = torch.clamp(targets_shifted, min=0)
        
        # NB distribution parameters
        total_count = theta
        probs = theta / (theta + mu_shifted)
        
        # Create NB distribution and compute loss
        nb_dist = NegativeBinomial(total_count=total_count, probs=probs)
        log_prob = nb_dist.log_prob(targets_shifted)
        
        return -log_prob.sum(dim=1).mean()


class HurdleCountNLLLoss(nn.Module):
    """
    Hurdle loss with a Bernoulli gate (zero / non-zero) and a *truncated*
    positive count distribution (Poisson or Negative-Binomial).

    log-likelihood is averaged over all entries (same scale as MSE).
    """

    def __init__(self, dist="nb", eps=1e-8):
        """
        dist ∈ {"poisson", "nb"}  –  positive component
        For NB we expect the model to output *mean* μ and *dispersion* θ>0
        such that  Var[Y] = μ + μ²/θ  (same parametrisation as scVI).
        """
        super().__init__()
        assert dist in {"poisson", "nb"}
        self.dist = dist
        self.eps  = eps

    # --------------------------------------------------------------------- #
    #                               helpers                                  #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _log_factorial(x):
        """log(k!) using lgamma(k+1).  x is an integer tensor."""
        return torch.lgamma(x + 1.0)

    def _trunc_poisson_logpmf(self, k, lam):
        """
        log P(Y=k | Y>0, λ)  where  Y∼Poisson(λ).
        """
        log_untrunc = k * torch.log(lam) - lam - self._log_factorial(k)
        log_p0      = -lam                          # log P(Y=0)
        return log_untrunc - torch.log1p(-torch.exp(log_p0))  # subtract log(1−P0)

    def _trunc_nb_logpmf(self, k, mu, theta):
        """
        log P(Y=k | Y>0)  for NB(μ,θ) with PMF
        P(Y=k)= Γ(k+θ)/(Γ(θ)k!) (μ/(μ+θ))^k (θ/(μ+θ))^θ
        """
        log_untrunc = (
            torch.lgamma(k + theta)
            - torch.lgamma(theta)
            - self._log_factorial(k)
            + k * torch.log(mu + self.eps)
            - k * torch.log(mu + theta + self.eps)
            + theta * torch.log(theta + self.eps)
            - theta * torch.log(mu + theta + self.eps)
        )
        log_p0 = theta * torch.log(theta / (mu + theta + self.eps))  # NB pmf at 0
        return log_untrunc - torch.log1p(-torch.exp(log_p0))

    # --------------------------------------------------------------------- #
    #                                forward                                #
    # --------------------------------------------------------------------- #

    def forward(self, zero_logits, mu, theta, targets):
        """
        Params
        ------
        zero_logits  : (B, G)  Bernoulli logits for P(Y>0)
        mu           : (B, G)  positive-part mean  (λ for Poisson, μ for NB)
        theta    : (B, G)  ignored for Poisson;  softplus() gives θ for NB
        targets      : (B, G)  integer counts (≥0)

        Returns
        -------
        scalar loss (mean over B×G)
        """
        # 0/1 masks
        is_zero     = (targets == 0)
        is_nonzero  = ~is_zero

        # Bernoulli gate
        p_nonzero = torch.sigmoid(zero_logits).clamp(self.eps, 1 - self.eps)
        p_zero    = 1.0 - p_nonzero

        # -- log-likelihood for zero entries --------------------------------
        ll_zero = is_zero.float() * torch.log(p_zero)

        # -- log-likelihood for non-zeros -----------------------------------
        if self.dist == "poisson":
            # lam   = F.softplus(mu) + self.eps               # ensure λ>0
            ll_pos = self._trunc_poisson_logpmf(targets, mu)
        else:                                               # Negative-Binomial
            # theta = F.softplus(aux_param) + self.eps        # dispersion >0
            # mu_   = F.softplus(mu) + self.eps               # mean >0
            ll_pos = self._trunc_nb_logpmf(targets, mu, theta)

        ll_nonzero = is_nonzero.float() * (
            torch.log(p_nonzero) + ll_pos
        )

        # average NLL ( *negative* log-likelihood )
        nll = -(ll_zero + ll_nonzero).sum() / targets.numel()

        # Optional: dispersion penalty to keep θ from blowing up
        if self.dist == "nb":
            het_loss = 1e-4 * torch.mean(theta)
            nll = nll + het_loss

        return nll
