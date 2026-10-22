"""
Hurdle loss functions: Zero-inflated and truncated Gaussian losses.

This module contains:
- HurdleGaussianMSELoss: Hurdle Gaussian with MSE
- HurdleGaussianLoss: Standard hurdle Gaussian loss
- HurdleTruncGaussLoss: Truncated Gaussian hurdle loss
- HurdleTruncGaussResidualLoss: Truncated Gaussian residual loss
- HurdleTruncGaussDeltaLoss: Truncated Gaussian delta loss
- Prediction utilities for hurdle models
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional

from .base import log_phi


class HurdleGaussianMSELoss(nn.Module):
    """
    Hurdle model: binary classification for zero/non-zero, 
    then Gaussian regression for non-zero values.
    Standard hurdle NLL + MSE loss.
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, zero_logits, mu, sigma, targets):
        """
        Args:
            zero_logits: (batch_size, n_genes) - logits for P(non-zero)
            mu: (batch_size, n_genes) - mean for non-zero values  
            sigma: (batch_size, n_genes) - std for non-zero values
            targets: (batch_size, n_genes) - observed values
        """
        # Masks
        is_zero = (targets == 0)
        is_nonzero = ~is_zero
        
        # Probabilities
        nonzero_prob = torch.sigmoid(zero_logits)
        zero_prob = 1 - nonzero_prob
        
        # Clamp for numerical stability
        zero_prob = torch.clamp(zero_prob, self.eps, 1 - self.eps)
        nonzero_prob = torch.clamp(nonzero_prob, self.eps, 1 - self.eps)
        sigma = torch.clamp(sigma, self.eps, float('inf'))
        
        # Standard hurdle NLL (no hyperparameters)
        # For zeros: log P(zero)
        zero_nll = -torch.sum(is_zero * torch.log(zero_prob))
        zero_nll = zero_nll.clamp(min=self.eps)  # Avoid log(0)
        
        # For non-zeros: log P(non-zero) + log P(value | non-zero, truncated Gaussian)
        if torch.any(is_nonzero):
            # Truncated Gaussian log prob for positive values
            normal_logprob = -0.5 * (math.log(2.0 * math.pi) + 2*torch.log(sigma)) - 0.5 * ((targets - mu) / sigma)**2
            
            # Normalization for truncation at 0 (log of survival function)
            # log_survival = torch.log(1 - 0.5 * (1 + torch.erf(-mu / (sigma * torch.sqrt(torch.tensor(2.0))))))
            log_survival   = log_phi(mu / sigma)
            truncated_logprob = normal_logprob - log_survival
            
            nonzero_nll = -torch.sum(is_nonzero * (torch.log(nonzero_prob) + truncated_logprob))
        else:
            nonzero_nll = torch.tensor(0.0, device=targets.device)
            
        nonzero_nll = nonzero_nll.clamp(min=self.eps)  # Avoid log(0)ßßß
        
        hurdle_nll = (zero_nll + nonzero_nll) / targets.numel()
        
        # MSE loss (for mean predictions)
        
        targets_hat = nonzero_prob * mu       
        mse_loss = torch.mean(((targets - targets_hat)[is_nonzero])**2)
        
        het_loss = torch.mean((sigma**2))
        # Combined loss
        print(f'Hurdle NLL: {hurdle_nll.item()}, MSE Loss: {mse_loss.item()}; sigma loss: {het_loss.item()}')
        total_loss = hurdle_nll + 25*mse_loss  + het_loss
        
        return total_loss


class HurdleGaussianLoss(nn.Module):
    """
    Hurdle model: binary classification for zero/non-zero, 
    then Gaussian regression for non-zero values.
    """
    
    def __init__(self, eps=1e-8, sigma_reg_weight=0.01, loss_balance_weight=1.):
        super().__init__()
        self.eps = eps
        self.sigma_reg_weight = sigma_reg_weight
        self.loss_balance_weight = loss_balance_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, zero_logits, mu, sigma, targets):
        """
        Args:
            zero_logits: (batch_size, n_genes) - logits for P(non-zero)
            mu: (batch_size, n_genes) - mean for non-zero values
            sigma: (batch_size, n_genes) - std for non-zero values
            targets: (batch_size, n_genes) - observed changes
        """
        # Binary labels: 1 if non-zero, 0 if zero
        is_nonzero = (targets != 0).float()
        
        # 1. Binary classification loss  
        binary_loss = self.bce(zero_logits, is_nonzero)
        
        # 2. Regression loss for ALL positions (not just non-zero)
        normal_dist = torch.distributions.Normal(mu, sigma)
        log_prob = normal_dist.log_prob(targets)
        
        # Only apply regression loss where targets are non-zero
        regression_loss = -torch.mean(is_nonzero * log_prob)
        
        # 3. Sigma regularization (critical!)
        sigma_reg = torch.mean(sigma**2)
        
        # 4. Zero-conditional sigma penalty
        zero_prob = torch.sigmoid(zero_logits)
        sigma_penalty = torch.mean(zero_prob * sigma)
        
        # 5. Balanced combination
        total_loss = (binary_loss + 
                     self.loss_balance_weight * regression_loss + 
                     0.01 * sigma_penalty)
        
        return total_loss


class HurdleGaussianMSELoss(nn.Module):
    """
    Hurdle model: binary classification for zero/non-zero, 
    then Gaussian regression for non-zero values.
    Standard hurdle NLL + MSE loss.
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, zero_logits, mu, sigma, targets, use_sampling=False):
        """
        Args:
            zero_logits: (batch_size, n_genes) - logits for P(non-zero)
            mu: (batch_size, n_genes) - mean for non-zero values  
            sigma: (batch_size, n_genes) - std for non-zero values
            targets: (batch_size, n_genes) - observed values
        """
        # Masks
        is_zero = (targets == 0)
        is_nonzero = ~is_zero
        
        # Probabilities
        nonzero_prob = torch.sigmoid(zero_logits)
        nonzero_prob = torch.clamp(nonzero_prob, self.eps, 1 - self.eps)
        zero_prob = torch.clamp(1 - nonzero_prob, self.eps, 1 - self.eps)
        
        sigma += 1e-3
        
        # Standard hurdle NLL (no hyperparameters)
        # For zeros: log P(zero)
        zero_nll = -torch.sum(is_zero * torch.log(zero_prob))
        # zero_nll = zero_nll.clamp(min=self.eps)  # Avoid log(0)
        
        # For non-zeros: log P(non-zero) + log P(value | non-zero, truncated Gaussian)
        if torch.any(is_nonzero):
            # Truncated Gaussian log prob for positive values
            normal_logprob = -0.5 * (math.log(2.0 * math.pi) + 2*torch.log(sigma)) \
                - 0.5 * ((targets - mu) / sigma)**2
            
            # Normalization for truncation at 0 (log of survival function)
            # log_survival = torch.log(1 - 0.5 * (1 + torch.erf(-mu / (sigma * torch.sqrt(torch.tensor(2.0))))))
            log_survival   = log_phi(mu / sigma)
            truncated_logprob = normal_logprob - log_survival
            
            nonzero_nll = -torch.sum(is_nonzero * (torch.log(nonzero_prob) + truncated_logprob))
        else:
            nonzero_nll = torch.tensor(0.0, device=targets.device)
            
            
        nonzero_nll = nonzero_nll.clamp(min=self.eps)  # Avoid log(0)ßßß
        
        hurdle_nll = (zero_nll + nonzero_nll) / targets.numel()
        
        
        # MSE loss (for mean predictions)
        if use_sampling:
            # Sample from Bernoulli for each prediction
            bernoulli_sample = torch.bernoulli(nonzero_prob.detach())  # Detach to avoid double gradients
            targets_hat = bernoulli_sample * mu
        else:
            targets_hat = nonzero_prob * mu       
        mse_loss = torch.mean(((targets - targets_hat))**2)
        
        # mse_nz = torch.mean(((targets[is_nonzero] - targets_hat[is_nonzero]) ** 2))
        if torch.any(is_nonzero):
            # Train mu to be "inflated" to compensate for probability weighting
            expected_scaling = 1.0 / (nonzero_prob[is_nonzero] + self.eps)
            scaled_targets = targets[is_nonzero] * expected_scaling
            mse_nz = torch.sum((scaled_targets - mu[is_nonzero])**2) / targets.numel()
        else:
            mse_nz = torch.tensor(0.0, device=targets.device)
        
        het_loss = torch.mean((sigma**2))
        # Combined loss
        print(f'Hurdle NLL: {hurdle_nll.item()}, MSE Loss: {mse_loss.item()}; MSE Nonzero loss: {mse_nz.item()}, sigma loss: {het_loss.item()}')
        total_loss = hurdle_nll + het_loss + mse_loss + mse_nz
        
        return total_loss
    
# class GaussianMSELoss(nn.Module):
#     """
#     Standard Gaussian regression loss with negative log-likelihood.
#     Predicts mean and standard deviation for continuous targets.
#     """
#     import numpy as np
#     def __init__(self, eps=1e-8, sigma_reg_weight=0.1):
#         super().__init__()
#         self.eps = eps
#         self.sigma_reg_weight = sigma_reg_weight
    
#     def forward(self, mu, sigma, targets):
#         """
#         Args:
#             mu: (batch_size, n_genes) - predicted mean values
#             sigma: (batch_size, n_genes) - predicted standard deviations
#             targets: (batch_size, n_genes) - observed target values
#         """
#         # Ensure sigma is positive and numerically stable
#         sigma = torch.clamp(sigma, min=self.eps)
#         mu = mu.float()
#         sigma = sigma.float()
#         targets = targets.float()
#         log_sigma = torch.log(sigma + self.eps)
        

#         normal_dist = torch.distributions.Normal(mu, sigma)
#         # Negative log-likelihood loss
#         nll = -1*normal_dist.log_prob(targets)
#         nll = torch.clamp(nll, min=1e-8)

#         nll_loss = torch.mean(nll)
        
#         het_loss = torch.mean((sigma**2))
        
#         total_loss = nll_loss
        
#         return total_loss


class HurdleTruncGaussLoss(nn.Module):
    """
    Hurdle: Bernoulli gate for zero / non-zero +
    *truncated* Gaussian on positive support, with μ>0 via softplus.
    """
    def __init__(self, eps=1e-6, sigma_reg_weight=0.01):
        super().__init__()
        self.eps = eps
        self.LOG2PI = math.log(2.0 * math.pi)
        self.sigma_reg_weight = sigma_reg_weight

    def forward(self, logits, mu, sigma, y):
        # Clamp sigma to prevent numerical issues
        # sigma = torch.clamp(sigma, min=1e-3, max=5.)
        sigma = torch.clamp(sigma, min=1e-3)
        # Gate probabilities
        p_nz = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        p_z = (1.0 - p_nz).clamp(self.eps, 1 - self.eps)

        # Masks for zero and non-zero values
        is_zero = (y == 0)
        is_nz = (y > 0)

        # ----- Log-likelihoods ------------------------------------------------
        # Zeros: log P(Y = 0) = log(p_z)
        ll_zero = is_zero.float() * torch.log(p_z)

        # Non-zeros: log P(Y = y | Y > 0) = log(p_nz) + log(truncated_normal)
        if is_nz.any():
            # Standard Gaussian log-likelihood
            z = (y - mu) / sigma
            ll_norm = -0.5 * (self.LOG2PI + 2*torch.log(sigma) + z**2)
            
            # Survival function: P(Z > -μ/σ) where Z ~ N(0,1)
            # Using erfc for numerical stability
            normalized_mu = mu / (sigma * math.sqrt(2))
            survival_prob = 0.5 * torch.erfc(-normalized_mu)
            log_surv = torch.log(survival_prob.clamp(min=self.eps))
            
            # Truncated likelihood: subtract log of survival probability
            ll_trunc = ll_norm - log_surv
            ll_nz = is_nz.float() * (torch.log(p_nz) + ll_trunc)
        else:
            ll_nz = torch.zeros_like(y)

        # Total negative log-likelihood
        total_ll = ll_zero + ll_nz
        nll = -total_ll.sum() / y.numel() 
        loss = nll + self.sigma_reg_weight * sigma_reg(sigma).mean()  # Regularization on sigma
        # loss = nll
        return loss


@torch.no_grad()
def predict_unconditional_mean(logits, mu, sigma):
    zero_logits = -logits
    # probability of exact zero
    pi_zero = torch.sigmoid(zero_logits)
    cond_mean = trunc_normal_cond_mean(mu, sigma)
    return (1.0 - pi_zero) * cond_mean


class HurdleTruncGaussResidualLoss(nn.Module):
    """
    Hurdle (Bernoulli at exact zero) + lower-truncated Normal on residuals:
      r = y - x_ref  ~  TN(mu_r, sigma_r^2; lower=a=-x_ref, upper=+inf)
    Guarantees y_hat = x_ref + E[r|r>=a] >= 0  without clamping or softplus(mu+x_ref).
    """
    def __init__(self, eps=1e-6, sigma_reg_weight=0):
        super().__init__()
        self.eps = float(eps)
        self.LOG2PI = math.log(2.0 * math.pi)
        self.sigma_reg_weight = float(sigma_reg_weight)

    @staticmethod
    def _log_std_normal_pdf(x):
        # log φ(x) = -0.5 * (x^2 + log(2π))
        return -0.5 * (x * x + math.log(2.0 * math.pi))

    def forward(self, logits, mu_r, sigma_r, y, x_ref, weights=None):
        """
        Args
          logits:  (...,)            Bernoulli logits for Y==0 vs >0
          mu_r:    (...,)            residual mean (unconstrained)
          sigma_r: (...,)            residual scale (pre-activation)
          y:       (...,)            observed expression (>=0)
          x_ref:   (...,)            perturbed reference baseline (>=0)
        Returns
          loss: scalar
        """


        # Gate
        p_nz = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)  # P(Y>0)
        p_z  = (1.0 - p_nz).clamp(self.eps, 1.0 - self.eps)           # P(Y=0)

        # Masks (treat tiny values as zeros to avoid fp issues)
        is_zero = (y <= self.eps)
        is_nz   = ~is_zero

        # Residual target and truncation bound
        r = (y - x_ref)                        # residuals (signed)
        a = (-x_ref).to(y.dtype)               # lower bound per element

        # ----- Zero part: log P(Y=0) -----
        ll_zero = torch.where(is_zero, torch.log(p_z), torch.zeros_like(y))

        # ----- Non-zero part: log P(Y=y | Y>0) + log P(Y>0) -----
        if is_nz.any():
            r_nz       = r[is_nz]
            mu_r_nz    = mu_r[is_nz]
            sigma_nz   = sigma_r[is_nz]
            a_nz       = a[is_nz]

            # Standardized variables
            z = (r_nz - mu_r_nz) / sigma_nz         # for pdf at r
            alpha = (a_nz - mu_r_nz) / sigma_nz     # truncation point

            # log φ(z)
            log_pdf = self._log_std_normal_pdf(z) - torch.log(sigma_nz)

            # Survival S = 1 - Φ(α) = Φ(-α) = 0.5 * erfc(α / sqrt(2))
            log_S = torch.log(0.5 * torch.erfc(alpha / math.sqrt(2.0)).clamp(min=self.eps))

            # Truncated Normal log-density
            log_tn = log_pdf - log_S

            ll_nz = torch.zeros_like(y)
            ll_nz[is_nz] = torch.log(p_nz[is_nz]) + log_tn
        else:
            ll_nz = torch.zeros_like(y)

        # ----- Total NLL -----
        total_ll = ll_zero + ll_nz
        if weights is not None:
            per_sample_ll = total_ll.mean(dim=1)      # (B,) average over genes
            w = weights.squeeze(-1).to(per_sample_ll)
            nll = -(w * per_sample_ll).sum() / (w.sum() + self.eps)
        else:
            nll = -total_ll.mean()

        # Optional mild regularization on sigma
        if self.sigma_reg_weight > 0.0:
            nll = nll + self.sigma_reg_weight * (sigma_r * sigma_r).mean()

        return nll

# class HurdleTruncGaussResidualLoss(nn.Module):
#     """
#     TUNED VERSION:
#     - Added 'nz_weight' to boost signal sensitivity (fixes "conservative" behavior).
#     - Reduced default sigma regularization.
#     """
#     def __init__(self, eps=1e-6, sigma_reg_weight=0.01, nz_weight=1.0):
#         super().__init__()
#         self.eps = float(eps)
#         self.LOG2PI = math.log(2.0 * math.pi)
        
#         # CHANGED: Lowered default reg weight to prevent "clamping" to mean
#         self.sigma_reg_weight = float(sigma_reg_weight) 
        
#         # NEW: Multiplier for non-zero loss term. 
#         # Set this to 2.0 - 5.0 to make model 'aggressive' like Model B.
#         self.nz_weight = float(nz_weight)

#     @staticmethod
#     def _log_std_normal_pdf(x):
#         return -0.5 * (x * x + math.log(2.0 * math.pi))

#     def forward(self, logits, mu_r, sigma_r, y, x_ref, weights=None):
#         """
#         logits:  (...,) Bernoulli logits
#         mu_r:    (...,) residual mean
#         sigma_r: (...,) residual scale
#         y:       (...,) observed expression
#         x_ref:   (...,) reference baseline
#         """

#         # Gate probabilities
#         p_nz = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
#         p_z  = (1.0 - p_nz).clamp(self.eps, 1.0 - self.eps)

#         # Masks
#         is_zero = (y <= self.eps)
#         is_nz   = ~is_zero

#         # Residuals
#         r = (y - x_ref)
#         a = (-x_ref).to(y.dtype)

#         # 1. Zero Part: log P(Y=0)
#         ll_zero = torch.where(is_zero, torch.log(p_z), torch.zeros_like(y))

#         # 2. Non-Zero Part: log P(Y=y | Y>0) + log P(Y>0)
#         if is_nz.any():
#             r_nz       = r[is_nz]
#             mu_r_nz    = mu_r[is_nz]
#             sigma_nz   = sigma_r[is_nz]
#             a_nz       = a[is_nz]

#             z = (r_nz - mu_r_nz) / sigma_nz
#             alpha = (a_nz - mu_r_nz) / sigma_nz

#             log_pdf = self._log_std_normal_pdf(z) - torch.log(sigma_nz)
            
#             # Stable survival function
#             log_S = torch.log(0.5 * torch.erfc(alpha / math.sqrt(2.0)).clamp(min=self.eps))

#             log_tn = log_pdf - log_S
            
#             # Combine gate + continuous likelihood
#             ll_nz_val = torch.log(p_nz[is_nz]) + log_tn
            
#             # NEW: Apply Signal Boost
#             # We multiply the likelihood contribution of non-zeros
#             ll_nz = torch.zeros_like(y)
#             ll_nz[is_nz] = ll_nz_val * self.nz_weight
            
#         else:
#             ll_nz = torch.zeros_like(y)

#         # Total NLL
#         # Note: We sum the weighted terms. 
#         # If nz_weight > 1, the optimizer will focus more on improving ll_nz.
#         total_ll = ll_zero + ll_nz
        
#         if weights is not None:
#             per_sample_ll = total_ll.mean(dim=1)
#             w = weights.squeeze(-1).to(per_sample_ll)
#             nll = -(w * per_sample_ll).sum() / (w.sum() + self.eps)
#         else:
#             nll = -total_ll.mean()

#         # Regularization on Sigma
#         if self.sigma_reg_weight > 0.0:
#             nll = nll + self.sigma_reg_weight * (sigma_r * sigma_r).mean()

#         return nll


@torch.no_grad()
def trunc_normal_cond_mean(mu, sigma):
    # E[Y | Y>0] for Y ~ TN(μ, σ^2; lower=0)
    # Do everything in float64 for stability
    mu64    = mu.double()
    sigma64 = sigma.double().clamp_min(1e-6)
    alpha   = (-mu64 / sigma64).clamp(-20.0, 20.0)          # α = (0-μ)/σ
    x       = (alpha / math.sqrt(2.0)).clamp(-14.0, 14.0)   # domain where erfcx is stable

    # λ(α) = sqrt(2/π) / erfcx(α/√2)
    lam = math.sqrt(2.0/math.pi) / torch.special.erfcx(x)

    cond_mean64 = mu64 + sigma64 * lam
    return cond_mean64.to(mu.dtype)




   
# def predict_unconditional_mean(logits, mu, sigma):
#     pi     = torch.sigmoid(logits)
#     # sigma  = F.softplus(sigma) + 1e-4        # same transform as during training
#     # conditional mean for lower-truncated N(μ,σ²)
#     alpha  = -mu / sigma
#     cond_mean = mu + sigma * torch.exp(-0.5*alpha**2) / torch.erfc(alpha / math.sqrt(2)) * math.sqrt(2/math.pi)
#     return pi * cond_mean                   # unconditional mean


def predict_unconditional_mean_residual(logits, mu_r, sigma_r, x_ref, eps=1e-6):
    # p(Y>0)
    p_nz = torch.sigmoid(logits).clamp(eps, 1 - eps)

    # lower bound a = -x_ref, alpha = (a - mu_r)/sigma_r
    sigma = sigma_r
    alpha = (-(x_ref) - mu_r) / sigma

    # lambda(alpha) = phi(alpha) / (1 - Phi(alpha))
    # = sqrt(2/pi) / erfcx(alpha/sqrt(2))
    t = alpha / math.sqrt(2.0)
    lam = math.sqrt(2.0 / math.pi) / torch.special.erfcx(t)

    # E[R | R>=a] and E[Y | Y>0]
    E_r_trunc = mu_r + sigma * lam
    E_y_given_nz = x_ref + E_r_trunc

    # Unconditional mean: P(Y>0) * E[Y | Y>0]  (mass at Y=0 contributes 0)
    return p_nz * E_y_given_nz


class HurdleTruncGaussDeltaLoss(nn.Module):
    """
    Residual r = y - x_ref is modeled as a spike-and-slab mixture on [a, +inf),
    where a = -x_ref (to ensure y >= 0). Both components are (lower-)truncated Normals:

      r ~  pΔ * TN(0, τ^2; lower=a)    +   (1-pΔ) * TN(μ_r, σ_r^2; lower=a)

    Gate: pΔ = sigmoid(logits_delta) is the 'no-change' mass.
    This captures sparse deltas while keeping positivity of y.
    """
    def __init__(self, eps=1e-6, sigma_floor=1e-3, tau=1e-2, learn_tau=False, sigma_reg=0.0):
        super().__init__()
        self.eps = float(eps)
        self.sigma_floor = float(sigma_floor)
        self.sigma_reg = float(sigma_reg)
        if learn_tau:
            # initialize tau near the provided value (log-space)
            self._log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        else:
            self.register_buffer("_tau_const", torch.tensor(float(tau)))
            self._log_tau = None

        self.LOG2PI = math.log(2.0 * math.pi)

    def _tau(self, like: torch.Tensor) -> torch.Tensor:
        if self._log_tau is not None:
            return torch.exp(self._log_tau).expand_as(like)
        else:
            return self._tau_const.expand_as(like)

    @staticmethod
    def _log_std_normal_pdf(z):
        # log φ(z) = -0.5 * (z^2 + log(2π))
        return -0.5 * (z*z + math.log(2.0*math.pi))

    def _log_trunc_normal(self, r, mu, sigma, a):
        """
        Log-density of lower-truncated Normal at r (elementwise).
        sigma must be > 0. a is the lower bound.
        """
        z = (r - mu) / sigma
        log_pdf = self._log_std_normal_pdf(z) - torch.log(sigma)
        alpha = (a - mu) / sigma
        # log survival S = log(1 - Φ(alpha)) = log(0.5*erfc(alpha/sqrt(2)))
        log_S = torch.log(0.5 * torch.erfc(alpha / math.sqrt(2.0)).clamp_min(self.eps))
        return log_pdf - log_S  # log TN(r | mu, sigma, lower=a)

    def forward(self, logits_delta, mu_r, sigma_r, y, x_ref):
        """
        Args:
          logits_delta: (B,G) logits for 'no-change' spike component pΔ
          mu_r:         (B,G) residual mean for slab
          sigma_r:      (B,G) residual scale for slab (pre-activated upstream; we clamp here)
          y:            (B,G) observed expression (>=0)
          x_ref:        (B,G) baseline/control reference (>=0)
        Returns:
          loss: scalar (mean NLL + optional reg)
        """
        # residuals and truncation
        r = y - x_ref
        a = -x_ref.to(r.dtype)

        # parameters
        p_delta = torch.sigmoid(logits_delta).clamp(self.eps, 1.0 - self.eps)
        sigma_r = sigma_r.clamp_min(self.sigma_floor)
        tau = self._tau(r).clamp_min(self.sigma_floor)

        # log-densities of spike and slab (both truncated on [a, +inf))
        log_spike = self._log_trunc_normal(r, mu=torch.zeros_like(r), sigma=tau,   a=a)
        log_slab  = self._log_trunc_normal(r, mu=mu_r,               sigma=sigma_r, a=a)

        # mixture log-likelihood for every element
        # log( pΔ * spike + (1-pΔ) * slab ) via logaddexp for stability
        log_mix = torch.logaddexp(torch.log(p_delta)   + log_spike,
                                  torch.log(1.0 - p_delta) + log_slab)

        nll = -(log_mix).mean()

        if self.sigma_reg > 0.0:
            nll = nll + self.sigma_reg * (sigma_r * sigma_r).mean()

        return nll


def trunc_mean(mu, sigma):
    alpha = (a - mu) / sigma
    # compute λ in float64 for stability, then cast back
    log_phi = (-0.5 * alpha.double()**2) - 0.5*math.log(2*math.pi)
    log_S   = torch.log(0.5 * torch.erfc(alpha.double()/math.sqrt(2.0)).clamp_min(1e-12))
    lam = torch.exp(log_phi - log_S).to(mu.dtype)
    return mu + sigma * lam

@torch.no_grad()
def predict_unconditional_mean_delta(logits_delta, mu_r, sigma_r, x_ref, tau=1e-2, eps=1e-6):
    """
    E[Y] under the same spike-and-slab residual model.
      r ~ pΔ * TN(0, τ^2; lower=a) + (1-pΔ) * TN(μ_r, σ_r^2; lower=a),  a=-x_ref
      Y = x_ref + r
    Returns:
      y_hat: (B,G)
    """
    p_delta = torch.sigmoid(logits_delta).clamp(eps, 1.0 - eps)
    sigma_r = sigma_r.clamp_min(1e-3)
    tau     = torch.as_tensor(tau, device=mu_r.device, dtype=mu_r.dtype).clamp_min(1e-3)
    if tau.dim() == 0:
        tau = tau.expand_as(mu_r)

    a = -x_ref
    # truncated Normal mean: E[r] = μ + σ * λ(α), α = (a-μ)/σ, λ = φ(α)/(1-Φ(α))

    Er_spike = trunc_mean(mu=torch.zeros_like(mu_r), sigma=tau)
    Er_slab  = trunc_mean(mu=mu_r, sigma=sigma_r)
    Er = p_delta * Er_spike + (1.0 - p_delta) * Er_slab
    return x_ref + Er


# ---------------------------------------------------------------
# Shuffle-dependency loss on parametric outputs (π, μ, σ)
# Encourages outputs under shuffled conditioning to differ from
# the original outputs by at least a margin δ.
# L_dep = relu( δ - ( ||π-π^shuf||_1 + α_μ||μ-μ^shuf||_1 + α_σ||logσ-logσ^shuf||_1 ) )
# ---------------------------------------------------------------
