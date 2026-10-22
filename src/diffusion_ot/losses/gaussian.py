"""
Gaussian loss functions: MSE, NLL, and mixture losses for continuous targets.

This module contains:
- ZINBMSELoss: Zero-inflated negative binomial MSE loss
- GaussianLoss: Standard Gaussian NLL loss
- MixtureOfGaussiansLoss: Mixture of Gaussians loss
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .base import log_phi


class ZINBMSELoss(nn.Module):
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
        
        zinb_nll = -zinb_log_prob(targets, mu, sigma, zero_logits.sigmoid(), self.eps).mean()
        
        # MSE loss (for mean predictions)
        mse_loss = torch.mean(((targets - mu)**2)[is_zero])
        
        # Combined loss
        print(f'ZINB NLL: {zinb_nll.item()}, MSE Loss: {mse_loss.item()}')
        total_loss = zinb_nll + 25*mse_loss
        
        return total_loss


def zinb_log_prob(x, mu, theta, pi, eps=1e-8):
    # x: integer counts (float tensor is fine)
    t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    nb_log_prob = t1 + t2 + t3                     # log NB(x | mu, theta)

    log_prob_zero = torch.log(pi + (1 - pi) * torch.exp(nb_log_prob))
    log_prob_nonzero = torch.log(1 - pi + eps) + nb_log_prob

    return torch.where(x < 1e-8, log_prob_zero, log_prob_nonzero)


class GaussianLoss(nn.Module):
    """
    Standard Gaussian regression loss with negative log-likelihood.
    Predicts mean and standard deviation for continuous targets.
    """
    
    def __init__(self, eps=1e-8, sigma_reg_weight=0.05):
        super().__init__()
        self.eps = eps
        self.sigma_reg_weight = sigma_reg_weight
    
    def forward(self, mu, sigma, targets):
        """
        Args:
            mu: (batch_size, n_genes) - predicted mean values
            sigma: (batch_size, n_genes) - predicted standard deviations
            targets: (batch_size, n_genes) - observed target values
        """
        # Ensure sigma is positive and numerically stable
        sigma = torch.clamp(sigma, min=self.eps, max=5)
        # Create normal distribution
        normal_dist = torch.distributions.Normal(mu, sigma)
        
        # Negative log-likelihood loss
        log_prob = normal_dist.log_prob(targets)
        nll_loss = -torch.mean(log_prob)
        
        # Optional sigma regularization to prevent overconfidence
        sigma_reg = torch.mean(sigma**2)
        
        # Total loss
        total_loss = nll_loss 
        
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
#         # sigma = torch.clamp(sigma, min=self.eps)
#         log_sigma = torch.log(sigma + self.eps)
        
#         # # Create normal distribution
#         # log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        
#         # Compute NLL directly using log_sigma
#         nll = 0.5 * (log_sigma + np.log(2 * np.pi) + 
#                      torch.exp(-log_sigma) * (targets - mu)**2)
        
#         # Negative log-likelihood loss
#         log_prob = normal_dist.log_prob(targets)
#         if torch.any(log_prob<0):
#             neg_mu = mu[log_prob<0]
#             neg_sigma = sigma[log_prob<0]
#             neg_targets = targets[log_prob<0]
#             print(f'Negative log prob detected: mu={neg_mu}, sigma={neg_sigma}, targets={neg_targets}')
#         nll_loss = -torch.mean(log_prob)
        
#         # Optional sigma regularization to prevent overconfidence
        
#         mse_loss = torch.mean(((targets - mu)**2))
#         het_loss = torch.mean((sigma**2))
#         # Total loss
#         print(f'Gaussian NLL: {nll_loss.item()}, MSE Loss: {mse_loss.item()}, sigma loss: {het_loss.item()}')
#         total_loss = nll_loss + 10*mse_loss + 10*het_loss
        
#         return total_loss


class MixtureOfGaussiansLoss(nn.Module):
    """
    Mixture of two Gaussians for modeling perturbation effects.
    - Component 1: For observed non-zero expression changes
    - Component 2: For observed zeros (which might be true zeros or dropouts)
    Currently don't work
    """
    
    def __init__(self, eps=1e-6, prior_dropout_rate=0.3):
        super().__init__()
        self.eps = eps
        self.prior_dropout_rate = prior_dropout_rate
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys:
                - 'mu1': (batch_size, n_genes) - mean for non-zero component
                - 'log_sigma1': (batch_size, n_genes) - log std for non-zero component
                - 'mu2': (batch_size, n_genes) - mean for zero component  
                - 'log_sigma2': (batch_size, n_genes) - log std for zero component
                - 'logit_pi': (batch_size, n_genes) - logit mixing probability
            targets: (batch_size, n_genes) - observed expression changes
        """
        # Extract predictions
        mu1 = predictions['mu1']
        log_sigma1 = predictions['log_sigma1']
        mu2 = predictions['mu2']
        log_sigma2 = predictions['log_sigma2']
        logit_pi = predictions['logit_pi']
        
        print(f'targets shape: {targets.shape}')
        print(f'mu1 shape: {mu1.shape}, log_sigma1 shape: {log_sigma1.shape}')        
        
        # Clamp log_sigma for stability
        log_sigma1 = torch.clamp(log_sigma1, min=-5, max=2)
        log_sigma2 = torch.clamp(log_sigma2, min=-5, max=2)
        
        sigma1 = torch.exp(log_sigma1)
        sigma2 = torch.exp(log_sigma2)
        
        # Mixing probabilities
        pi = torch.sigmoid(logit_pi)  # P(component 1)
        
        # Compute log probabilities for each component
        log_prob1 = -0.5 * (torch.log(2 * torch.pi * sigma1**2) + 
                           ((targets - mu1) / sigma1)**2)
        log_prob2 = -0.5 * (torch.log(2 * torch.pi * sigma2**2) + 
                           ((targets - mu2) / sigma2)**2)
        
        # Mixture log probability using log-sum-exp trick for stability
        log_pi1 = torch.log(pi + self.eps)
        log_pi2 = torch.log(1 - pi + self.eps)
        
        # Log of weighted sum: log(π₁ * p₁ + π₂ * p₂)
        # max_log_prob = torch.maximum(log_pi1 + log_prob1, log_pi2 + log_prob2)
        # log_prob_mixture = max_log_prob + torch.log(
        #     torch.exp(log_pi1 + log_prob1 - max_log_prob) + 
        #     torch.exp(log_pi2 + log_prob2 - max_log_prob)
        # )
        log_prob_mixture = torch.logaddexp(
            log_pi1 + log_prob1,
            log_pi2 + log_prob2
        )
        
        print(f'log_prob_mixture shape: {log_prob_mixture.shape}')
        # Negative log-likelihood
        nll = -torch.mean(log_prob_mixture)
        
        # Additional regularization terms
        # 1. Encourage separation between components
        separation_loss = -torch.mean(torch.abs(mu1 - mu2))
        
        # 2. Prior on mixing probability based on expected dropout rate
        is_zero = (targets == 0).float()
        zero_rate = is_zero.mean()
        # prior_loss = F.binary_cross_entropy(pi.mean(), 
        #                                      torch.tensor(1 - self.prior_dropout_rate, device=pi.device, dtype=pi.dtype)
        #                                      )
        
        prior_loss = F.binary_cross_entropy_with_logits(
            logit_pi.mean(), 
            torch.tensor(1 - self.prior_dropout_rate, device=logit_pi.device, dtype=logit_pi.dtype)
        )
        
        # Total loss
        total_loss = nll + 0.1 * separation_loss + 0.05 * prior_loss
        
        # Logging
        with torch.no_grad():
            # Posterior probability of each component given the data
            posterior_pi1 = torch.exp(log_pi1 + log_prob1 - log_prob_mixture)
            
            # Statistics for non-zero vs zero targets
            nonzero_mask = targets != 0
            zero_mask = ~nonzero_mask
            
            if nonzero_mask.any():
                nonzero_posterior1 = posterior_pi1[nonzero_mask].mean()
                print(f"P(comp1|nonzero): {nonzero_posterior1:.3f}")
            
            if zero_mask.any():
                zero_posterior1 = posterior_pi1[zero_mask].mean()
                print(f"P(comp1|zero): {zero_posterior1:.3f}")
                
            print(f"NLL: {nll:.3f}, Sep: {separation_loss:.3f}, Prior: {prior_loss:.3f}")
        
        return total_loss, {
            'nll': nll.item(),
            'separation_loss': separation_loss.item(),
            'prior_loss': prior_loss.item(),
            'posterior_pi1': posterior_pi1.detach()
        }
