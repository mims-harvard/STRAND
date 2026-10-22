"""
Diffusion model components: core diffusion models, schedulers, and sampling functions.

This module contains:
- Core diffusion models (scDiffusion1DStable, scDiffusion1DStableMoE)
- Sampling functions (ddpm_sample_i2sb)
- Schedule scalers and schedulers
- Helper components (TimeEmbedding, MLPBlock, ResidualScale)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from scipy.special import betainc as scipy_betainc

# Import scheduler utilities from the refactored package (no legacy fallback)
from ..schedulers import make_symmetric_beta_bases, _safe_norm_pos

# Import attention components
from .attention import (
    TimeEmbedding, MLPBlock, ResidualScale, BilinearFiLM, 
    ControlDNAGate, CondBlock, timestep_embedding
)



def sigma_reg(s, delta=3.0):
    """Smooth L1 regularization on sigma: quadratic near 0, linear after delta."""
    x = s
    return torch.where(x < delta,
                       0.5*x*x,
                       delta*(x - 0.5*delta))


@torch.no_grad()
def ddpm_sample_i2sb(model, 
                     x_T, 
                     betas, 
                     alphas_cumprod, 
                     T, 
                     eta=1.0,
                     cond=None,
                     cond_null=None,
                     guidance_scale=1.0,
                     schedule_scaler=None,
                     weightLoss=False,
                     verbose=False):
    """
    I2SB-consistent DDPM posterior sampler with a monotone, EDM-like dynamic index schedule.
    
    Critical change vs previous attempt: we build a STRICTLY MONOTONE index path using
    torch.searchsorted so the reverse recursion is valid.
    
    Args:
        model: Diffusion model that predicts noise/epsilon
        x_T: Starting noise tensor (B, 1, D) or (B, D)
        betas: Noise schedule (T,) or (1, T)
        alphas_cumprod: Cumulative product of alphas (T,)
        T: Number of diffusion steps
        eta: Noise level for sampling (1.0 = DDPM, 0.0 = DDIM)
        cond: Conditional tokens (B, L, D) or None
        cond_null: Null conditional tokens for CFG
        guidance_scale: Classifier-free guidance scale
        schedule_scaler: Optional schedule scaler module
        weightLoss: Whether to use weighted loss (legacy)
        verbose: Whether to print debug info
        
    Returns:
        x_0: Denoised sample (B, 1, D) or (B, D)
    """
    x = x_T.clone()  # (B, 1, D) or (B, D) depending on your pipeline
    device = x.device
    B = x.size(0)

    # ---- per-sample betas (optionally scaled) ----
    if schedule_scaler is not None and cond is not None:
        cond_mean = cond.mean(dim=1, keepdim=True)               # (B,1,·)
        s_p, _ = schedule_scaler(cond_mean)                       # (B,1)
        s_p = s_p.view(B,1)
        betas = (betas.view(1, T).to(device).expand(B, T) * s_p)  # (B,T)
    else:
        betas = betas.view(1, T).to(device).expand(B, T)          # (B,T)

    betas = betas.clamp(1e-10, 0.999)
    idx_b = torch.arange(B, device=device)

    # ---- cumulative variances A_n ----
    A_table = torch.cumsum(betas, dim=1)                          # (B, T)
    A_T = A_table[:, -1]                                          # (B,)
    sigma_max = A_T.sqrt()                                        # (B,)

    # ---- dynamic time sampling: EDM-like grid over sigma with rho=3 ----
    rho = 3.0
    u = torch.linspace(0, 1, T, device=device)                    # (T,)
    # sigma grid DESCENDING from sigma_max to 0
    sigma_grid = sigma_max.view(B,1) * (1.0 - u.view(1,T)).pow(rho)   # (B,T)
    A_target = sigma_grid.pow(2)                                  # (B,T), DESCENDING along T

    # To build a VALID reverse path, we need indices INCREASING with variance (0→A_T).
    # So: sort targets ASCENDING, map to indices via searchsorted, then reverse in the loop.
    A_target_asc, _ = torch.sort(A_target, dim=1)                 # (B,T), ASCENDING 0..A_T

    # Build per-sample monotone index schedule with searchsorted
    schedule_idx = torch.empty((B, T), device=device, dtype=torch.long)
    for b in range(B):
        # positions of A_target_asc[b] in the cumulative A_table[b]
        # searchsorted returns first j with A_table[b, j] >= val
        j = torch.searchsorted(A_table[b], A_target_asc[b], right=False)
        # clamp to [0, T-1]; also ensure monotonicity explicitly (non-decreasing)
        j = j.clamp(min=0, max=T-1)
        # make strictly non-decreasing by cumulative max (prevents tiny dips)
        j = torch.maximum(j, torch.nn.functional.pad(j[:-1], (1,0), value=j[0]).cummax(dim=0).values)
        schedule_idx[b] = j

    # ---- main reverse loop over the dynamic index schedule ----
    for k in reversed(range(T)):
        t = schedule_idx[:, k]                                     # (B,), non-decreasing over k ascending
        t_rev = t.to(torch.float32) / (T - 1 + 1e-8)               # feed exactly as trained
        progress = (T - k) / (T - 1) if T > 1 else 1.0
        current_guidance_scale = guidance_scale
        # current_guidance_scale = guidance_scale * (1.0 - progress) + 0.01 * progress
        # predict eps (with CFG if provided)
        if cond is None:
            eps = model(x, t_rev)
        else:
            # Match old behavior: unconditional call doesn't use z_ctrl
            eps_u = model(x, t_rev, cond=cond_null)
            eps_c = model(x, t_rev, cond=cond, z_ctrl=x_T)
            # eps   = eps_u + guidance_scale * (eps_c - eps_u)
            eps = eps_u + current_guidance_scale * (eps_c - eps_u)

        # x0 prediction (I2SB: target ~(x_t - x_0)/sqrt(A_t) → x0 = x - sqrt(A_t) * eps)
        A_n = A_table[idx_b, t].view(B,1,1)
        x0_pred = x - (A_n + 1e-10).sqrt() * eps

        # FINAL step: return x0
        if k == 0:
            x = x0_pred
            break

        # reverse posterior p(x_{prev} | x, x0):
        t_prev = schedule_idx[:, k-1]                               # (B,)
        # IMPORTANT: enforce t_prev <= t (monotone). Our construction guarantees this,
        # but in case of equal indices, Delta=0 acts as an identity step.
        A_prev = A_table[idx_b, t_prev].view(B,1,1).clamp_min(0.0)
        Delta  = (A_n - A_prev).clamp_min(0.0)  # no negatives now

        w0      = Delta / (A_n + 1e-10)
        wn      = A_prev / (A_n + 1e-10)
        mu_prev = w0 * x0_pred + wn * x
        std_prev= ((A_prev * Delta) / (A_n + 1e-10)).sqrt()

        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x = mu_prev + eta * std_prev * noise

    return x



class ControlDNAGate(nn.Module):
    # gate g = f(control, DNA, time), no LayerNorms
    def __init__(self, ctrl_dim, model_dim, gmax=1.0):
        super().__init__()
        self.lin_t = nn.Linear(model_dim, model_dim)   # time → D  (zero-init)
        nn.init.zeros_(self.lin_t.weight); nn.init.zeros_(self.lin_t.bias)
        self.mlp   = nn.Sequential(
            nn.Linear(ctrl_dim + 2*model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1)
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias,  -0.5)   # start gate < 0.5
        self.gmax = gmax

    def forward(self, ctrl_vec, dna_sum, time_vec):  # (B,Cc),(B,D),(B,D)
        
        h = torch.cat([ctrl_vec, dna_sum, self.lin_t(time_vec)], dim=-1)
        g = torch.sigmoid(self.mlp(h))               # (B,1)
        return self.gmax * g.view(-1,1,1)            # → (B,1,1)
    
    
    
def i2sb_training_step(model, x_0, x_1, t, beta_schedule, cond=None,
                       EPS=1e-6, schedule_scaler=None, weightLoss=None, 
                       return_x0_pred=False, alpha=1, verbose=False, 
                       dna_tokens=None, p_uncond=0.05, scaler=None,
                       force_unconditional=False, disable_cfg_dropout=False):
    """ Train I2SB using the closed-form posterior q(x_t | x_0, x_1).
    Args:
        model: The neural network ε_θ(x_t, t)
        x_0: Clean image (start of the bridge)
        x_1: Degraded image (end of the bridge)
        t: Time step tensor of shape (B,) with float values in [0, 1]
        beta_schedule: (betas, alphas, alphas_cumprod) from cosine schedule
        cond: Optional conditioning information
    """
    device = x_0.device
    B = x_0.size(0)
    betas, _, _ = beta_schedule
    # Compute σ_t and σ̄_t for each timestep
    T = len(betas)
    if schedule_scaler is not None and cond is not None:
        # s_p = schedule_scaler(cond).view(B,1) # (B, 1)
        cond_mean = cond.mean(dim=1,keepdim=True) # (B, 1, D)
        # betas, _, aux = schedule_scaler(cond_mean)
        s_p, kl = schedule_scaler(cond_mean) # (B, 1)
        # s_p = schedule_scaler(cond_mean) # (B, 1)
        s_p = s_p.view(B,1) # (B, 1)
        betas = (betas.view(1,T).expand(B,T)).to(device) # (B, T) - ensure on device
        betas = betas * s_p.view(B,1)
    else:
        betas = betas.view(1,T).expand(B,T).to(device)
    t_scaled = (t * (T - 1)).long().clamp(0, T-1).to(device) # Scale t to [0, T-1] - ensure on device
    ## q-sampler
    # Ensure indexing shapes
    A = torch.cumsum(betas, dim=1)
    A_T = A[:, -1].view(B,1,1)
    sigma_t2 = A[torch.arange(B, device=A.device), t_scaled].view(B,1,1)
    sigma_bar_t2 = (A_T - sigma_t2).clamp_min(1e-12)
    # Posterior mean and std (Equation 11 in the paper)
    mu_t = (sigma_bar_t2 / (sigma_t2 + sigma_bar_t2 + 1e-12)) * x_0 + (sigma_t2 / (sigma_t2 + sigma_bar_t2 + 1e-12)) * x_1
    sigma_t = (sigma_t2 * sigma_bar_t2 / (sigma_t2 + sigma_bar_t2 + 1e-12)).sqrt()
    # Sample x_t from posterior
    noise = torch.randn_like(x_0)
    x_t = mu_t + sigma_t * noise
    ## sigma_t scale by sqrt(s_p), mu_t is not scaled
    # Regression target: (x_t - x_0) / sigma_t
    if verbose:
        print(f'alpha being used: {alpha}')
        
        
    ## stable version ###
    pred_noise = model(x_t, t, 
                       cond=cond, alpha=alpha, 
                       z_ctrl=x_1, 
                       force_unconditional=force_unconditional, 
                       disable_cfg_dropout=disable_cfg_dropout,
                       )
    target = (x_t - x_0) / (sigma_t2.sqrt()+EPS) ## (B, 1, D)
    target = target.detach()
    # target = noise.detach()
    if schedule_scaler is not None and cond is not None:
        loss_weight = s_p.detach()
        loss_per_sample = F.mse_loss(pred_noise, target.detach(), reduction='none')
        loss_per_sample = loss_per_sample.mean(dim=[1, 2])  # (B,)
        # Apply weight to normalize expected loss
        loss_diff = (loss_per_sample * loss_weight.squeeze(-1)).mean()
    elif weightLoss is not None:
        w = weightLoss.squeeze(-1)  # shape (B,)
        loss_per_sample = F.mse_loss(pred_noise, target.detach(), reduction='none').mean(dim=[1, 2])
        loss_diff = (w * loss_per_sample).sum() / w.sum()
    else:
        loss_diff = F.mse_loss(pred_noise, target)
    ########################################################
    
    
    
    moe_term = 0.0
    aux = getattr(model, "moe_aux", None)
    # if aux is not None and "lb_loss" in aux and moe_lb_coef > 0:
    #     moe_term = moe_lb_coef * aux["lb_loss"]
    loss_val = (loss_diff + moe_term)
    loss = {
        'loss_diff': loss_val,
        'x_t': x_t,
    }
    if return_x0_pred:
        x0_pred = x_t - (sigma_t2.sqrt()+EPS) * pred_noise
        return loss, x0_pred
    else:
        return loss



class scDiffusion1DStable(nn.Module):
    def __init__(self, embedding_dim=128, num_layers=4,
                 use_dna_attn=True, dna_dim=1536, ctrl_dim=128,
                 cfg_dropout_prob = 0.1, 
                 use_local_post_gate=False, use_attn_temperature=False,
                 log_gate_stats=False, **kw):
        super().__init__()
        self.dim = embedding_dim
        self.use_dna_attn = use_dna_attn
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self._last_cfg_drop_mask = None
        self.log_gate_stats = log_gate_stats
        self._gate_stats_buffer = []  # for logging

        # self.dna_proj   = nn.Linear(dna_dim, embedding_dim)
        self.ctrl_proj  = nn.Linear(ctrl_dim, embedding_dim)
        self.ctrl_norm = nn.LayerNorm(embedding_dim)
        self.ctrl_dna_gate = ControlDNAGate(ctrl_dim=embedding_dim, model_dim=embedding_dim, gmax=1.0)
        

        self.time_mlp = TimeEmbedding(embedding_dim)
        self.in_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.blocks   = nn.ModuleList([
            CondBlock(embedding_dim, heads=4, use_xattn=use_dna_attn,
                     use_local_post_gate=use_local_post_gate,
                     use_attn_temperature=use_attn_temperature) 
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.null_cfg_diff = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.null_cfg_diff, std=1e-4)
        
        
    @staticmethod
    def _bernoulli_mask(B, device, p, force_unconditional=False, disable=False):
        if force_unconditional:
            return torch.ones(B, 1, 1, dtype=torch.bool, device=device)
        if disable or p <= 0.0:
            return torch.zeros(B, 1, 1, dtype=torch.bool, device=device)
        return (torch.rand(B, 1, 1, device=device) < p)
    

    def _apply_cfg_dropout(self, cond_tokens, drop_mask, null_vec):
        if cond_tokens is None:
            return None
        if (drop_mask is None) or (not drop_mask.any()):
            return cond_tokens
        B, Lc, D = cond_tokens.shape
        null_seq = null_vec.view(1, 1, D).to(cond_tokens.dtype).to(cond_tokens.device).expand(B, Lc, D)
        return torch.where(drop_mask.expand(B, Lc, 1), null_seq, cond_tokens)
    
    
    def make_cfg_drop_mask(self, batch_size, device, *, training: bool,
                           force_unconditional: bool = False, disable_dropout: bool = False):
        p = self.cfg_dropout_prob if training else 0.0
        mask = self._bernoulli_mask(batch_size, device, p,
                                    force_unconditional=force_unconditional,
                                    disable=disable_dropout)
        self._last_cfg_drop_mask = mask
        return mask
    
    def get_last_cfg_drop_mask(self):
        return self._last_cfg_drop_mask
    
    def get_gate_stats(self):
        """Return aggregated gate statistics from the logging buffer."""
        if not self._gate_stats_buffer:
            return None
        stats = {
            'mean': np.mean([s['mean'] for s in self._gate_stats_buffer]),
            'std': np.mean([s['std'] for s in self._gate_stats_buffer]),
            'min': np.min([s['min'] for s in self._gate_stats_buffer]),
            'max': np.max([s['max'] for s in self._gate_stats_buffer]),
            'n_batches': len(self._gate_stats_buffer)
        }
        return stats
    
    def clear_gate_stats(self):
        """Clear the gate statistics buffer."""
        self._gate_stats_buffer.clear()

    # simple time gate: 1.0 (keep it trivial; you can replace with a schedule later)
    def time_gate(self, t):
        return 1.0

    def forward(self, x, t, alpha=1.0, cond=None, z_ctrl=None, force_unconditional = False, disable_cfg_dropout = False):
        # x: (B,L,D)
        x = self._forward_body(x, t, cond, z_ctrl, force_unconditional, disable_cfg_dropout)
        return self.out_proj(x)

    def _forward_body(self, x, t, cond=None, z_ctrl=None, force_unconditional = False, disable_cfg_dropout = False):
        B, L, D = x.shape
        x = self.in_proj(x)                               # no input LN
        time_vec = self.time_mlp(t)                       # (B,D)

        # --- control vector (accept (B,Cc) or (B,1,Cc)) ---
        ctrl_vec = None
        if z_ctrl is not None:
            if z_ctrl.dim() == 3 and z_ctrl.size(1) == 1:
                z_ctrl = z_ctrl.squeeze(1)                # (B,Cc)
            ctrl_vec = self.ctrl_proj(z_ctrl)             # (B,D)
            
            
        drop_mask = self.make_cfg_drop_mask(
            B, x.device, training=self.training,
            force_unconditional=force_unconditional,
            disable_dropout=disable_cfg_dropout
        )
            
        cond = self._apply_cfg_dropout(cond, drop_mask, self.null_cfg_diff)

        # --- joint gate g(control, DNA, time) ---
        gate = None
        if (ctrl_vec is not None) and (cond is not None):
            gate = self.ctrl_dna_gate(self.ctrl_norm(ctrl_vec), torch.mean(cond, dim=1), time_vec)  # (B,1,1) added the layernorm on Oct 27th, 2025
            gate = gate * self.time_gate(t)
            
            # Log gate statistics if enabled (for monitoring convergence)
            if self.log_gate_stats and self.training:
                gate_mean = gate.mean().item()
                gate_std = gate.std().item()
                gate_min = gate.min().item()
                gate_max = gate.max().item()
                self._gate_stats_buffer.append({
                    'mean': gate_mean, 'std': gate_std,
                    'min': gate_min, 'max': gate_max
                })
                # Keep buffer size manageable (last 100 batches)
                if len(self._gate_stats_buffer) > 100:
                    self._gate_stats_buffer.pop(0)

        # --- blocks ---
        for blk in self.blocks:
            x = blk(x, time_vec, cond, gate)
        return x


class StopGradOnModule:
    """
    Context manager to stop gradients on a module's parameters while allowing
    gradients to flow through the module's forward pass (for inputs).
    
    This is useful for isolating gradient updates - e.g., allowing gradients
    to flow to input conditions while preventing updates to model weights.
    
    The key insight: parameter hooks receive the gradient during backward pass.
    By returning None from the hook, we stop the gradient from updating the parameter.
    However, gradients still flow through the forward computation graph to inputs.
    """
    def __init__(self, module):
        self.module = module
        self.hooks = []
    
    def __enter__(self):
        # Register hooks on all parameters to stop gradients
        # The hook function receives the gradient and returns zeros to prevent parameter updates
        # while still allowing gradients to flow through the computation graph to inputs
        for param in self.module.parameters():
            if param.requires_grad:
                # Return zeros to zero out the gradient, preventing parameter updates
                # but the computation graph is preserved so gradients can flow to inputs
                hook_handle = param.register_hook(lambda g: torch.zeros_like(g) if g is not None else None)
                self.hooks.append(hook_handle)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove hooks - PyTorch hook handles have a remove() method
        for hook_handle in self.hooks:
            hook_handle.remove()
        self.hooks.clear()





