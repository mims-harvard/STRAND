"""
Prediction heads: modules for predicting RNA expression from latent representations.

This module contains:
- PredictionHead: Main prediction head with zero-inflation support
- PredictionHeadMixture: Prediction head with mixture-of-variances
- DeltaHead: Predicts delta (perturbation - control) from latent delta
- PerturbHeadFromZ: Predicts perturbed expression directly from z_pert
- ResidualOnControlHead: Residual prediction on control cell
- DNAFullPredictor: Full predictor combining adaptor and prediction head
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .attention import CrossAttentionBlock

# Import loss functions from the refactored package (no legacy fallback)
from ..losses import (
    HurdleTruncGaussResidualLoss, 
    GaussianLoss
)



class MLP(nn.Module):
    """Two-layer GELU MLP with optional dropout (used inside DeltaHead)."""
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.0):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)

class ResidualOnControlHead(nn.Module):
    def __init__(self, d_dna, d_ctrl, d_out, h=512):
        super().__init__()
        # two-tower + low-rank interaction, zero-meanable
        self.dna_tower  = nn.Sequential(nn.LayerNorm(d_dna),
                                        nn.Linear(d_dna, h), nn.GELU(),
                                        nn.Linear(h, h))
        self.ctrl_tower = nn.Sequential(nn.LayerNorm(d_ctrl),
                                        nn.Linear(d_ctrl, h), nn.GELU(),
                                        nn.Linear(h, h))
        self.bilin = nn.Linear(h, d_out, bias=False)          # for dna⊙ctrl
        self.res   = nn.Linear(h, d_out)                      # dna-only path
        # logits/sigma heads as you already have
        self.zero_head = nn.Linear(h, d_out)                  # p_nz logits
        self.sigma_head = nn.Linear(h, 1)                     # shared σ

        # init: small residual, no baseline drift
        for m in [self.bilin, self.res, self.zero_head, self.sigma_head]:
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=1e-3)
            if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)

    def forward(self, h_dna, x_ctrl):
        # h_ctrl can be x_ctrl or a compact embedding of it
        h_ctrl = x_ctrl
        a = self.dna_tower(h_dna)               # (B,h)
        b = self.ctrl_tower(h_ctrl)             # (B,h)
        r = self.res(a) + self.bilin(a * b)     # (B,G) small at init
        mu = x_ctrl.detach() + r                # residual on control (stopgrad)
        logits = self.zero_head(a)              # keep gate tied to dna (optional)
        sigma  = F.softplus(self.sigma_head(a)) + 1e-4
        sigma = sigma.expand_as(mu)
        return logits, mu, sigma


class PredictionHeadMixture(nn.Module):
    """
    Extension of PredictionHead with a mixture-of-variances for σ.
    
    Shared encoder trunk produces hidden representation h.
    Separate heads:
      - mu_head:      mean (per gene)
      - zero_head:    zero-inflation logits (if losstype='zi')
      - sigma_mix:    mixture of shared variance components

    losstype = 'zi'     → Bernoulli-trunc-Normal (hurdle)
    losstype = 'gauss'  → plain Gaussian
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 512,
                 losstype: str = 'zi',
                 num_var_components: int = 4,
                 eps: float = 1e-6,
                 sigma_min: float = 0.01,
                 sigma_max: float = 0.5):
        super().__init__()
        assert losstype in ('zi', 'gauss')

        self.losstype = losstype
        self.eps = eps
        self.output_dim = output_dim
        self.num_var_components = num_var_components
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # ── shared representation trunk ────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )

        # ── mean head ─────────────────────────────────────────────────
        self.mu_head = nn.Linear(hidden_dim, output_dim)

        # ── variance mixture parameters ───────────────────────────────
        self.gene_emb = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.02)
        self.gate_layer = nn.Linear(hidden_dim, num_var_components, bias=False)
        self.var_components = nn.Parameter(torch.full((num_var_components,), -1.5))  # log σ init ~0.22
        self.register_buffer("temp", torch.tensor(1.0))  # softmax temperature

        # ── zero-inflation head ───────────────────────────────────────
        if losstype == 'zi':
            self.zero_head = nn.Linear(hidden_dim, output_dim)

    # -----------------------------------------------------------------
    def forward(self, x):
        """
        x : (B, D) → returns tuple(s) sized (B, output_dim)
        """
        h = self.trunk(x)   # (B, hidden_dim)

        # Mean prediction
        mu = self.mu_head(h)  # (B, G)

        # Variance mixture
        # Gating per gene & component
        # logits: (B, G, K)
        logits = torch.einsum("ge,be->bg", self.gene_emb, h)    # (B,G)
        logits = logits.unsqueeze(-1) + self.gate_layer(h).unsqueeze(1)  # broadcast to (B,G,K)
        pi = torch.softmax(logits / self.temp, dim=-1)          # (B,G,K)

        var_k = F.softplus(self.var_components)**2  # (K,) variance
        var   = (pi * var_k.view(1,1,-1)).sum(-1)   # (B,G)
        sigma = var.sqrt().clamp(self.sigma_min, self.sigma_max)

        if self.losstype == 'zi':
            zero_logits = self.zero_head(h)  # (B, G)
            return zero_logits, mu, sigma
        else:
            return mu, sigma


class PredictionHead(nn.Module):
    """
    Shared encoder + independent heads for:
      • zero/non-zero logits p_z
      • mean μ   (positive via softplus)
      • scale σ  (positive via softplus, small ε added)
    
    losstype = 'zi'     → Bernoulli-trunc-Normal (hurdle)
    losstype = 'gauss'  → plain Gaussian (μ, σ only)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, losstype: str = 'zi', eps: float = 1e-6):
        super().__init__()
        assert losstype in ('zi', 'gauss')
        self.eps = eps
        self.losstype = losstype

        # ── existing trunks ──
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2*hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.LayerNorm(2*hidden_dim),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.trunk_delta = nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.LayerNorm(2*hidden_dim),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ── new baseline trunk + head (use control-only context) ──
        self.base_trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim*2),
            nn.GELU(), nn.Dropout(0.1),
            nn.LayerNorm(2*hidden_dim),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.x_base_head = nn.Linear(hidden_dim, output_dim)  # predicts x_ref

        # ── delta heads ──
        self.mu_head      = nn.Linear(hidden_dim, output_dim)      # μ_r (additive delta)
        # self.sigma_head   = nn.Linear(hidden_dim, output_dim)
        self.sigma_head   = nn.Linear(hidden_dim, output_dim)
        self.gamma_scale  = nn.Parameter(torch.ones(output_dim))    # keep if you want mult. effect

        if losstype == 'zi':
            self.zero_head = nn.Linear(hidden_dim, output_dim)

    # ---------------------------------------------------------------------
    def forward(self, z, z_ctrl, x_ctrl=None):
        """
        z : (B, D) latent representation
        z_ctrl : (B, D) control latent representation  
        x_ctrl : (B, D) control input features
        →  returns tuple(s) sized (B, output_dim)
        """
        # print(f'input shape: {z.shape}')
        z_delta = z - z_ctrl
        h = self.trunk(z)        # (B, hidden_dim)
        h_delta = self.trunk_delta(z_delta)        # (B, hidden_dim)
        # print(f'h shape: {h.shape}')
        # μ  (strictly positive)
        # mu = F.softplus(self.mu_head(h))    # softplus ≈ log(1+e^x)
        mu_delta = self.mu_head(h_delta)    # no activation, for hurdle loss
        
        
        h_base = self.base_trunk(z_ctrl)
        x_ref  = self.x_base_head(h_base)          # (B, G)
        # mu = mu_delta + x_ref
        mu = mu_delta
        # mu = beta + gamma * x_ctrl
        # σ  (strictly positive, add ε for numerical floor)
        sigma = F.softplus(self.sigma_head(h_delta)) + 1e-4
        # sigma = sigma.expand(z.shape[0], -1)
        # sigma = F.softplus(self.log_sigma_g) + 1e-4
        if self.losstype == 'zi':
            zero_logits = self.zero_head(h)     # no activation
            return zero_logits, mu, sigma, x_ref      # for hurdle loss
        else:                                   # plain Gaussian
            return mu, sigma, x_ref
        
        
    
def HardConcreteGate(gate_logit):
    s = torch.sigmoid((torch.log(torch.rand_like(gate_logit)) -
                           torch.log1p(-torch.rand_like(gate_logit)) +
                           gate_logit) / 0.1)
    z = torch.clamp(s * 1.1 - 0.1, 0, 1)   # stretch to [0,1]
    return z


def log_phi(x: Tensor) -> Tensor:
    """Stable log Φ(x)."""
    return torch.special.log_ndtr(x.double())


    



class DeltaHead(nn.Module):
    def __init__(
        self,
        d_latent: int,        # dimension of z_ctrl / z_pert
        d_hidden: int = 512,  # hidden width for content backbone
        d_out: int = 512,     # output (e.g., genes)
        layers: int = 1,      # number of FiLM modulation blocks
        dropout: float = 0.1,
        use_orthogonalize: bool = False,  # optional residualize z_delta vs z_ctrl
        verbose: bool = False,
    ):
        """
        DNA-driven Δ-head: content from z_delta, context from z_ctrl (FiLM).

        Args:
            d_latent: Dimension of latent embeddings (z_ctrl, z_pert).
            d_hidden: Hidden size for the delta content pathway.
            d_out: Output dimension (e.g., #genes for x_delta).
            layers: Number of FiLM blocks applied to the delta content.
            dropout: Dropout applied inside the MLPs.
            use_orthogonalize: If True, learn a linear projector W to remove the
                component of z_delta predictable from z_ctrl: z_delta <- z_delta - W z_ctrl.
            verbose: If True, print tensor shapes in forward (debug).
        """
        super().__init__()
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.layers = layers
        self.use_orthogonalize = use_orthogonalize
        self.verbose = verbose

        # ── Normalizations ───────────────────────────────────────────────────
        self.ctrl_norm  = nn.LayerNorm(d_latent)
        self.delta_norm = nn.LayerNorm(d_latent)

        # ── Content path: z_delta → hidden ───────────────────────────────────
        # First block expands z_delta; subsequent FiLM-modulated residual MLPs operate at d_hidden.
        self.P_delta_in = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Optional stack of residual MLPs for depth after FiLM
        self.delta_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_hidden),
                MLP(d_hidden, d_hidden * 4, d_out=d_hidden, dropout=dropout)
            ) for _ in range(max(layers - 1, 0))
        ])

        # ── FiLM from z_ctrl: produce [gamma, beta] for each layer ───────────
        self.ctrl_film = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_latent, d_hidden * 4),
                nn.GELU(),
                nn.Linear(d_hidden * 4, d_hidden * 2)
            ) for _ in range(layers)
        ])
        for i in range(layers):
            nn.init.zeros_(self.ctrl_film[i][-1].weight)
            nn.init.zeros_(self.ctrl_film[i][-1].bias)

        # Small learned scales to keep FiLM stable (tanh-squashed)
        self.c_scale = nn.Parameter(torch.ones(2))  # [scale_gamma, scale_beta]

        # ── Optional orthogonalization: learn W so z_delta <- z_delta - W z_ctrl ─
        if use_orthogonalize:
            self.W = nn.Linear(d_latent, d_latent, bias=False)
            nn.init.zeros_(self.W.weight)  # start with no residualization

        # ── Output projection ────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out)
        )
        nn.init.kaiming_uniform_(self.out_proj[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, z_ctrl: torch.Tensor, z_pert: torch.Tensor):
        """
        Forward pass.

        Args:
            z_ctrl: [B, D] control latent embedding
            z_pert: [B, D] perturbed latent embedding

        Returns:
            x_delta_hat: [B, d_out] predicted raw-space delta
        """
        assert z_ctrl.dim() == 2 and z_pert.dim() == 2, "Expect [B, D] tensors."
        # Compute and (optionally) residualize delta
        z_delta = z_pert - z_ctrl
        if self.use_orthogonalize:
            # Remove component of z_delta predictable from z_ctrl
            z_delta = z_delta - self.W(z_ctrl)

        # Normalize
        z_ctrl_n  = self.ctrl_norm(z_ctrl)
        z_delta_n = self.delta_norm(z_delta)

        # Content backbone
        h = self.P_delta_in(z_delta_n)  # [B, H]

        # Apply FiLM per layer
        # Layer 0: FiLM directly on h
        gamma_beta = self.ctrl_film[0](z_ctrl_n)           # [B, 2H]
        gamma, beta = gamma_beta.chunk(2, dim=-1)          # [B, H], [B, H]
        gamma = 1.0 + self.c_scale[0] * torch.tanh(gamma)  # stabilized scale
        beta  =        self.c_scale[1] * torch.tanh(beta)
        h = gamma * h + beta

        # Deeper layers (residual MLP + FiLM)
        for i, block in enumerate(self.delta_blocks, start=1):
            h_res = block(h)                 # residual transform at hidden size
            h = h + h_res                    # pre-FiLM residual connection
            # FiLM from control (layer i)
            gamma_beta = self.ctrl_film[i](z_ctrl_n)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = 1.0 + self.c_scale[0] * torch.tanh(gamma)
            beta  =        self.c_scale[1] * torch.tanh(beta)
            h = gamma * h + beta

        # Output head
        x_delta_hat = self.out_proj(h)       # [B, d_out]

        if self.verbose:
            print(f"[DeltaHead] z_ctrl: {z_ctrl.shape}, z_pert: {z_pert.shape}, "
                  f"h: {h.shape}, x_delta_hat: {x_delta_hat.shape}")

        return x_delta_hat
    
    

class PerturbHeadFromZ(nn.Module):
    def __init__(
        self,
        d_latent: int,        # dim of z_pert
        d_hidden: int = 512,  # hidden width
        d_out: int = 978,     # e.g., #genes for x_pert
        layers: int = 2,      # residual depth at hidden size
        dropout: float = 0.1,
        verbose: bool = False,
    ):
        """
        Predict x_pert directly from z_pert (no z_ctrl/DNA paths).

        Args:
            d_latent: Dimension of z_pert.
            d_hidden: Hidden size for backbone.
            d_out: Output dimension (e.g., genes).
            layers: Number of residual MLP blocks at hidden size.
            dropout: Dropout rate inside MLPs.
            verbose: If True, prints tensor shapes in forward.
        """
        super().__init__()
        self.verbose = verbose
        self.z_norm = nn.LayerNorm(d_latent)

        # z_pert → hidden
        self.in_proj = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # residual stack at hidden size
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_hidden),
                MLP(d_hidden, 4 * d_hidden, d_out=d_hidden, dropout=dropout)
            ) for _ in range(max(layers - 1, 0))
        ])

        # head
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out)
        )
        nn.init.kaiming_uniform_(self.out_proj[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, z_pert: torch.Tensor):
        """
        Args:
            z_pert: [B, D_latent] perturbed latent embedding

        Returns:
            x_pert_hat: [B, d_out] predicted raw expression under perturbation
        """
        assert z_pert.dim() == 2, "z_pert must be [B, D]."
        # z = self.z_norm(z_pert)
        h = self.in_proj(z_pert)
        for block in self.blocks:
            h = h + block(h)
        x_pert_hat = self.out_proj(h)

        if self.verbose:
            print(f"[PerturbHeadFromZ] z_pert: {z_pert.shape} -> x_pert_hat: {x_pert_hat.shape}")
        return x_pert_hat













class Inverse_Weighted_MSE(nn.Module):
    '''
    variance detached from gradient
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, mu_r, sigma_r, y, x_ref):
        delta = y - x_ref
        mse = ((mu_r - delta) ** 2)
        inv_var = 1.0 / (sigma_r.detach() ** 2 + 1e-6)
        inv_var = inv_var / (inv_var.mean() + 1e-8)
        loss = (mse * inv_var).mean() 
        return loss

    
class DNAFullPredictor(nn.Module):
    def __init__(self, dna_dim, ctrl_dim, RNA_dim, hidden_dim, 
                 num_heads, num_layers=1, loss_metric='htg_nll', verbose=False):
        super().__init__()
        '''
        htg_null: Hurdle Truncated Gaussian Negative Log Likelihood
        nll: Negative Log Likelihood
        mse: Mean Squared Error
        '''
        
        allowed_losses = {'htg_nll', 'nll', 'mse'}
        if loss_metric not in allowed_losses:
            raise ValueError(f"Invalid loss: {loss_metric}. Choose from {allowed_losses}.")
        
        self.loss_metric = loss_metric
        
        # self.adaptor = PerturbHeadFromZ(
        #     d_latent=ctrl_dim,
        #     d_hidden=hidden_dim,
        #     d_out = hidden_dim,
        #     layers=num_layers,
        # )
        
        self.prediction_head = PredictionHead(
            input_dim=hidden_dim, output_dim=RNA_dim, hidden_dim=hidden_dim
        )
        
        # self.prediction_head = PredictionHeadMixture(
        #     input_dim=hidden_dim,
        #     output_dim=RNA_dim,
        #     hidden_dim=hidden_dim,
        #     num_var_components=8,
        # )
        
        
        # self.cross_attn = CrossAttentionBlock(
        #     hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1
        # )
        
        self.ln_z = nn.LayerNorm(hidden_dim)

        # self.loss_fn = MSELoss()  # Use the new sparse MSE loss
        if loss_metric == 'htg_nll':
            # self.loss_fn = HurdleTruncGaussLoss()
            self.loss_fn = HurdleTruncGaussResidualLoss()
            # self.loss_fn = HurdleTruncGaussDeltaLoss()
        elif loss_metric == 'nll':
            self.loss_fn = GaussianLoss()
        elif loss_metric == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f"Invalid loss metric: {loss_metric}. Choose from 'htg_nll', 'nll', or 'mse'.")
        
        self.ivw_mse_loss = Inverse_Weighted_MSE()
        
    def forward(self, dna_emb, x_ctrl, ctrl_emb=None, pert_emb=None, x_pert=None, time_emb=None, weights=None):
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        t0 = time.time()
        if x_pert is not None:
            # print(f"Using perturbation: {x_pert.shape}")
            x_delta = x_pert - x_ctrl

        
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        t1 = time.time()
        # print(f"RNA_projector: {t1 - t0:.4f} sec")
        if ctrl_emb.dim() ==3:
            ctrl_emb = ctrl_emb.squeeze(1)
        if pert_emb.dim() ==3:
            pert_emb = pert_emb.squeeze(1)
        # z_delta = self.adaptor(ctrl_emb, pert_emb)
        # z_pert = self.adaptor(pert_emb)
        
        # z_delta = pert_emb - ctrl_emb
        
        # logits, mu, sigma = self.prediction_head(z, x_ctrl)
        logits, mu, sigma, mu_ref = self.prediction_head(pert_emb, ctrl_emb, x_ctrl)
        # mu = F.softplus(mu + x_ctrl)
        # mu = F.softplus(mu)
        
        
        t4 = time.time()
        # print(f"Prediction head: {t4 - t3:.4f} sec")

        gate = torch.sigmoid(logits)
        
        # if self.loss_metric == 'htg_nll':
        #     mu = F.softplus(mu)  # ensure positivity for hurdle loss
        
        t5 = time.time()
        # print(f"Gate + mu transform: {t5 - t4:.4f} sec")
        if x_pert is None:
            if self.loss_metric == 'htg_nll':
                return {
                    'logits': logits,
                    'mu': mu,
                    'sigma': sigma,
                    'mu_ref': mu_ref,
                }
            elif self.loss_metric == 'nll':
                return {
                    'mu': mu + x_ctrl,
                    'sigma': sigma,
                }
            elif self.loss_metric == 'mse':
                return {
                    'mu': mu + x_ctrl,
                } 
        else:
            # loss = 0.05 * z_reg
            loss = 0.0
            if self.loss_metric == 'htg_nll':
                loss += self.loss_fn(logits, mu, sigma, x_pert, mu_ref, weights=weights)
                predict_dict = {
                    'logits': logits,
                    'mu': mu,
                    'sigma': sigma,
                }
                
                # delta_true = x_pert - x_ctrl
                # pi_nz = torch.sigmoid(logits)
                # a = -mu / (sigma + 1e-8)
                # lambda_a = torch.exp(-0.5*a**2) / (math.sqrt(2*math.pi) * (0.5*torch.erfc(a/math.sqrt(2)) + 1e-8))
                # cond_mean = mu + sigma * lambda_a
                # x_hat = pi_nz * cond_mean  # E[Y]
                
                # delta_hat  = x_hat - x_ctrl
                # with torch.no_grad():
                #     thr = torch.quantile(delta_true.abs(), 0.75)
                #     w = torch.sigmoid((delta_true.abs() - thr) / (0.15 * (thr + 1e-6)))
                # L_delta = (w * (delta_hat - delta_true).pow(2)).sum() / (w.sum() + 1e-8)
                # loss += L_delta *0.05
                
            elif self.loss_metric == 'nll':
                # loss += self.loss_fn(mu, sigma, x_delta)
                loss += self.loss_fn(mu, sigma)
                predict_dict = {
                    'mu': mu + x_ctrl,
                    'sigma': sigma,
                }
                
            elif self.loss_metric == 'mse':
                loss += self.loss_fn(mu, x_delta)
                predict_dict = {
                    'mu': mu + x_ctrl,
                }
                
            t6 = time.time()
            # print(f"Loss computation: {t6 - t5:.4f} sec")

            return loss, predict_dict



def _resample_cumulative_A(A_eff, T_new):
    """
    A_eff: (B, T_old) cumulative, monotone, A_T=1
    Returns:
      A_new: (B, T_new) cumulative on a uniform grid in [0,1]
    """
    B, T_old = A_eff.shape
    if T_new == T_old:
        return A_eff
    # linear 1D interpolation on the cumulative (preserves monotonicity)
    A_new = F.interpolate(
        A_eff.unsqueeze(1), size=T_new, mode="linear", align_corners=True
    ).squeeze(1)
    # small clamp to keep strictly increasing after numeric ops
    A_new = A_new.clamp_min(0.0)
    A_new[:, -1] = 1.0
    return A_new

def _A_to_betas(A):
    """A: (B,T) -> betas: (B,T) forward diffs with non-negativity clamp"""
    B, T = A.shape
    pad = torch.zeros(B, 1, device=A.device, dtype=A.dtype)
    betas = (A - torch.cat([pad, A[:, :-1]], dim=1)).clamp(1e-10, 0.999)
    return betas  



    
    
    
def get_sinusoidal_pos_embedding(seq_len, dim, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))  # [dim/2]
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, dim]

def timestep_embedding(t, dim, max_period=10000):
    """
    Sinusoidal timestep embedding from DDPM.
    t : (B,) int/float   timestep 0…T-1  (or continuous σ)
    returns : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]          # (B, half)
    emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:                                      # zero-pad if dim odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb




def init_half_sum_linear(linear: nn.Linear):
    """
    Initialize a [2D → D] linear so that both halves (null, delta) survive.
    """
    D_out, D_in = linear.weight.shape
    assert D_in == 2 * D_out, f"Expected in=2*D, got {D_in} vs {D_out}"
    with torch.no_grad():
        linear.weight.zero_()
        # pass-through both halves with 0.5 * I
        linear.weight[:, :D_out].copy_(0.5 * torch.eye(D_out))
        linear.weight[:, D_out:].copy_(0.5 * torch.eye(D_out))
        if linear.bias is not None:
            linear.bias.zero_()


def teacher_force_ratio(ep, start=0, end=9):
    return max(0.0, 1.0 - (ep - start) / max(1, end-start))

import math
from collections import defaultdict
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
