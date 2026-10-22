"""
Attention components: cross-attention blocks, FiLM, DeepSet, and related modules.

This module contains:
- Basic components: TimeEmbedding, MLPBlock, ResidualScale
- Attention blocks: CondBlock, CrossAttentionBlock
- FiLM and gating: BilinearFiLM, ControlDNAGate
- DeepSet encoders: MultiSequenceDeepSet, MultiSequenceDeepSetCond
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def timestep_embedding(t, dim, max_period=10000):
    """
    Sinusoidal timestep embedding from DDPM.
    
    Args:
        t: (B,) int/float timestep 0…T-1 (or continuous σ)
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal embedding
        
    Returns:
        emb: (B, dim) timestep embeddings
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


class TimeEmbedding(nn.Module):
    """Time embedding module for diffusion models."""
    def __init__(self, dim, T=2000, mult=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mult),
            nn.SiLU(),
            nn.Linear(dim * mult, dim),
        )
        self.T = T
        
    def forward(self, t):
        emb = timestep_embedding(t, self.mlp[0].in_features)  # shape (B, dim)
        return self.mlp(emb)


class MLPBlock(nn.Module):
    """MLP block with layer norm and feedforward network."""
    def __init__(self, embed_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.ff(self.norm(x))


class ResidualScale(nn.Module):
    """Residual connection with learnable scaling: x + α·F(x)."""
    def __init__(self, dim, init_alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(init_alpha * torch.ones(dim))

    def forward(self, x, f_out):
        return x + self.alpha * f_out


class BilinearFiLM(nn.Module):
    """
    FiLM(cond, state) where state is pooled from current token activations x.
    gamma, beta = W(  V(cond) ⊙ G(state)  ), with zero-init so modulation starts at 0.
    """
    def __init__(self, cond_dim: int, state_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or max(64, min(out_dim, 4 * ((cond_dim + state_dim) // 8 + 1)))
        # Branch projections (bias-free; LN for scale hygiene)
        self.e_proj = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, h, bias=False))
        self.s_proj = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, h, bias=False))
        # Output to 2*out_dim (gamma, beta), zero-initialized → start as identity
        self.out = nn.Linear(h, 2 * out_dim, bias=False)
        nn.init.zeros_(self.out.weight)
        # State pooling (mean over tokens)
        self.pool = lambda x: x.mean(dim=1)   # x: (B, L, D) → (B, D)

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
    def forward(self, x_tokens: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_tokens: (B, L, D) current block input (pre/post LN is fine)
        cond:     (B, 1, cond_dim) perturbation embedding
        returns:  gamma, beta shaped (B, 1, D_out) for broadcasting over tokens
        """
        s = self.pool(x_tokens)                 # (B, D_state)
        u = self.e_proj(cond.squeeze(1)) * self.s_proj(s)  # (B, h)  ← Hadamard interaction
        gamma, beta = self.out(u).chunk(2, dim=-1)  # (B, D_out) each
        return gamma.unsqueeze(1), beta.unsqueeze(1)  # (B,1,D_out)


class ControlDNAGate(nn.Module):
    """Gate g = f(control, DNA, time), no LayerNorms."""
    def __init__(self, ctrl_dim, model_dim, gmax=1.0):
        super().__init__()
        self.lin_t = nn.Linear(model_dim, model_dim)   # time → D  (zero-init)
        nn.init.zeros_(self.lin_t.weight)
        nn.init.zeros_(self.lin_t.bias)
        self.mlp = nn.Sequential(
            nn.Linear(ctrl_dim + 2*model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1)
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, -0.5)   # start gate < 0.5
        self.gmax = gmax

    def forward(self, ctrl_vec, dna_sum, time_vec):  # (B,Cc),(B,D),(B,D)
        h = torch.cat([ctrl_vec, dna_sum, self.lin_t(time_vec)], dim=-1)
        g = torch.sigmoid(self.mlp(h))               # (B,1)
        return self.gmax * g.view(-1,1,1)            # → (B,1,1)


class CondBlock(nn.Module):
    """Conditional block with cross-attention and time modulation."""
    def __init__(self, dim, heads=4, use_xattn=True,
                 gate_attn_init=0.05, gate_res_init=0.05,
                 use_local_post_gate=False, use_attn_temperature=False):
        super().__init__()
        self.use_xattn = use_xattn
        self.use_local_post_gate = use_local_post_gate
        self.use_attn_temperature = use_attn_temperature

        # (A) token MLP residual
        self.mlp = MLPBlock(dim)
        self.scale_res = ResidualScale(dim, init_alpha=gate_res_init)

        # (B) cross-attn residual
        if use_xattn:
            # NEW: self-attention on cond_tokens (enriches conditioning before cross-attn)
            self.cond_self_ln = nn.LayerNorm(dim, elementwise_affine=False)
            self.cond_self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.scale_cond_self = ResidualScale(dim, init_alpha=gate_attn_init)
            
            self.xattn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.scale_attn = ResidualScale(dim, init_alpha=gate_attn_init)

            # light pre-norms for q/k only (no affine to avoid shifting stats)
            self.q_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(dim, elementwise_affine=False)

            # --- post-attn gain g(time, cond) (identity-preserving) ---
            if use_local_post_gate:
                self.gate_mlp = nn.Sequential(
                    nn.Linear(2*dim, dim), nn.SiLU(),
                    nn.Linear(dim, 1)                    # → (B,1)
                )
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.zeros_(self.gate_mlp[-1].bias)
                self.alpha_cond = nn.Parameter(torch.tensor(1e-2))  # tiny LayerScale for gain
            else:
                self.gate_mlp = None
                self.alpha_cond = None

            # --- attention temperature τ(time, cond) (≥ τ_min) ---
            if use_attn_temperature:
                self.temp_mlp = nn.Sequential(
                    nn.Linear(2*dim, dim), nn.SiLU(),
                    nn.Linear(dim, 1)                     # → (B,1) log-τ raw
                )
                nn.init.zeros_(self.temp_mlp[-1].weight)
                nn.init.zeros_(self.temp_mlp[-1].bias)
                self.tau_min = 0.7                        # keep softmax from collapsing
                self.alpha_tau = nn.Parameter(torch.tensor(1e-2))  # small learning rate into τ
            else:
                self.temp_mlp = None
                self.tau_min = None
                self.alpha_tau = None

        # (C) very light AdaLN-zero on time
        self.adaln = nn.Linear(dim, 2*dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        
        self.alpha_s = nn.Parameter(torch.tensor(0.05))  # increased from 1e-2
        self.alpha_b = nn.Parameter(torch.tensor(0.05))  # increased from 1e-2

    def forward(self, x, time_vec, cond_tokens=None, gate=None):
        # ---------- time-only modulation of x (unchanged spirit) ----------
        sc, sh = self.adaln(time_vec).chunk(2, dim=-1)          # (B,D)
        gx = x * (1 + self.alpha_s * sc).unsqueeze(1) + self.alpha_b * sh.unsqueeze(1)

        # token MLP residual
        x = self.scale_res(x, self.mlp(gx))

        # ---------- conditional residual path (CLEAN AMPLIFICATION) ----------
        if self.use_xattn and cond_tokens is not None and gate is not None:
            # NEW: self-attention on cond_tokens (enriches conditioning representation)
            cond_sa = self.cond_self_ln(cond_tokens).float()
            cond_sa_out, _ = self.cond_self_attn(cond_sa, cond_sa, cond_sa, need_weights=False)
            cond_tokens = self.scale_cond_self(cond_tokens, cond_sa_out.to(cond_tokens.dtype))
            
            # (1) pre-norm for stable SDPA
            q = self.q_norm(x).float()                # (B,L,D)
            k = self.k_norm(cond_tokens).float()      # (B,Lc,D)  <- now using enriched cond
            v = k

            # (2) Compute time×cond summaries if needed for temperature or local gate
            tc = None
            if self.use_attn_temperature or self.use_local_post_gate:
                c_sum = cond_tokens.mean(dim=1)           # (B,D)
                tc = torch.cat([time_vec, c_sum], dim=-1) # (B,2D)

            # (3) Apply attention temperature if enabled
            if self.use_attn_temperature and tc is not None:
                # learned attention temperature τ(time,cond)
                log_tau_raw = self.temp_mlp(tc) * self.alpha_tau      # (B,1)
                tau = F.softplus(log_tau_raw) + self.tau_min      # (B,1), ≥ τ_min
                # implement temperature by scaling q (equivalently k) before MHA
                q = q / tau.view(-1, 1, 1)

            # (4) SDPA
            attn_out, _ = self.xattn(q, k, v, need_weights=False) # (B,L,D)
            
            # (5) Apply local post-attn gate if enabled, otherwise use only global gate
            if self.use_local_post_gate and tc is not None:
                g_logits = self.gate_mlp(tc)                          # (B,1), zero-init → 0
                g_tc = torch.sigmoid(self.alpha_cond * g_logits).view(-1, 1, 1)  # (B,1,1)
                attn_out = (attn_out * gate * g_tc).to(x.dtype)       # combine with global gate
            else:
                # Use only global gate from ControlDNAGate
                attn_out = (attn_out * gate).to(x.dtype)

            # (6) residual add with LayerScale
            x = self.scale_attn(x, attn_out)

        return x


class CrossAttentionBlock(nn.Module):
    """Proper cross-attention block with residual and normalization"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.normkv = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key_value):
        # Cross attention with residual
        normed_kv = self.normkv(key_value)
        attn_out = self.cross_attn(self.norm1(query), normed_kv, normed_kv)[0]
        query = (query + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(query))
        query = (query + ffn_out)
        
        return query



class MultiSequenceDeepSetCond(nn.Module):
    """
    DeepSets encoder with *control-conditioned* attention pooling.

    • φ encodes tokens per sequence.
    • Attention pooling uses Q that blends a global learned query and a small MLP
      of the control embedding: Q = (1-λ)·Q_global + λ·Q_ctrl.
    • Keys/values remain DNA-token features only → no direct content injection from control.

    Extras:
      - control_dropout: randomly drop control in pooling (fall back to global query).
      - stopgrad_ctrl_in_q: stop gradient through the control path to avoid degenerate co-adaptation.
      - tau_min/max: clamp attention temperature.
      - entropy_reg: return an optional entropy loss to keep pooling non-degenerate.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        phi_layers: int = 1,
        rho_layers: int = 1,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        ln_eps: float = 1e-5,
        # --- new knobs ---
        ctrl_dim: int = 128,
        q_mix_lambda: float = 0.5,          # blend between global and control queries
        control_dropout: float = 0.1,       # chance to ignore control in pooling
        stopgrad_ctrl_in_q: bool = True,    # cut grad on control → Q_ctrl
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        return_entropy: bool = False,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.return_entropy = return_entropy
        self.q_mix_lambda = float(q_mix_lambda)
        self.control_dropout = float(control_dropout)
        self.stopgrad_ctrl_in_q = bool(stopgrad_ctrl_in_q)
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)

        # φ: per-token encoder (shared)
        self.ln_in = nn.LayerNorm(input_dim, eps=ln_eps) if use_layernorm else nn.Identity()
        phi = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(phi_layers - 1):
            phi += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        self.phi = nn.Sequential(*phi)

        # ρ heads
        self.rho = self._make_mlp(hidden_dim, output_dim, rho_layers, dropout, use_layernorm, ln_eps)
        self.rho_pair = self._make_mlp(2 * hidden_dim, output_dim, max(1, rho_layers), dropout, use_layernorm, ln_eps)

        # global query (learned parameter)
        self.q_global = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # small control→query MLP (tiny capacity)
        self.q_ctrl_mlp = nn.Sequential(
            nn.LayerNorm(ctrl_dim),
            nn.Linear(ctrl_dim, hidden_dim),
            nn.Tanh(),                        # bound magnitude to avoid softmax blow-ups
        )

        self.ln_after_pool = nn.LayerNorm(hidden_dim, eps=ln_eps) if use_layernorm else nn.Identity()

        # learnable (log) temperature for attention; clamped at runtime
        self.log_tau = nn.Parameter(torch.zeros(()))  # start at τ≈1.0

    @staticmethod
    def _make_mlp(in_dim, out_dim, n_layers, dropout, use_ln, ln_eps):
        layers = [nn.Linear(in_dim, in_dim)]
        if use_ln: layers += [nn.LayerNorm(in_dim, eps=ln_eps)]
        layers += [nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(in_dim, in_dim)]
            if use_ln: layers += [nn.LayerNorm(in_dim, eps=ln_eps)]
            layers += [nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(in_dim, out_dim)]
        return nn.Sequential(*layers)

    def _attn_pool_cond(self, h, ctrl_vec=None, mask=None):
        """
        h:    (B, L, H) token features after φ
        ctrl: (B, Cc)   control embedding (optional)
        """
        B, L, H = h.shape
        # Build query
        qg = self.q_global.expand(B, 1, H)  # (B,1,H)
        if (ctrl_vec is None) or (self.training and torch.rand(()) < self.control_dropout):
            q = qg
        else:
            c = ctrl_vec
            if self.stopgrad_ctrl_in_q:
                c = c.detach()
            qc = self.q_ctrl_mlp(c).unsqueeze(1)  # (B,1,H)
            # blend with a fixed λ (could also make λ learnable and clamp to [0,1])
            q = (1.0 - self.q_mix_lambda) * qg + self.q_mix_lambda * qc  # (B,1,H)

        # scaled dot-product attention (single-head)
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        scores = (q @ h.transpose(1, 2)) / (tau * (H ** 0.5))  # (B,1,L)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        w = torch.softmax(scores, dim=-1)                      # (B,1,L)
        pooled = w @ h                                         # (B,1,H)

        ent = None
        if self.return_entropy:
            # entropy across L (higher is more uniform)
            p = w.clamp_min(1e-8)
            ent = -(p * p.log()).sum(dim=-1).mean()  # scalar (avg over batch & head=1)

        return pooled, ent

    def forward(
        self,
        sequences: List[torch.Tensor],                 # list of (B, L_i, D)
        masks: Optional[List[torch.Tensor]] = None,    # list of (B, L_i)
        pool: str = "sum",                             # 'sum', 'mean', 'diff'
        diff_mode: str = "anchor_and_diff",
        ctrl_vec: Optional[torch.Tensor] = None        # (B, Cc) control embedding
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """
        Returns: (B, 1, output_dim) and optional entropy regularizer.
        """
        assert pool in {"sum", "mean", "diff"}
        masks = masks or [None] * len(sequences)

        per_seq = []
        entropies = []
        for seq, m in zip(sequences, masks):
            seq = self.ln_in(seq) if isinstance(self.ln_in, nn.LayerNorm) else seq
            h = self.phi(seq)                               # (B, L_i, H)
            pooled, ent = self._attn_pool_cond(h, ctrl_vec=ctrl_vec, mask=m)  # (B,1,H)
            pooled = self.ln_after_pool(pooled)             # (B,1,H)
            per_seq.append(pooled.squeeze(1))               # (B,H)
            if ent is not None:
                entropies.append(ent)

        if pool in {"sum", "mean"}:
            stacked = torch.stack(per_seq, dim=1)           # (B, Nseq, H)
            set_repr = stacked.sum(dim=1) if pool == "sum" else stacked.mean(dim=1)  # (B,H)
            out = self.rho(set_repr)                        # (B, out_dim)

        else:  # 'diff'
            assert len(per_seq) == 2, "diff pooling requires exactly 2 sequences (A,B)."
            c_A, c_B = per_seq[0], per_seq[1]               # (B,H)
            c_diff = c_B - c_A
            if diff_mode == "diff_only":
                out = self.rho(c_diff)
            elif diff_mode == "anchor_and_diff":
                out = self.rho_pair(torch.cat([c_A, c_diff], dim=-1))
            else:
                raise ValueError("diff_mode must be 'diff_only' or 'anchor_and_diff'.")

        entropy_reg = None
        if self.return_entropy and len(entropies):
            # encourage non-collapsed pooling: maximize entropy → add as negative loss outside
            entropy_reg = torch.stack(entropies).mean()

            return out.unsqueeze(1), entropy_reg
        return out.unsqueeze(1)

class MultiSequenceDeepSet(nn.Module):
    """
    Flexible DeepSets encoder for multiple variable-length sequences.
    Pool is chosen at forward: 'sum', 'mean', 'diff'.

    For pool='diff' (requires exactly 2 sequences):
      - diff_mode='diff_only': encode c_diff = c_B - c_A with rho
      - diff_mode='anchor_and_diff': encode concat([c_A, c_diff]) with rho_pair  (recommended)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        phi_layers: int = 1,
        rho_layers: int = 1,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        # φ: per-token encoder (shared)
        self.use_layernorm = use_layernorm
        self.ln_in = nn.LayerNorm(input_dim, eps=ln_eps) if use_layernorm else nn.Identity()
        phi_layers_list = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(phi_layers - 1):
            phi_layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.phi = nn.Sequential(*phi_layers_list)

        # ρ (single-vector head): used for 'sum'/'mean' and 'diff_only'
        self.rho = self._make_mlp(hidden_dim, output_dim, rho_layers, dropout, use_layernorm, ln_eps)

        # ρ_pair (two-vector head): used for 'anchor_and_diff' on 'diff'
        # input dim = 2 * hidden_dim because we feed [c_A, c_diff]
        self.rho_pair = self._make_mlp(2 * hidden_dim, output_dim, max(1, rho_layers), dropout, use_layernorm, ln_eps)
        self.query = nn.Parameter(torch.randn(1, hidden_dim))  # (1,H)
        
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.tau_min, self.tau_max = 0.5, 2.0
        
        self.ln_after_pool = nn.LayerNorm(hidden_dim, eps=ln_eps) if use_layernorm else nn.Identity()

    @staticmethod
    def _make_mlp(in_dim, out_dim, n_layers, dropout, use_ln, ln_eps):
        layers = [nn.Linear(in_dim, in_dim)]
        if use_ln: layers += [nn.LayerNorm(in_dim, eps=ln_eps)]
        layers += [nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(in_dim, in_dim)]
            if use_ln: layers += [nn.LayerNorm(in_dim, eps=ln_eps)]
            layers += [nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(in_dim, out_dim)]
        return nn.Sequential(*layers)

    def attn_pool(self, h, mask=None):
        # h: (B,L,H)
        h_raw = h
        q = self.query.unsqueeze(0).expand(h.size(0), -1, -1)   # (B,1,H)
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        attn_scores = torch.matmul(q, h.transpose(1,2)) / (tau * h.size(-1)**0.5)  # (B,1,L)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1)==0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)       # (B,1,L)
        pooled = torch.bmm(attn_weights, h_raw).squeeze(1)          # (B,H)
        return pooled

    def forward(
        self,
        sequences: List[torch.Tensor],          # list of (B, L_i, D)
        masks: Optional[List[torch.Tensor]] = None,  # list of (B, L_i)
        pool: str = "sum",                      # 'sum', 'mean', 'diff'
        diff_mode: str = "anchor_and_diff",           # 'diff_only' or 'anchor_and_diff' (only for pool='diff')
    ) -> torch.Tensor:
        """
        Returns: (B, 1, output_dim)
        """
        assert pool in {"sum", "mean", "diff"}

        masks = masks or [None] * len(sequences)
        # φ on each sequence, then pool within sequence
        per_seq = []
        for seq, m in zip(sequences, masks):
            seq = self.ln_in(seq) if self.use_layernorm else seq
            h = self.phi(seq)                    # (B, L_i, H)
            pooled = self.attn_pool(h, m)             # (B, H)
            pooled = self.ln_after_pool(pooled)       # (B, H)
            per_seq.append(pooled)  # (B, H)

        if pool in {"sum", "mean"}:
            stacked = torch.stack(per_seq, dim=1)  # (B, Nseq, H)
            set_repr = stacked.sum(dim=1) if pool == "sum" else stacked.mean(dim=1)  # (B, H)
            out = self.rho(set_repr)  # (B, out_dim)

        else:  # pool == 'diff'
            assert len(per_seq) == 2, "diff pooling requires exactly 2 sequences (A, B)."
            c_A, c_B = per_seq[0], per_seq[1]      # (B, H) each
            c_diff = c_B - c_A                      # (B, H)

            if diff_mode == "diff_only":
                out = self.rho(c_diff)              # encode only the delta
            elif diff_mode == "anchor_and_diff":
                pair = torch.cat([c_A, c_diff], dim=-1)  # keep anchor + delta
                out = self.rho_pair(pair)
            else:
                raise ValueError("diff_mode must be 'diff_only' or 'anchor_and_diff'.")

        return out.unsqueeze(1)  # (B, 1, out_dim)

