'''
This script provides DNA foundation model adaptation strategies.
With independent masking channel
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint_sequential
from functools import partial

# Core dependency: Borzoi/Flashzoi backbone
try:
    from borzoi_pytorch import Borzoi
except Exception as e:
    raise ImportError(
        "Missing dependency `borzoi-pytorch`. Install it from PyPI (see "
        "`https://github.com/johahi/borzoi-pytorch`) and ensure it is importable."
    ) from e

# Optional dependencies: used for speed (flash-attn) and PEFT fine-tuning.
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None

try:
    from flash_attn import flash_attn_func
    from flash_attn.layers.rotary import RotaryEmbedding
except Exception:
    flash_attn_func = None
    RotaryEmbedding = None
import time
import math
import numpy as np
import random
import os
import sys
import hashlib
from collections import OrderedDict



class CellSpecificCCREBias(nn.Module):
    """
    Learn cell- and distance-specific additive logit bias for cCRE tokens.

    - Uses: token embedding X, cell embedding z_cell, coarse distance bin (short/mid/long).
    - Outputs: bias_ij in [-max_bias, max_bias] for each position j in batch i.
    - Applied *only* to cCRE tokens; non-cCRE and padded tokens get 0 bias.
    - Optionally mean-centers bias over cCREs per sample to focus on relative priorities.
    """
    def __init__(self, dim=1536, cell_dim=128, num_distance_bins=3,
                 max_bias: float = 3.0, normalize_per_sample: bool = True):
        super().__init__()
        self.max_bias = float(max_bias)
        self.normalize_per_sample = bool(normalize_per_sample)

        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        # Distance bin embeddings (e.g. short/mid/long)
        self.distance_embed = nn.Embedding(num_distance_bins, num_distance_bins)

        # Predictor: [X || cell_ctx || dist_embed] -> scalar bias
        in_dim = dim + 128 + num_distance_bins
        self.importance_predictor = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, X, z_cell, distances_bp, ccre_mask):
        """
        Args:
            X:            (B, L, dim)     token embeddings
            z_cell:       (B, cell_dim)   cell embedding
            distances_bp: (B, L)         bp distances from perturbation anchor (>= 0)
            ccre_mask:    (B, L) bool    True = cCRE token (not perturb, not padding)

        Returns:
            ccre_bias:    (B, L) float   additive logit bias for cCRE tokens
        """
        B, L, dim = X.shape
        device = X.device
        dtype = X.dtype

        # encode cell: (B, 128) -> broadcast to (B, L, 128)
        z_cell_enc = self.cell_encoder(z_cell)                         # (B, 128)
        z_cell_exp = z_cell_enc.unsqueeze(1).expand(-1, L, -1)        # (B, L, 128)

        # distance bins: 0 (short), 1 (mid), 2 (long)
        d_kb = (distances_bp.to(dtype) / 1000.0)                       # (B, L)
        dist_bins = torch.zeros(B, L, device=device, dtype=torch.long)  # default short

        mid_mask = (d_kb >= 10.0) & (d_kb < 100.0)
        long_mask = (d_kb >= 100.0)
        dist_bins[mid_mask] = 1
        dist_bins[long_mask] = 2                                      # (B, L)

        dist_embed = self.distance_embed(dist_bins)                   # (B, L, num_distance_bins)

        # predictor input: (B, L, dim + 128 + num_distance_bins)
        predictor_input = torch.cat([X, z_cell_exp, dist_embed.to(dtype)], dim=-1)
        raw = self.importance_predictor(predictor_input).squeeze(-1)  # (B, L)

        # map to [-max_bias, max_bias]
        bias = self.max_bias * torch.tanh(raw)                        # (B, L)

        # apply only to cCRE tokens
        ccre_mask_f = ccre_mask.float()
        bias = bias * ccre_mask_f                                     # (B, L)

        # optional per-sample centering over cCREs
        if self.normalize_per_sample:
            denom = ccre_mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_bias = (bias * ccre_mask_f).sum(dim=-1, keepdim=True) / denom
            bias = (bias - mean_bias) * ccre_mask_f                   # (B, L)

        return bias


class StructuredCCREDropout(nn.Module):
    """
    Biologically motivated dropout that keeps the closest cCREs to the perturbation
    while randomly dropping redundant distal ones during training.
    """

    def __init__(self, keep_closest_n: int = 2, base_drop_prob: float = 0.2):
        super().__init__()
        self.keep_closest_n = max(0, int(keep_closest_n))
        self.base_drop_prob = float(base_drop_prob)

    def forward(
        self,
        ccre_mask: torch.Tensor,
        perturbation_mask: torch.Tensor,
        distances_bp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ccre_mask: (B, L) bool tensor indicating current cCRE tokens (excludes perturbation).
            perturbation_mask: (B, L) bool tensor marking perturbation bins.
            distances_bp: (B, L) float tensor with distances to perturbation anchor in bp.

        Returns:
            (B, L) bool tensor with a subset of distal cCREs dropped.
        """
        if (
            (not self.training)
            or self.keep_closest_n <= 0
            or self.base_drop_prob <= 0.0
        ):
            return ccre_mask

        B, _ = ccre_mask.shape
        device = ccre_mask.device
        augmented = ccre_mask.clone()

        for b in range(B):
            valid = ccre_mask[b] & (~perturbation_mask[b])
            ccre_indices = torch.where(valid)[0]
            num_ccres = ccre_indices.numel()
            if num_ccres <= self.keep_closest_n:
                continue  # nothing to drop

            dists = distances_bp[b, ccre_indices]
            sorted_indices = torch.argsort(dists)
            keep_core = ccre_indices[sorted_indices[: self.keep_closest_n]]
            distal = ccre_indices[sorted_indices[self.keep_closest_n :]]

            if distal.numel() == 0:
                continue

            drop_prob = min(self.base_drop_prob, 2.0 / float(num_ccres))
            # sample Bernoulli per distal cCRE
            drop_mask = torch.rand(
                distal.shape[0], device=device, dtype=distances_bp.dtype
            ) < drop_prob
            if drop_mask.any():
                augmented[b, distal[drop_mask]] = False

            # ensure closest cCREs are always kept
            augmented[b, keep_core] = True

        return augmented



def build_attn_bias(pos_q: torch.Tensor, pos_k: torch.Tensor, tau: float = 16.0, quadratic: bool = False):
    """
    pos_q: (B, Lq) continuous coords centered at perturbation
    pos_k: (B, Lk) continuous coords centered at perturbation
    Returns: (B,1,Lq,Lk) additive bias to feed into SDPA. Larger |d| => more negative.
    """
    d = pos_q[:, None, :, None] - pos_k[:, None, None, :]  # (B,1,Lq,Lk)
    if quadratic:
        return -(d * d) / (2.0 * (tau ** 2))
    else:
        return -d.abs() / tau


def build_rope_cache(L: int, Dh: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """
    Returns cos, sin with shapes (1, 1, L, Dh) for half-dim rotations.
    Assumes Dh is even. Compute in fp32 and cast later for stability.
    """
    assert Dh % 2 == 0, "RoPE head dim must be even."
    device = device or torch.device('cpu')

    # frequencies for each pair (even/odd)
    half = Dh // 2
    # inv_freq: (half,) — standard RoPE geometric progression
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))  # (half,)

    # positions: (L,)
    t = torch.arange(L, device=device, dtype=dtype)

    # angles: (L, half)
    angles = torch.einsum('l,d->ld', t, inv_freq)

    # expand to (1,1,L,half) for broadcasting over (B,H)
    cos = torch.cos(angles)[None, None, :, :]  # (1,1,L,half)
    sin = torch.sin(angles)[None, None, :, :]  # (1,1,L,half)

    # we'll use these to rotate [x_even, x_odd]
    return cos, sin

def apply_rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0):
    """
    x:   (B, H, L, Dh)  — queries or keys
    pos: (B, L) or (1, L) integer positions for this stream
    Returns rotated x with same shape. Computes cos/sin in fp32 then casts to x.dtype.

    NOTE: If L varies per batch, you can pre-compute a max-length cache and slice.
    """
    B, H, L, Dh = x.shape
    assert Dh % 2 == 0

    # Build cos/sin cache at fp32 for numeric stability
    cos, sin = build_rope_cache(L=L, Dh=Dh, base=base, device=x.device, dtype=torch.float32)

    # Allow per-batch offsets by indexing cos/sin with pos if needed.
    # If pos is not the trivial [0..L-1], we "gather" positions; else, fast path.
    # pos: (B,L) — clamp just in case
    if pos.dim() == 2 and (not torch.equal(pos, torch.arange(L, device=pos.device).expand(B, -1))):
        pos = pos.clamp_(0, L-1)
        # gather cos/sin per batch position: (B,1,L,half)
        cos = cos.expand(B, 1, L, Dh//2).gather(2, pos[:, None, :, None].expand(B,1,L,Dh//2))
        sin = sin.expand(B, 1, L, Dh//2).gather(2, pos[:, None, :, None].expand(B,1,L,Dh//2))
    else:
        # broadcast to (B,1,L,half)
        cos = cos.expand(B, 1, L, Dh//2)
        sin = sin.expand(B, 1, L, Dh//2)

    # cast to x.dtype (bf16-safe after computing in fp32)
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    # split even/odd
    x_even = x[..., 0::2]   # (B,H,L,half)
    x_odd  = x[..., 1::2]   # (B,H,L,half)

    # rotation: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    # interleave back to (B,H,L,Dh)
    # stack & reshape to original layout
    x_out = torch.empty_like(x)
    x_out[..., 0::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out

def build_ccre_mask(L, ccre_intervals_rel, bp_per_token=128.0, device=None):
    """
    Build cCRE mask on GPU (shared across batch).

    Args:
        L:                 number of tokens along sequence
        ccre_intervals_rel: list[(start_bp, end_bp)] in bp, relative to window
        bp_per_token:      bp covered by a single token at current resolution
        device:            torch.device

    Returns:
        (L,) bool tensor on GPU, all ops GPU-native.
    """
    if ccre_intervals_rel is None or len(ccre_intervals_rel) == 0:
        return torch.zeros(L, device=device, dtype=torch.bool)
    
    iv = torch.tensor(ccre_intervals_rel, device=device, dtype=torch.float32)  # (N, 2)
    starts = (iv[:, 0] / bp_per_token).round().to(torch.long)
    ends   = (iv[:, 1] / bp_per_token).round().to(torch.long)
    
    s = torch.clamp(torch.minimum(starts, ends), 0, L - 1)
    e = torch.clamp(torch.maximum(starts, ends), 0, L - 1)
    
    diff = torch.zeros(L + 1, device=device, dtype=torch.int32)
    if s.numel() > 0:
        diff.scatter_add_(0, s, torch.ones_like(s, dtype=torch.int32))
        e1 = torch.clamp(e + 1, max=L)
        diff.scatter_add_(0, e1, -torch.ones_like(e1, dtype=torch.int32))
    return diff.cumsum(0)[:L] > 0  # (L,) bool



def compress_after_transformer_fast(
    feat,
    mask_128,
    ccre_intervals_rel=None,
    distal_stride_tokens=1,
    bp_per_token=None,
):
    """
    Fast compression with adaptive resolution support:
      - full-resolution perturbation region
      - pooled flanks at stride k
      - cCRE + perturbation bins marked via keep_regions_mask
      - adaptive resolution: bp_per_token can be float or tensor (B, L) or (L,)

    feat:      (B, L, C)  # after transformer at *current* resolution
    mask_128:  (B, 1, L)  # mask at same resolution as feat
    ccre_intervals_rel: list of (start_bp, end_bp) tuples
    distal_stride_tokens: pooling stride for distal (non-perturbation) regions
    bp_per_token: bp covered by a single token of `feat`
                  - float: uniform resolution (default: 128.0)
                  - tensor (B, L) or (L,): per-token adaptive resolution

    Returns: (feat_comp, mask_comp, width_comp_bp, keep_regions_mask)
      feat_comp:      (B, L_out, C)     compressed features
      mask_comp:      (B, 1, L_out)     pooled perturbation mask
      width_comp_bp:  (B, 1, L_out)     width per token in bp
      keep_regions_mask: (B, 1, L_out)  True = cCRE/pert bin, used by SPE
    """
    B, L, C = feat.shape
    device = feat.device
    k = distal_stride_tokens

    # Handle adaptive resolution: normalize bp_per_token to (B, L) tensor
    if bp_per_token is None:
        bp_per_token = 128.0
    
    if isinstance(bp_per_token, (int, float)):
        # Uniform resolution: broadcast to (B, L)
        bp_per_token_tensor = torch.full((B, L), float(bp_per_token), device=device, dtype=torch.float32)
        bp_per_token_scalar = float(bp_per_token)
    elif isinstance(bp_per_token, torch.Tensor):
        if bp_per_token.dim() == 1 and bp_per_token.shape[0] == L:
            # (L,) -> broadcast to (B, L)
            bp_per_token_tensor = bp_per_token.unsqueeze(0).expand(B, -1).to(device=device, dtype=torch.float32)
            bp_per_token_scalar = bp_per_token.mean().item()  # For backward compat with build_ccre_mask
        elif bp_per_token.dim() == 2 and bp_per_token.shape == (B, L):
            # (B, L) -> use as is
            bp_per_token_tensor = bp_per_token.to(device=device, dtype=torch.float32)
            bp_per_token_scalar = bp_per_token.mean().item()  # For backward compat with build_ccre_mask
        else:
            raise ValueError(f"bp_per_token tensor must be (L,) or (B, L), got {bp_per_token.shape}")
    else:
        raise TypeError(f"bp_per_token must be float or tensor, got {type(bp_per_token)}")

    # ---------- 1. Build keep_mask = perturbation OR cCRE ----------
    pert_mask = (mask_128.squeeze(1) != 0)  # (B, L)
    keep_mask = pert_mask.clone()           # (B, L)

    # Detect per-sample vs shared cCRE intervals
    is_per_sample = (
        ccre_intervals_rel is not None
        and isinstance(ccre_intervals_rel, (list, tuple))
        and len(ccre_intervals_rel) == B
        and any(
            (samp is not None)
            and isinstance(samp, (list, tuple))
            and (len(samp) == 0 or (isinstance(samp[0], (list, tuple)) and len(samp[0]) == 2))
            for samp in ccre_intervals_rel
        )
    )

    if ccre_intervals_rel is not None and len(ccre_intervals_rel) > 0 and not is_per_sample:
        # Single shared cCRE definition
        valid_intervals = [
            iv for iv in ccre_intervals_rel
            if iv is not None and isinstance(iv, (list, tuple)) and len(iv) == 2
        ]
        if valid_intervals:
            # Use mean resolution for cCRE mask building (backward compat)
            ccre_mask = build_ccre_mask(
                L,
                valid_intervals,
                bp_per_token=bp_per_token_scalar,
                device=device,
            )  # (L,)
            keep_mask = keep_mask | ccre_mask.unsqueeze(0)  # (B, L)

    if not is_per_sample:
        has_any_keep = keep_mask.any(dim=1)  # (B,)
        if not has_any_keep.all():
            center = L // 2
            span = min(L, 512)
            fallback = torch.zeros(L, device=device, dtype=torch.bool)
            fallback[max(0, center - span // 2):min(L, center + span // 2)] = True
            keep_mask[~has_any_keep] = fallback

    # ---------- 2. Helper to pool a segment (left/right flanks) with adaptive resolution ----------
    def pool_segment(feat_seg, mask_seg, keep_seg, k, bp_per_token_seg):
        """
        feat_seg: (L_seg, C)
        mask_seg: (1, L_seg)
        keep_seg: (L_seg,) bool
        bp_per_token_seg: (L_seg,) tensor of bp per token for this segment

        Returns:
            feat_p:  (L_seg_out, C)
            mask_p:  (1, L_seg_out)
            keep_p:  (L_seg_out,) bool
            width_p: (1, L_seg_out) in bp
        """
        L_seg = feat_seg.shape[0]
        if L_seg == 0:
            return (
                feat_seg.new_empty(0, C),
                mask_seg.new_empty(1, 0),
                keep_seg.new_empty(0, dtype=torch.bool),
                mask_seg.new_empty(1, 0),
            )

        n_full = (L_seg // k) * k
        parts_f, parts_m, parts_k, parts_w = [], [], [], []

        if n_full > 0:
            feat_main = feat_seg[:n_full].view(n_full // k, k, C).mean(dim=1)         # (L_main, C)
            mask_main = mask_seg[:, :n_full].view(1, n_full // k, k).amax(dim=-1)     # (1, L_main)
            keep_main = keep_seg[:n_full].view(n_full // k, k).any(dim=-1)            # (L_main,)
            # Adaptive resolution: sum bp_per_token for each pooled block
            bp_main = bp_per_token_seg[:n_full].view(n_full // k, k).sum(dim=1)       # (L_main,)
            width_main = bp_main.unsqueeze(0)                                          # (1, L_main)
            parts_f.append(feat_main)
            parts_m.append(mask_main)
            parts_k.append(keep_main)
            parts_w.append(width_main)

        if n_full < L_seg:
            feat_tail = feat_seg[n_full:].mean(dim=0, keepdim=True)                   # (1, C)
            mask_tail = mask_seg[:, n_full:].amax(dim=-1, keepdim=True)              # (1, 1)
            keep_tail = keep_seg[n_full:].any().unsqueeze(0)                          # (1,)
            # Adaptive resolution: sum bp_per_token for tail
            width_tail = bp_per_token_seg[n_full:].sum().unsqueeze(0).unsqueeze(0)   # (1, 1)
            parts_f.append(feat_tail)
            parts_m.append(mask_tail)
            parts_k.append(keep_tail)
            parts_w.append(width_tail)

        feat_p = torch.cat(parts_f, dim=0)               # (L_seg_out, C)
        mask_p = torch.cat(parts_m, dim=1)               # (1, L_seg_out)
        keep_p = torch.cat(parts_k, dim=0)               # (L_seg_out,)
        width_p = torch.cat(parts_w, dim=1)              # (1, L_seg_out)
        return feat_p, mask_p, keep_p, width_p

    # ---------- 3. Per-sample processing ----------
    feat_chunks, mask_chunks, width_chunks, keep_chunks = [], [], [], []

    for b in range(B):
        feat_b = feat[b]          # (L, C)
        mask_b = mask_128[b]      # (1, L)
        pert_b = (mask_b[0] != 0) # (L,)
        bp_per_token_b = bp_per_token_tensor[b]  # (L,) per-token resolution for this sample
        
        if is_per_sample:
            ccre_b = ccre_intervals_rel[b] if b < len(ccre_intervals_rel) else []
        else:
            ccre_b = ccre_intervals_rel if ccre_intervals_rel else []

        keep_b = pert_b.clone()   # (L,)

        if is_per_sample and ccre_b:
            _valid_ccre = [
                iv for iv in ccre_b
                if iv is not None and isinstance(iv, (list, tuple)) and len(iv) == 2
            ]
            if _valid_ccre:
                # Use mean resolution for this sample's cCRE mask
                ccre_mask_b = build_ccre_mask(
                    L,
                    _valid_ccre,
                    bp_per_token=bp_per_token_b.mean().item(),
                    device=device,
                )
                keep_b = keep_b | ccre_mask_b
        elif not is_per_sample:
            keep_b = keep_mask[b]

        if not keep_b.any():
            center = L // 2
            span = min(L, 512)
            keep_b[max(0, center - span // 2):min(L, center + span // 2)] = True

        if not pert_b.any():
            feat_p, mask_p, keep_p, width_p = pool_segment(feat_b, mask_b, keep_b, k, bp_per_token_b)
            feat_chunks.append(feat_p)
            mask_chunks.append(mask_p)
            width_chunks.append(width_p)
            keep_chunks.append(keep_p)
            continue

        pert_idx = pert_b.nonzero(as_tuple=True)[0]
        if pert_idx.numel() > 0:
            p_start = int(pert_idx.min().item())
            p_end   = int(pert_idx.max().item())
        else:
            p_start, p_end = 0, -1

        # Left flank
        if p_start > 0:
            feat_left, mask_left, keep_left, width_left = pool_segment(
                feat_b[:p_start],
                mask_b[:, :p_start],
                keep_b[:p_start],
                k,
                bp_per_token_b[:p_start],
            )
        else:
            feat_left  = feat_b.new_empty(0, C)
            mask_left  = mask_b.new_empty(1, 0)
            keep_left  = keep_b.new_empty(0, dtype=torch.bool)
            width_left = mask_b.new_empty(1, 0)

        # Middle (pert span full-res at current resolution)
        feat_mid = feat_b[p_start:p_end+1]
        mask_mid = mask_b[:, p_start:p_end+1]
        keep_mid = keep_b[p_start:p_end+1]
        # Adaptive resolution: use per-token bp values
        width_mid = bp_per_token_b[p_start:p_end+1].unsqueeze(0)  # (1, L_mid)

        # Right flank
        if p_end + 1 < L:
            feat_right, mask_right, keep_right, width_right = pool_segment(
                feat_b[p_end+1:],
                mask_b[:, p_end+1:],
                keep_b[p_end+1:],
                k,
                bp_per_token_b[p_end+1:],
            )
        else:
            feat_right  = feat_b.new_empty(0, C)
            mask_right  = mask_b.new_empty(1, 0)
            keep_right  = keep_b.new_empty(0, dtype=torch.bool)
            width_right = mask_b.new_empty(1, 0)

        feat_b_comp  = torch.cat([feat_left, feat_mid, feat_right], dim=0)
        mask_b_comp  = torch.cat([mask_left, mask_mid, mask_right], dim=1)
        keep_b_comp  = torch.cat([keep_left, keep_mid, keep_right], dim=0)
        width_b_comp = torch.cat([width_left, width_mid, width_right], dim=1)

        feat_chunks.append(feat_b_comp)
        mask_chunks.append(mask_b_comp)
        width_chunks.append(width_b_comp)
        keep_chunks.append(keep_b_comp)

    # ---------- 4. Pad batch ----------
    L_out = max(f.shape[0] for f in feat_chunks)

    feat_comp = torch.stack(
        [F.pad(f, (0, 0, 0, L_out - f.shape[0])) for f in feat_chunks], dim=0
    )  # (B, L_out, C)

    mask_comp = torch.stack(
        [F.pad(m, (0, L_out - m.shape[1])) for m in mask_chunks], dim=0
    )  # (B, 1, L_out)

    width_comp_bp = torch.stack(
        [F.pad(w, (0, L_out - w.shape[1])) for w in width_chunks], dim=0
    )  # (B, 1, L_out)

    keep_regions_mask = torch.stack(
        [F.pad(k.float(), (0, L_out - k.shape[0])) for k in keep_chunks], dim=0
    ).bool()  # (B, L_out)
    keep_regions_mask = keep_regions_mask.unsqueeze(1)          # (B, 1, L_out)

    return feat_comp, mask_comp, width_comp_bp, keep_regions_mask


# def compress_after_transformer(feat, mask_128, ccre_intervals_rel=None, distal_stride_tokens=32):
#     """
#     Compress features after Borzoi transformer, respecting cCRE and perturbation regions.
#     GPU-native implementation: all operations stay on GPU, no CPU-GPU sync.
#     
#     feat:      (B, L, C)  # after transformer at 128bp resolution
#     mask_128:  (B, 1, L)  # mask at 128bp resolution
#     ccre_intervals_rel: list of (start_bp, end_bp) tuples relative to window (base-pair coordinates)
#     distal_stride_tokens: pooling stride in tokens at 128bp resolution
#     Returns: (feat_comp, mask_comp, width_comp_bp, keep_regions_mask) where:
#         - feat_comp: (B, Lmax, C) compressed features (padded to common length)
#         - mask_comp: (B, 1, Lmax) compressed mask
#         - width_comp_bp: (B, 1, Lmax) bp width per compressed token
#         - keep_regions_mask: (B, 1, Lmax) bool - True = keep (cCRE/pert), False = compress
#     """
#     B, L, C = feat.shape
#     device = feat.device
#     bp_per_token = 128.0
#     
#     # Shared cCRE mask across batch (GPU-native)
#     ccre_mask_1d = build_ccre_mask(L, ccre_intervals_rel, bp_per_token=bp_per_token, device=device)  # (L,)
#     
#     def pool_feat_gap(x):
#         """Pool features along sequence dimension: (C, len_gap) -> (C, len_gap//k)"""
#         if x.numel() == 0:
#             return x
#         k = distal_stride_tokens
#         n = (x.shape[-1] // k) * k
#         main = x[:, :n].reshape(C, -1, k).mean(-1) if n > 0 else x.new_empty(C, 0)
#         tail = x[:, n:].mean(-1, keepdim=True) if n < x.shape[-1] else x.new_empty(C, 0)
#         return torch.cat([main, tail], dim=-1)
#     
#     def pool_mask_gap(x):
#         """Pool mask along sequence dimension: (1, len_gap) -> (1, len_gap//k)"""
#         if x.numel() == 0:
#             return x
#         k = distal_stride_tokens
#         n = (x.shape[-1] // k) * k
#         main = x[:, :n].reshape(1, -1, k).amax(-1) if n > 0 else x.new_empty(1, 0)
#         tail = x[:, n:].amax(-1, keepdim=True) if n < x.shape[-1] else x.new_empty(1, 0)
#         return torch.cat([main, tail], dim=-1)
#     
#     def build_widths(n_tokens, k):
#         """Build width tensor in tokens (will be converted to bp later)"""
#         if n_tokens == 0:
#             return torch.empty(1, 0, device=device, dtype=torch.int32)
#         n_full = (n_tokens // k) * k
#         if k > 1:
#             w_main = torch.full((1, n_full // k), k, device=device, dtype=torch.int32)
#         else:
#             w_main = torch.full((1, n_full), 1, device=device, dtype=torch.int32)
#         w_tail = torch.full((1, 1), n_tokens - n_full, device=device, dtype=torch.int32) if n_full < n_tokens else torch.empty(1, 0, device=device, dtype=torch.int32)
#         return torch.cat([w_main, w_tail], dim=-1)
#     
#     feat_chunks, mask_chunks, width_chunks, keep_mask_chunks = [], [], [], []
#     
#     for b in range(B):
#         feat_b = feat[b].transpose(0, 1)        # (C, L)
#         mask_b = mask_128[b]                    # (1, L)
#         pert_mask_b = (mask_b[0] != 0)          # (L,) bool
#         
#         # keep tokens = perturbation OR cCRE
#         keep_b = pert_mask_b | ccre_mask_1d     # (L,) bool
#         
#         # If no keep tokens, fallback to full central region
#         if not bool(keep_b.any()):
#             keep_b = torch.zeros(L, device=device, dtype=torch.bool)
#             center = L // 2
#             span = L  # or some fixed central span if you want
#             l = max(0, center - span // 2)
#             r = min(L, center + span // 2)
#             keep_b[l:r] = True
#         
#         # Convert keep_b into contiguous intervals (all on GPU)
#         keep_int = keep_b.to(torch.int8)                       # (L,)
#         keep_pad = F.pad(keep_int, (1, 1))                     # (L+2,)
#         edges   = keep_pad[1:] - keep_pad[:-1]                 # (L+1,)
#         starts  = torch.nonzero(edges == 1,  as_tuple=False).flatten()  # (n_int,)
#         ends    = torch.nonzero(edges == -1, as_tuple=False).flatten()  # (n_int,)
#         # Each interval is [starts[i], ends[i]) in token indices
#         
#         parts_f, parts_m, parts_w, parts_keep = [], [], [], []
#         prev_end = 0
#         
#         # iterate over intervals; we *do* use Python loop but everything sliced as tensors
#         for i in range(starts.numel()):
#             ks = int(starts[i])
#             ke = int(ends[i])
#             
#             # gap before this keep interval
#             if ks > prev_end:
#                 gap_f = feat_b[:, prev_end:ks]      # (C, gap_len)
#                 gap_m = mask_b[:, prev_end:ks]      # (1, gap_len)
#                 gap_f_p = pool_feat_gap(gap_f)
#                 gap_m_p = pool_mask_gap(gap_m)
#                 gap_w   = build_widths(gap_f.shape[-1], distal_stride_tokens)
#                 gap_keep = torch.zeros(1, gap_f_p.shape[-1], device=device, dtype=torch.bool)
#                 
#                 parts_f.append(gap_f_p)
#                 parts_m.append(gap_m_p)
#                 parts_w.append(gap_w)
#                 parts_keep.append(gap_keep)
#             
#             # keep region full-res
#             keep_f = feat_b[:, ks:ke]              # (C, keep_len)
#             keep_m = mask_b[:, ks:ke]
#             keep_w = torch.full((1, keep_f.shape[-1]), 1, device=device, dtype=torch.int32)  # 1 token per position
#             keep_mask = torch.ones(1, keep_f.shape[-1], device=device, dtype=torch.bool)
#             
#             parts_f.append(keep_f)
#             parts_m.append(keep_m)
#             parts_w.append(keep_w)
#             parts_keep.append(keep_mask)
#             
#             prev_end = ke
#         
#         # gap after last keep interval
#         if prev_end < L:
#             gap_f = feat_b[:, prev_end:]
#             gap_m = mask_b[:, prev_end:]
#             gap_f_p = pool_feat_gap(gap_f)
#             gap_m_p = pool_mask_gap(gap_m)
#             gap_w   = build_widths(gap_f.shape[-1], distal_stride_tokens)
#             gap_keep = torch.zeros(1, gap_f_p.shape[-1], device=device, dtype=torch.bool)
#             
#             parts_f.append(gap_f_p)
#             parts_m.append(gap_m_p)
#             parts_w.append(gap_w)
#             parts_keep.append(gap_keep)
#         
#         # stitch per-sample
#         if parts_f:
#             feat_b_comp  = torch.cat(parts_f, dim=-1)          # (C, L_comp_b)
#             mask_b_comp  = torch.cat(parts_m, dim=-1)          # (1, L_comp_b)
#             width_b_comp = torch.cat(parts_w, dim=-1)           # (1, L_comp_b) int32, converted to float later
#             keep_b_comp  = torch.cat(parts_keep, dim=-1)       # (1, L_comp_b)
#         else:
#             feat_b_comp  = feat_b                              # fallback
#             mask_b_comp  = mask_b
#             width_b_comp = torch.ones(1, L, device=device, dtype=torch.int32)  # 1 token per position
#             keep_b_comp  = torch.zeros(1, L, device=device, dtype=torch.bool)
#         
#         feat_chunks.append(feat_b_comp)
#         mask_chunks.append(mask_b_comp)
#         width_chunks.append(width_b_comp)
#         keep_mask_chunks.append(keep_b_comp)
#     
#     # Pad to common length across batch
#     Lmax = max(f.shape[-1] for f in feat_chunks)
#     feat_comp = torch.stack([F.pad(f, (0, Lmax - f.shape[-1])) for f in feat_chunks], dim=0).transpose(1, 2)  # (B,Lmax,C)
#     mask_comp = torch.stack([F.pad(m, (0, Lmax - m.shape[-1])) for m in mask_chunks], dim=0)                  # (B,1,Lmax)
#     # Convert width from tokens to bp: all widths are in tokens, multiply by bp_per_token
#     width_comp_bp = torch.stack([F.pad(w, (0, Lmax - w.shape[-1])) for w in width_chunks], dim=0).float() * bp_per_token  # (B,1,Lmax)
#     keep_regions_mask = torch.stack([F.pad(k.float(), (0, Lmax - k.shape[-1])) for k in keep_mask_chunks], dim=0).bool()  # (B,1,Lmax)
#     
#     return feat_comp.contiguous(), mask_comp.contiguous(), width_comp_bp.contiguous(), keep_regions_mask.contiguous()



def compress_after_stem(feat, mask_raw, keep_span_bp=512, distal_stride_tokens=8, ccre_intervals_rel=None):
    """
    feat:      (B, C, L1)  # after conv_dna + first pool
    mask_raw:  (B, 1, Lraw)
    keep_span_bp: biological span to keep near center (e.g., 512 bp)
    distal_stride_tokens: pooling stride in tokens at this stem resolution (power of 2 recommended)
    ccre_intervals_rel: list of (start_bp, end_bp) tuples relative to window (base-pair coordinates)
                        If None or empty, falls back to default behavior
    Returns: (feat_comp, mask_comp, width_comp, keep_regions_mask) where keep_regions_mask indicates full-res tokens
    """
    B, C, L1 = feat.shape
    device = feat.device


    
    # Downsample mask to stem length with 'max' semantics (don't lose thin edits)
    # Estimate bp_per_token at stem by relative lengths (works for arbitrary inputs)
    Lraw = mask_raw.shape[-1]
    # avoid division by 0
    bp_per_token_stem = max(1.0, float(Lraw) / float(L1))
    
    
    def build_widths(n_tokens, k):
        if n_tokens == 0: 
            return torch.empty(1,0, device=device, dtype=torch.int32)
        n_full = (n_tokens // k) * k
        w_main = torch.full((1, n_full // k), k, device=device, dtype=torch.int32) if k > 1 else torch.full((1, n_full), 1, device=device, dtype=torch.int32)
        w_tail = torch.full((1, 1), n_tokens - n_full, device=device, dtype=torch.int32) if n_full < n_tokens else torch.empty(1,0, device=device, dtype=torch.int32)
        return torch.cat([w_main, w_tail], dim=-1)  # (1,*)

    def downsample_mask_max(mask, L_target):
        if L_target == mask.shape[-1]: return mask
        k = mask.shape[-1] // L_target
        if k >= 2 and mask.shape[-1] % L_target == 0:
            return mask.unfold(-1, k, k).amax(-1)
        return (F.interpolate(mask.float(), size=L_target, mode='nearest') > 0.5).to(mask.dtype)

    mask_stem = downsample_mask_max(mask_raw, L1)  # (B,1,L1)

    # Center at stem resolution
    with torch.no_grad():
        m = (mask_stem != 0).float()
        idx = torch.arange(L1, device=device).view(1,1,L1)
        centers = ((m*idx).sum(-1) / m.sum(-1).clamp_min(1e-6)).round().squeeze(1).long()
        centers.clamp_(0, L1-1)

    # Convert keep span from bp to tokens at stem
    keep_tokens = max(1, int(round(keep_span_bp / bp_per_token_stem)))
    half = keep_tokens // 2

    # pool distal features (avg) and mask (max) by the same stride
    def pool_feat(x):
        if x.numel()==0: return x
        k = distal_stride_tokens
        n = (x.shape[-1] // k) * k
        main = x[:, :n].contiguous().reshape(C, -1, k).mean(-1)               # (C, n/k)
        tail = x[:, n:].mean(-1, keepdim=True) if n < x.shape[-1] else x.new_empty(C,0)
        return torch.cat([main, tail], dim=-1)                    # (C, *)

    def pool_mask(x):
        if x.numel()==0: return x
        k = distal_stride_tokens
        n = (x.shape[-1] // k) * k
        main = x[:, :n].reshape(1, -1, k).amax(-1)               # (1, n/k)
        tail = x[:, n:].amax(-1, keepdim=True) if n < x.shape[-1] else x.new_empty(1,0, device=x.device)
        return torch.cat([main, tail], dim=-1)                    # (1, *)

    feat_chunks, mask_chunks, width_chunks, keep_mask_chunks = [], [], [], []
    for b in range(B):
        c_tensor = centers[b]
        c = int(c_tensor.item())
        l = max(0, c - half)
        r = min(L1, c + half)

        # Identify perturbation region tokens (from mask) - keep on GPU
        pert_mask = (mask_stem[b, 0, :] != 0)  # (L1,) bool tensor on GPU
        pert_tokens = torch.where(pert_mask)[0].cpu().numpy().tolist()  # Convert only when needed
        
        # Identify cCRE region tokens (convert bp to tokens)
        ccre_tokens_set = set()
        if ccre_intervals_rel is not None and len(ccre_intervals_rel) > 0:
            for ccre_start_bp, ccre_end_bp in ccre_intervals_rel:
                # Convert bp to stem token coordinates
                ccre_start_token = max(0, int(round(ccre_start_bp / bp_per_token_stem)))
                ccre_end_token = min(L1 - 1, int(round(ccre_end_bp / bp_per_token_stem)))
                if ccre_end_token >= ccre_start_token:
                    ccre_tokens_set.update(range(ccre_start_token, ccre_end_token + 1))
        
        # Merge perturbation + cCRE regions
        keep_tokens_set = set(pert_tokens) | ccre_tokens_set
        
        # If no cCREs and no perturbation, fall back to default (central region)
        if len(keep_tokens_set) == 0:
            keep_tokens_set = set(range(l, r))
        
        # Convert to sorted list of intervals
        if keep_tokens_set:
            keep_tokens_sorted = sorted(keep_tokens_set)
            # Merge adjacent tokens into intervals
            keep_intervals = []
            start = keep_tokens_sorted[0]
            for i in range(1, len(keep_tokens_sorted)):
                if keep_tokens_sorted[i] != keep_tokens_sorted[i-1] + 1:
                    keep_intervals.append((start, keep_tokens_sorted[i-1] + 1))
                    start = keep_tokens_sorted[i]
            keep_intervals.append((start, keep_tokens_sorted[-1] + 1))
        else:
            # Fallback: use central region
            keep_intervals = [(l, r)]

        # Split sequence into chunks: keep regions (full res) vs. gaps (compressed)
        feat_parts, mask_parts, width_parts, keep_mask_parts = [], [], [], []
        prev_end = 0
        
        for keep_start, keep_end in keep_intervals:
            # Gap before this keep region (compress)
            if keep_start > prev_end:
                gap_f = feat[b, :, prev_end:keep_start]
                gap_m = mask_stem[b, :, prev_end:keep_start]
                gap_f_p = pool_feat(gap_f) if gap_f.numel() > 0 else gap_f
                gap_m_p = pool_mask(gap_m) if gap_m.numel() > 0 else gap_m
                gap_w = build_widths(gap_f.shape[-1], distal_stride_tokens)
                gap_keep = torch.zeros(1, gap_f_p.shape[-1], device=device, dtype=torch.bool)
                
                feat_parts.append(gap_f_p)
                mask_parts.append(gap_m_p)
                width_parts.append(gap_w)
                keep_mask_parts.append(gap_keep)
            
            # Keep region (full resolution)
            keep_f = feat[b, :, keep_start:keep_end]
            keep_m = mask_stem[b, :, keep_start:keep_end]
            keep_w = torch.ones(1, keep_f.shape[-1], device=device, dtype=torch.int32)
            keep_mask = torch.ones(1, keep_f.shape[-1], device=device, dtype=torch.bool)
            
            feat_parts.append(keep_f)
            mask_parts.append(keep_m)
            width_parts.append(keep_w)
            keep_mask_parts.append(keep_mask)
            
            prev_end = keep_end
        
        # Gap after last keep region (compress)
        if prev_end < L1:
            gap_f = feat[b, :, prev_end:]
            gap_m = mask_stem[b, :, prev_end:]
            gap_f_p = pool_feat(gap_f) if gap_f.numel() > 0 else gap_f
            gap_m_p = pool_mask(gap_m) if gap_m.numel() > 0 else gap_m
            gap_w = build_widths(gap_f.shape[-1], distal_stride_tokens)
            gap_keep = torch.zeros(1, gap_f_p.shape[-1], device=device, dtype=torch.bool)
            
            feat_parts.append(gap_f_p)
            mask_parts.append(gap_m_p)
            width_parts.append(gap_w)
            keep_mask_parts.append(gap_keep)
        
        # Stitch all parts
        feat_b = torch.cat(feat_parts, dim=-1) if feat_parts else feat[b]
        mask_b = torch.cat(mask_parts, dim=-1) if mask_parts else mask_stem[b]
        width_b = torch.cat(width_parts, dim=-1) if width_parts else torch.ones(1, L1, device=device, dtype=torch.int32)
        keep_mask_b = torch.cat(keep_mask_parts, dim=-1) if keep_mask_parts else torch.zeros(1, L1, device=device, dtype=torch.bool)
        
        feat_chunks.append(feat_b)
        mask_chunks.append(mask_b)
        width_chunks.append(width_b)
        keep_mask_chunks.append(keep_mask_b)

    # pad to a common length
    Lmax = max(t.shape[-1] for t in feat_chunks)
    feat_comp = torch.stack([F.pad(t, (0, Lmax - t.shape[-1])) for t in feat_chunks], 0)  # (B,C,Lmax)
    mask_comp = torch.stack([F.pad(t, (0, Lmax - t.shape[-1])) for t in mask_chunks], 0)  # (B,1,Lmax)
    width_comp = torch.stack([F.pad(w.float(), (0, Lmax - w.shape[-1])) for w in width_chunks], 0)  # (B,1,Lmax)
    keep_regions_mask = torch.stack([F.pad(k.float(), (0, Lmax - k.shape[-1])) for k in keep_mask_chunks], 0)  # (B,1,Lmax)
    
    return feat_comp.contiguous(), mask_comp.contiguous(), width_comp.contiguous(), keep_regions_mask.contiguous()

def _fwd_bwd_dt(m):  # m: (B,L) bool
    B, L = m.shape
    inf = L + 1
    # forward pass
    d = torch.full((B, L), inf, device=m.device, dtype=torch.int32)
    d[:, 0] = torch.where(m[:, 0], 0, inf)
    for i in range(1, L):
        d[:, i] = torch.where(m[:, i], 0, torch.minimum(d[:, i-1] + 1, torch.tensor(inf, device=m.device, dtype=d.dtype)))
    # backward pass
    for i in range(L-2, -1, -1):
        d[:, i] = torch.minimum(d[:, i], d[:, i+1] + 1)
    return d

def signed_nearest_distance(mask_128):
    """
    Vectorized signed distance to the nearest nonzero mask position per token.
    Tie-breaking matches original behavior (tie-left => negative).
    Returns:
        d_signed: (B, L) float32
        has_pert: (B,) bool
    """
    B, _, L = mask_128.shape
    m = (mask_128 != 0).squeeze(1)  # (B,L)
    has_pert = m.any(dim=1)         # (B,)

    # Early exit: entire batch empty
    if not has_pert.any():
        d_signed = torch.full((B, L), float(L), device=mask_128.device, dtype=torch.float32)
        return d_signed, has_pert

    idx = torch.arange(L, device=mask_128.device, dtype=torch.int32).expand(B, -1)  # (B,L)

    # Left distance (last True at or before i) via prefix max of indices
    neg_sentinel = torch.full((B, L), -2**30, device=mask_128.device, dtype=torch.int32)
    left_idx_src = torch.where(m, idx, neg_sentinel)
    last_true_idx, _ = left_idx_src.cummax(dim=1)
    left_dist = idx - last_true_idx
    left_dist = torch.where(last_true_idx < 0, torch.full_like(left_dist, L + 1), left_dist)

    # Right distance (next True at or after i) via suffix min of indices
    pos_sentinel = torch.full((B, L), 2**30, device=mask_128.device, dtype=torch.int32)
    right_idx_src = torch.where(m, idx, pos_sentinel)
    rev_min, _ = torch.flip(right_idx_src, dims=[1]).cummin(dim=1)
    next_true_idx = torch.flip(rev_min, dims=[1])
    right_dist = next_true_idx - idx
    right_dist = torch.where(next_true_idx >= L, torch.full_like(right_dist, L + 1), right_dist)

    # Tie-left signed rule
    d_signed_int = torch.where(left_dist <= right_dist, -left_dist, right_dist)
    d_signed = d_signed_int.to(torch.float32)

    # Rows with no perturbation -> large positive constant
    d_signed = torch.where(has_pert.view(-1, 1), d_signed, torch.full_like(d_signed, float(L)))
    return d_signed, has_pert


class DistanceKernel(nn.Module):
    """Learnable distance-dependent decay kernel for regulatory effects."""
    def __init__(self):
        super().__init__()
        # Sigmoid -> [0,1] for mixing ratio
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        # Softplus -> >0, in units of 128bp tokens  
        self.ell_raw = nn.Parameter(torch.tensor(32.0))  # ~4kb default
        # Softplus -> >0, power law decay
        self.beta_raw = nn.Parameter(torch.tensor(0.8))
    
    @staticmethod
    def _pos(x, eps=1e-6):
        return F.softplus(x) + eps
    
    def forward(self, d_tokens):
        """
        Args:
            d_tokens: (B,1,L) signed or absolute distances in tokens (128bp each)
        Returns:
            (B,1,L) weights in [0,1]
        """
        a = torch.sigmoid(self.alpha_raw)
        ell = self._pos(self.ell_raw)
        bet = self._pos(self.beta_raw)
        
        x = d_tokens.abs()
        # Mixture of exponential and power law decay
        exp_decay = torch.exp(-x/ell)
        pow_decay = (1 + x/ell).pow(-bet)
        
        return a * exp_decay + (1-a) * pow_decay
    
    
class DistanceBias(nn.Module):
    def __init__(self, init_ell=32.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))     # exp weight
        self.beta  = nn.Parameter(torch.tensor(1.0))     # power weight
        self.ell   = nn.Parameter(torch.tensor(init_ell))# scale in tokens
        # self.gamma = nn.Parameter(torch.tensor(0.0))     # signed tilt

    def forward(self, d_signed, dtype):
        # d_signed: (B,L) signed distances in tokens
        x   = d_signed.abs().to(dtype)
        ell = F.softplus(self.ell) + 1e-6
        a   = F.softplus(self.alpha)
        b   = F.softplus(self.beta)
        # negative bias (more negative further away)
        # shape → (B,1,1,L) for broadcasting over heads & queries
        neg = a * (x / ell) + b * torch.log1p(x / ell)
        # tilt = self.gamma.to(dtype) * torch.sign(d_signed).to(dtype).abs() * 0.0  # keep available if you want signed bias
        return -(neg)[..., None, None, :]  # (B,1,1,L)

    
    
class SinglePerturbEmbedding(nn.Module):
    """
    Single-perturbation embedding with:
      - perturbation-centered anchor query
      - optional global distance bias (DistanceBias over token/bp distance)
      - optional cell-specific cCRE logit bias (CellSpecificCCREBias)
      - keep_regions_mask + width_comp_bp aware

    Ablations:
      - use_distance_bias (init-level)
      - use_cell_ccre_bias (init-level)
      - ablate_distance_bias (forward-level)
      - ablate_ccre_bias (forward-level)
      - cell_bias_mode: 'normal' | 'noise' | 'shuffle'
    """
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 8,
        out_dim: int = 512,
        max_L: int = 4096,
        cell_dim: int = 128,
        attn_dim: int = 512,
        ccre_max_bias: float = 3.0,
        ccre_normalize_per_sample: bool = True,
        use_distance_bias: bool = True,
        use_cell_ccre_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert attn_dim % num_heads == 0, f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.heads = num_heads
        self.attn_dim = attn_dim
        self.dh = dim // num_heads
        self.attn_dh = attn_dim // num_heads  # Head dimension for attention projections

        # config switches
        self.use_distance_bias = bool(use_distance_bias)
        self.use_cell_ccre_bias_cfg = bool(use_cell_ccre_bias)
        self.cell_dim = cell_dim

        # Q/K/V for single-query attention
        self.Wq = nn.Linear(dim, attn_dim, bias=False)
        self.Wk = nn.Linear(dim, attn_dim, bias=False)
        self.Wv = nn.Linear(dim, attn_dim, bias=False)
        self.attn_out = nn.Linear(attn_dim, dim, bias=False)

        # Global distance kernel (over token-ish distances)
        self.distance_bias = DistanceBias(init_ell=32.0)

        # Local anchor preproc
        self.anchor_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//16),
            nn.GELU(),
            nn.Linear(dim//16, dim),
        )

        # Final projector to compact embedding
        self.proj = nn.Sequential(
            nn.LayerNorm(dim * 2 + 2),
            nn.Linear(dim * 2 + 2, out_dim//16),
            nn.GELU(),
            nn.Linear(out_dim//16, out_dim),
        )
        nn.init.xavier_uniform_(self.proj[1].weight, gain=0.1)
        nn.init.zeros_(self.proj[1].bias)

        # Cell-specific cCRE bias
        self.use_cell_ccre_bias = (
            self.use_cell_ccre_bias_cfg and (cell_dim is not None) and (cell_dim > 0)
        )
        if self.use_cell_ccre_bias:
            self.cell_ccre_bias = CellSpecificCCREBias(
                dim=dim,
                cell_dim=cell_dim,
                num_distance_bins=3,
                max_bias=ccre_max_bias,
                normalize_per_sample=ccre_normalize_per_sample,
            )

    # -------- helpers --------
    def _shape_heads(self, x):  # (B,C) or (B,L,C) -> (B,H,1,Dh) or (B,H,L,Dh)
        # Use attn_dh since Q/K/V project to attn_dim
        if x.dim() == 2:
            B = x.size(0)
            return x.view(B, self.heads, 1, self.attn_dh)
        else:
            B, L, _ = x.shape
            return x.view(B, L, self.heads, self.attn_dh).transpose(1, 2).contiguous()

    def _compute_bp_distances(self, m, d_abs, width_comp_bp, keep_regions_mask):
        """
        Compute bp distances from perturbation anchor.

        Args:
            m:               (B,1,L) mask at current resolution
            d_abs:           (B,L) token distance (abs, in tokens)
            width_comp_bp:   (B,1,L) or (B,L), bp width per token (may include padding)
            keep_regions_mask:(B,1,L) or (B,L) bool, True = real token, False = padding

        Returns:
            bp_distances:    (B,L) >=0, bp distance from perturbation anchor
            p_anchor_bp:     (B,1) anchor position in bp (for debug)
        """
        B, L = d_abs.shape

        # standardize width_comp_bp to (B,L)
        if width_comp_bp.dim() == 3:
            w_bp = width_comp_bp.squeeze(1)   # (B,L)
        else:
            w_bp = width_comp_bp              # (B,L)

        if w_bp.shape[-1] != L:
            w_bp = F.interpolate(w_bp.unsqueeze(1), size=L, mode='nearest').squeeze(1)

        # zero padded tokens
        if keep_regions_mask is not None:
            if keep_regions_mask.dim() == 3:
                keep_mask = keep_regions_mask.squeeze(1)  # (B,L)
            else:
                keep_mask = keep_regions_mask             # (B,L)
            if keep_mask.shape[-1] != L:
                keep_mask = F.interpolate(
                    keep_mask.float().unsqueeze(1), size=L, mode='nearest'
                ).squeeze(1).bool()
            else:
                keep_mask = keep_mask.bool()
            w_bp = w_bp * keep_mask.float()

        # token centers in bp
        p_mid = torch.cumsum(w_bp, dim=-1) - 0.5 * w_bp  # (B,L)

        # perturbation anchor from mask + exponential halo
        m_bp = m.squeeze(1).float()                      # (B,L)
        tau = 8.0
        g_bp = torch.exp(-d_abs / tau)                   # (B,L)
        w_loc_bp = (g_bp * m_bp).clamp_min(1e-12)        # (B,L)
        w_loc_sum = w_loc_bp.sum(-1, keepdim=True).clamp_min(1e-12)  # (B,1)

        has_pert = (m_bp.sum(-1, keepdim=True) > 1e-12)  # (B,1)
        # fallback to geometric center if no perturbation in this sample
        p_anchor_bp = torch.where(
            has_pert,
            (w_loc_bp * p_mid).sum(-1, keepdim=True) / w_loc_sum,
            p_mid[:, L // 2:L // 2 + 1],
        )  # (B,1)

        bp_distances = (p_mid - p_anchor_bp).abs()       # (B,L)
        return bp_distances, p_anchor_bp

    def _ablate_cell(self, z_ctrl_flat: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply ablation on cell embedding.

        mode:
          - 'normal'  : use real z_ctrl
          - 'noise'   : N(0, I) with same shape
          - 'shuffle' : random permutation across batch
        """
        if mode == "normal":
            return z_ctrl_flat
        B, D = z_ctrl_flat.shape
        device = z_ctrl_flat.device
        dtype = z_ctrl_flat.dtype

        if mode == "noise":
            return torch.randn(B, D, device=device, dtype=dtype)
        elif mode == "shuffle":
            perm = torch.randperm(B, device=device)
            return z_ctrl_flat[perm]
        else:
            raise ValueError(f"Unknown cell_bias_mode='{mode}'")

    # -------- main forward --------
    def forward(
        self,
        X,                 # (B,L,C)
        mask_raw,          # (B,1,L_raw) or (B,1,L)
        interaction: bool = True,
        keep_regions_mask=None,  # (B,1,L) or (B,L), True = cCRE/pert/real
        width_comp_bp=None,      # (B,1,L) or (B,L) bp width per token
        return_p_anchor: bool = False,
        z_ctrl=None,             # (B,cell_dim) or (B,1,cell_dim)
        # ablations:
        ablate_distance_bias: bool = False,
        ablate_ccre_bias: bool = False,
        cell_bias_mode: str = "normal",   # 'normal' | 'noise' | 'shuffle'
    ):
        """
        Returns:
            if interaction == False:
                q: (B, dim) perturbation-anchored local summary
            else:
                z: (B, out_dim) single perturbation embedding
                [p_anchor_bp]: (B,1) if return_p_anchor=True
        """
        B, L, C = X.shape
        dtype = X.dtype
        device = X.device

        # 1) interpolated mask & signed/token distances
        if mask_raw is None:
            m = torch.zeros(B, 1, L, device=device, dtype=dtype)
        else:
            m = F.interpolate(mask_raw.float(), size=L, mode='nearest')  # (B,1,L)

        d_signed, _ = signed_nearest_distance(m)              # (B,L)
        d_abs = d_signed.abs().to(dtype)                      # (B,L)

        # 2) local anchor query q at perturb region (exp halo)
        tau = 8.0
        g = torch.exp(-(d_abs / tau))                         # (B,L)
        m_squeezed = m.squeeze(1).to(dtype)                   # (B,L)
        w_loc = (g * m_squeezed).clamp_min(1e-12)             # (B,L)
        den = w_loc.sum(-1, keepdim=True).clamp_min(1e-12)    # (B,1)
        q = (w_loc[..., None] * X).sum(dim=1) / den           # (B,C)

        if not interaction:
            if return_p_anchor:
                # approximate anchor in bp with a dummy constant (if you care, compute bp distances)
                p_anchor_bp = d_abs.new_zeros(B, 1)
                return q, p_anchor_bp
            return q

        # 4) anchor MLP for query
        q = self.anchor_mlp(q)                                # (B,C)

        # 5) K/V from base features (no multiplicative cell gating)
        qh = self._shape_heads(self.Wq(q))                    # (B,H,1,Dh)
        kh = self._shape_heads(self.Wk(X))                    # (B,H,L,Dh)
        vh = self._shape_heads(self.Wv(X))                    # (B,H,L,Dh)

        # 6) global distance bias (DistanceBias)
        if width_comp_bp is not None:
            bp_distances, p_anchor_bp = self._compute_bp_distances(
                m, d_abs, width_comp_bp, keep_regions_mask
            )                                                 # (B,L), (B,1)
            d_tokens_for_bias = (bp_distances / 128.0)
        else:
            bp_distances = d_abs * 128.0                      # approx, for cCRE bias
            p_anchor_bp = None
            d_tokens_for_bias = d_signed.to(dtype)            # token distances

        if self.use_distance_bias and not ablate_distance_bias:
            base_dbias = self.distance_bias(d_tokens_for_bias, dtype=dtype)  # (B,1,1,L)
        else:
            base_dbias = torch.zeros(B, 1, 1, L, device=device, dtype=dtype)

        dbias = base_dbias                                    # (B,1,1,L)

        # 7) cell-specific cCRE bias (additive on top of distance bias)
        if (
            self.use_cell_ccre_bias
            and (not ablate_ccre_bias)
            and (z_ctrl is not None)
            and (keep_regions_mask is not None)
        ):
            # Ensure z_ctrl is on the same device as X
            if z_ctrl.device != device:
                z_ctrl = z_ctrl.to(device, non_blocking=True)
            # normalize z_ctrl shape: (B,cell_dim)
            if z_ctrl.dim() == 3:
                z_ctrl_flat = z_ctrl.squeeze(1)
            else:
                z_ctrl_flat = z_ctrl

            # apply cell ablation mode
            z_ctrl_flat = self._ablate_cell(z_ctrl_flat, cell_bias_mode)

            # build cCRE mask: keep_regions & not perturbation
            if keep_regions_mask.dim() == 3:
                keep_mask = keep_regions_mask.squeeze(1)      # (B,L)
            else:
                keep_mask = keep_regions_mask                 # (B,L)
            if keep_mask.shape[-1] != L:
                keep_mask = F.interpolate(
                    keep_mask.float().unsqueeze(1), size=L, mode='nearest'
                ).squeeze(1).bool()
            else:
                keep_mask = keep_mask.bool()

            pert_mask = (m.squeeze(1) != 0)                   # (B,L)
            ccre_mask = keep_mask & (~pert_mask)              # (B,L)

            # compute cell-specific bias in [-max_bias, max_bias] on cCREs
            ccre_bias_flat = self.cell_ccre_bias(
                X=X,
                z_cell=z_ctrl_flat,
                distances_bp=bp_distances.detach(),           # (B,L)
                ccre_mask=ccre_mask,
            )                                                 # (B,L)

            # expand to (B,1,1,L) and add to dbias
            ccre_dbias = ccre_bias_flat.unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
            dbias = dbias + ccre_dbias

        # 8) raw attention logits
        logits = (qh @ kh.transpose(-2, -1)) / (self.attn_dh ** 0.5)    # (B,H,1,L)

        # 9) mask out non-keep / padding regions if keep_regions_mask given
        if keep_regions_mask is not None:
            if keep_regions_mask.dim() == 3:
                keep_mask = keep_regions_mask.squeeze(1)           # (B,L)
            else:
                keep_mask = keep_regions_mask                      # (B,L)
            if keep_mask.shape[-1] != L:
                keep_mask = F.interpolate(
                    keep_mask.float().unsqueeze(1), size=L, mode='nearest'
                ).squeeze(1).bool()
            else:
                keep_mask = keep_mask.bool()

            pad_mask = ~keep_mask                                  # True = ignore
            neg_large = torch.tensor(-1e9, dtype=torch.float32, device=logits.device)
            neg_large = neg_large.to(logits.dtype)
            logits = logits.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), neg_large)

        # 10) attention & aggregation
        attn = torch.softmax(logits + dbias, dim=-1)               # (B,H,1,L)
        h_attn = (attn @ vh).squeeze(2).transpose(1, 2).reshape(B, self.attn_dim)                                 # (B,H,Dh)
        h = self.attn_out(h_attn)                        # (B,C)

        # 11) distance moments of attended distribution
        a = attn.mean(dim=1).squeeze(1)                            # (B,L)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        mu1 = (a * d_abs).sum(-1, keepdim=True)                    # (B,1)
        mu2 = (a * (d_abs ** 2)).sum(-1, keepdim=True)             # (B,1)

        # 12) fuse & project
        z = self.proj(torch.cat([q, h, mu1, mu2], dim=-1))         # (B,out_dim)

        if return_p_anchor:
            return z, p_anchor_bp
        return z




class CrossAttentionSDPA(nn.Module):
    """
    Cross-attn with optional RoPE on Q/K. Compatible with FlashAttention SDPA.
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.rope_base = rope_base

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _reshape(self, x):  # (B, L, C) -> (B, H, L, Dh)
        B, L, C = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        x_q: torch.Tensor,              # (B,Lq,C)
        x_kv: torch.Tensor,             # (B,Lk,C)
        pos_q: torch.Tensor | None = None,   # (B,Lq) or (1,Lq) integer positions
        pos_k: torch.Tensor | None = None,   # (B,Lk) or (1,Lk)
        attn_bias: torch.Tensor | None = None,   # (B,H,Lq,Lk) additive bias (optional)
        pad_mask: torch.Tensor | None = None,    # (B,Lk) True=ignore (optional)
        pad_mask_query: torch.Tensor | None = None,  # queries
        is_causal: bool = False
    ) -> torch.Tensor:

        B, Lq, _ = x_q.shape
        B2, Lk, _ = x_kv.shape
        assert B == B2

        q = self._reshape(self.q_proj(x_q))  # (B,H,Lq,Dh)
        k = self._reshape(self.k_proj(x_kv)) # (B,H,Lk,Dh)
        v = self._reshape(self.v_proj(x_kv)) # (B,H,Lk,Dh)


        # === build additive mask: combine bias + padding ===
        additive = None
        if attn_bias is not None:
            additive = attn_bias.to(dtype=q.dtype, device=q.device)  # (B,H,Lq,Lk)
        if pad_mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            pad_k = torch.where(pad_mask[:,None,None,:], neg_inf, 0.).to(q.dtype)
            additive = pad_k if additive is None else additive + pad_k

        # query padding  (mask full row)
        if pad_mask_query is not None:
            neg_inf = torch.finfo(q.dtype).min
            pad_q = torch.where(pad_mask_query[:,None,:,None], neg_inf, 0.).to(q.dtype)
            additive = pad_q if additive is None else additive + pad_q

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=is_causal
            )  # (B,H,Lq,Dh)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.dim)  # (B,Lq,C)
        return self.out_proj(out)



class shallowEncoder(nn.Module):
    def __init__(self, input_dim=1536, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, out_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    
    
    
class BorzoiPerturbAdapter(nn.Module):
    """
    Extract Borzoi features and produce a single perturbation embedding per sample.
    """
    def __init__(self, model_name='johahi/flashzoi-replicate-0',
                 local_window=16,  # still used by compress_after_stem
                 out_dim=1536,     # summarized embedding size
                 max_L=4096,       # max tokens at 128bp for pos-embed inside SPE
                 use_pos_embed=True,
                 jitter_target_length=4000,  # target length for jittered embeddings
                 pre_transformer_compression: bool = False,
                 compress_stride_after_transformer: int = 1,
                 stem_keep_span_bp: int = 512,
                 stem_distal_stride_tokens: int = 1,
                 optimize_backbone_bf16: bool = False,
                 enable_tf32: bool = True,
                 ablate_distance_bias: bool = False,
                 ablate_ccre_bias: bool = False,
                 cell_bias_mode: str = "normal",   # 'normal' | 'noise' 
                 cell_dim=128,  # dimension of z_ctrl (control cell embedding)
                 ccre_dropout_keep_closest_n: int = 2,
                 ccre_dropout_base_prob: float = 0.2,
                 load_backbone: bool = True,
                 base_bp_per_token: float = 128.0,
                 cache_pool_k: int = 1,
                 ):
        super().__init__()

        # Load + freeze Borzoi
        if load_backbone:
            self.borzoi = Borzoi.from_pretrained(model_name)
            self.borzoi.eval()
            for p in self.borzoi.parameters():
                p.requires_grad_(False)
        else:
            self.borzoi = None

        # Store ablation parameters as instance variables (forward-level, not init-level)
        self.ablate_distance_bias = bool(ablate_distance_bias)
        self.ablate_ccre_bias = bool(ablate_ccre_bias)
        self.cell_bias_mode = str(cell_bias_mode)
        
        # Single-vector perturbation summarizer (create single instance, pass ablation params in forward)
        self.perturbation = SinglePerturbEmbedding(
            dim=1536,
            num_heads=8,
            out_dim=out_dim,
            max_L=max_L,
            cell_dim=cell_dim,
        )

        # Optional global gain if you'll regress from z later
        self.delta_gain = nn.Parameter(torch.tensor(1.0))

        self.local_window = local_window
        self.jitter_target_length = jitter_target_length
        self.pre_transformer_compression = pre_transformer_compression
        self.compress_stride_after_transformer = int(compress_stride_after_transformer)
        self.stem_keep_span_bp = int(stem_keep_span_bp)
        self.stem_distal_stride_tokens = int(stem_distal_stride_tokens)
        self.optimize_backbone_bf16 = bool(optimize_backbone_bf16)
        self.enable_tf32 = bool(enable_tf32)
        
        self.base_bp_per_token = float(base_bp_per_token)
        self.cache_pool_k = int(cache_pool_k)
        if self.enable_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        self._cache_hits = 0
        self._cache_misses = 0
        
        # self.post_z_norm = nn.LayerNorm(out_dim)
        if ccre_dropout_base_prob > 0.0:
            self.ccre_dropout = StructuredCCREDropout(
                keep_closest_n=ccre_dropout_keep_closest_n,
                base_drop_prob=ccre_dropout_base_prob,
            )
        else:
            self.ccre_dropout = None

    def forward(self, x, return_components=False, verbose=False, debug=False, cache=None, cache_key=None, cache_keys=None, ccre_intervals_rel=None, jitter_window=0, z_ctrl=None):
        """
        Args:
            x: (B, 5, L) with DNA one-hot (4ch) + mask (1ch). Can be dummy for cache hits.
            ccre_intervals_rel: list of (start_bp, end_bp) tuples (shared) OR list of lists (per-sample)
            cache: LRUCache or dict for caching full-resolution Borzoi embeddings
            cache_key: (deprecated) single key for cache - use cache_keys instead
            cache_keys: List[str] of length B (one per sample, None if no cache)
            jitter_window: maximum jitter offset in tokens (e.g., 48 for ±48 tokens)
            z_ctrl: (B, cell_dim) or (B, 1, cell_dim) - control cell embedding (optional)
        Returns:
            z: (B, 1, out_dim) single perturbation embedding per sample.
        """
        dna = x[:, :4, :]
        mask = x[:, 4:, :]
        B, L = x.shape[0], x.shape[2]
        device = dna.device
        bp_per_token_effective = self.base_bp_per_token * self.cache_pool_k
        # Backward compatibility: convert single cache_key to list format
        if cache_keys is None:
            if cache_key is not None:
                cache_keys = [cache_key] * B
            else:
                cache_keys = [None] * B
        
        # Backward compatibility: convert single ccre_intervals_rel to per-sample format
        if ccre_intervals_rel is not None and len(ccre_intervals_rel) > 0:
            # Determine if it's already per-sample robustly (some samples may be None/empty)
            is_per_sample_format = (
                isinstance(ccre_intervals_rel, (list, tuple))
                and len(ccre_intervals_rel) == B
                and any(
                    (samp is not None)
                    and isinstance(samp, (list, tuple))
                    and (len(samp) == 0 or (isinstance(samp[0], (list, tuple)) and len(samp[0]) == 2))
                    for samp in ccre_intervals_rel
                )
            )
            if not is_per_sample_format:
                # Single shared list - convert to per-sample format
                ccre_intervals_rel = [ccre_intervals_rel] * B
        else:
            ccre_intervals_rel = [None] * B
        if verbose:
            timings = {}
            def _tic():
                # if device.type == 'cuda': torch.cuda.synchronize()
                return time.perf_counter()
            def _toc(t0, key):
                # if device.type == 'cuda': torch.cuda.synchronize()
                timings[key] = timings.get(key, 0.0) + (time.perf_counter() - t0)
            t_all = _tic()
            # Enable inner SPE timing
            try:
                self.perturbation._verbose = True
            except Exception:
                pass
        
        if verbose:
            # Defer expensive operations until after GPU work
            nz = int(torch.count_nonzero(mask).item())
            delta_gain_val = float(self.delta_gain.item())
            ccre_count = len(ccre_intervals_rel[0]) if (ccre_intervals_rel and ccre_intervals_rel[0]) else 0
            print(f'mask nonzero bins (raw res): {nz}')
            print(f'self.delta_gain: {delta_gain_val}')
            if ccre_count > 0:
                print(f'number of cCRE intervals (sample 0): {ccre_count}')

        # ===== Batched embedding extraction (cache hits + Borzoi passes) =====
        feat_full_per_sample: list[torch.Tensor | None] = [None] * B
        mask_full_per_sample: list[torch.Tensor | None] = [None] * B
        ccre_full_per_sample: list = [None] * B
        miss_indices: list[int] = []
        cache_hit_logged = False
        cache_miss_logged = False

        # Batch cache lookups: pre-check all keys to identify hits/misses
        hit_mask = [False] * B
        cached_data_list = [None] * B
        if cache is not None:
            for i in range(B):
                key_i = cache_keys[i]
                if key_i is not None:
                    if hasattr(cache, "get"):
                        cached_data_list[i] = cache.get(key_i)
                    elif isinstance(cache, dict):
                        cached_data_list[i] = cache.get(key_i, None)
                    hit_mask[i] = cached_data_list[i] is not None
                else:
                    hit_mask[i] = False

        # Process cache hits: keep on GPU instead of CPU
        # Separate hits and misses upfront for better batching
        hit_indices = [i for i in range(B) if hit_mask[i]]
        miss_indices = [i for i in range(B) if not hit_mask[i]]

        if hit_indices:
            # Process all cache hits
            for i in hit_indices:
                cached_data = cached_data_list[i]
                if isinstance(cached_data, tuple):
                    feat_full, mask_full, ccre_cached = cached_data
                else:
                    feat_full = cached_data
                    mask_full = mask[i:i+1]
                    ccre_cached = ccre_intervals_rel[i]

                # Convert to tensor if needed (but don't force device yet)
                if not isinstance(feat_full, torch.Tensor):
                    feat_full = torch.as_tensor(feat_full)
                if not isinstance(mask_full, torch.Tensor):
                    mask_full = torch.as_tensor(mask_full)

                # Handle shape normalization first
                if feat_full.dim() == 3 and feat_full.shape[0] == 1:
                    feat_full = feat_full.squeeze(0)
                if mask_full.dim() == 3 and mask_full.shape[0] == 1:
                    mask_store = mask_full.squeeze(0)
                elif mask_full.dim() == 2:
                    mask_store = mask_full
                else:
                    raise ValueError(f"Unexpected mask shape {mask_full.shape} for sample {i}")

                # Only convert dtype if needed
                if feat_full.dtype != torch.float32:
                    feat_full = feat_full.to(torch.float32)
                if mask_store.dtype != torch.float32:
                    mask_store = mask_store.to(torch.float32)

                # Apply pooling if cache_pool_k > 1
                # Pool features and masks with the same factor k to maintain shape consistency
                if self.cache_pool_k > 1:
                    k = self.cache_pool_k
                    L_orig = feat_full.shape[0]
                    C = feat_full.shape[1]
                    
                    # Pool features: average pooling along sequence dimension
                    # feat_full: (L, C) -> (L//k, C)
                    n_full = (L_orig // k) * k
                    if n_full > 0:
                        feat_pooled = feat_full[:n_full].view(n_full // k, k, C).mean(dim=1)  # (L//k, C)
                        if n_full < L_orig:
                            # Handle remainder: average the tail
                            feat_tail = feat_full[n_full:].mean(dim=0, keepdim=True)  # (1, C)
                            feat_pooled = torch.cat([feat_pooled, feat_tail], dim=0)  # (L//k + 1, C)
                    else:
                        # Entire sequence is remainder
                        feat_pooled = feat_full.mean(dim=0, keepdim=True)  # (1, C)
                    feat_full = feat_pooled
                    
                    # Pool mask: max pooling along sequence dimension
                    # Normalize mask_store to (1, L) format for consistent pooling
                    if mask_store.dim() == 1:
                        # (L,) -> (1, L)
                        mask_store = mask_store.unsqueeze(0)
                    elif mask_store.dim() == 3:
                        # (1, 1, L) -> (1, L)
                        mask_store = mask_store.squeeze(0)
                    # Now mask_store is (1, L) or (L,) - ensure (1, L)
                    if mask_store.dim() == 1:
                        mask_store = mask_store.unsqueeze(0)
                    
                    L_mask = mask_store.shape[-1]
                    n_full_mask = (L_mask // k) * k
                    if n_full_mask > 0:
                        # mask_store: (1, L) -> view as (1, L//k, k) -> max -> (1, L//k)
                        mask_pooled = mask_store[:, :n_full_mask].view(1, n_full_mask // k, k).amax(dim=-1)  # (1, L//k)
                        if n_full_mask < L_mask:
                            # Handle remainder: max of the tail
                            mask_tail = mask_store[:, n_full_mask:].amax(dim=-1, keepdim=True)  # (1, 1)
                            mask_pooled = torch.cat([mask_pooled, mask_tail], dim=-1)  # (1, L//k + 1)
                    else:
                        # Entire sequence is remainder
                        mask_pooled = mask_store.amax(dim=-1, keepdim=True)  # (1, 1)
                    mask_store = mask_pooled
                    
                    # Ensure shapes match: feat_full.shape[0] == mask_store.shape[-1]
                    # This ensures feat_full_per_sample[i].shape[0] == mask_full_per_sample[i].shape[-1]
                    # (sequence length dimension must match)
                    L_feat = feat_full.shape[0]
                    L_mask = mask_store.shape[-1]
                    if L_feat != L_mask:
                        # Truncate to minimum length to ensure consistency
                        min_len = min(L_feat, L_mask)
                        feat_full = feat_full[:min_len]
                        mask_store = mask_store[:, :min_len]
                    
                    # Ensure mask_store is in (1, L) format for consistent stacking later
                    if mask_store.dim() == 1:
                        mask_store = mask_store.unsqueeze(0)
                    elif mask_store.dim() == 3:
                        mask_store = mask_store.squeeze(0)

                # Ensure tensors are on target device before storing
                # This ensures all cache hits and misses are on the same device for stacking
                if feat_full.device != device:
                    feat_full = feat_full.to(device)
                if mask_store.device != device:
                    mask_store = mask_store.to(device)

                # Store contiguous tensors
                feat_full_per_sample[i] = feat_full.contiguous()
                mask_full_per_sample[i] = mask_store.contiguous()
                ccre_full_per_sample[i] = ccre_cached if ccre_cached else None

            if verbose and hit_indices:
                print(f"Cache hits for {len(hit_indices)} samples")
            try:
                self._cache_hits += len(hit_indices)
            except Exception:
                pass

        if miss_indices:
            if self.borzoi is None:
                missing_key = cache_keys[miss_indices[0]] if cache_keys else None
                raise RuntimeError(
                    f"Cache miss for sample {miss_indices[0]} (key: {missing_key}) but load_backbone=False. Cannot compute embedding."
                )

            dna_miss = dna[miss_indices]
            mask_miss = mask[miss_indices]

            with torch.no_grad():
                if self.optimize_backbone_bf16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        feat_miss = self.borzoi.conv_dna(dna_miss)
                        feat_miss = self.borzoi._max_pool(feat_miss)
                else:
                    feat_miss = self.borzoi.conv_dna(dna_miss)
                    feat_miss = self.borzoi._max_pool(feat_miss)

                if self.optimize_backbone_bf16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        for block in self.borzoi.res_tower:
                            feat_miss = block(feat_miss)
                else:
                    for block in self.borzoi.res_tower:
                        feat_miss = block(feat_miss)

                if self.optimize_backbone_bf16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        feat_miss = self.borzoi.unet1(feat_miss).transpose(1, 2)
                else:
                    feat_miss = self.borzoi.unet1(feat_miss).transpose(1, 2)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat_miss = feat_miss.to(torch.bfloat16)
                    feat_miss = self.borzoi.transformer(feat_miss)
                feat_miss = feat_miss.to(torch.float32)

            L128 = feat_miss.shape[1]
            mask_128_miss = F.interpolate(mask_miss.float(), size=L128, mode="nearest")

            # Keep on GPU for now, only move to CPU when needed for cache storage
            feat_miss_gpu = feat_miss.detach()
            mask_128_miss_gpu = mask_128_miss.detach()

            # Batch cache writes: collect all payloads first
            cache_payloads = []
            cache_keys_to_store = []

            for local_idx, sample_idx in enumerate(miss_indices):
                # Store on GPU for immediate use
                feat_full_per_sample[sample_idx] = feat_miss_gpu[local_idx].contiguous()
                mask_full_per_sample[sample_idx] = mask_128_miss_gpu[local_idx].contiguous()
                ccre_full_per_sample[sample_idx] = ccre_intervals_rel[sample_idx]

                # Prepare cache payloads (will move to CPU only for cache storage)
                key = cache_keys[sample_idx]
                if cache is not None and key is not None:
                    cache_keys_to_store.append(key)
                    cache_payloads.append((
                        feat_miss_gpu[local_idx].unsqueeze(0).cpu(),  # Move to CPU only for cache
                        mask_128_miss_gpu[local_idx].unsqueeze(0).cpu(),
                        ccre_intervals_rel[sample_idx],
                    ))

                if verbose and not cache_miss_logged:
                    print(f"Cache miss for {key}, computed full-resolution embedding")
                    cache_miss_logged = True
                try:
                    self._cache_misses += 1
                except Exception:
                    pass

            # Batch cache writes
            if cache is not None and len(cache_payloads) > 0:
                for key, payload in zip(cache_keys_to_store, cache_payloads):
                    if hasattr(cache, "set"):
                        cache.set(key, payload)
                    elif isinstance(cache, dict):
                        cache[key] = payload

        if any(f is None for f in feat_full_per_sample) or any(m is None for m in mask_full_per_sample):
            raise RuntimeError("Failed to assemble DNA embeddings for all samples.")

        # Stack tensors (should already be on target device from cache hit/miss processing)
        feat_full_batch = torch.stack([t for t in feat_full_per_sample], dim=0)  # (B, L_full, C)
        mask_full_batch = torch.stack([t for t in mask_full_per_sample], dim=0)  # (B, 1, L_full)

        # Safety check: ensure tensors are on target device (should already be there)
        if feat_full_batch.device != device:
            feat_full_batch = feat_full_batch.to(device, non_blocking=True)
        if mask_full_batch.device != device:
            mask_full_batch = mask_full_batch.to(device, non_blocking=True)

        L_full = feat_full_batch.shape[1]
        target_length = L_full
        use_jitter = jitter_window > 0 and self.jitter_target_length > 0
        if use_jitter:
            target_length = min(L_full, self.jitter_target_length)

        if use_jitter and target_length < L_full:
            max_offset = min(jitter_window, max(0, (L_full - target_length) // 2))
            if max_offset > 0:
                offsets = torch.randint(-max_offset, max_offset + 1, (B,), device=device)
            else:
                offsets = torch.zeros(B, device=device, dtype=torch.long)
            centers = torch.full((B,), L_full // 2, device=device, dtype=torch.long)
            start_idx = (centers - target_length // 2 + offsets).clamp_(0, L_full - target_length)
            gather_idx = start_idx[:, None] + torch.arange(target_length, device=device, dtype=torch.long)

            feat = torch.gather(
                feat_full_batch,
                1,
                gather_idx[:, :, None].expand(-1, -1, feat_full_batch.size(-1)),
            )
            mask_128 = torch.gather(mask_full_batch, 2, gather_idx[:, None, :])
            actual_center = start_idx + target_length // 2
            effective_offset = (actual_center - centers).to(torch.long)
        else:
            feat = feat_full_batch
            mask_128 = mask_full_batch
            effective_offset = torch.zeros(B, device=device, dtype=torch.long)

        feat = feat.contiguous()
        mask_128 = mask_128.contiguous()

        # Keep effective_offset on GPU, only convert to CPU when needed
        effective_offset_gpu = effective_offset.to(torch.long)
        bp_per_token_effective = self.base_bp_per_token * self.cache_pool_k
        # window_size_bp = float(L_full * 128.0)
        window_size_bp = float(L_full * bp_per_token_effective)
        ccre_intervals_rel_batched = []
        for sample_idx in range(B):
            intervals = ccre_full_per_sample[sample_idx]
            if intervals and use_jitter and target_length < L_full:
                offset_bp = float(-effective_offset_gpu[sample_idx].item() * 128.0)
                adjusted = []
                for iv in intervals:
                    if (
                        iv is None
                        or not isinstance(iv, (list, tuple))
                        or len(iv) != 2
                    ):
                        continue
                    start_bp = max(0.0, iv[0] + offset_bp)
                    end_bp = min(window_size_bp, iv[1] + offset_bp)
                    adjusted.append((start_bp, end_bp))
                ccre_intervals_rel_batched.append(adjusted if adjusted else None)
            else:
                ccre_intervals_rel_batched.append(intervals if intervals else None)
        
        # ===== Apply compression after transformer =====
        if verbose: t = _tic()
        feat_comp, mask_comp, width_comp_bp, keep_regions_mask = compress_after_transformer_fast(
            feat, mask_128, 
            ccre_intervals_rel=ccre_intervals_rel_batched,
            distal_stride_tokens=self.compress_stride_after_transformer,
            bp_per_token=bp_per_token_effective,
        )
        if verbose: _toc(t, "compress_after_transformer")

        if (
            self.training
            and self.ccre_dropout is not None
            and keep_regions_mask is not None
        ):
            keep_mask_bool = keep_regions_mask.squeeze(1).bool()
            pert_mask_bool = (mask_comp != 0).squeeze(1).bool()
            ccre_only_mask = keep_mask_bool & (~pert_mask_bool)

            if ccre_only_mask.any():
                with torch.no_grad():
                    d_signed_comp, _ = signed_nearest_distance(mask_comp)
                    d_abs_comp = d_signed_comp.abs().to(feat_comp.dtype)
                    bp_distances_dropout, _ = self.perturbation._compute_bp_distances(
                        m=mask_comp,
                        d_abs=d_abs_comp,
                        width_comp_bp=width_comp_bp,
                        keep_regions_mask=keep_regions_mask,
                    )

                augmented_ccre_mask = self.ccre_dropout(
                    ccre_only_mask,
                    pert_mask_bool,
                    bp_distances_dropout.detach(),
                )
                keep_regions_mask = (
                    (pert_mask_bool | augmented_ccre_mask).unsqueeze(1)
                ).to(keep_regions_mask.dtype)

        

        # ===== Single perturbation embedding =====
        if verbose: t = _tic()
        pert_result = self.perturbation(
            feat_comp, mask_comp, 
            interaction=True, 
            keep_regions_mask=keep_regions_mask,
            width_comp_bp=width_comp_bp,
            return_p_anchor=(return_components or debug),
            z_ctrl=z_ctrl,
            ablate_distance_bias=self.ablate_distance_bias,
            ablate_ccre_bias=self.ablate_ccre_bias,
            cell_bias_mode=self.cell_bias_mode
        )  # (B, out_dim) or ((B, out_dim), p_anchor)
        if isinstance(pert_result, tuple):
            z, p_anchor = pert_result
        else:
            z = pert_result
            p_anchor = None
        if verbose: _toc(t, "spe_forward")
        z = self.delta_gain * (z)              # (B, out_dim)
        # z = self.post_z_norm(z)
        
        if verbose:
            print(f'feat_comp shape: {feat_comp.shape}')  # (B, L_comp, 1536)
            print(f'inside the function -- dna_ref_embeddings: {z}')
            print(f'z shape is {z.shape}')  # (B, out_dim)
            _toc(t_all, "adapter_total")
            ms = {k: round(v*1000.0, 3) for k, v in timings.items()}
            print(f"Adapter timings (ms): {ms}")
            try:
                print(f"[Cache] hits={self._cache_hits}, misses={self._cache_misses}")
            except Exception:
                pass
        
        if z.dim() == 2:
            z = z.unsqueeze(1) ## (B, 1, out_dim)
        
        if return_components or debug:
            debug_dict = {
                "base_features": feat_comp, 
                "mask_comp": mask_comp, 
                "keep_regions_mask": keep_regions_mask, 
                "width_comp_bp": width_comp_bp
            }
            if p_anchor is not None:
                debug_dict["p_anchor_bp"] = p_anchor  # (B, 1) bp anchor position
            return z, debug_dict

        return z
    


# ----------------------------------------------------------------------
# Correctness tests and micro-benchmark for signed distance (optional)
# Guarded by env var: TEST_SIGNED_DIST=1
# ----------------------------------------------------------------------
def _signed_nearest_distance_ref(mask_128: torch.Tensor):
    """
    Reference implementation using the loop-based distance transform for validation.
    """
    B, _, L = mask_128.shape
    m = (mask_128 != 0).squeeze(1)
    has_pert = m.any(dim=1)
    left = _fwd_bwd_dt(torch.flip(m, dims=[1]))
    left = torch.flip(left, dims=[1])
    right = _fwd_bwd_dt(m)
    d_signed = torch.where(left <= right, -left, right).to(torch.float32)
    d_signed = torch.where(has_pert.view(-1,1), d_signed, torch.full_like(d_signed, L, dtype=d_signed.dtype))
    return d_signed, has_pert


def _validate_signed_distance(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cases = [
        (1, 64), (2, 129), (4, 512), (2, 1024)
    ]
    for B, L in cases:
        for pattern in ["zeros", "ones", "single", "alt", "random"]:
            mask = torch.zeros(B, 1, L, device=device)
            if pattern == "ones":
                mask[:] = 1
            elif pattern == "single":
                idx = torch.randint(0, L, (B,), device=device)
                for b in range(B):
                    mask[b, 0, int(idx[b].item())] = 1
            elif pattern == "alt":
                mask[:, 0, ::2] = 1
            elif pattern == "random":
                mask = (torch.rand(B, 1, L, device=device) > 0.8).float()
            # zeros handled by default
            d_ref, _ = _signed_nearest_distance_ref(mask)
            d_new, _ = signed_nearest_distance(mask)
            max_err = (d_ref - d_new).abs().max().item()
            assert max_err == 0.0, f"Mismatch for B={B}, L={L}, pattern={pattern}, max_err={max_err}"
    print("signed_nearest_distance: correctness OK on basic patterns.")


def _bench_signed_distance(B=2, L=4096, reps=20, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mask = (torch.rand(B, 1, L, device=device) > 0.95).float()
    # Warmup
    signed_nearest_distance(mask)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        signed_nearest_distance(mask)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Vectorized signed distance: {(t1 - t0)*1000.0/reps:.3f} ms per call (B={B}, L={L})")

# ----------------------------------------------------------------------
# example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TEST_LORA = False
    TEST_Adaptor = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optional: run correctness and micro-benchmark for signed distance
    if os.environ.get("TEST_SIGNED_DIST", "0") == "1":
        _validate_signed_distance(device)
        _bench_signed_distance(device=device)
        TEST_Adaptor = False

    if TEST_Adaptor:
        print(f'###### TESTING Borzoi model with adaptation ######')
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            model = BorzoiPerturbAdapter().to(device)
            
            # === Calculate trainable parameters ===
            print("\n" + "="*60)
            print("ADAPTER TRAINABLE PARAMETERS ANALYSIS")
            print("="*60)
            
            # Total trainable params in adapter
            adapt_params = [p for p in model.parameters() if p.requires_grad]
            total_trainable = sum(p.numel() for p in adapt_params)
            print(f"Total adapter trainable params: {total_trainable:,}")
            
            # Verify Borzoi is frozen
            if hasattr(model, 'borzoi') and model.borzoi is not None:
                borzoi_params = [p for p in model.borzoi.parameters() if p.requires_grad]
                borzoi_trainable = sum(p.numel() for p in borzoi_params)
                print(f"Borzoi trainable params (should be 0): {borzoi_trainable:,}")
                if borzoi_trainable > 0:
                    print("⚠️  WARNING: Some Borzoi parameters are trainable!")
                else:
                    print("✓ Borzoi is correctly frozen")
            
            # Breakdown by module
            print("\nModule breakdown:")
            for name, module in model.named_children():
                if name == 'borzoi':
                    continue  # Skip frozen Borzoi
                mod_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if mod_params > 0:
                    print(f"  {name}: {mod_params:,} params")
            
            # Detailed breakdown of SinglePerturbEmbedding if it exists
            if hasattr(model, 'perturbation'):
                print("\nSinglePerturbEmbedding breakdown:")
                spe = model.perturbation
                for name, module in spe.named_children():
                    mod_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    if mod_params > 0:
                        print(f"    {name}: {mod_params:,} params")
            
            print("="*60 + "\n")
            
            encode = shallowEncoder(input_dim=1536, out_dim=128).to(device)
            # model.train_mask_only()         # freeze Borzoi, learn mask branch first
            # x = torch.randn(2, 5, 524_288).to(device)  # dummy DNA+mask
            B, L = 2, 524_288
            indices = torch.randint(4, (1, L), device=device)
            dna = F.one_hot(indices, num_classes=4).permute(0, 2, 1).float()
            dna = dna.repeat(B, 1, 1)  # same DNA for all in batch
            dna_2  = torch.randn(B, 4, L, device=device)        # random DNA bases
            
            dna = torch.cat([dna, dna_2], 0)  # (B, 4, L)
            
            mask = torch.zeros(2*B, 1, L, device=device)        # *all zeros*  ← control
            centre = L//2
            mask[0, 0, 0:128] = 1
            mask[1, 0, 1128:1256] = 1
            mask[2, 0, 0:128] = 1
            mask[3, 0, 1128:1256] = 1
            
            print(f'mask sum: {mask.sum(dim=(-1,-2))}')
            # mask[0, 0, 100000:100000+1000] = 1
            
            x = torch.cat([dna, mask], 1)
            out, debug = model(x, debug=True,verbose=True)  # (B, L', C), debug_dict
            
            
            print(f'out shape is {out.shape}')
        
            out_pool = out.mean(dim=1)  # (B, C)
            # out_pool = out
            normalized_pool = F.normalize(out_pool, p=2, dim=1)
            ## compute cosine similarity of out_pool
            cosine_sim_matrix = normalized_pool @ normalized_pool.T
            print(f'cosine similarity of out_pool: {cosine_sim_matrix}')
            
            out_project = encode(out_pool)
            normalized_project = F.normalize(out_project, p=2, dim=1)
            cosine_sim_matrix = normalized_project @ normalized_project.T
            print(f'projected cosine similarity of out_pool: {cosine_sim_matrix}')
            
            print(out.shape)                # → (2, L', 1536)
            print(f'sum of out: {out.sum()}')
            print(out)
        del model
        del out
    if TEST_LORA:
        print(f'###### TESTING Borzoi model with adaptation and LORA ######')
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            model = BorzoiPerturbAdapter(enable_lora=True).to(device)
            model.train_mask_only()         # freeze Borzoi, learn mask branch first
            x = torch.randn(2, 5, 524_288).to(device)  # dummy DNA+mask
            out = model(x)
            
        print(out.shape)                # → (2, L', 1536)
        # print(out)
