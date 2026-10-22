"""
Weighting utilities: Adaptive per-label weighting for imbalanced datasets.

This module contains:
- PerturbationWeighter: Adaptive weighting with EMA-smoothed counts
"""

import torch
import torch.distributed as dist
from typing import Optional


class PerturbationWeighter:
    """
    Adaptive per-label weighting without knowing num_labels upfront.
    - Grows self.run_counts on demand (contiguous non-negative label IDs).
    - Uses EMA-smoothed global batch histograms across DDP ranks.
    - Supports inverse-power (alpha) or Class-Balanced (beta) weighting.
    - Normalizes per-batch weights to mean=1 to keep loss scale stable.
    """
    def __init__(self, alpha=0.5, beta=None, ema_gamma=1e-3,
                 w_cap=5.0, prior=10.0, device=None):
        """
        alpha: exponent in counts^{-alpha} (set beta=None to use this)
        beta:  Class-Balanced effective-number param (use instead of alpha)
        ema_gamma: EMA step size for running counts (1e-3 ~ 1e-2 typical)
        w_cap: cap relative to median weight (e.g., 5.0); set None to disable
        prior: initial pseudo-count for any new label (keeps early behavior ~unweighted)
        device: torch device (defaults to current device of labels)
        """
        self.alpha = alpha
        self.beta = beta
        self.ema_gamma = float(ema_gamma)
        self.w_cap = w_cap
        self.prior = float(prior)
        self.device = device
        # Start with zero-length run_counts; grow as needed
        self.run_counts = torch.empty(0, device=device, dtype=torch.float32)

    def _ensure_capacity(self, needed: int, device: torch.device):
        cur = self.run_counts.numel()
        if cur >= needed:
            # move to device if device changed
            if self.run_counts.device != device:
                self.run_counts = self.run_counts.to(device)
            return
        grow = needed - cur
        add = torch.full((grow,), self.prior, device=device, dtype=self.run_counts.dtype)
        if cur == 0:
            self.run_counts = add
        else:
            self.run_counts = torch.cat([self.run_counts.to(device), add], dim=0)

    @torch.no_grad()
    def update_and_get_weights(self, labels: torch.LongTensor, loss_device=None):
        """
        labels: LongTensor shape [B], values in [0, ...], contiguous non-negative.
                (If your raw labels aren't contiguous, factorize them upstream.)
        """
        device = labels.device if self.device is None else self.device

        # ---- Find global max label to decide capacity (DDP-safe) ----
        max_local = torch.as_tensor(labels.max().item() if labels.numel() > 0 else -1,
                                    device=device, dtype=torch.long)
        max_global = max_local.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(max_global, op=dist.ReduceOp.MAX)
        needed = int(max_global.item()) + 1  # if -1 => needed = 0

        # ---- Grow run_counts to fit new labels ----
        self._ensure_capacity(needed, device=device)

        # ---- Build per-batch histogram (length = current capacity) ----
        # If no labels this step (edge case), make a zero hist
        if labels.numel() == 0:
            hist_local = torch.zeros_like(self.run_counts)
        else:
            hist_local = torch.bincount(labels.to(device),
                                        minlength=self.run_counts.numel()).float()

        # ---- All-reduce to get GLOBAL batch histogram ----
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(hist_local, op=dist.ReduceOp.SUM)

        # ---- EMA update of running counts ----
        if self.run_counts.numel() == 0:
            # nothing to update (shouldn't happen if needed>0), but be safe
            self.run_counts = hist_local.clone()
        else:
            self.run_counts.mul_(1.0 - self.ema_gamma).add_(self.ema_gamma * hist_local)

        # ---- Convert counts -> per-label weights ----
        counts = self.run_counts.clamp_min_(1.0)  # avoid div-by-zero / pow(0)
        if self.beta is not None:
            # Cui et al., Class-Balanced (effective number of samples)
            # w_i = (1 - beta) / (1 - beta^{counts_i})
            w = (1.0 - self.beta) / (1.0 - torch.pow(self.beta, counts))
        else:
            # inverse power law: w_i = counts_i^{-alpha}
            w = torch.pow(counts, -self.alpha)

        # ---- Cap extremes relative to median to avoid chasing noise ----
        if self.w_cap is not None:
            median = torch.median(w)
            # If no labels yet, median might be nan; guard it
            if torch.isfinite(median):
                w = torch.minimum(w, median * self.w_cap)

        # ---- Gather weights for current batch and normalize to mean=1 ----
        if labels.numel() == 0:
            w_batch = torch.ones(0, device=device, dtype=w.dtype)
        else:
            w_batch = w[labels.to(device)]
            w_batch = w_batch / w_batch.mean().clamp_min(1e-8)

        # ---- Match loss device if provided ----
        if (loss_device is not None) and (w_batch.device != loss_device):
            w_batch = w_batch.to(loss_device)

        return w_batch
