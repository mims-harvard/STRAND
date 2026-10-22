"""
Learning rate scheduler utilities: Group-based and phase-aware scheduling.

This module contains:
- scheduler_step_all_groups: Step scheduler for all parameter groups
- GroupPlateau: Independent ReduceLROnPlateau for selected parameter groups
- Phase-aware learning rate management
"""

import torch
from torch.optim import Optimizer
from typing import Optional


_saved_lrs = None


def init_saved_lrs(optimizer: Optimizer):
    """Initialize saved learning rates for phase-aware scheduling."""
    global _saved_lrs
    _saved_lrs = [g['lr'] for g in optimizer.param_groups]


def refresh_saved_lrs_from_optimizer(optimizer: Optimizer):
    """Refresh saved learning rates after scheduler step."""
    global _saved_lrs
    for i, g in enumerate(optimizer.param_groups):
        _saved_lrs[i] = g['lr'] if g['lr'] > 0 else _saved_lrs[i]


def get_saved_lrs() -> Optional[list]:
    """Get the saved learning rates for all parameter groups.
    
    Returns:
        List of saved learning rates, or None if not initialized.
    """
    global _saved_lrs
    return _saved_lrs


def toggle_grad(params, enable: bool = True):
    """Enable/disable grads in-place (fast: no new tensors are created)."""
    for p in params:
        p.requires_grad_(enable)


def set_phase(phase: str, optimizer: Optimizer, diff_params, adapt_params, pred_params):
    """Toggle grads and set LR=0 for frozen groups, restoring the last non-zero LR for active groups."""
    assert phase in ("adapt", "diff")
    toggle_grad(adapt_params, phase == "adapt")
    toggle_grad(diff_params,  phase == "diff")
    toggle_grad(pred_params,  phase == "adapt")

    # groups: [diff, adapt, pred]
    if phase == "adapt":
        optimizer.param_groups[0]['lr'] = 0.0
        optimizer.param_groups[1]['lr'] = _saved_lrs[1]  # restore adapt
        optimizer.param_groups[2]['lr'] = _saved_lrs[2]
    else:  # diff
        optimizer.param_groups[0]['lr'] = _saved_lrs[0]  # restore diff
        optimizer.param_groups[1]['lr'] = 0.0
        optimizer.param_groups[2]['lr'] = 0.0  # restore pred


def scheduler_step_all_groups(scheduler, optimizer, metric):
    # Save current (phase-masked) LRs
    masked = [g['lr'] for g in optimizer.param_groups]
    # Expose real saved LRs so scheduler can act on all groups
    for i, g in enumerate(optimizer.param_groups):
        g['lr'] = _saved_lrs[i]
    scheduler.step(metric)
    # Capture any new decayed LRs
    refresh_saved_lrs_from_optimizer(optimizer)
    # Re-apply phase mask
    for g, m in zip(optimizer.param_groups, masked):
        g['lr'] = m


def hard_freeze_borzoi_adapter(adapt):
    adapt.eval()  # avoid any stochastic layers if present
    for p in adapt.parameters():
        p.requires_grad_(False)


    """Independent ReduceLROnPlateau for selected optimizer param groups.

    Tracks a metric and reduces LR for the specified group indices when the
    metric plateaus according to patience/threshold, mirroring PyTorch's
    ReduceLROnPlateau behavior but scoped to groups.
    """
    def __init__(self, optimizer, group_ids, mode='min', factor=0.5, patience=3,
                 threshold=1e-4, cooldown=0, min_lr=1e-7, verbose=True):
        self.opt = optimizer
        self.group_ids = tuple(group_ids)
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.verbose = bool(verbose)
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, metric):
        val = float(metric)
        if self.best is None:
            self.best = val
            self.num_bad_epochs = 0
            return
        improved = (val < self.best - self.threshold) if self.mode == 'min' else (val > self.best + self.threshold)
        if improved:
            self.best = val
            self.num_bad_epochs = 0
            return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            for gi in self.group_ids:
                g = self.opt.param_groups[gi]
                old_lr = float(g['lr'])
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    g['lr'] = new_lr
                    if self.verbose:
                        print(f"ReduceLROnPlateau[group {gi}]: {old_lr:.3e} -> {new_lr:.3e} (metric={val:.5g})")
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown



class GroupPlateau:
    """Independent ReduceLROnPlateau for selected optimizer param groups.

    Tracks a metric and reduces LR for the specified group indices when the
    metric plateaus according to patience/threshold, mirroring PyTorch's
    ReduceLROnPlateau behavior but scoped to groups.
    """
    def __init__(self, optimizer, group_ids, mode='min', factor=0.5, patience=3,
                 threshold=1e-4, cooldown=0, min_lr=1e-7, verbose=True):
        self.opt = optimizer
        self.group_ids = tuple(group_ids)
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.verbose = bool(verbose)
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, metric):
        val = float(metric)
        if self.best is None:
            self.best = val
            self.num_bad_epochs = 0
            return
        improved = (val < self.best - self.threshold) if self.mode == 'min' else (val > self.best + self.threshold)
        if improved:
            self.best = val
            self.num_bad_epochs = 0
            return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            for gi in self.group_ids:
                g = self.opt.param_groups[gi]
                old_lr = float(g['lr'])
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    g['lr'] = new_lr
                    if self.verbose:
                        print(f"ReduceLROnPlateau[group {gi}]: {old_lr:.3e} -> {new_lr:.3e} (metric={val:.5g})")
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
