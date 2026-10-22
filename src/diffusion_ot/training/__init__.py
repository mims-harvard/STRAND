"""
Training utilities: DDP setup, schedulers, and configuration.

This module provides:
- DDP setup and cleanup functions
- Learning rate scheduler utilities
- Configuration loading and flattening
"""

from .ddp_setup import (
    setup_ddp,
    cleanup,
    find_free_port,
    ddp_sum,
    gather_numpy_arrays_across_ranks,
    toggle_grad,
    flatten_dict,
)
from .scheduler import (
    init_saved_lrs,
    refresh_saved_lrs_from_optimizer,
    get_saved_lrs,
    set_phase,
    scheduler_step_all_groups,
    hard_freeze_borzoi_adapter,
    GroupPlateau,
)
from .config import (
    load_config,
)

__all__ = [
    # DDP setup
    'setup_ddp',
    'cleanup',
    'find_free_port',
    'ddp_sum',
    'gather_numpy_arrays_across_ranks',
    'toggle_grad',
    'flatten_dict',
    # Scheduler
    'init_saved_lrs',
    'refresh_saved_lrs_from_optimizer',
    'get_saved_lrs',
    'set_phase',
    'scheduler_step_all_groups',
    'hard_freeze_borzoi_adapter',
    'GroupPlateau',
    # Config
    'load_config',
]

