"""
Base loss utilities: Common functions and base classes.

This module contains:
- sigma_reg: Regularization for sigma parameters
- log_phi: Stable log of standard normal CDF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
import math
from torch import Tensor


def sigma_reg(s, delta=3.0):
    # smooth L1 on sigma: quadratic near 0, linear after delta
    x = s
    return torch.where(x < delta,
                       0.5*x*x,
                       delta*(x - 0.5*delta))

def sigma_reg(s, delta=3.0):
    # smooth L1 on sigma: quadratic near 0, linear after delta
    x = s
    return torch.where(x < delta,
                       0.5*x*x,
                       delta*(x - 0.5*delta))


def log_phi(x: Tensor) -> Tensor:
    """Stable log Φ(x)."""
    return torch.special.log_ndtr(x)
