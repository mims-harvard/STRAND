"""STRAND: Sequence-Conditioned Transport for Single-Cell Perturbations.

``strand.locus`` and ``strand.metrics`` are lightweight (NumPy/SciPy). ``strand.api``
loads the trained model and imports PyTorch and the DNA foundation model on first use.
"""

from .locus import (
    BIN_BP,
    DEFAULT_HALF_WIDTH,
    N_TOKENS,
    WINDOW_BP,
    Locus,
    bp_mask,
    masked_tokens,
    scan,
    token_mask,
    window_coordinates,
)

__version__ = "1.0.0"

__all__ = [
    "BIN_BP",
    "DEFAULT_HALF_WIDTH",
    "N_TOKENS",
    "WINDOW_BP",
    "Locus",
    "bp_mask",
    "masked_tokens",
    "scan",
    "token_mask",
    "window_coordinates",
    "STRAND",
    "ControlCells",
    "Prediction",
    "SamplerConfig",
]


def __getattr__(name):
    if name in {"STRAND", "ControlCells", "Prediction", "SamplerConfig"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)
