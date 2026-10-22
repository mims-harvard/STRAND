"""Genomic coordinates, sequence windows and perturbation masks.

This module has no heavy dependencies (NumPy only) so the coordinate contract can be
inspected and tested without PyTorch, the DNA foundation model, or a reference genome.
It mirrors the logic the model uses at inference time:

* ``Locus.label`` produces the interval label parsed by
  ``src.diffusion_ot.data.dna_matching.parse_interval_label``.
* ``window_coordinates`` reproduces the window placement in
  ``dna_matching.gen_interval_centered_sequence`` (no jitter, as at inference).
* ``bp_mask`` reproduces ``dna_matching.aggregate_mask_from_interval``.
* ``token_mask`` reproduces the nearest-neighbour downsampling
  (``F.interpolate(mask, size=4096, mode="nearest")``) in ``PModule/DNA_modeling.py``.

Coordinates are 1-based and inclusive, as in GTF files and in the interval labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np

#: Sequence context seen by the frozen DNA encoder (Flashzoi/Borzoi input length).
WINDOW_BP = 524_288
#: Output-bin size of the DNA encoder; this is the model resolution.
BIN_BP = 128
#: Number of DNA context tokens (WINDOW_BP / BIN_BP).
N_TOKENS = WINDOW_BP // BIN_BP
#: Default CRISPRi-accessible half-width used throughout the paper (mask = center +/- 200 bp).
DEFAULT_HALF_WIDTH = 200

_LABEL_RE = re.compile(
    r"^(chr[0-9A-Za-z]+):(\d[\d,]*)-(\d[\d,]*)(?:\(([+-])\))?(?:-([^-]+)-([^-]+))?$"
)


@dataclass(frozen=True)
class Locus:
    """A perturbation target: a genomic interval on a reference assembly.

    The interval is marked by the perturbation mask ``m``. The DNA sequence itself is the
    unmodified reference sequence; STRAND does not edit nucleotides.
    """

    chrom: str
    start: int
    end: int
    strand: str = "+"
    tag: str = "locus"
    dataset: str = "query"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"chr[0-9A-Za-z]+", self.chrom):
            raise ValueError(f"chrom must look like 'chr1' or 'chrX', got {self.chrom!r}")
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"invalid interval {self.start}-{self.end} (1-based, inclusive)")
        if self.strand not in {"+", "-"}:
            raise ValueError(f"strand must be '+' or '-', got {self.strand!r}")
        for name in ("tag", "dataset"):
            value = getattr(self, name)
            if not value or "-" in value:
                raise ValueError(f"{name} must be non-empty and contain no '-', got {value!r}")

    @classmethod
    def centered(
        cls,
        chrom: str,
        center: int,
        half_width: int = DEFAULT_HALF_WIDTH,
        strand: str = "+",
        tag: str = "locus",
        dataset: str = "query",
    ) -> "Locus":
        """Interval ``[center - half_width, center + half_width]``, as in the paper's scans."""
        tag = str(tag).replace("-", "_")
        return cls(chrom, max(1, int(center) - half_width), int(center) + half_width, strand, tag, dataset)

    @classmethod
    def from_label(cls, label: str) -> "Locus":
        m = _LABEL_RE.match(label)
        if not m:
            raise ValueError(f"not an interval label: {label!r}")
        chrom, start, end, strand, tag, dataset = m.groups()
        return cls(
            chrom,
            int(start.replace(",", "")),
            int(end.replace(",", "")),
            strand or "+",
            tag or "locus",
            dataset or "query",
        )

    @property
    def center(self) -> int:
        """Window center used by the model: ``floor((start + end) / 2)``."""
        return (self.start + self.end) // 2

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    @property
    def label(self) -> str:
        """Interval label consumed by ``STRANDModel`` (``embed_backend='interval'``)."""
        return f"{self.chrom}:{self.start}-{self.end}({self.strand})-{self.tag}-{self.dataset}"

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start:,}-{self.end:,}({self.strand})"


@dataclass(frozen=True)
class Window:
    """Placement of the 524,288 bp encoder window around a locus."""

    chrom: str
    proxy_start: int  # 1-based genomic coordinate of the first window base
    proxy_end: int  # 1-based genomic coordinate of the last window base
    rel_start: int  # 0-based window index of the first masked base
    rel_end: int  # 0-based window index of the last masked base (inclusive)

    @property
    def length(self) -> int:
        return self.proxy_end - self.proxy_start + 1


def window_coordinates(locus: Locus, chrom_length: int, window_bp: int = WINDOW_BP) -> Window:
    """Place the encoder window around ``locus`` exactly as the model does at inference.

    The window is centered on ``locus.center`` with ``window_bp // 2`` bases to the left and
    ``window_bp - window_bp // 2 - 1`` to the right, shifted inward near chromosome ends.
    """
    if chrom_length < window_bp:
        raise ValueError("chromosome shorter than the encoder window")
    mid = locus.center
    left = window_bp // 2
    right = window_bp - left - 1
    if mid - left <= 0:
        residual = left - mid + 1
        left = mid - 1
        right += residual
    elif mid + right > chrom_length:
        residual = mid + right - chrom_length
        right = chrom_length - mid
        left += residual
    proxy_start = mid - left
    proxy_end = mid + right
    rel_start = max(0, locus.start - proxy_start)
    rel_end = min(window_bp - 1, locus.end - proxy_start)
    return Window(locus.chrom, proxy_start, proxy_end, rel_start, rel_end)


def bp_mask(window: Window, window_bp: int = WINDOW_BP) -> np.ndarray:
    """Base-resolution mask ``m``: 1 on masked positions (inclusive), 0 elsewhere."""
    mask = np.zeros(window_bp, dtype=np.float32)
    a = max(0, window.rel_start)
    b = min(window_bp - 1, window.rel_end)
    if b >= a:
        mask[a : b + 1] = 1.0
    return mask


def token_mask(mask_bp: np.ndarray, n_tokens: int = N_TOKENS) -> np.ndarray:
    """Downsample a base-resolution mask to encoder tokens (nearest neighbour).

    Token ``i`` takes the mask value at base ``floor(i * len(mask_bp) / n_tokens)``; with the
    default sizes that is the first base of each 128 bp bin.
    """
    idx = np.floor(np.arange(n_tokens) * (mask_bp.shape[-1] / n_tokens)).astype(np.int64)
    return mask_bp[..., idx]


def masked_tokens(locus: Locus, chrom_length: int) -> np.ndarray:
    """Indices of the DNA context tokens selected by the perturbation mask."""
    return np.flatnonzero(token_mask(bp_mask(window_coordinates(locus, chrom_length))))


def offset_from(position: int, reference: int, strand: str) -> int:
    """Signed distance in transcription direction (downstream positive)."""
    return (position - reference) if strand == "+" else (reference - position)


def scan(
    chrom: str,
    start: int,
    end: int,
    step: int = 100,
    half_width: int = DEFAULT_HALF_WIDTH,
    strand: str = "+",
    tag: str = "scan",
    dataset: str = "query",
) -> List[Locus]:
    """Sliding-window loci with centers ``start, start + step, ...`` while ``center < end``.

    ``step`` is the scan interval (100 bp in the paper's gene-body scans, 50 bp in the
    e-HDAC6 region scan). It is not the model resolution, which is set by the 128 bp bins.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    return [
        Locus.centered(chrom, c, half_width, strand, f"{tag}_w{i}", dataset)
        for i, c in enumerate(range(int(start), int(end), int(step)))
    ]


def iter_labels(loci: List[Locus]) -> Iterator[str]:
    for locus in loci:
        yield locus.label


__all__ = [
    "WINDOW_BP",
    "BIN_BP",
    "N_TOKENS",
    "DEFAULT_HALF_WIDTH",
    "Locus",
    "Window",
    "window_coordinates",
    "bp_mask",
    "token_mask",
    "masked_tokens",
    "offset_from",
    "scan",
    "iter_labels",
]
