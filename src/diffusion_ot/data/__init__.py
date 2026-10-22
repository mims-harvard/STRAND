"""
Data module: datasets, preprocessing, caching, and DNA matching.

This module provides:
- Dataset classes for perturbation-control paired data
- Preprocessing utilities for gene name conversion and normalization
- LRU cache for DNA embeddings
- DNA embedding matching functions
"""

from .cache import LRUCache
from .preprocessing import (
    convert_gene_name,
    convert_to_ensembl,
    gaussian_copula_transform,
    debug_conversion_reduction
)
from .datasets import (
    perturb_control_dataloader,
    ScalablePerturbControlDataLoader,
    perturb_control_dataloader_syn
)
from .dna_matching import (
    match_gene_dna_embeddings,
    match_gene_pert_mask_dna_embeddings,
    match_interval_pert_mask_dna_embeddings,
    match_gene_pert_mask_dna_embeddings_cached,
    match_interval_pert_mask_dna_embeddings_cached,
    match_gene_pert_mask_dna_embeddings_minimal,
    parse_interval_label,
    gen_interval_centered_sequence,
    aggregate_mask_from_interval,
    _as_BLD,
    _to_LD_per_sample
)

__all__ = [
    # Cache
    'LRUCache',
    # Preprocessing
    'convert_gene_name',
    'convert_to_ensembl',
    'gaussian_copula_transform',
    'debug_conversion_reduction',
    # Datasets
    'perturb_control_dataloader',
    'ScalablePerturbControlDataLoader',
    'perturb_control_dataloader_syn',
    # DNA Matching
    'match_gene_dna_embeddings',
    'match_gene_pert_mask_dna_embeddings',
    'match_interval_pert_mask_dna_embeddings',
    'parse_interval_label',
    'gen_interval_centered_sequence',
    'aggregate_mask_from_interval',
    '_as_BLD',
    '_to_LD_per_sample',
]

