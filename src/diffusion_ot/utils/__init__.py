"""
Utility modules: DNA, gene, distributed, weighting, and factorization utilities.

This module provides:
- DNA sequence manipulation and annotation loading
- Gene annotation utilities
- Distributed training helpers
- Adaptive weighting for imbalanced datasets
- Label factorization for DDP
"""

from .dna_utils import (
    get_DNA_annot,
    get_dna_record,
    load_ccres_bed,
    get_exons,
    random_dna_sequence,
    perturb_sequence,
    perturb_sequence_exon,
    searchPAM,
    gen_perturb_DNA,
    gen_gene_TSS_sequence,
    aggregate_perturbation,
    aggregate_mask_perturbation,
    rc,
    _c,
    pam_to_forward,
    build_regex,
    find_site,
    targeted_perturbation,
    target_perturbation_batch,
)
from .gene_utils import curate_perturb_tag
from .distributed import (
    gather_and_concatenate_tensors,
    gather_and_concatenate_list,
)
from .weighter import PerturbationWeighter
from .factorizer import StringLabelFactorizer

__all__ = [
    # DNA utilities
    'get_DNA_annot',
    'get_dna_record',
    'load_ccres_bed',
    'get_exons',
    'random_dna_sequence',
    'perturb_sequence',
    'perturb_sequence_exon',
    'searchPAM',
    'gen_perturb_DNA',
    'gen_gene_TSS_sequence',
    'aggregate_perturbation',
    'aggregate_mask_perturbation',
    'rc',
    '_c',
    'pam_to_forward',
    'build_regex',
    'find_site',
    'targeted_perturbation',
    'target_perturbation_batch',
    # Gene utilities
    'curate_perturb_tag',
    # Distributed utilities
    'gather_and_concatenate_tensors',
    'gather_and_concatenate_list',
    # Weighting
    'PerturbationWeighter',
    # Factorization
    'StringLabelFactorizer',
]

