"""
Dataset classes: PyTorch datasets for perturbation-control paired data.

This module contains:
- perturb_control_dataloader: Standard Dataset for perturbation-control pairs
- ScalablePerturbControlDataLoader: IterableDataset for large-scale data loading
- perturb_control_dataloader_syn: Dataset for synthetic data
"""

import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import scipy
import scipy.sparse
import pickle
import h5py
from typing import List, Dict, Optional, Tuple, Iterator, Union, Literal
import random
import warnings
import torch.distributed as dist
import os
from collections import defaultdict
import re
import time
import math

from ..data.preprocessing import convert_to_ensembl, convert_gene_name
from ..data.cache import LRUCache

# Import config for default paths
try:
    from ..config.defaults import (
        PROCESSED_DATA_ROOT, SCIMILARITY_MODEL_PATH,
        H5_SUFFIX, IDX_SUFFIX
    )
except ImportError:
    # Fallback when the config package cannot be imported
    PROCESSED_DATA_ROOT = os.environ.get("DATA_ROOT", "./data/processed")
    SCIMILARITY_MODEL_PATH = os.environ.get("SCIMILARITY_MODEL_PATH", "")
    H5_SUFFIX = "X_scimilarity.h5"
    IDX_SUFFIX = "trainsub.idx.csv"

# Import scimilarity utilities (will be refactored later)
try:
    from scimilarity.utils import align_dataset
    from scimilarity import CellQuery
except ImportError:
    # Fallback - these may not be available in all environments
    align_dataset = None
    CellQuery = None


def _translate_gene_name(label: str) -> str:
    """
    Translates gene names in labels for DNA embedding lookup.
    Currently translates "AC118549.1" -> "ZZZ3".
    
    Args:
        label: Gene label in format "{gene_name}-{dataset}" or "{gene_name}-{batch}-{dataset}"
               or "{gene_name}+{other_genes}-{dataset}"
    
    Returns:
        Label with translated gene name, preserving all suffixes
    """
    if not label or label == 'control':
        return label
    
    # Handle multi-gene labels (e.g., "GENE1+GENE2-dataset")
    if '+' in label:
        # Split by '+' to get individual genes, then by '-' to separate from suffix
        parts = label.split('-', 1)
        if len(parts) == 2:
            gene_part, suffix = parts
            genes = gene_part.split('+')
            # Translate each gene (case-insensitive comparison)
            translated_genes = []
            for gene in genes:
                if gene.upper() == 'AC118549.1'.upper():
                    translated_genes.append('ZZZ3')
                else:
                    translated_genes.append(gene)
            return '+'.join(translated_genes) + '-' + suffix
        else:
            # No suffix, just genes
            genes = label.split('+')
            translated_genes = []
            for gene in genes:
                if gene.upper() == 'AC118549.1'.upper():
                    translated_genes.append('ZZZ3')
                else:
                    translated_genes.append(gene)
            return '+'.join(translated_genes)
    
    # Single gene label - parse format: {gene_name}-{dataset} or {gene_name}-{batch}-{dataset}
    parts = label.rsplit('-', 2)
    if len(parts) == 3 and parts[1].startswith('batch_'):
        # Format: gene-batch-dataset
        gene_name, batch_name, dataset_name = parts
        if gene_name.upper() == 'AC118549.1'.upper():
            gene_name = 'ZZZ3'
        return f'{gene_name}-{batch_name}-{dataset_name}'
    elif len(parts) == 2:
        # Format: gene-dataset
        gene_name, dataset_name = parts
        if gene_name.upper() == 'AC118549.1'.upper():
            gene_name = 'ZZZ3'
        return f'{gene_name}-{dataset_name}'
    elif len(parts) == 3:
        # Has 3 parts but second part doesn't start with "batch_" - treat as 2-part
        # This handles cases like "RBM14-RBM4-rpe1" where gene name has a hyphen
        gene_name = '-'.join(parts[:2])
        dataset_name = parts[2]
        # Check if the first part is AC118549.1
        if parts[0].upper() == 'AC118549.1'.upper():
            gene_name = 'ZZZ3' + '-' + parts[1] if len(parts) > 1 else 'ZZZ3'
        return f'{gene_name}-{dataset_name}'
    else:
        # Single gene name without suffix
        if label.upper() == 'AC118549.1'.upper():
            return 'ZZZ3'
        return label


def _should_filter_label(label: str) -> bool:
    """
    Returns True if label should be filtered out (excluded from training).
    Currently filters out labels containing "MTRNR2L1".
    
    Args:
        label: Gene label string
    
    Returns:
        True if label should be filtered out, False otherwise
    """
    if not label:
        return False
    # Case-insensitive check
    return 'MTRNR2L1' in label.upper()


def curate_perturb_tag_obs_only(adata, data_name):
    '''
    Adds curated perturbation labels to adata.obs without modifying the underlying data.
    Compatible with backed AnnData objects since it only modifies .obs columns.
    
    Parameters:
    -----------
    adata : AnnData
        Input data (can be backed) - will not be filtered or copied
    data_name : str
        Dataset name for labeling
        
    Returns:
    --------
    perturb_dict : dict
        Mapping of curated tags to raw entries
    adata : AnnData
        Original AnnData with new obs columns added (same object, not copied)
    '''
    
    perturb_dict = defaultdict(list)
    perturb_to_label = {}
    
    # Work with perturbation column, handling gene_target for HCT116
    # Avoid unnecessary copy() operations
    if 'HCT116_filtered_dual_guide_cells' in data_name and 'gene_target' in adata.obs.columns:
        perturbation_col = adata.obs['gene_target']
        perturbation_col = convert_to_ensembl(perturbation_col, suffix_token=0, verbose=True) ## convert to ensembl id
        perturbation_col = pd.Series(perturbation_col, index=adata.obs.index)
    elif 'jurkat' in data_name or 'k562' in data_name or 'rpe1' in data_name or 'combined_perturbqa_hvg2k' in data_name:
        perturbation_col = adata.obs['gene']
    else:
        perturbation_col = adata.obs['perturbation']
        perturbation_col = convert_to_ensembl(perturbation_col, suffix_token=0, verbose=True)
        perturbation_col = pd.Series(perturbation_col, index=adata.obs.index)
    
    # Get unique perturbations (only from non-null entries for dict creation)
    perturb_type = set(perturbation_col.dropna().values)
    
    # Ensure batch column exists
    if 'batch' not in adata.obs.columns:
        adata.obs['batch'] = '0'
    
    # Initialize output columns with appropriate default values
    adata.obs['perturbation-batch'] = ''
    adata.obs['curated_perturbation'] = ''
    
    if data_name == 'AdamsonWeissman2016_GSM2406675_10X001':
        batch_id = '0'
        batch_suffix = f'+batch_{batch_id}'
        
        # Build mapping dictionary
        for entry in perturb_type:
            if '62(mod)' in str(entry):
                label = f'control-{data_name}-batch_{batch_id}'
            else:
                if '_' in str(entry):
                    gene_name = str(entry).split('_')[0]
                    label = f'{gene_name}-{data_name}-batch_{batch_id}'
                else:
                    print(f"Error parsing entry: {entry}")
                    label = f'unknown-{data_name}-batch_{batch_id}'
            
            entry_with_batch = f'{entry}{batch_suffix}'
            perturb_dict[label].append(entry_with_batch)
            perturb_to_label[entry_with_batch] = label
        
        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + batch_suffix
        adata.obs['batch'] = batch_id

    elif data_name == 'NormanWeissman2019_filtered':
        if 'gemgroup' in adata.obs.columns:
            batch_ids = adata.obs['gemgroup'].unique()
            batch_col = adata.obs['gemgroup'].astype(str)
        else:
            batch_ids = ['0']
            batch_col = pd.Series(['0'] * len(adata.obs), index=adata.obs.index)
        
        # Build mapping dictionary
        for batch_id in batch_ids:
            batch_suffix = f'+batch_{batch_id}'
            for entry in perturb_type:
                entry_clean = str(entry).replace('_', '+')
                label = f'{entry_clean}-{data_name}-batch_{batch_id}'
                entry_with_batch = f'{entry}{batch_suffix}'
                perturb_dict[label].append(entry_with_batch)
                perturb_to_label[entry_with_batch] = label
        
        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + '+batch_' + batch_col
        adata.obs['batch'] = batch_col

    elif data_name == 'DixitRegev2016':
        batch_id = '0'
        batch_suffix = f'+batch_{batch_id}'
        
        # Build mapping dictionary
        for entry in perturb_type:
            components = str(entry).split('_')
            genes = set()
            is_control = True

            for comp in components:
                if 'INTERGENIC' in comp:
                    continue
                elif 'sg' in comp:
                    gene = comp.replace('sg', '').rsplit('-', 1)[0]
                    gene = gene.replace('p-', '')
                    genes.add(gene)
                    is_control = False

            if is_control:
                label = f'control-{data_name}-batch_{batch_id}'
            else:
                label = '+'.join(sorted(genes)) + f'-{data_name}-batch_{batch_id}'
            
            entry_with_batch = f'{entry}{batch_suffix}'
            perturb_dict[label].append(entry_with_batch)
            perturb_to_label[entry_with_batch] = label
        
        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + batch_suffix
        adata.obs['batch'] = batch_id
        
    elif data_name == 'LaraAstiasoHuntly2023_exvivo':
        # Create batch labels for each cell - compute once and reuse
        cell_phases = adata.obs['cellcycle_phase'].fillna('unknownPhase').astype(str)
        time_points = adata.obs['time'].fillna('unknownTime').astype(str)
        batch_labels = cell_phases + '_' + time_points
        unique_batch_labels = batch_labels.unique()  # Compute once
        
        # Build mapping dictionary
        for entry in perturb_type:
            for batch_label in unique_batch_labels:
                if pd.isna(entry) or 'nan' in str(entry).lower():
                    label = f'control-{data_name}-batch_{batch_label}'
                else:
                    gene_name = str(entry).split('_')[0]
                    label = f'{gene_name}-{data_name}-batch_{batch_label}'

                entry_with_batch = f'{entry}+batch_{batch_label}'
                perturb_dict[label].append(entry_with_batch)
                perturb_to_label[entry_with_batch] = label

        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + '+batch_' + batch_labels
        adata.obs['batch'] = batch_labels
        
    elif data_name in ['TianKampmann2021_CRISPRi', 'TianKampmann2021_CRISPRa', 'TianKampmann2019_day7neuron']:
        batch_ids = adata.obs['batch'].unique()  # Compute once
        
        # Build mapping dictionary
        for batch_id in batch_ids:
            batch_suffix = f'+batch_{batch_id}'
            for entry in perturb_type:
                label = f'{entry}-{data_name}-batch_{batch_id}'
                entry_with_batch = f'{entry}{batch_suffix}'
                perturb_dict[label].append(entry_with_batch)
                perturb_to_label[entry_with_batch] = label
        
        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + '+batch_' + adata.obs['batch'].astype(str)
        
    elif data_name == 'LaraAstiasoHuntly2023_leukemia':
        # Create batch labels for each cell - compute once and reuse
        cell_phases = adata.obs['cellcycle_phase'].fillna('unknownPhase').astype(str)
        samples = adata.obs['sample'].fillna('unknownTime').astype(str)
        batch_labels = cell_phases + '_' + samples
        unique_batch_labels = batch_labels.unique()  # Compute once
        
        # Build mapping dictionary
        for entry in perturb_type:
            for batch_label in unique_batch_labels:
                if pd.isna(entry) or 'nan' in str(entry).lower():
                    label = f'control-{data_name}-batch_{batch_label}'
                else:
                    gene_name = str(entry).split('_')[0]
                    label = f'{gene_name}-{data_name}-batch_{batch_label}'

                entry_with_batch = f'{entry}+batch_{batch_label}'
                perturb_dict[label].append(entry_with_batch)
                perturb_to_label[entry_with_batch] = label

        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + '+batch_' + batch_labels
        adata.obs['batch'] = batch_labels
        
    elif 'HCT116_filtered_dual_guide_cells' in data_name:
        batch_ids = adata.obs['sample'].unique()  # Compute once
        
        # Build mapping dictionary
        for batch_id in batch_ids:
            batch_suffix = f'+batch_{batch_id}'
            for entry in perturb_type:
                entry_clean = str(entry).replace('_', '+') 
                if entry=='Non-Targeting':
                    label = f'control-{data_name}-batch_{batch_id}'
                else:
                    label = f'{entry_clean}-{data_name}-batch_{batch_id}'
                entry_with_batch = f'{entry}{batch_suffix}'
                perturb_dict[label].append(entry_with_batch)
                perturb_to_label[entry_with_batch] = label
        
        # Apply to all cells using vectorized operations
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + '+batch_' + adata.obs['sample'].astype(str)
        adata.obs['batch'] = adata.obs['sample'].astype(str)
        
    elif data_name == 'ReplogleWeissman2022_rpe1':
        batch_ids = adata.obs['batch'].astype(str).unique()  # Compute once
        for batch_id in batch_ids:
            batch_suffix = f'+batch_{batch_id}'
            for entry in perturb_type:
                if entry=='non-targeting':
                    label = f'control-{data_name}-batch_{batch_id}'
                else:
                    label = f'{entry}-{data_name}-batch_{batch_id}'
                entry_with_batch = f'{entry}{batch_suffix}'
                perturb_dict[label].append(entry_with_batch)
                perturb_to_label[entry_with_batch] = label
        adata.obs['perturbation-batch'] = perturbation_col.astype(str) + '+batch_' + adata.obs['batch'].astype(str)
    
    elif data_name.lower() in ['jurkat', 'k562', 'rpe1', 'combined_perturbqa_hvg2k']:
        for entry in perturb_type:
            label = f'{entry}-{data_name}'
            perturb_dict[label].append(entry)
            perturb_to_label[entry] = label
        adata.obs['perturbation-batch'] = perturbation_col.astype(str)
        
    else:
        raise ValueError(f"Dataset: {data_name} QC is not implemented")

    # Map curated perturbation labels for all cells using vectorized operations
    adata.obs['curated_perturbation'] = adata.obs['perturbation-batch'].map(perturb_to_label)
    
    # Handle any unmapped entries (e.g., NaN perturbations)
    unmapped_mask = adata.obs['curated_perturbation'].isna()
    if unmapped_mask.any():
        print(f"Warning: {unmapped_mask.sum()} cells could not be mapped to curated perturbations")
        adata.obs.loc[unmapped_mask, 'curated_perturbation'] = 'unmapped'

    return perturb_dict, adata


class perturb_control_dataloader(Dataset):
    def __init__(self, namelist, perturbation=None, excld_perturbation=None, 
                 root_dir=None,
                 idx_paths=None,
                 h5_suffix='X_scimilarity.h5',
                 idx_suffix='trainsub.idx.csv',
                 aug=False,
                 whitening=False,
                 scale: Union[bool, StandardScaler] = False,
                 textemb=False,
                 suffix=None,
                 gene_le=None,
                 hvg_mask=None,
                 count_transform=False,
                 cache_size=3,
                 convert_to_ensembl=False,
                 seed: int = 42,
                 memory_policy: Literal['eager', 'lazy'] = 'lazy',
                 partition_by_perturbation: bool = False,
                 cell_line_mapping: Optional[Dict[str, int]] = None,
                 anndata_path: Optional[str] = None, 
                 h5_dataset_name: Optional[Union[str, Dict[str, str]]] = None,
                 anndata_dataset_name: Optional[Union[str, Dict[str, str]]] = None,
                 allow_unknown_cell_lines: bool = False):
        '''
        namelist: list of dataset names
        idx_paths: list of paths to index files (matching ScalablePerturbControlDataLoader format)
        h5_suffix: suffix for H5 files containing X_scimilarity embeddings
        idx_suffix: suffix for index CSV files
        convert_to_ensembl: if True, convert gene names in labels to Ensembl IDs. If False (default), 
                           keep labels in original format from curated_perturbation.
        labels are read as: {gene_name(s)}-{dataset_name}-batch_{batch_id}
        '''
        if root_dir is None:
            root_dir = PROCESSED_DATA_ROOT
        if idx_paths is not None:
            assert len(namelist) == len(idx_paths), "namelist and idx_paths must have same length"
        
        # Initialize caches for H5 files
        self.h5_cache: Dict[str, h5py.File] = {}
        self.cache_order: List[str] = []
        self.cache_size = cache_size
        
        self.namelist = namelist
        self.idx_paths = idx_paths if idx_paths else [root_dir] * len(namelist)
        self.root_dir = root_dir
        self.anndata_path = anndata_path
        self.h5_suffix = h5_suffix
        self.idx_suffix = idx_suffix
        self.suffix = suffix  # Keep for backward compatibility but prefer idx_paths
        
        self.h5_dataset_name = h5_dataset_name
        self.anndata_dataset_name = anndata_dataset_name
        
        # Store sample metadata for efficient access
        self.sample_table = []
        self.dataset_names = []  # Track which dataset each sample belongs to
        
        self.raw_data = []
        self.data = []
        self.raw_control = []
        self.control = []
        self.labels = []
        self.perturb_type = []
        self.text_emb = {}
        self.count_transform = count_transform
        self.hvg_mask = hvg_mask
        self.convert_to_ensembl = convert_to_ensembl
        
        # Initialize scaling-related attributes
        self.scale_flag = bool(scale)
        self._provided_scaler = scale if isinstance(scale, StandardScaler) else None
        self.seed = seed
        self.scaler = self._provided_scaler
        # Lazy-loading configuration and containers
        self.memory_policy = memory_policy
        self._h5_paths: Dict[str, str] = {}
        self._adata_paths: Dict[str, str] = {}
        self._h5_handles: Dict[str, h5py.File] = {}
        self._adata_handles: Dict[str, sc.AnnData] = {}
        self._global_pairs: List[Tuple[str, int, int]] = []  # (dataset_name, pert_row, ctrl_row)
        self._embed_indices_by_name: Dict[str, List[int]] = {}
        self.length = 0

        # DDP partitioning configuration
        self.partition_by_perturbation = partition_by_perturbation
        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Initialize cell_line tracking
        self.cell_lines = []
        self.cell_line_ids = None
        if cell_line_mapping is not None:
            # Convert keys to lowercase for case-insensitive matching
            self.cell_line_to_id = {k.lower(): v for k, v in cell_line_mapping.items()}
            self.num_cell_lines = len(self.cell_line_to_id)
            # Create reverse mapping for debugging
            self.id_to_cell_line = {v: k for k, v in self.cell_line_to_id.items()}
        else:
            # Default: assume single "unknown" cell line with id 0
            self.cell_line_to_id = {"unknown": 0}
            self.num_cell_lines = 1
            self.id_to_cell_line = {0: "unknown"}

        # Determine if using new idx_paths format or legacy suffix format
        use_idx_format = (idx_paths is not None) or (suffix and 'idx' in suffix)
        
        for i, name in enumerate(namelist):
            if use_idx_format:
                # New format: use idx_paths and h5_suffix
                idx_path = self.idx_paths[i]
                root_dir = self.root_dir
                idx_file = f'{idx_path}/{name}{self.idx_suffix}'
                
                h5_name = self._get_dataset_name_override(name, self.h5_dataset_name)
                h5_file = f'{root_dir}/{h5_name}/{h5_name}{self.h5_suffix}'
                
                
                anndata_base = self.anndata_path if self.anndata_path is not None else self.root_dir
                adata_name = self._get_dataset_name_override(name, self.anndata_dataset_name)
                adata_full_path = f'{anndata_base}/{adata_name}/{adata_name}.h5ad'
                
                if not os.path.exists(idx_file):
                    raise FileNotFoundError(f"Index file not found: {idx_file}")
                if not os.path.exists(h5_file):
                    raise FileNotFoundError(f"H5 file not found: {h5_file}")
                
                pd_idx = pd.read_csv(idx_file)  
                
                pert_indices = pd_idx['perturbation_index'].values
                control_indices = pd_idx['control_index'].values

                # Filter by perturbation if specified
                if perturbation is not None:
                    print(f"⚡️ Optimized subset loading for {name}...")
                    # Load AnnData in backed mode for filtering
                    adata_backed = sc.read(adata_full_path, backed='r')
                    # _, adata_backed = curate_perturb_tag_obs_only(adata_backed, data_name=name)
                    if 'curated_perturbation' in adata_backed.obs.columns:
                        labels_series = adata_backed.obs['curated_perturbation']
                    else:
                        # Fall back to local curation only when needed
                        _, adata = curate_perturb_tag_obs_only(adata_backed, data_name=name)
                        labels_series = adata_backed.obs['curated_perturbation']
                    
                    # Get labels for all perturbation indices
                    pert_labels = adata_backed.obs['curated_perturbation'].iloc[pert_indices]
                    pert_labels.index = pd_idx.index
                    
                    # Create mask for desired perturbations
                    mask = pd.Series(False, index=pd_idx.index)
                    for pert_name in perturbation:
                        mask |= pert_labels.str.contains(pert_name, na=False)
                    
                    # Filter the CSV to get subset
                    pd_idx_filtered = pd_idx[mask].copy()
                    adata_backed.file.close()
                else:
                    pd_idx_filtered = pd_idx.copy()

                pert_indices_final = pd_idx_filtered['perturbation_index'].values
                control_indices_final = pd_idx_filtered['control_index'].values
                
                # Hybrid default: Eager embeddings, lazy raw counts
                # 1) Eager load embeddings into RAM (per dataset)
                h5f = self._load_h5(h5_file)
                X_scimilarity_full = np.asarray(h5f['X_scimilarity'][:])
                cache_key = h5_file
                h5f.close()
                if cache_key in self.h5_cache:
                    del self.h5_cache[cache_key]
                if cache_key in self.cache_order:
                    self.cache_order.remove(cache_key)
                X_scim_pert = X_scimilarity_full[pert_indices_final]
                X_scim_control = X_scimilarity_full[control_indices_final]
                self.data.append(X_scim_pert)
                self.control.append(X_scim_control)
                # 2) Labels and types from backed AnnData without materializing X
                adata = sc.read(adata_full_path, backed='r')
                # _, adata = curate_perturb_tag_obs_only(adata, data_name=name)
                if 'curated_perturbation' in adata.obs.columns:
                    labels_series = adata.obs['curated_perturbation']
                else:
                    # Fall back to local curation only when needed
                    _, adata = curate_perturb_tag_obs_only(adata, data_name=name)
                    labels_series = adata.obs['curated_perturbation']
                labels_arr = np.array(adata.obs['curated_perturbation'].iloc[pert_indices_final].values)
                # if "perturbation_type" not in adata.obs.columns:
                ################################################################################
                ## at this point, let's hardcode the perturbation type to CRISPR-interference ##
                adata.obs['perturbation_type'] = "CRISPR-interference"
                ################################################################################
                types_arr = np.array(adata.obs['perturbation_type'].iloc[pert_indices_final].values)
                self.labels.append(labels_arr)
                self.perturb_type.append(types_arr)
                # Read cell_line column
                if 'cell_line' not in adata.obs.columns:
                    raise ValueError(f"'cell_line' column not found in {adata_full_path}")
                cell_lines_arr = np.array(adata.obs['cell_line'].iloc[pert_indices_final].values)
                # Check for NaN values
                nan_mask = pd.isna(cell_lines_arr)
                if nan_mask.any():
                    raise ValueError(f"Found {nan_mask.sum()} NaN values in cell_line column for {name}")
                self.cell_lines.append(cell_lines_arr)
                # 3) Build global pairs and store adata path for lazy raw per-row
                self._adata_paths[name] = adata_full_path
                for pr, cr in zip(pert_indices_final, control_indices_final):
                    self._global_pairs.append((name, int(pr), int(cr)))
                # Track dataset names aligned with pairs
                self.dataset_names.extend([name] * len(pert_indices_final))
                adata.file.close()
                
            else:
                # Legacy format: use suffix (for backward compatibility)
                if suffix is None:
                    suffix = 'scim_ot_match.train'
                    
                if 'idx' in suffix:
                    adata_full_path = f'{root_dir}{name}.full.scim.zero.h5ad'
                    pd_idx = pd.read_csv(f'{root_dir}{name}_{suffix}.csv')

                    

                    pert_indices = pd_idx['perturbation_index'].values
                    control_indices = pd_idx['control_index'].values

                    if perturbation is not None:
                        print(f"⚡️ Optimized subset loading for {name}...")
                        adata_backed = sc.read(adata_full_path, backed='r')
                        pert_labels = adata_backed.obs['curated_perturbation'].iloc[pert_indices]
                        pert_labels.index = pd_idx.index
                        
                        mask = pd.Series(False, index=pd_idx.index)
                        for pert_name in perturbation:
                            mask |= pert_labels.str.contains(pert_name, na=False)
                        
                        pd_idx_filtered = pd_idx[mask].copy()
                        adata_backed.file.close()
                    else:
                        pd_idx_filtered = pd_idx.copy()

                    pert_indices_final = pd_idx_filtered['perturbation_index'].values
                    control_indices_final = pd_idx_filtered['control_index'].values
                    
                    # Legacy idx path: embeddings live in obsm, no H5
                    adata = sc.read(adata_full_path, backed='r')
                    _, adata = curate_perturb_tag_obs_only(adata, data_name=name)
                    # Eager embeddings from obsm
                    X_scim_pert = adata.obsm['X_scimilarity'][pert_indices_final]
                    X_scim_control = adata.obsm['X_scimilarity'][control_indices_final]
                    self.data.append(X_scim_pert)
                    self.control.append(X_scim_control)
                    # Labels/types
                    labels_arr = np.array(adata.obs['curated_perturbation'].iloc[pert_indices_final].values)
                    # if "perturbation_type" not in adata.obs.columns:
                    ################################################################################
                    ## at this point, let's hardcode the perturbation type to CRISPR-interference ##
                    adata.obs['perturbation_type'] = "CRISPR-interference"
                    ################################################################################
                    
                    types_arr = np.array(adata.obs['perturbation_type'].iloc[pert_indices_final].values)
                    self.labels.append(labels_arr)
                    self.perturb_type.append(types_arr)
                    # Read cell_line column
                    if 'cell_line' not in adata.obs.columns:
                        raise ValueError(f"'cell_line' column not found in {adata_full_path}")
                    cell_lines_arr = np.array(adata.obs['cell_line'].iloc[pert_indices_final].values)
                    # Check for NaN values
                    nan_mask = pd.isna(cell_lines_arr)
                    if nan_mask.any():
                        raise ValueError(f"Found {nan_mask.sum()} NaN values in cell_line column for {name}")
                    self.cell_lines.append(cell_lines_arr)
                    # Pairs and adata path
                    self._adata_paths[name] = adata_full_path
                    for pr, cr in zip(pert_indices_final, control_indices_final):
                        self._global_pairs.append((name, int(pr), int(cr)))
                    self.dataset_names.extend([name] * len(pert_indices_final))
                    adata.file.close()
                else:
                    # Non-idx suffix format
                    adata = sc.read(f'{root_dir}{name}_{suffix}.h5ad')
                    self.labels.append(np.array(adata.obs['curated_perturbation'].values))
                    self.data.append(adata.X)
                    self.control.append(adata.obsm['X_control'])
                    with open(f'{root_dir}/perturb_text_emb/{name}_desc_dict.pkl', 'rb') as f:
                        self.text_emb |= pickle.load(f)
                    if "perturbation_type" not in adata.obs.columns:
                        adata.obs['perturbation_type'] = "CRISPR-interference"
                    self.perturb_type.append(adata.obs['perturbation_type'].values)
                    # Read cell_line column
                    if 'cell_line' not in adata.obs.columns:
                        raise ValueError(f"'cell_line' column not found in {root_dir}{name}_{suffix}.h5ad")
                    cell_lines_arr = np.array(adata.obs['cell_line'].values)
                    # Check for NaN values
                    nan_mask = pd.isna(cell_lines_arr)
                    if nan_mask.any():
                        raise ValueError(f"Found {nan_mask.sum()} NaN values in cell_line column for {name}")
                    self.cell_lines.append(cell_lines_arr)
                    
                    n_samples = len(adata)
                    self.dataset_names.extend([name] * n_samples)

        # Concatenate embeddings and labels (eager embeddings)
        if len(self.data) > 0:
            print(f"Data shape: {self.data[0].shape}")
            self.data = np.concatenate(self.data, axis=0)
            self.control = np.concatenate(self.control, axis=0)
            # labels/types were appended per-dataset as arrays
            self.labels = np.concatenate(self.labels, axis=0)
            self.perturb_type = np.concatenate(self.perturb_type, axis=0)
            # Concatenate cell_lines if available
            if len(self.cell_lines) > 0:
                self.cell_lines = np.concatenate(self.cell_lines, axis=0)
        else:
            raise ValueError("No data loaded")
        
        # Validate and map cell_lines using config mapping
        if self.cell_line_to_id is not None and len(self.cell_lines) > 0:
            # Check if using default "unknown" mapping
            is_default_mapping = (self.cell_line_to_id == {"unknown": 0})
            
            if is_default_mapping:
                # Default case: assign all cells to id 0 regardless of actual cell_line value
                self.cell_line_ids = np.zeros(len(self.cell_lines), dtype=np.int64)
                print(f'Using default cell line mapping: all cells assigned to "unknown" (id=0)')
            else:
                # Normal case: validate and map using provided mapping
                # Convert to lowercase for case-insensitive matching
                cell_lines_lower = []
                for cl in self.cell_lines:
                    if pd.isna(cl):
                        raise ValueError(f"Found NaN value in cell_line column. All cell lines must be specified.")
                    cell_lines_lower.append(str(cl).lower())
                cell_lines_lower = np.array(cell_lines_lower, dtype=object)
                
                
                # Check for any cell_lines not in config
                unique_cell_lines = set(cell_lines_lower)
                unknown_cell_lines = unique_cell_lines - set(self.cell_line_to_id.keys())
                if unknown_cell_lines:
                    if not allow_unknown_cell_lines:
                        raise ValueError(f"Found unknown cell_lines not in config: {unknown_cell_lines}")
                    # Map all unknowns to an arbitrary seen id (first key), since model will use mix weights
                    warnings.warn(
                        f"Unknown cell_lines {unknown_cell_lines} -> mapping to first known cell line "
                        f"(override with mix weights at inference). Found {unknown_cell_lines} in data."
                    )
                    # Map unknowns to first known key
                    fallback_name = next(iter(self.cell_line_to_id.keys()))
                    mask = np.isin(cell_lines_lower, list(unknown_cell_lines))
                    cell_lines_lower[mask] = fallback_name
                
                # Pre-compute cell_line_ids using config mapping
                self.cell_line_ids = np.array([self.cell_line_to_id[cl] for cl in cell_lines_lower], dtype=np.int64)
                print(f'Cell line mapping: {self.id_to_cell_line}')
                print(f'Found {len(np.unique(self.cell_line_ids))} unique cell lines in data')
        
        # Apply gene name translation (AC118549.1 -> ZZZ3)
        self.labels = np.array([_translate_gene_name(str(lbl)) for lbl in self.labels])
        
        # Apply filtering (exclude MTRNR2L1) — filter embeddings and metadata; raw is lazy
        keep_mask = np.array([not _should_filter_label(str(lbl)) for lbl in self.labels])
        n_filtered = (~keep_mask).sum()
        if n_filtered > 0:
            print(f'Filtering out {n_filtered} samples with MTRNR2L1')
            self.data = self.data[keep_mask]
            self.control = self.control[keep_mask]
            self.labels = self.labels[keep_mask]
            self.perturb_type = self.perturb_type[keep_mask]
            if self.cell_line_ids is not None:
                self.cell_line_ids = self.cell_line_ids[keep_mask]
            if hasattr(self, 'dataset_names'):
                self.dataset_names = [self.dataset_names[i] for i in range(len(keep_mask)) if keep_mask[i]]
            # Filter global pairs to keep alignment with embeddings
            self._global_pairs = [self._global_pairs[i] for i in range(len(keep_mask)) if keep_mask[i]]
        
        print(f'perturbation labels are: {np.unique(self.labels)}')
        print(f'perturb_type are: {np.unique(self.perturb_type)}')
        
        # Determine RNA_dim (raw is lazy; only shape)
        if self.hvg_mask is not None:
            self.RNA_dim = int(self.hvg_mask.sum())
        else:
            if len(self._adata_paths) > 0:
                first = next(iter(self._adata_paths.values()))
                ad = sc.read(first, backed='r')
                self.RNA_dim = int(ad.shape[1])
                ad.file.close()
            else:
                self.RNA_dim = 0
        
        # Keep textemb and gene_le for backward compatibility (not used in output)
        self.textemb = textemb
        self.gene_le = gene_le
        if not textemb and gene_le is not None:
            # Store encode_labels for backward compatibility if needed
            self.encode_labels = gene_le.transform(self.labels.reshape(-1, 1))
        self.length = len(self.labels)
        
        # Process labels (convert gene names to ensembl if requested)
        if self.convert_to_ensembl:
            labels = pd.Series(self.labels)
            split_labels = labels.str.split('-', n=1, expand=True)
            gene_names = split_labels[0]
            suffix = split_labels[1]
            
            # Convert gene names to Ensembl
            labels = pd.Series(convert_to_ensembl(gene_names, suffix_token=0))
            
            # Handle different suffix formats row by row
            def reconstruct_label(row):
                ensembl_id, suff = row[0], row[1]
                if pd.isna(suff):
                    # Format: {gene_name} (no suffix)
                    return ensembl_id
                else:
                    # Split suffix to get parts
                    suff_parts = str(suff).split('-')
                    if len(suff_parts) >= 2:
                        # Format: {gene_name}-{dataset_name}-batch_{batch_id}
                        return f'{ensembl_id}-{suff_parts[0]}-{suff_parts[1]}'
                    elif len(suff_parts) == 1:
                        # Format: {gene_name}-{dataset_name}
                        return f'{ensembl_id}-{suff_parts[0]}'
                    else:
                        # Shouldn't happen, but handle gracefully
                        return ensembl_id
            
            labels = pd.DataFrame({'ensembl': labels, 'suffix': suffix}).apply(reconstruct_label, axis=1)
            self.labels = labels.values
            print(f'labels are: {self.labels[:10]}')
        else:
            # Keep labels as-is (already in curated_perturbation format)
            print(f'labels are: {self.labels[:10]}')
        
        # Pre-factorize labels to avoid DDP object collectives during training
        # Build label-to-ID mapping once (no DDP needed here)
        all_label_strings = [str(label) for label in self.labels]
        unique_labels = sorted(set(all_label_strings))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = unique_labels
        
        # Pre-compute integer IDs for all samples
        self.label_ids = np.array([self.label_to_id[str(label)] for label in self.labels], dtype=np.int64)
        
        print(f'Pre-factorized {len(unique_labels)} unique labels into integer IDs (0-{len(unique_labels)-1})')
        
        
        # --- DDP label-based sharding (true per-perturbation partition) ---
        if self.partition_by_perturbation and self.world_size > 1:
            uniq_ids = np.unique(self.label_ids)
            chunks = np.array_split(uniq_ids, self.world_size)
            my_ids = set(chunks[self.rank].tolist())

            # Mask: keep only samples whose label_id belongs to this rank
            mask = np.isin(self.label_ids, list(my_ids))

            kept = int(mask.sum())
            total = int(mask.shape[0])
            uniq_local = int(np.unique(self.label_ids[mask]).shape[0])
            print(f"[Rank {self.rank}] final shard: kept {kept}/{total} samples, "
                f"{uniq_local} unique perts out of {len(uniq_ids)} total")

            # Apply mask to all per-sample arrays / lists
            self.data        = self.data[mask]
            self.control     = self.control[mask]
            self.labels      = self.labels[mask]
            self.perturb_type = self.perturb_type[mask]
            self.label_ids   = self.label_ids[mask]
            if self.cell_line_ids is not None:
                self.cell_line_ids = self.cell_line_ids[mask]

            # _global_pairs and dataset_names are Python lists; index them manually
            self._global_pairs = [self._global_pairs[i] for i in range(total) if mask[i]]
            if hasattr(self, 'dataset_names'):
                self.dataset_names = [self.dataset_names[i] for i in range(total) if mask[i]]

            self.length = len(self.labels)
        else:
            self.length = len(self.labels)
        # --- end DDP sharding ---

        
        # Fit scaler on entire loaded data if needed
        if self.scale_flag and self._provided_scaler is None:
            # Fit scaler on combined data (both perturb and control embeddings)
            if dist.is_initialized() and self.partition_by_perturbation:
                # When partitioning, each GPU only has a subset of data
                # Use distributed all-reduce to compute global statistics
                self.scaler = self._fit_scaler_distributed()
            elif dist.is_initialized() and not self.partition_by_perturbation:
                # Original logic: fit on rank 0, broadcast to others
                rank = dist.get_rank()
                if rank == 0:
                    # Combine all embeddings for fitting
                    all_embeddings = np.vstack([self.data, self.control])
                    self.scaler = StandardScaler()
                    self.scaler.fit(all_embeddings)
                    
                    scaler_params = {
                        'mean_': self.scaler.mean_,
                        'scale_': self.scaler.scale_,
                        'var_': self.scaler.var_,
                        'n_samples_seen_': self.scaler.n_samples_seen_
                    }
                else:
                    scaler_params = None
                
                broadcast_list = [scaler_params]
                dist.broadcast_object_list(broadcast_list, src=0)
                scaler_params = broadcast_list[0]
                
                if rank != 0:
                    self.scaler = StandardScaler()
                    for key, value in scaler_params.items():
                        setattr(self.scaler, key, value)
            else:
                # Single GPU: standard fit
                # Combine all embeddings for fitting
                all_embeddings = np.vstack([self.data, self.control])
                self.scaler = StandardScaler()
                self.scaler.fit(all_embeddings)
        
        # Apply scaling to entire arrays at once if scaler is available
        if self.scaler is not None:
            self.data = self.scaler.transform(self.data)
            self.control = self.scaler.transform(self.control)
        
        # No whitening (data already normalized)
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def _manage_cache(self, dataset_name: str):
        """Manage H5 file cache with LRU eviction"""
        if dataset_name in self.cache_order:
            self.cache_order.remove(dataset_name)
        elif len(self.cache_order) >= self.cache_size:
            old = self.cache_order.pop(0)
            if old in self.h5_cache:
                self.h5_cache[old].close()
                del self.h5_cache[old]
        self.cache_order.append(dataset_name)

    def _load_h5(self, h5_path: str) -> h5py.File:
        """Load H5 file containing X_scimilarity data with caching"""
        # Use path as cache key
        cache_key = h5_path
        
        if cache_key in self.h5_cache:
            self._manage_cache(cache_key)
            h5f = self.h5_cache[cache_key]
            # Check if file is still valid
            try:
                _ = h5f["X_scimilarity"].shape
                return h5f
            except (OSError, KeyError, RuntimeError):
                # File might be corrupted or closed, remove from cache and reload
                print(f"Warning: HDF5 file {cache_key} appears to be corrupted or closed, reloading...")
                if cache_key in self.h5_cache:
                    try:
                        self.h5_cache[cache_key].close()
                    except:
                        pass
                    del self.h5_cache[cache_key]
                if cache_key in self.cache_order:
                    self.cache_order.remove(cache_key)
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        try:
            h5f = h5py.File(h5_path, 'r')
            # Verify the file has the expected dataset
            if "X_scimilarity" not in h5f:
                h5f.close()
                raise KeyError(f"HDF5 file {h5_path} does not contain 'X_scimilarity' dataset")
            
            self._manage_cache(cache_key)
            self.h5_cache[cache_key] = h5f
            return h5f
        except Exception as e:
            print(f"Error loading HDF5 file {h5_path}: {e}")
            raise e

    def _concat_sparse(self, mat_list, keep_sparse=True):
        if len(mat_list) == 0:
            return None
        if keep_sparse:
            from scipy.sparse import vstack
            return vstack(mat_list)
        else:
            return np.concatenate([m.toarray() if hasattr(m, 'toarray') else m for m in mat_list], axis=0)

    def __len__(self):
        return self.length if self.length else len(self.data)

    def __getitem__(self, idx):
        # Eager embeddings from RAM
        z_0 = torch.tensor(self.data[idx], dtype=torch.float32)
        z_ctrl = torch.tensor(self.control[idx], dtype=torch.float32)
        # Lazy raw counts per-row from AnnData
        name, pr, cr = self._global_pairs[idx]
        if name not in self._adata_handles and name in self._adata_paths:
            self._adata_handles[name] = sc.read(self._adata_paths[name], backed='r')
        ad = self._adata_handles.get(name, None)
        if ad is None:
            raise RuntimeError(f"Backed AnnData handle missing for dataset {name}")
        x_0 = ad.X[pr]
        x_ctrl = ad.X[cr]
        x_0 = x_0.toarray().ravel() if scipy.sparse.issparse(x_0) else np.asarray(x_0).ravel()
        x_ctrl = x_ctrl.toarray().ravel() if scipy.sparse.issparse(x_ctrl) else np.asarray(x_ctrl).ravel()
        if self.hvg_mask is not None:
            x_0 = x_0[self.hvg_mask]
            x_ctrl = x_ctrl[self.hvg_mask]
        x_0 = torch.tensor(x_0, dtype=torch.float32)
        x_ctrl = torch.tensor(x_ctrl, dtype=torch.float32)
        label = str(self.labels[idx])
        label_id = int(self.label_ids[idx])  # Pre-computed integer ID
        perturb_type = str(self.perturb_type[idx])
        dataset_name = self.dataset_names[idx] if hasattr(self, 'dataset_names') and idx < len(self.dataset_names) else self.namelist[0]
        cell_line_id = int(self.cell_line_ids[idx]) if self.cell_line_ids is not None else None
        return {
            'perturb': z_0, 
            'control': z_ctrl, 
            'perturb_raw': x_0, 
            'control_raw': x_ctrl,
            'label': label,
            'label_id': label_id,  # Pre-factorized integer ID for fast weighting
            'perturb_type': perturb_type,
            'dataset_name': dataset_name,
            'cell_line_id': cell_line_id  # New field integer ids
        }
    
    def __getstate__(self):
        """
        Called when pickling the object to send to workers.
        We must remove the open h5py handles from the state.
        """
        state = self.__dict__.copy()
        # Reset caches to empty dicts for the pickled state
        state['h5_cache'] = {}
        # Do not pickle lazy handles
        if '_h5_handles' in state:
            state['_h5_handles'] = {}
        if '_adata_handles' in state:
            state['_adata_handles'] = {}
        return state

    def __setstate__(self, state):
        """
        Called in the new worker process to restore state.
        """
        self.__dict__.update(state)
        # Ensure caches are initialized empty in the new worker
        self.h5_cache = {}
        self.cache_order = []
        if not hasattr(self, '_h5_handles'):
            self._h5_handles = {}
        if not hasattr(self, '_adata_handles'):
            self._adata_handles = {}
    
    def __del__(self):
        """Clean up H5 files when dataloader is destroyed"""
        for h5f in self.h5_cache.values():
            try:
                h5f.close()
            except:
                pass
        # Close lazy handles too
        try:
            for h in getattr(self, '_h5_handles', {}).values():
                try:
                    h.close()
                except:
                    pass
            for a in getattr(self, '_adata_handles', {}).values():
                try:
                    a.file.close()
                except:
                    pass
        except Exception:
            pass

    def _fit_scaler_lazy(self, batch_size: int = 4096):
        """
        Fit StandardScaler in a streaming way over H5 embeddings referenced by indices.
        Rank-0 fits and broadcasts to other ranks if using torch.distributed.
        """
        # Deprecated in hybrid mode; embeddings are eager so use eager scaler fit downstream
        pass

    def _fit_scaler_distributed(self):
        """
        Compute global mean and variance across all DDP shards manually
        to create a unified StandardScaler.
        """
        # 1. Local data concatenation
        local_data = np.vstack([self.data, self.control])
        n_samples = torch.tensor([local_data.shape[0]], device='cuda')
        
        # 2. Local Sum and Sum Squares
        local_sum = torch.tensor(local_data.sum(axis=0), device='cuda')
        local_sq_sum = torch.tensor((local_data ** 2).sum(axis=0), device='cuda')
        
        # 3. Global All-Reduce
        dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
        
        # 4. Compute Global Stats
        total_n = n_samples.item()
        global_mean = (local_sum / total_n).cpu().numpy()
        global_sq_mean = (local_sq_sum / total_n).cpu().numpy()
        global_var = global_sq_mean - (global_mean ** 2)
        
        # 5. Create Scaler
        scaler = StandardScaler()
        scaler.mean_ = global_mean
        scaler.var_ = global_var
        scaler.scale_ = np.sqrt(global_var)
        scaler.n_samples_seen_ = total_n
        
        if self.rank == 0:
            print(f"Distributed Scaler fitted on {total_n} total samples.")
            
        return scaler
    
    def _get_dataset_name_override(self, original_name: str, override: Optional[Union[str, Dict[str, str]]]) -> str:
        """
        Helper method to resolve dataset name override.
        
        Args:
            original_name: Original dataset name from namelist
            override: None, str (applies to all), or dict (name -> override mapping)
        
        Returns:
            Override name if provided, otherwise original_name
        """
        if override is None:
            return original_name
        elif isinstance(override, str):
            return override
        elif isinstance(override, dict):
            return override.get(original_name, original_name)
        else:
            return original_name


class ScalablePerturbControlDataLoader(IterableDataset):
    def __init__(
        self,
        namelist: List[str],
        idx_paths: List[str],
        root_dir: str = PROCESSED_DATA_ROOT,
        batch_size: int = 32,
        shuffle: bool = True,
        scale: Union[bool, StandardScaler] = True,
        textemb: bool = False,
        gene_le: Optional[OneHotEncoder] = None,
        hvg_mask: Optional[np.ndarray] = None,
        count_transform: bool = True,
        perturb_type_le: Optional[LabelEncoder] = None,
        cache_size: int = 3,
        seed: int = 42,
        scaler_sample_size: int = 10000,
        emb_model_path: str = SCIMILARITY_MODEL_PATH,
        h5_suffix: str = 'X_scimilarity.h5',  # New parameter for h5 file suffix
        idx_suffix: str = 'trainsub.idx.csv',
        hvg_thresh: float = 1e-2,
        drop_last: bool = False,
        dna_record=None,
        gene_annot=None,
        ccre_intervals_dict=None,
        jitter_window=0,
        promoter_len=200,
        drop_mask_prob=0.0,
    ):
        assert len(namelist) == len(idx_paths), "namelist and idx_paths must have same length"
        self.h5_cache: Dict[str, h5py.File] = {}  # New cache for h5 files
        self.namelist = namelist
        self.idx_paths = idx_paths
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.textemb = textemb
        self.gene_le = gene_le
        self.hvg_mask = hvg_mask
        self.count_transform = count_transform
        self.cache_size = cache_size
        self.seed = seed
        self.emb_model_path = emb_model_path
        self.h5_suffix = h5_suffix
        self.idx_suffix = idx_suffix
        # Scaling
        self.scale_flag = bool(scale)
        self._provided_scaler = scale if isinstance(scale, StandardScaler) else None
        self.scaler_sample_size = scaler_sample_size
        self.scaler = self._provided_scaler
        self.reorder_map = {}
        # Label encoder
        self.perturb_type_le = perturb_type_le
        self.gene_order = pd.read_csv(f'{emb_model_path}/geneID_order_converted.tsv', header=None).values.flatten()
        self.RNA_dim = len(self.gene_order)
        # Metadata
        self.dataset_info = []
        self.total_samples = 0
        self._initialize_metadata()
        self.collapse_mat = {}
        # Caches
        self.adata_cache: Dict[str, sc.AnnData] = {}
       
        self.cache_order: List[str] = []
        self.text_emb_cache: Dict[str, Dict] = {}
        self.gRNAs = {}
        self.hvg_thresh = hvg_thresh
        
        self.dna_record = dna_record
        self.gene_annot = gene_annot
        self.ccre_intervals_dict = ccre_intervals_dict
        self.jitter_window = jitter_window
        self.promoter_len = promoter_len
        self.drop_mask_prob = drop_mask_prob
        # Fit scaler on random control samples if needed
        if self.scale_flag and self._provided_scaler is None:
            self._fit_scaler()
        
        
        self.gene_sd = None
        
        
        if self.hvg_mask is None:                      # allow user to override
            # keep genes whose SD is above a quantile threshold
            thresh = self.hvg_thresh
            self._estimate_gene_disp()
            self.hvg_mask = self.hvg_score > thresh
            
            self.gene_sd    = self.gene_sd[self.hvg_mask]
            
        
            print(f'{self.hvg_mask.sum()} genes out of {len(self.gene_sd)} are kept after hvg_mask')
        self.RNA_dim    = int(self.hvg_mask.sum())

        
        
        self.sample_table = []
        for di, info in enumerate(self.dataset_info):
            df = pd.read_csv(f'{info["idx_path"]}/{info["name"]}{self.idx_suffix}')
            self.sample_table.extend(
                [(di, info['idx_path'], df['perturbation_index'].iloc[i], df['control_index'].iloc[i]) for i in range(info['n_samples'])]
            )
            
        self.drop_last = drop_last

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        

    def _initialize_metadata(self):
        for name, idx_path in zip(self.namelist, self.idx_paths):
            if not os.path.exists(idx_path):
                raise FileNotFoundError(f"Index file not found: {idx_path}")
            print(f'{idx_path}/{name}{self.idx_suffix}')
            df = pd.read_csv(f'{idx_path}/{name}{self.idx_suffix}')
            n = len(df)
            self.dataset_info.append({
                'name': name, 
                'idx_path': idx_path, 
                'n_samples': n, 
                'cumsum': self.total_samples
            })
            self.total_samples += n
            
            
    def _sample_gene_vectors(self, mode: str = "both") -> np.ndarray:
        """
        Return (≤ scaler_sample_size, G) log1p-normalized gene vectors aligned to self.gene_order.
        mode: "perturb" | "control" | "both"
        """
        assert mode in {"perturb", "control", "both"}
        rng = np.random.RandomState(self.seed)
        picked = []

        for info in self.dataset_info:
            idx_path, name, n = info['idx_path'], info['name'], info['n_samples']
            k = int(self.scaler_sample_size * (n / self.total_samples)) + 1
            df = pd.read_csv(f'{idx_path}/{name}{self.idx_suffix}')
            k = min(k, len(df))
            rows = rng.choice(len(df), size=k, replace=False)

            for j in rows:
                p = int(df['perturbation_index'].iloc[j])
                c = int(df['control_index'].iloc[j])
                if mode in {"perturb", "both"}:
                    picked.append((name, idx_path, p, "perturb"))
                if mode in {"control", "both"}:
                    picked.append((name, idx_path, c, "control"))

        rng.shuffle(picked)
        picked = picked[: self.scaler_sample_size]

        vecs = []
        for ds_name, idx_path, abs_idx, _kind in picked:
            ad = self._load_adata(ds_name)  # ensures collapse_mat built
            row = ad[abs_idx].X
            row = row.toarray().squeeze() if scipy.sparse.issparse(row) else np.asarray(row)
            row = self.collapse_mat[ds_name].dot(np.asarray(row).reshape(-1))

            if self.count_transform:
                s = row.sum()
                if s > 0:
                    row = row / s * 1e4
                row = np.log1p(row)
                

            vecs.append(row)

        if not vecs:
            raise RuntimeError("No rows sampled for HVG estimation.")
        return np.vstack(vecs)
    
    
    def _binwise_zscore(self, mean_vec, disp_vec, n_bins=20, robust=True):
        """
        Seurat/Scanpy 'seurat' flavor:
        - bin genes by mean
        - standardize dispersion within each mean bin
        Returns: z_disp (same shape as disp_vec)
        """
        eps = 1e-8
        # define bins on mean; use quantiles for roughly equal bin sizes
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(mean_vec, qs)
        # ensure strictly increasing edges (guard rare ties)
        edges = np.unique(edges)
        if len(edges) < 3:  # degenerate case: fall back to single-bin z
            m = np.median(disp_vec)
            s = np.median(np.abs(disp_vec - m)) + eps if robust else (disp_vec.std() + eps)
            return (disp_vec - m) / s

        bin_ids = np.digitize(mean_vec, edges[1:-1], right=False)  # 0..(len(edges)-2)

        z = np.zeros_like(disp_vec)
        for b in np.unique(bin_ids):
            idx = (bin_ids == b)
            if idx.sum() < 20:  # tiny bin; skip or copy raw centered
                m = np.median(disp_vec[idx]) if robust else disp_vec[idx].mean()
                s = (np.median(np.abs(disp_vec[idx] - m)) + eps) if robust else (disp_vec[idx].std() + eps)
                z[idx] = (disp_vec[idx] - m) / s
                continue
            if robust:
                m = np.median(disp_vec[idx])
                s = np.median(np.abs(disp_vec[idx] - m)) + eps
            else:
                m = disp_vec[idx].mean()
                s = disp_vec[idx].std() + eps
            z[idx] = (disp_vec[idx] - m) / s
        return z
    
    def _estimate_gene_disp(self, n_bins=20, robust=True, n_top_genes=None, top_quantile=0.10):
        """
        Compute per-gene mean/variance on log1p(CPM) data (controls+pert, same as now if you switched).
        Then compute Seurat/Scanpy 'seurat' flavor mean-corrected dispersion and store:
        - self.gene_mu           (mean in log1p space)
        - self.gene_var          (variance in log1p space)
        - self.gene_sd           (sqrt(var), for compatibility)
        - self.hvg_score         (bin-standardized dispersion)
        Finally set self.hvg_mask from top-K or top-quantile of hvg_score.
        """
        B = 256
        # If you already added the "both" sampler earlier, use it. Otherwise keep your current sampler.
        sample_mat = self._sample_gene_vectors(mode="both")  # ← surgical: combined cells
        n, G = sample_mat.shape

        # Welford running stats (same as before)
        mu  = np.zeros(G, dtype=np.float64)
        M2  = np.zeros(G, dtype=np.float64)
        cnt = 0

        for i in range(0, n, B):
            X = sample_mat[i : i + B]        # (b, G), already log1p(CPM)
            b = X.shape[0]
            cnt_new = cnt + b
            delta   = X.mean(0) - mu
            mu     += delta * b / cnt_new
            M2     += ((X - mu)**2).sum(0) + (delta**2) * cnt * b / cnt_new
            cnt     = cnt_new

        var = M2 / (cnt - 1 + 1e-8)
        sd  = np.sqrt(np.clip(var, 1e-12, None))

        # --- Seurat/Scanpy 'seurat' flavor dispersion (on log1p space) ---
        eps = 1e-8
        disp = var / (mu + eps)   # matches Scanpy flavor='seurat' (variance/mean) after log1p
        # bin-wise standardization
        z_disp = self._binwise_zscore(mu, disp, n_bins=n_bins, robust=robust)

        # store
        self.gene_mu    = mu
        self.gene_var   = var
        self.gene_sd    = sd
        self.hvg_score  = z_disp
    
    
    def _estimate_gene_sd(self):
        """
        Compute per‑gene standard deviation on a random subset of perturbation cells
        using numerically stable one‑pass Welford updates.  Stores result in
        self.gene_sd (shape: G,).
        """
        B = 256                               # process in mini‑chunks to limit RAM
        # sample_mat = self._sample_perturb_vectors()
        sample_mat = self._sample_gene_vectors(mode="both")
        n, G = sample_mat.shape

        mu  = np.zeros(G, dtype=np.float64)
        M2  = np.zeros(G, dtype=np.float64)
        cnt = 0

        for i in range(0, n, B):
            X = sample_mat[i : i + B]         # (b, G)
            b = X.shape[0]
            cnt_new = cnt + b
            delta   = X.mean(0) - mu
            mu     += delta * b / cnt_new
            M2     += ((X - mu)**2).sum(0) + (delta**2) * cnt * b / cnt_new
            cnt     = cnt_new

        var = M2 / (cnt - 1)
        self.gene_sd = np.sqrt(np.clip(var, 1e-8, None))   # avoid exact zeros
        self.gene_mu = mu
        
    def _sample_pooled_h5_vectors(self) -> np.ndarray:
        """
        Return an ndarray of shape (≤ scaler_sample_size, D) where each row is an
        X_scimilarity embedding taken from BOTH perturbation and control indices,
        proportionally across datasets. This matches the model input space.
        """
        rng = np.random.RandomState(self.seed)
        picked = []

        # 1) proportional sampling of PAIRS per dataset
        for info in self.dataset_info:
            idx_path, name, n = info['idx_path'], info['name'], info['n_samples']
            k_pairs = int(self.scaler_sample_size * (n / self.total_samples)) + 1
            df = pd.read_csv(f'{idx_path}/{name}{self.idx_suffix}')
            k_pairs = min(k_pairs, len(df))
            chosen_rows = rng.choice(len(df), size=k_pairs, replace=False)

            # collect BOTH perturbation and control indices
            for j in chosen_rows:
                p = int(df['perturbation_index'].iloc[j])
                c = int(df['control_index'].iloc[j])
                picked.append((name, idx_path, p))
                picked.append((name, idx_path, c))

        # 2) flatten, shuffle, dedupe, hard cap
        rng.shuffle(picked)
        # dedupe by (dataset, abs_index)
        seen = set()
        uniq = []
        for ds_name, idx_path, abs_idx in picked:
            key = (ds_name, idx_path, abs_idx)
            if key not in seen:
                seen.add(key)
                uniq.append((ds_name, idx_path, abs_idx))
            if len(uniq) >= self.scaler_sample_size:
                break

        # 3) materialize rows per dataset with one H5 read each
        per_ds = defaultdict(list)
        for ds_name, idx_path, abs_idx in uniq:
            per_ds[(ds_name, idx_path)].append(abs_idx)

        rows_all = []
        for (ds_name, idx_path), abs_list in per_ds.items():
            h5f = self._load_h5(f'{idx_path}/{ds_name}{self.h5_suffix}')
            abs_arr = np.array(sorted(set(abs_list)), dtype=np.int64)

            # read unique rows, then map back in original order if needed
            X_unique = np.asarray(h5f['X_scimilarity'][abs_arr])
            # simple mapping when we don't need exact order restoration:
            # (we're just stacking for scaler fit, order does not matter)
            rows_all.append(X_unique)

        if not rows_all:
            raise RuntimeError("No rows sampled for pooled scaler.")

        return np.vstack(rows_all)
            
    def _sample_control_vectors(self):
        sample_indices = []
        for info in self.dataset_info:
            idx_path = info['idx_path']
            name = info['name']
            idx_df = pd.read_csv(f'{idx_path}/{name}{self.idx_suffix}')
            n = len(idx_df)
            k = int(self.scaler_sample_size * (n / self.total_samples)) + 1
            chosen = np.random.RandomState(self.seed).choice(n, size=min(k, n), replace=False)
            for idx in chosen:
                sample_indices.append((name, idx_path, idx_df['control_index'].iloc[idx]))

        if len(sample_indices) > self.scaler_sample_size:
            sample_indices = sample_indices[:self.scaler_sample_size]
        random.shuffle(sample_indices)

        controls = []
        for ds_name, idx_path, ctrl_idx in sample_indices:
            h5f = self._load_h5(f'{idx_path}/{ds_name}{self.h5_suffix}')
            raw = h5f["X_scimilarity"][ctrl_idx]
            if scipy.sparse.issparse(raw):
                raw = raw.toarray().squeeze()
            arr = np.asarray(raw).reshape(-1)
            controls.append(arr)

        return np.vstack(controls)
    
    def _build_collapse_mat(self, mapping, n_input_genes):
        rows, cols = [], []
        for new_i, old_idx_list in enumerate(mapping):
            rows.extend([new_i]*len(old_idx_list))
            cols.extend(old_idx_list)
        data = np.ones(len(rows), dtype=np.float32)
        M = scipy.sparse.csr_matrix((data, (rows, cols)),
                                    shape=(len(mapping), n_input_genes))
        return M

    def _fit_scaler(self):
        """Fit scaler on random control samples across all datasets"""
        
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                # control_matrix = self._sample_control_vectors()
                pooled_matrix = self._sample_pooled_h5_vectors()
                # Existing scaler fitting code
                self.scaler = StandardScaler()
                # self.scaler.fit(control_matrix)
                self.scaler.fit(pooled_matrix)

                scaler_params = {
                    'mean_': self.scaler.mean_,
                    'scale_': self.scaler.scale_,
                    'var_': self.scaler.var_,
                    'n_samples_seen_': self.scaler.n_samples_seen_
                }
            else:
                scaler_params = None
                
            broadcast_list = [scaler_params]
            dist.broadcast_object_list(broadcast_list, src=0)
            scaler_params = broadcast_list[0]
            
            if rank != 0:
                self.scaler = StandardScaler()
                for key, value in scaler_params.items():
                    setattr(self.scaler, key, value)
        else:                       
            # control_matrix = self._sample_control_vectors()
            pooled_matrix = self._sample_pooled_h5_vectors()
            self.scaler = StandardScaler()
            # self.scaler.fit(control_matrix)
            self.scaler.fit(pooled_matrix)
            

    def _manage_cache(self, dataset_name: str):
        """Manage both adata and h5 caches"""
        if dataset_name in self.cache_order:
            self.cache_order.remove(dataset_name)
        elif len(self.cache_order) >= self.cache_size:
            old = self.cache_order.pop(0)
            # Remove from both caches
            if old in self.adata_cache:
                del self.adata_cache[old]
            if old in self.h5_cache:
                self.h5_cache[old].close()
                del self.h5_cache[old]
        self.cache_order.append(dataset_name)

    def _load_adata(self, dataset_name: str) -> sc.AnnData:
        """Load AnnData file (still needed for metadata)"""
        if dataset_name in self.adata_cache:
            self._manage_cache(dataset_name)
            return self.adata_cache[dataset_name]
        
        path = f"{self.root_dir}{dataset_name}.h5ad"
        adata = sc.read(path, backed='r')
        _, adata = curate_perturb_tag_obs_only(adata, data_name=dataset_name)
        if self.count_transform:
            model_path = self.emb_model_path
            
            cq_genes_converted = self.gene_order
            adata_genes_converted = convert_to_ensembl(adata.var.index,suffix_token=0)
            # adata_genes_converted = convert_gene_name(adata.var.index,suffix_token=0)
            # cq_genes_converted = convert_gene_name(gene_order,suffix_token=0)
            self.total_genes = len(cq_genes_converted)
            # Create mapping that collects all indices for each gene name (handling duplicates)
            gene2indices = defaultdict(list)
            for i, g in enumerate(adata_genes_converted):
                gene2indices[g].append(i)
            
            # Create mapping for each cq gene to all matching indices
            mapping = []
            for g in cq_genes_converted:
                if g in gene2indices:
                    mapping.append(gene2indices[g])  # List of indices for this gene
                else:
                    mapping.append([])  # Empty list for missing genes
            print(f'data name: {dataset_name} has {len(mapping)} matched genes, out of {adata.n_vars} total genes')
            self.collapse_mat[dataset_name] = self._build_collapse_mat(mapping, adata.n_vars)
            self.reorder_map[dataset_name] = mapping
            # adata = align_dataset(adata, cq.gene_order)
            # Note: log normalization is now done per-sample in _process_sample
        
        self._manage_cache(dataset_name)
        
        self.adata_cache[dataset_name] = adata
        return adata

    def _load_h5(self, dataset_name: str) -> h5py.File:
        """Load h5 file containing X_scimilarity data"""
        if dataset_name in self.h5_cache:
            self._manage_cache(dataset_name)
            h5f = self.h5_cache[dataset_name]
            # Check if the file is still valid
            try:
                # Test if we can access the dataset
                _ = h5f["X_scimilarity"].shape
                return h5f
            except (OSError, KeyError, RuntimeError):
                # File might be corrupted or closed, remove from cache and reload
                print(f"Warning: HDF5 file {dataset_name} appears to be corrupted or closed, reloading...")
                if dataset_name in self.h5_cache:
                    try:
                        self.h5_cache[dataset_name].close()
                    except:
                        pass
                    del self.h5_cache[dataset_name]
                if dataset_name in self.cache_order:
                    self.cache_order.remove(dataset_name)
        
        path = f"{dataset_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"H5 file not found: {path}")
        
        try:
            h5f = h5py.File(path, 'r')
            # Verify the file has the expected dataset
            if "X_scimilarity" not in h5f:
                h5f.close()
                raise KeyError(f"HDF5 file {path} does not contain 'X_scimilarity' dataset")
            
            self._manage_cache(dataset_name)
            self.h5_cache[dataset_name] = h5f
            return h5f
        except Exception as e:
            print(f"Error loading HDF5 file {path}: {e}")
            raise e



    def _get_sample_indices(self) -> Iterator[Tuple[int,int,int,int]]:
        lst = []
        # for i, info in enumerate(self.dataset_info):
        #     idx_path = info['idx_path']
        #     name = info['name']
        #     df = pd.read_csv(f'{idx_path}/{name}{self.idx_suffix}')
        #     for j in range(info['n_samples']):
        #         lst.append((i, j, df['perturbation_index'].iloc[j], df['control_index'].iloc[j]))
        lst = self.sample_table
        if self.shuffle:
            random.Random(self.seed + getattr(self, 'epoch', 0)).shuffle(lst)
            
        if dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
            lst = lst[rank::world_size]
        
        
        return iter(lst)

    def _encode_sample(self, s: Dict) -> Dict:
        if self.scaler is not None:
            s['perturb'] = self.scaler.transform(s['perturb'].reshape(1,-1)).squeeze()
            s['control'] = self.scaler.transform(s['control'].reshape(1,-1)).squeeze()
        
        # No encoding - keep original string values as they are
        return s

    def _create_batch(self, batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        # Separate numeric tensors from string fields
        # numeric_fields = ['perturb', 'control', 'perturb_raw', 'control_raw', 'gene_sd']
        numeric_fields = ['perturb', 'control', 'perturb_raw', 'control_raw']
        string_fields = ['label', 'perturb_type', 'dataset_name', 'dna_seq', 'direction']
        
        result = {}
        
        # Handle numeric fields that should be converted to tensors
        for k in numeric_fields:
            if k in batch[0]:  # Check if field exists in the first sample
                result[k] = torch.stack([
                    torch.tensor(s[k], dtype=torch.float32)
                    for s in batch
                ])
        
        # Handle string fields that should remain as lists
        for k in string_fields:
            if k in batch[0]:  # Check if field exists in the first sample
                result[k] = [s[k] for s in batch]
                
        if 'dna_mask' in batch[0]:
            # Keep as list of arrays, model will handle OneHot + concatenation
            result['dna_mask'] = [s['dna_mask'] for s in batch]
            result['valid_dna'] = torch.tensor([s['valid_dna'] for s in batch])
        
        return result
    
    
    def _process_batch_h5(self, dataset_path, pair_list):
        """
        pair_list: [(p_idx, c_idx), ...]  all from the same dataset_path
        returns:   dict { (p_idx,c_idx) -> (Xp, Xc) }
        """
        try:
            h5f = self._load_h5(f"{dataset_path}{self.h5_suffix}")
            flat = np.array([idx for p,c in pair_list for idx in (p,c)], dtype=np.int64)

            # Debug: Check if indices are valid
            dataset_shape = h5f["X_scimilarity"].shape
            if np.any(flat < 0) or np.any(flat >= dataset_shape[0]):
                invalid_indices = flat[(flat < 0) | (flat >= dataset_shape[0])]
                raise ValueError(f"Invalid indices found: {invalid_indices}. Dataset shape: {dataset_shape}, pair_list: {pair_list}")

            # Check for duplicates and handle them
            unique_indices, inverse_indices = np.unique(flat, return_inverse=True)
            # Ensure increasing order for h5py fancy indexing (np.unique returns sorted asc by default)
            # Still enforce ascending order explicitly
            unique_sorted = np.sort(unique_indices, kind='mergesort')
            # Map from sorted positions back to unique order
            # Build position map: index value -> position in unique_sorted
            pos = {v: i for i, v in enumerate(unique_sorted)}
            inv_sorted = np.array([pos[v] for v in unique_indices], dtype=np.int64)
            
            # Read unique indices in sorted order
            time1 = time.time()
            Xrows_unique_sorted = np.asarray(h5f["X_scimilarity"][unique_sorted])
            # Restore np.unique order
            Xrows_unique = Xrows_unique_sorted[inv_sorted]
            time2 = time.time()
            # print(f"Time to read unique indices: {time2 - time1:.4f} seconds")
            
            # Map back to original positions
            Xrows = Xrows_unique[inverse_indices]
            
            # Create iterator and return pairs
            it = iter(Xrows)
            return {(p, c): (next(it), next(it)) for p, c in pair_list}
        except Exception as e:
            print(f"Error in _process_batch_h5 for dataset_path: {dataset_path}")
            print(f"pair_list: {pair_list}")
            print(f"flat indices: {flat if 'flat' in locals() else 'not created'}")
            if 'h5f' in locals():
                print(f"HDF5 dataset shape: {h5f['X_scimilarity'].shape}")
            raise e

    def _validate_sample_indices(self, batch_meta):
        """Validate that all indices in batch_meta are within bounds"""
        for di, _, pi, ci in batch_meta:
            info = self.dataset_info[di]
            dataset_name = info['name']
            
            # Load AnnData to check bounds
            ad = self._load_adata(dataset_name)
            if pi >= len(ad) or ci >= len(ad):
                raise ValueError(f"Index out of bounds for dataset {dataset_name}: pi={pi}, ci={ci}, dataset_size={len(ad)}")
            
            # Load HDF5 to check bounds
            h5f = self._load_h5(f"{info['idx_path']}/{dataset_name}{self.h5_suffix}")
            h5_shape = h5f["X_scimilarity"].shape[0]
            if pi >= h5_shape or ci >= h5_shape:
                raise ValueError(f"HDF5 index out of bounds for dataset {dataset_name}: pi={pi}, ci={ci}, h5_shape={h5_shape}")

    def _create_batch_from_meta(self, batch_meta):
        """
        batch_meta: list of tuples (dataset_name, idx_path, perturbation_index, control_index)
        """
        try:
            # Validate indices first
            self._validate_sample_indices(batch_meta)
            
            per_ds = defaultdict(list)
            for di, _,pi, ci in batch_meta:
                info = self.dataset_info[di]
                per_ds[info['name']].append((pi, ci))
            
            ## bulk read h5 rows
            h5_blocks = {}
            for ds, pairs in per_ds.items():
                info = self.dataset_info[self.namelist.index(ds)]
                try:
                    h5_blocks[ds] = self._process_batch_h5(info['idx_path']+'/'+ds, pairs)
                except Exception as e:
                    print(f"Failed to process dataset {ds} with pairs {pairs}")
                    raise e
            
            def to_array(x): 
                return x.toarray().squeeze() if scipy.sparse.issparse(x) else x
            
            ## assemble samples
            samples = []
            for di, li, pi, ci in batch_meta:
                info = self.dataset_info[di]
                ad = self._load_adata(info['name'])
                p_meta = ad[pi]
                c_meta = ad[ci]
                X, Xc = h5_blocks[info['name']][(pi, ci)]

                time3 = time.time()
                X_raw = np.asarray(to_array(p_meta.X)).reshape(-1)
                Xc_raw = np.asarray(to_array(c_meta.X)).reshape(-1)
                time4 = time.time()
                # print(f"Time to convert to array: {time4 - time3:.4f} seconds")
                X_raw = self.collapse_mat[info['name']].dot(X_raw)
                Xc_raw = self.collapse_mat[info['name']].dot(Xc_raw)
                
                
                time5 = time.time()
                # print(f"Time to collapse mat: {time5 - time3:.4f} seconds")
                if self.count_transform:
                    # Normalize to 10,000 reads per cell
                    X_raw_sum = np.sum(X_raw)
                    Xc_raw_sum = np.sum(Xc_raw)
                    
                    if X_raw_sum > 0:
                        X_raw = X_raw / X_raw_sum * 10000
                    if Xc_raw_sum > 0:
                        Xc_raw = Xc_raw / Xc_raw_sum * 10000
                        
                if self.hvg_mask is not None:
                    X_raw = X_raw[self.hvg_mask]
                    Xc_raw = Xc_raw[self.hvg_mask]
                
                X_raw = np.log1p(X_raw)
                Xc_raw = np.log1p(Xc_raw)
                
                dna_seq = ""
                dna_mask = None
                direction = None
                valid_dna = False
                
                # Extract gene name from label
                lbl = p_meta.obs['curated_perturbation'].values[0]
                
                # Apply filtering: skip samples with MTRNR2L1
                if _should_filter_label(str(lbl)):
                    continue  # Skip this sample
                
                # Apply gene name translation (AC118549.1 -> ZZZ3)
                lbl = _translate_gene_name(str(lbl))
                
                if lbl != 'control':
                    try:
                        # Parse gene name (adjust split logic as needed for your data)
                        gene_name_str = lbl.rsplit('-', 2)[0].split('+')[0]
                        
                        # Call the generation utility (ensure imported)
                        seq_info = gen_gene_TSS_sequence(
                            gene_name_str, 
                            dna_record=self.dna_record,
                            all_annots_transcript=self.gene_annot,
                            jitter_window=self.jitter_window,
                            ccre_intervals_dict=self.ccre_intervals_dict
                        )
                        
                        if seq_info is not None:
                            seq_info[0]['promoter_len'] = self.promoter_len
                            
                            # Heavy string op
                            ref_seq, mask_seq = aggregate_mask_perturbation(seq_info[0])
                            
                            if np.random.random() < self.drop_mask_prob:
                                mask_seq = np.zeros_like(mask_seq)
                                
                            dna_seq = ref_seq
                            dna_mask = mask_seq # numpy array
                            direction = seq_info[0]['Direction']
                            valid_dna = True
                    except Exception as e:
                        # print(f"DNA gen failed for {lbl}: {e}")
                        pass
                
                sample = {
                    'perturb': X,
                    'control': Xc,
                    'perturb_raw': X_raw,
                    'control_raw': Xc_raw,
                    'label': lbl,
                    'perturb_type': (c_meta.obs['perturbation_type'].values[0] 
                               if 'perturbation_type' in c_meta.obs.columns 
                               else 'CRISPR-interference'),
                    'dataset_name': info['name'],
                    
                    # Add new fields
                    'dna_seq': dna_seq,
                    'dna_mask': dna_mask,
                    'direction': direction,
                    'valid_dna': valid_dna
                }
                
                sample = self._encode_sample(sample)
                samples.append(sample)
            return self._create_batch(samples)
        
        except Exception as e:
            print(f"Error in _create_batch_from_meta with batch_meta: {batch_meta}")
            raise e
    
    
    def __iter__(self):
        batch_meta = []
        batch_count = 0

        # 1. Get all indices
        it_list = list(self._get_sample_indices())
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: 
            # Split workload among workers
            per_worker = int(math.ceil(len(it_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(it_list))
            it_list = it_list[iter_start:iter_end]
        
        if self.drop_last:
            # trim to a multiple of batch_size so every rank yields the same #batches
            it_list = it_list[: (len(it_list) // self.batch_size) * self.batch_size]

        for quatlist in it_list:
            batch_meta.append(quatlist)
            if len(batch_meta) == self.batch_size:
                try:
                    batch_count += 1
                    yield self._create_batch_from_meta(batch_meta)
                finally:
                    # ensure we clear even if _create_batch_from_meta raises
                    batch_meta = []

        # Emit the final partial batch only when NOT dropping the last
        if (not self.drop_last) and batch_meta:
            yield self._create_batch_from_meta(batch_meta)
            
            
            
    def __getstate__(self):
        """
        Called when pickling the object to send to workers.
        We must remove the open h5py handles from the state.
        """
        state = self.__dict__.copy()
        # Reset caches to empty dicts for the pickled state
        state['h5_cache'] = {} 
        state['adata_cache'] = {}
        return state

    def __setstate__(self, state):
        """
        Called in the new worker process to restore state.
        """
        self.__dict__.update(state)
        # Ensure caches are initialized empty in the new worker
        self.h5_cache = {}
        self.adata_cache = {}
    
    def __del__(self):
        """Clean up h5 files when dataloader is destroyed"""
        for h5f in self.h5_cache.values():
            h5f.close()



    


class perturb_control_dataloader_syn(Dataset):
    def __init__(self, namelist, perturbation=None, cond=None, 
                 root_dir=PROCESSED_DATA_ROOT,
                 aug=False,
                 scale=True):
        '''
        namelist: list of dataset names
        '''
        self.data = []
        self.control = []
        self.labels = []
        
        for name in namelist:
            adata = sc.read(f'{root_dir}{name}.h5ad')
            self.labels.append(adata.obs['curated_perturbation'].values)
            self.data.append(adata.X)
            self.control.append(adata.obsm['X_control'])
            
        print(self.data[0].shape)
       
        self.data = np.concatenate(self.data, axis=0)
        self.control = np.concatenate(self.control, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        if perturbation is not None:
            ## take a subset of the data
            sub_data = []
            sub_labels = []
            sub_control = []
            for pert in perturbation:
                # dname = pert.split('_')[0]
                data_idx = self.labels==pert
                print(f'size of data_idx for {pert}: {np.sum(data_idx)}')
                sub_data.append(self.data[data_idx])
                sub_labels.append(self.labels[data_idx])
                sub_control.append(self.control[data_idx])
            
            self.data = np.concatenate(sub_data, axis=0)
            self.control = np.concatenate(sub_control, axis=0)
            self.labels = np.concatenate(sub_labels, axis=0)
            
        
       
        
        if aug:
            DE = np.mean(self.data, axis=0) - np.mean(self.control, axis=0)
            stds = np.std(self.control, axis=0)
            ctrl_aug = np.random.normal(loc=0, scale=1, size=(5000, self.control.shape[1]))
            pert_aug = ctrl_aug*stds + DE
            self.data = np.concatenate([self.data, pert_aug], axis=0)
            self.control = np.concatenate([self.control, ctrl_aug], axis=0)
            self.labels = np.concatenate([self.labels, ['aug']*pert_aug.shape[0]], axis=0)
        
        
        if scale:
            self.scaler = StandardScaler()
            
            self.scaler.fit(self.control)
            self.data = self.scaler.transform(self.data)
            self.control = self.scaler.transform(self.control)
        # self.control = gaussian_copula_transform(self.control)
            
        
        le = LabelEncoder()
        self.encode_labels = le.fit_transform(self.labels)
        

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_0 = self.data[idx]
        x_0 = torch.tensor(x_0, dtype=torch.float32)
        x_ctrl = self.control[idx]
        x_ctrl = torch.tensor(x_ctrl, dtype=torch.float32)
        label = self.labels[idx]
        label_encode = torch.tensor(self.encode_labels[idx], dtype=torch.long)
        return {'perturb': x_0, 'control': x_ctrl, 'label':label, 'label_encode': label_encode}
    


# def curate_perturb_tag(adata, data_name):
#     '''
#     Adds a curated perturbation label to adata.obs['curated_perturbation']
    
#     Each label is:
#         - 'control-<dataset_name>' for negative controls
#         - '<gene>' or 'gene1+gene2+...' for gene perturbations
#     Also returns:
#         - perturb_dict: dict mapping curated tags to list of raw entries
#     '''

#     perturb_dict = defaultdict(list)
#     perturb_to_label = {}

#     perturb_type = set(adata.obs['perturbation'].values)

#     if data_name == 'AdamsonWeissman2016_GSM2406675_10X001':
#         for entry in perturb_type:
#             if pd.isna(entry):
#                 continue
#             if '62(mod)' in entry:
#                 label = f'control-{data_name}'
#             else:
#                 if '_' in entry:
#                     gene_name = entry.split('_')[0]
#                     label = f'{gene_name}-{data_name}'
#                 else:
#                     print(f"Error parsing entry: {entry}")
#                     label = f'unknown-{data_name}'
#             perturb_dict[label].append(entry)
#             perturb_to_label[entry] = label

#     elif data_name == 'NormanWeissman2019_filtered':
#         for entry in perturb_type:
#             entry = entry.replace('_', '+')
#             label = f'{entry}-{data_name}'
#             # label = label.replace('_', '+')
#             perturb_dict[label].append(entry)
#             perturb_to_label[entry] = label

#     elif data_name == 'DixitRegev2016':
#         for entry in perturb_type:
#             components = entry.split('_')
#             genes = set()
#             is_control = True

#             for comp in components:
#                 if 'INTERGENIC' in comp:
#                     continue
#                 elif 'sg' in comp:
#                     gene = comp.replace('sg', '').rsplit('-', 1)[0]
#                     gene = gene.replace('p-', '')
#                     genes.add(gene)
#                     is_control = False

#             if is_control:
#                 label = f'control-{data_name}'
#             else:
#                 label = '+'.join(sorted(genes)) + f'-{data_name}'

#             perturb_dict[label].append(entry)
#             perturb_to_label[entry] = label

#     else:
#         raise ValueError(f"Dataset: {data_name} QC is not implemented")

#     # Add curated label to AnnData
#     adata.obs['curated_perturbation'] = adata.obs['perturbation'].map(perturb_to_label)

#     return perturb_dict
