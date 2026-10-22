#!/usr/bin/env python3
"""
Extract standardized subsets from three cell line datasets (K562, RPE1, Jurkat).

This script:
1. Filters each dataset to keep only perturbQA perturbations + "non-targeting"
2. For individual files: Reformats using cell-line specific unique_genes
3. For concatenation: Reformats using union_all_three for gene alignment
4. Saves individual reformatted files (with cell-line specific genes)
5. Concatenates all three datasets (using union_all_three aligned versions)
6. Selects top 2k HVG using Seurat v3 flavor (based on control cells only)
7. Saves combined HVG-selected dataset

Raw mode (--mode raw):
- Saves raw untransformed data (no normalization, no log transformation)
- Keeps original gene order (no gene reformatting)
- Uses same cell filtering as standard mode
- Saves as {cell_line}_raw.h5ad (e.g., k562_raw.h5ad)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import gc
from pathlib import Path

# Settings
sc.settings.verbosity = 3

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_REPO_ROOT / "data" / "processed"))).expanduser()

# These are intentionally overridable via CLI/env for portability.
dataset_path = os.environ.get("PERTURBQA_META_PATH", "")
data_path = os.environ.get("SCRPERTURB_DATA_PATH", "")
output_dir = os.environ.get("CURATION_OUTPUT_DIR", str(_DEFAULT_DATA_ROOT))


def get_gene_names(adata):
    """Get gene names from adata.var, handling both index and gene_name column"""
    if 'gene_name' in adata.var.columns:
        return adata.var.gene_name.values
    else:
        return adata.var.index.values


def reformat_adata_genes(adata, canonical_genes, cell_line_name):
    """
    Reformat adata to match canonical gene order.
    For genes in canonical_genes but not in adata: add zeros
    For genes in adata but not in canonical_genes: remove them
    
    Optimized to avoid expensive COO reconstruction when possible.
    """
    if adata.isbacked:
        adata = adata.to_memory()
    
    current_genes = get_gene_names(adata)
    current_genes_set = set(current_genes)
    gene_to_idx = {gene: idx for idx, gene in enumerate(current_genes)}
    
    n_cells = adata.shape[0]
    n_canonical = len(canonical_genes)
    
    # Ensure sparse format
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    
    # Check if all canonical genes exist (reordering only) vs some missing
    canonical_genes_set = set(canonical_genes)
    missing_genes = canonical_genes_set - current_genes_set
    all_genes_present = len(missing_genes) == 0
    
    # Build column mapping
    col_mapping = [gene_to_idx.get(gene) for gene in canonical_genes]
    valid_indices = [i for i, idx in enumerate(col_mapping) if idx is not None]
    source_indices = [idx for idx in col_mapping if idx is not None]
    
    # Optimized path: Use CSR operations instead of COO reconstruction
    if all_genes_present:
        # Case 1: All genes present, just reordering
        # Check if columns are already in canonical order
        # source_indices extracts columns in canonical_genes order, so if they're already
        # in that order in adata, we can avoid remapping
        X_subset = adata.X[:, source_indices]
        
        # Check if reordering is needed by comparing current gene order to canonical order
        current_subset_genes = [current_genes[i] for i in source_indices]
        needs_reordering = current_subset_genes != list(canonical_genes)
        
        if needs_reordering:
            print(f"  Reordering genes for {cell_line_name} (all genes present)...")
            # Create column remapping array: map from position in X_subset to position in canonical_genes
            # valid_indices[i] gives the canonical position for the i-th column of X_subset
            col_remap = np.array(valid_indices, dtype=np.int32)
            
            # Remap columns using COO format
            coo = X_subset.tocoo()
            new_cols = col_remap[coo.col]
            new_X = sparse.coo_matrix((coo.data, (coo.row, new_cols)), shape=(n_cells, n_canonical)).tocsr()
        else:
            # Columns already in correct order, no remapping needed
            print(f"  Genes already in canonical order for {cell_line_name} (all genes present)...")
            new_X = X_subset
    else:
        # Case 2: Some genes missing, need to add zero columns
        print(f"  Reformatting genes for {cell_line_name} ({len(missing_genes)} missing genes)...")
        
        # Extract existing columns
        X_subset = adata.X[:, source_indices]
        
        # Build column remapping: map from position in X_subset to position in canonical_genes
        # valid_indices[i] gives the canonical position for the i-th column of X_subset
        col_remap = np.array(valid_indices, dtype=np.int32)
        
        # Remap columns using COO format (necessary when we have missing genes)
        coo = X_subset.tocoo()
        new_cols = col_remap[coo.col]
        
        # Missing genes are implicitly zero (no entries), so we just need correct shape
        new_X = sparse.coo_matrix((coo.data, (coo.row, new_cols)), shape=(n_cells, n_canonical)).tocsr()
    
    # Optimize var construction using vectorized operations
    print(f"  Constructing var metadata for {cell_line_name}...")
    
    # Pre-compute mappings for efficient lookup
    if 'gene_name' in adata.var.columns:
        gene_name_series = adata.var['gene_name']
        gene_name_to_var_idx = {name: idx for idx, name in enumerate(gene_name_series)}
        
        # Vectorized var construction
        var_indices = []
        for gene in canonical_genes:
            if gene in gene_name_to_var_idx:
                var_indices.append(gene_name_to_var_idx[gene])
            else:
                var_indices.append(None)
        
        # Build var DataFrame efficiently
        var_rows = []
        for i, gene in enumerate(canonical_genes):
            if var_indices[i] is not None:
                var_rows.append(adata.var.iloc[var_indices[i]])
            else:
                # Create minimal row for missing gene
                var_rows.append(pd.Series({'gene_name': gene}, name=gene))
        
        new_var = pd.DataFrame(var_rows)
        new_var.index = canonical_genes
    else:
        # Fallback: index-based
        var_indices = []
        for gene in canonical_genes:
            if gene in gene_to_idx:
                var_indices.append(gene_to_idx[gene])
            else:
                var_indices.append(None)
        
        var_rows = []
        for i, gene in enumerate(canonical_genes):
            if var_indices[i] is not None:
                var_rows.append(adata.var.iloc[var_indices[i]])
            else:
                var_rows.append(pd.Series({}, name=gene))
        
        new_var = pd.DataFrame(var_rows)
        new_var.index = canonical_genes
    
    # Sanitize var dtypes for HDF5 write without losing semantics
    if 'in_matrix' in new_var.columns:
        new_var['in_matrix'] = (
            new_var['in_matrix']
            .astype('boolean')
            .fillna(False)
            .astype(bool)
        )
    for col in new_var.columns:
        s = new_var[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        # If categorical: ensure categories are plain Python strings (object), not StringDtype
        if pd.api.types.is_categorical_dtype(s):
            new_var[col] = s.astype(str).astype('category')
            continue
        # If pandas nullable string dtype: convert to plain object strings
        try:
            is_pandas_string = pd.api.types.is_dtype_equal(s.dtype, pd.StringDtype())
        except Exception:
            is_pandas_string = False
        if is_pandas_string:
            new_var[col] = s.fillna("").astype(str)
            continue
        if s.dtype == object:
            non_na = s.dropna()
            try:
                unique_vals = pd.unique(non_na)
            except Exception:
                unique_vals = non_na.unique()
            as_str = {str(v) for v in unique_vals}
            if as_str.issubset({'True', 'False', 'true', 'false', '1', '0'}):
                def _to_bool(x):
                    if isinstance(x, (bool, np.bool_)):
                        return bool(x)
                    if isinstance(x, (int, np.integer)):
                        return bool(int(x))
                    if isinstance(x, str):
                        x_lower = x.lower()
                        if x_lower in ('true', '1'):
                            return True
                        if x_lower in ('false', '0'):
                            return False
                    return False
                new_var[col] = s.map(_to_bool).fillna(False).astype(bool)
            else:
                new_var[col] = s.fillna("").astype(str)
    
    return ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)


def process_and_save(cell_line, adata_path, unique_pert, cell_line_genes, union_genes):
    """
    Process, save, and return path to temp union file to keep memory clean.
    """
    print(f"Processing {cell_line}...")
    
    # Load Backed
    print(f"  Loading {cell_line} dataset (backed mode)...")
    adata_full = sc.read_h5ad(adata_path, backed='r')
    print(f"  Dataset shape: {adata_full.shape}")
    
    target_genes = set(unique_pert) | {"non-targeting"}
    print(f"  Filtering cells for {len(target_genes)} target perturbations...")
    filter_mask = adata_full.obs['gene'].isin(target_genes)
    n_filtered = filter_mask.sum()
    print(f"  Found {n_filtered:,} cells matching target perturbations (out of {len(filter_mask):,} total)")
    
    # Load only subset to memory
    print(f"  Loading filtered subset to memory...")
    adata_filtered = adata_full[filter_mask].to_memory()
    print(f"  Filtered dataset shape: {adata_filtered.shape}")
    del adata_full
    gc.collect()
    
    # 1. Save Individual (Cell-line specific genes)
    path_indiv = f'{output_dir}/{cell_line.lower()}_perturbqa_reformatted.h5ad'
    adata_indiv = reformat_adata_genes(adata_filtered, cell_line_genes, cell_line)
    adata_indiv.obs['cell_line'] = cell_line
    
    # Normalize and log-transform to match HVG file processing
    print(f"  Normalizing and log-transforming {cell_line} individual file...")
    sc.pp.normalize_total(adata_indiv, target_sum=1e4)
    sc.pp.log1p(adata_indiv)
    
    adata_indiv.write(path_indiv)
    del adata_indiv
    gc.collect()
    
    # 2. Save Union (Union genes) - Save to temp file for concatenation later
    path_union = f'{output_dir}/temp_{cell_line.lower()}_union.h5ad'
    adata_union = reformat_adata_genes(adata_filtered, union_genes, f"{cell_line}_union")
    adata_union.obs['cell_line'] = cell_line
    adata_union.write(path_union)
    
    print(f"Saved temp union file: {path_union}")
    
    del adata_filtered, adata_union
    gc.collect()
    
    return path_union


def process_and_save_raw(cell_line, adata_path, unique_pert):
    """
    Process and save raw (untransformed) data without normalization or log transformation.
    Keeps original gene order and same cell filtering as process_and_save().
    Loads filtered subset to memory (more efficient than streaming copy for large datasets).
    """
    print(f"Processing {cell_line} (raw mode)...")
    
    # Load Backed
    print(f"  Loading {cell_line} dataset (backed mode)...")
    adata_full = sc.read_h5ad(adata_path, backed='r')
    print(f"  Dataset shape: {adata_full.shape}")
    
    target_genes = set(unique_pert) | {"non-targeting"}
    print(f"  Filtering cells for {len(target_genes)} target perturbations...")
    filter_mask = adata_full.obs['gene'].isin(target_genes)
    n_filtered = filter_mask.sum()
    print(f"  Found {n_filtered:,} cells matching target perturbations (out of {len(filter_mask):,} total)")
    
    # Load filtered subset to memory (more efficient than streaming copy for large datasets)
    print(f"  Loading filtered subset to memory...")
    adata_filtered = adata_full[filter_mask].to_memory()
    print(f"  Filtered dataset shape: {adata_filtered.shape}")
    del adata_full
    gc.collect()
    
    # Add cell_line metadata
    adata_filtered.obs['cell_line'] = cell_line
    
    # Write final file
    path_raw = f'{output_dir}/{cell_line.lower()}_raw.h5ad'
    print(f"  Writing raw file...")
    adata_filtered.write(path_raw)
    
    print(f"  Saved raw file: {path_raw}")
    
    del adata_filtered
    gc.collect()
    
    return path_raw


def load_metadata_and_compute_union():
    """Load perturbQA metadata and compute gene union across all datasets"""
    print("=" * 60)
    print("Loading perturbQA data and computing gene union")
    print("=" * 60)
    
    rpe1 = pd.read_csv(os.path.join(dataset_path, "rpe1-de.csv"))
    k562 = pd.read_csv(os.path.join(dataset_path, "k562-de.csv"))
    jurkat = pd.read_csv(os.path.join(dataset_path, "jurkat-de.csv"))
    
    unique_genes_rpe1 = rpe1['gene'].unique()
    unique_pert_rpe1 = rpe1['pert'].unique()
    
    unique_genes_k562 = k562['gene'].unique()
    unique_pert_k562 = k562['pert'].unique()
    
    unique_genes_jurkat = jurkat['gene'].unique()
    unique_pert_jurkat = jurkat['pert'].unique()
    
    print(f'Unique genes RPE1: {len(unique_genes_rpe1)}')
    print(f'Unique genes K562: {len(unique_genes_k562)}')
    print(f'Unique genes Jurkat: {len(unique_genes_jurkat)}')
    
    # Compute union of all three cell lines based on adata.var
    # Memory efficient: only reads var metadata, not the full X matrix
    print('\nComputing union of genes from all datasets (reading var only)...')
    
    # K562 genes (only var is accessed, X stays on disk)
    print("Reading K562 genes from var...")
    tmp_k562 = sc.read_h5ad(f'{data_path}/ReplogleWeissman2022_K562_gwps.h5ad', backed='r')
    genes_k562 = get_gene_names(tmp_k562)
    del tmp_k562
    gc.collect()
    
    # RPE1 genes
    print("Reading RPE1 genes from var...")
    tmp_rpe1 = sc.read_h5ad(f'{data_path}/ReplogleWeissman2022_rpe1.h5ad', backed='r')
    genes_rpe1 = get_gene_names(tmp_rpe1)
    del tmp_rpe1
    gc.collect()
    
    # Jurkat genes
    print("Reading Jurkat genes from var...")
    tmp_jurkat = sc.read_h5ad(f'{data_path}/GSE264667_jurkat_raw_singlecell_01.h5ad', backed='r')
    genes_jurkat = get_gene_names(tmp_jurkat)
    del tmp_jurkat
    gc.collect()
    
    # Compute union
    union_all_three = np.union1d(genes_k562, genes_rpe1)
    union_all_three = np.union1d(union_all_three, genes_jurkat)
    
    print(f'Union of all three cell lines (from adata.var): {len(union_all_three)} genes')
    
    # Free memory
    del genes_k562, genes_rpe1, genes_jurkat
    gc.collect()
    
    metadata = {
        'rpe1': {'genes': unique_genes_rpe1, 'pert': unique_pert_rpe1},
        'k562': {'genes': unique_genes_k562, 'pert': unique_pert_k562},
        'jurkat': {'genes': unique_genes_jurkat, 'pert': unique_pert_jurkat},
        'union_genes': union_all_three
    }
    
    # Free memory for CSV data
    del rpe1, k562, jurkat
    gc.collect()
    
    return metadata


def load_metadata_minimal():
    """Load only perturbation metadata (skip gene union computation for raw mode)"""
    print("=" * 60)
    print("Loading perturbQA perturbation metadata")
    print("=" * 60)
    
    rpe1 = pd.read_csv(os.path.join(dataset_path, "rpe1-de.csv"))
    k562 = pd.read_csv(os.path.join(dataset_path, "k562-de.csv"))
    jurkat = pd.read_csv(os.path.join(dataset_path, "jurkat-de.csv"))
    
    unique_pert_rpe1 = rpe1['pert'].unique()
    unique_pert_k562 = k562['pert'].unique()
    unique_pert_jurkat = jurkat['pert'].unique()
    
    print(f'Unique perturbations RPE1: {len(unique_pert_rpe1)}')
    print(f'Unique perturbations K562: {len(unique_pert_k562)}')
    print(f'Unique perturbations Jurkat: {len(unique_pert_jurkat)}')
    
    metadata = {
        'rpe1': {'pert': unique_pert_rpe1},
        'k562': {'pert': unique_pert_k562},
        'jurkat': {'pert': unique_pert_jurkat}
    }
    
    # Free memory for CSV data
    del rpe1, k562, jurkat
    gc.collect()
    
    return metadata


def process_single_dataset(cell_line_name, metadata):
    """Process a single dataset"""
    print("=" * 60)
    print(f"Processing {cell_line_name} dataset")
    print("=" * 60)
    
    cell_line_lower = cell_line_name.lower()
    cell_line_info = metadata[cell_line_lower]
    
    # Map cell line names to file paths
    dataset_paths = {
        'k562': f'{data_path}/ReplogleWeissman2022_K562_gwps.h5ad',
        'rpe1': f'{data_path}/ReplogleWeissman2022_rpe1.h5ad',
        'jurkat': f'{data_path}/GSE264667_jurkat_raw_singlecell_01.h5ad'
    }
    
    adata_path = dataset_paths[cell_line_lower]
    union_genes = metadata['union_genes']
    
    # Process and save
    union_path = process_and_save(
        cell_line_name,
        adata_path,
        cell_line_info['pert'],
        cell_line_info['genes'],
        union_genes
    )
    
    print(f"\n{cell_line_name} processing complete!")
    print(f"  Individual file: {output_dir}/{cell_line_lower}_perturbqa_reformatted.h5ad")
    print(f"  Union temp file: {union_path}")
    
    return union_path


def process_single_dataset_raw(cell_line_name, metadata):
    """Process a single dataset in raw mode"""
    print("=" * 60)
    print(f"Processing {cell_line_name} dataset (raw mode)")
    print("=" * 60)
    
    cell_line_lower = cell_line_name.lower()
    cell_line_info = metadata[cell_line_lower]
    
    # Map cell line names to file paths
    dataset_paths = {
        'k562': f'{data_path}/ReplogleWeissman2022_K562_gwps.h5ad',
        'rpe1': f'{data_path}/ReplogleWeissman2022_rpe1.h5ad',
        'jurkat': f'{data_path}/GSE264667_jurkat_raw_singlecell_01.h5ad'
    }
    
    adata_path = dataset_paths[cell_line_lower]
    
    # Process and save raw data
    raw_path = process_and_save_raw(
        cell_line_name,
        adata_path,
        cell_line_info['pert']
    )
    
    print(f"\n{cell_line_name} raw processing complete!")
    print(f"  Raw file: {raw_path}")
    
    return raw_path


def concatenate_and_compute_hvg():
    """Concatenate all datasets and compute HVG selection"""
    print("=" * 60)
    print("Concatenating datasets and computing HVG selection")
    print("=" * 60)
    
    # Expected union temp files
    union_paths = [
        f'{output_dir}/temp_k562_union.h5ad',
        f'{output_dir}/temp_rpe1_union.h5ad',
        f'{output_dir}/temp_jurkat_union.h5ad'
    ]
    
    # Check that all files exist
    missing_files = [p for p in union_paths if not os.path.exists(p)]
    if missing_files:
        print(f"ERROR: Missing required union files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please ensure all individual datasets have been processed first.")
        sys.exit(1)
    
    # --- Step 1: Iterative Concatenation ---
    print("\n" + "=" * 60)
    print("Step 1: Iterative Concatenation...")
    print("=" * 60)
    
    # Initialize with the first file
    print(f"Loading {union_paths[0]}...")
    adata_combined = sc.read_h5ad(union_paths[0])
    
    # Append the others one by one
    for path in union_paths[1:]:
        print(f"Merging {path}...")
        ad_next = sc.read_h5ad(path)
        # In-memory concatenation (creates new object, but we overwrite immediately)
        adata_combined = ad.concat([adata_combined, ad_next], join='outer', fill_value=0)
        del ad_next
        gc.collect()
        
    print(f"Combined shape: {adata_combined.shape}")

    # --- Step 2: HVG Selection (Corrected Logic) ---
    print("\n" + "=" * 60)
    print("Step 2: HVG Selection (Seurat v3 flavor)...")
    print("=" * 60)
    
    # 1. Identify Control Cells
    control_mask = adata_combined.obs['gene'] == 'non-targeting'
    
    # 2. Extract Counts for Seurat v3 (Must use raw counts!)
    # We need a copy because seurat_v3 modifies the object with mean/var info
    adata_control = adata_combined[control_mask].copy()
    
    # Ensure we have integers/counts if possible, though float counts are usually accepted
    # DO NOT NORMALIZE OR LOG HERE for seurat_v3
    
    print("Computing HVGs on RAW counts of control cells...")
    sc.pp.highly_variable_genes(adata_control, n_top_genes=2000, flavor='seurat_v3')
    
    # Get the boolean mask of genes to keep
    hvg_mask = adata_control.var['highly_variable']
    
    # 3. Apply to Main Dataset
    # Now we filter the main dataset genes
    adata_combined.var['highly_variable'] = False
    adata_combined.var.loc[hvg_mask.index, 'highly_variable'] = hvg_mask
    
    # Subset to HVGs
    adata_combined_hvg = adata_combined[:, adata_combined.var['highly_variable']].copy()
    
    # Free the massive combined raw object
    del adata_combined, adata_control
    gc.collect()
    
    # 4. NOW Normalize and Log the final HVG dataset
    print("Normalizing and Log-transforming final HVG dataset...")
    sc.pp.normalize_total(adata_combined_hvg, target_sum=1e4)
    sc.pp.log1p(adata_combined_hvg)
    
    # Save
    output_hvg_path = f'{output_dir}/combined_perturbqa_hvg2k.h5ad'
    adata_combined_hvg.write(output_hvg_path)
    print(f"Saved final HVG dataset to {output_hvg_path}")
    
    # Clean up temp files
    for path in union_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed temp file: {path}")
    
    print("\n" + "=" * 60)
    print("Concatenation and HVG selection complete!")
    print("=" * 60)
    print(f"\nOutput file saved to: {output_dir}")
    print("  - combined_perturbqa_hvg2k.h5ad")


def main():
    """Main processing pipeline with support for parallel execution"""
    global dataset_path, data_path, output_dir
    parser = argparse.ArgumentParser(
        description='Extract standardized subsets from cell line datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets sequentially (original behavior)
  python curate_perturbqa.py --mode all
  
  # Process only K562 dataset
  python curate_perturbqa.py --mode single --dataset k562
  
  # Process only RPE1 dataset
  python curate_perturbqa.py --mode single --dataset rpe1
  
  # Process only Jurkat dataset
  python curate_perturbqa.py --mode single --dataset jurkat
  
  # Concatenate and compute HVG (after all datasets are processed)
  python curate_perturbqa.py --mode concat
  
  # Save raw data for all datasets
  python curate_perturbqa.py --mode raw
  
  # Save raw data for single dataset
  python curate_perturbqa.py --mode raw --dataset k562
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'single', 'concat', 'raw'],
        default='all',
        help='Processing mode: "all" (sequential), "single" (one dataset), "concat" (concatenate only), or "raw" (save raw untransformed data)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['k562', 'rpe1', 'jurkat'],
        default=None,
        help='Dataset to process (required if mode=single)'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default=dataset_path,
        help='Path to PerturbQA metadata CSVs directory (contains e.g. k562-de.csv). '
             'Can also be set via env PERTURBQA_META_PATH.',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=data_path,
        help='Path to raw input h5ad files directory. Can also be set via env SCRPERTURB_DATA_PATH.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=output_dir,
        help='Output directory for curated h5ad files. Defaults to env CURATION_OUTPUT_DIR or DATA_ROOT.',
    )
    
    args = parser.parse_args()

    # Update globals used throughout helper functions.
    dataset_path = args.dataset_path
    data_path = args.data_path
    output_dir = args.output_dir

    if not dataset_path:
        raise ValueError(
            "dataset_path is not set. Pass --dataset_path or set env PERTURBQA_META_PATH."
        )
    if not data_path:
        raise ValueError(
            "data_path is not set. Pass --data_path or set env SCRPERTURB_DATA_PATH."
        )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if args.mode == 'single':
        if args.dataset is None:
            parser.error("--dataset is required when --mode=single")
        
        # Load metadata and compute union (needed for processing)
        metadata = load_metadata_and_compute_union()
        
        # Process single dataset
        cell_line_map = {
            'k562': 'K562',
            'rpe1': 'RPE1',
            'jurkat': 'Jurkat'
        }
        process_single_dataset(cell_line_map[args.dataset], metadata)
        
    elif args.mode == 'concat':
        # Only concatenate and compute HVG
        concatenate_and_compute_hvg()
        
    elif args.mode == 'raw':
        # Load minimal metadata (only perturbations, no gene union)
        metadata = load_metadata_minimal()
        
        if args.dataset is not None:
            # Process single dataset in raw mode
            cell_line_map = {
                'k562': 'K562',
                'rpe1': 'RPE1',
                'jurkat': 'Jurkat'
            }
            process_single_dataset_raw(cell_line_map[args.dataset], metadata)
        else:
            # Process all datasets in raw mode
            process_single_dataset_raw('K562', metadata)
            process_single_dataset_raw('RPE1', metadata)
            process_single_dataset_raw('Jurkat', metadata)
        
        print("\n" + "=" * 60)
        print("Raw data processing complete!")
        print("=" * 60)
        print(f"\nOutput files saved to: {output_dir}")
        if args.dataset is not None:
            print(f"  - {args.dataset}_raw.h5ad")
        else:
            print("  - k562_raw.h5ad")
            print("  - rpe1_raw.h5ad")
            print("  - jurkat_raw.h5ad")
        
    else:  # mode == 'all'
        # Original sequential processing
        metadata = load_metadata_and_compute_union()
        
        # Process all datasets
        union_paths = []
        union_paths.append(process_single_dataset('K562', metadata))
        union_paths.append(process_single_dataset('RPE1', metadata))
        union_paths.append(process_single_dataset('Jurkat', metadata))
        
        # Concatenate and compute HVG
        concatenate_and_compute_hvg()
        
        print("\n" + "=" * 60)
        print("All processing complete!")
        print("=" * 60)
        print(f"\nOutput files saved to: {output_dir}")
        print("  - k562_perturbqa_reformatted.h5ad")
        print("  - rpe1_perturbqa_reformatted.h5ad")
        print("  - jurkat_perturbqa_reformatted.h5ad")
        print("  - combined_perturbqa_hvg2k.h5ad")


if __name__ == "__main__":
    main()
