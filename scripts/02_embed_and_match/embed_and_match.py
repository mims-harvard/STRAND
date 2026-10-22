# Load necessary extensions and libraries
#%%
# %load_ext autoreload
# %autoreload 2

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import anndata as ad
import os
import datetime
import re
import argparse
from ot_utils import bootstrap_cells
import gc

# SCimilarity is only required when generating embeddings.
try:
    from scimilarity.utils import lognorm_counts, align_dataset
    from scimilarity import CellQuery
except Exception:
    lognorm_counts = None
    align_dataset = None
    CellQuery = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(_repo_root() / "data" / "processed"))).expanduser()


def _maybe_add_hiref_to_syspath(hiref_src: str):
    """
    External dependency: HiRef (OT solver) is not vendored in this repository.
    Provide `--hiref_src /path/to/HiRef/src` (or env HIREF_SRC).
    """
    if not hiref_src:
        return
    p = str(Path(hiref_src).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


parser = argparse.ArgumentParser(description="Compute SCimilarity embeddings + OT matching indices.")
parser.add_argument(
    "--data_root",
    type=str,
    default=str(_default_data_root()),
    help="Root folder containing per-dataset directories (see scripts/01_curate/make_layout.py).",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="k562",
    choices=["k562", "jurkat", "rpe1", "combined_perturbqa_hvg2k"],
)
parser.add_argument("--chunk_id", type=int, default=0, help="Chunk id for embedding generation.")
parser.add_argument(
    "--splits",
    nargs="*",
    default=["train", "test"],
    help="Splits to process for OT matching (default: train test).",
)
parser.add_argument(
    "--split_column",
    type=str,
    default="split",
    help="obs column containing split labels (default: split).",
)
parser.add_argument(
    "--scim_model_path",
    type=str,
    default=os.environ.get("SCIMILARITY_MODEL_PATH", ""),
    help="Path to SCimilarity model directory (required for --gen_embeddings).",
)
parser.add_argument(
    "--hiref_src",
    type=str,
    default=os.environ.get("HIREF_SRC", ""),
    help="Path to HiRef 'src' directory providing HR_OT and rank_annealing (required for --ot_match).",
)
parser.add_argument(
    "--ot_ids_tag",
    type=str,
    default="line_specific",
    help="Subfolder name under <DATA_ROOT>/ot_ids/ for writing OT index CSVs.",
)
parser.add_argument("--gen_embeddings", action="store_true", help="Generate SCimilarity embeddings (chunked).")
parser.add_argument("--agg_embeddings", action="store_true", help="Aggregate chunked embeddings into final h5.")
parser.add_argument("--ot_match", action="store_true", help="Compute OT-matched control/perturbation indices.")

args = parser.parse_args()

data_dir = Path(args.data_root).expanduser().resolve()
data_name = args.dataset
chunk_id = int(args.chunk_id)
splits_to_process = list(args.splits) if args.splits else ["train", "test"]

# Default mode: if user didn't pass any flags, do OT matching only.
Gen_data = bool(args.gen_embeddings)
Agg_data = bool(args.agg_embeddings)
OT_match = bool(args.ot_match) or (not Gen_data and not Agg_data)

# Initialize device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")
    print("Using CPU")

# External dependency: HiRef for OT
_maybe_add_hiref_to_syspath(args.hiref_src)
try:
    import HR_OT
    import rank_annealing
except Exception as e:
    if OT_match:
        raise ImportError(
            "HiRef OT dependency not found. Install/clone HiRef and set env HIREF_SRC or pass "
            "`--hiref_src /path/to/HiRef/src` so that `import HR_OT` and `import rank_annealing` work."
        ) from e
    HR_OT = None
    rank_annealing = None

# Dataset candidates
dataset_cand = [
    ('k562', 'k562', 'CRISPR-interference'),
    ('jurkat', 'jurkat', 'CRISPR-interference'),
    ('rpe1', 'rpe1', 'CRISPR-interference'),
    ('combined_perturbqa_hvg2k', 'combined_perturbqa_hvg2k', 'CRISPR-interference'),
]

RNA_embeddings = []
Text_embeddings = []
String_list = []
adata_dict = {}
perturb_dict_all = defaultdict(list)
data_idx = 0

# Print informative metadata at the start of the run
print("\n================ METADATA ================",flush=True)
print(f"Script: {os.path.basename(__file__)}",flush=True)
print(f"Timestamp: {datetime.datetime.now().isoformat()}",flush=True)
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",flush=True)
print(f"Python: {sys.version}",flush=True)
print(f"Working Directory: {os.getcwd()}",flush=True)
print(f"Selected Dataset: {data_name}",flush=True)
print("==========================================\n")


#%%

def get_gene_names(adata):
    """Get gene names from adata.var, handling both index and gene_name column"""
    if 'gene_name' in adata.var.columns:
        return adata.var.gene_name.values
    else:
        return adata.var.index.values


# Process the requested dataset
pert_type = dict((a, c) for a, _, c in dataset_cand).get(data_name, "CRISPR-interference")
dataset_dir = data_dir / data_name
dataset_dir.mkdir(parents=True, exist_ok=True)
embedding_path = dataset_dir / f"{data_name}.X_scimilarity.h5"

if Gen_data and not args.scim_model_path:
    raise ValueError("--scim_model_path is required when --gen_embeddings is enabled (or set env SCIMILARITY_MODEL_PATH).")

if Gen_data:
    if CellQuery is None or align_dataset is None or lognorm_counts is None:
        raise ImportError(
            "SCimilarity is required for embedding generation. Install it (e.g. `pip install scimilarity`) "
            "and ensure it is importable."
        )
    cq = CellQuery(args.scim_model_path)
    if Gen_data:
        
        print(f'Starting processing for dataset: {data_name} with perturbation type: {pert_type}',flush=True)
        # Load raw file for embedding generation (not standardized, perfect for scimilarity normalization)
        h5ad_file = data_dir / f"{data_name}"/ f"{data_name}_raw_with_split.h5ad"
        if not h5ad_file.exists():
            raise FileNotFoundError(f"Raw file not found: {h5ad_file}")
        adata = sc.read_h5ad(h5ad_file, backed='r')
        print(f'Loaded dataset {data_name} with shape {adata.shape}',flush=True)
        if "perturbation_type" not in adata.obs.columns: ## add perturbation type if not present (newly added)
            adata.obs['perturbation_type'] = pert_type
        n_cells = adata.shape[0]

        n_chunks = 10
        chunk_size = (n_cells + n_chunks - 1) // n_chunks  # ceil division

        batch_size = 10000

        start = chunk_id * chunk_size
        end   = min(n_cells, start + chunk_size)
        print(f'original adata shape: {adata.shape}',flush=True)
        
        # Data is already curated - verify curated_perturbation column exists
        if 'curated_perturbation' not in adata.obs.columns:
            raise ValueError(f"'curated_perturbation' column not found in {data_name}. Data must be pre-curated.")
        if 'split' not in adata.obs.columns:
            raise ValueError(f"'split' column not found in {data_name}. Data must have train/test splits.")
        print(f'Verified curated data with shape {adata.shape}',flush=True)
        
        adata_dict[data_name] = [adata, pert_type]
        # perturb_dict_all = perturb_dict_all | perturb_dict
        
        # Plotting
        curated_perturbation = adata.obs['curated_perturbation'].dropna().values
        perturb_type, perturb_count = np.unique(curated_perturbation, return_counts=True)
        
        
        # Process in batches
        print(f'Encoding....',flush=True)
        embedding_dim = None
        embeddings = None
        # embeddings_list = []
        print(f'Start initializing embeddings array',flush=True)
        
        # Write chunked embeddings under the dataset folder.
        chunks_dir = dataset_dir / "_scim_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out_file = chunks_dir / f"{data_name}.chunk-{chunk_id:04d}.h5"
        if end > start:
            with h5py.File(out_file, 'w') as h5f:
                dset = None
                for start_idx in range(start, end, batch_size):
                    end_idx = min(start_idx + batch_size, end)
                    print(f"Processing batch {start_idx//batch_size + 1}: cells {start_idx}-{end_idx}", flush=True)
                    batch_slice = slice(start_idx, end_idx)
                    adata_filter_batch = adata[batch_slice].to_memory()
                    # Convert gene names to gene symbols before alignment
                    gene_names = get_gene_names(adata_filter_batch)
                    # converted_genes = convert_gene_name(list(gene_names))
                    gene_names = pd.Index(gene_names).astype(str)

                    # Make unique: add suffix to duplicates
                    gene_series = gene_names.to_series()
                    counts = gene_series.groupby(gene_series).cumcount()
                    gene_names = pd.Index([f"{name}_{count}" if count > 0 else name 
                                        for name, count in zip(gene_series, counts)])

                    adata_filter_batch.var.index = gene_names
                    
                    # Load only this batch
                    adata_filter_batch = align_dataset(adata_filter_batch, cq.gene_order)
                    print(f'adata_filter_batch.X.shape: {adata_filter_batch.X.shape}')
                    if hasattr(adata_filter_batch.X, 'toarray'):
                        batch_X = adata_filter_batch.X.toarray()
                    else:
                        batch_X = adata_filter_batch.X
                    
                    # Apply lognorm to batch
                    adata_filter_batch.layers['counts'] = batch_X.copy()
                    adata_filter_batch = lognorm_counts(adata_filter_batch)
                    
                    # Get embeddings for this batch
                    batch_embeddings = cq.get_embeddings(adata_filter_batch.X)
                    if dset is None:
                        embedding_dim = batch_embeddings.shape[1]
                        dset = h5f.create_dataset(
                            "X_scimilarity",
                            shape=(end - start, embedding_dim),
                            dtype='float32',
                            chunks=(min(batch_size, end - start), embedding_dim),
                            compression='lzf'
                        )
                        
                    dset[start_idx - start : end_idx - start] = batch_embeddings


                    
                    # Clear batch memory
                    del batch_X, batch_embeddings
                    gc.collect()

if Agg_data:
    print(f'Aggregating embeddings for dataset: {data_name}',flush=True)
    tmp_dir = dataset_dir / "_scim_chunks"
    out_file = embedding_path

    chunk_files = sorted(tmp_dir.glob(f'{data_name}.chunk-*.h5'),
                     key=lambda x: int(re.search(r'chunk-(\d+)', x.name).group(1)))
    
    if not chunk_files:
        print(f"ERROR: No chunk files found in {tmp_dir}. Make sure Gen_data=True was executed first.", flush=True)
        raise FileNotFoundError(f"No chunk files found for {data_name}")
    
    n_cells = 0
    embedding_dim = None
    for f in tqdm(chunk_files):
        with h5py.File(f, 'r') as h5f:
            if 'X_scimilarity' not in h5f:
                print(f"Skipping empty chunk file: {f}")
                continue
            dset = h5f['X_scimilarity']
            if embedding_dim is None:
                embedding_dim = dset.shape[1]
            n_cells += dset.shape[0]

    # Create output file
    with h5py.File(out_file, 'w') as h5f_out:
        dset_out = h5f_out.create_dataset(
            "X_scimilarity",
            shape=(n_cells, embedding_dim),
            dtype='float32',
            compression='lzf'
        )
        start = 0
        for f in chunk_files:
            with h5py.File(f, 'r') as h5f_in:
                if 'X_scimilarity' not in h5f_in:
                    print(f"Skipping empty chunk file: {f}")
                    continue
                data = h5f_in['X_scimilarity'][:]
                end = start + data.shape[0]
                dset_out[start:end] = data
                start = end
                
if OT_match:
    print(f'OT matching for dataset: {data_name}',flush=True)
    # Try loading with _with_split suffix first, fallback to original name
    h5ad_file = dataset_dir / f"{data_name}.h5ad"
    # if not h5ad_file.exists():
    #     h5ad_file = data_dir / f"{data_name}"/ f"{data_name}.h5ad"
    adata = sc.read_h5ad(h5ad_file, backed='r')
    # Verify curated data
    if 'curated_perturbation' not in adata.obs.columns:
        raise ValueError(f"'curated_perturbation' column not found in {data_name}. Data must be pre-curated.")
    if args.split_column not in adata.obs.columns:
        raise ValueError(
            f"Split column '{args.split_column}' not found in {data_name}.h5ad; "
            "create it with scripts/01_curate/make_layout.py"
        )
    print(f'Loaded curated data with shape {adata.shape}',flush=True)
    ctrl_indices = []
    pert_indices = []
    perturbation_info = []
    pair_adata_whole = []
    # adata = adata_dict[data_name]
    scaler = StandardScaler()
    
    # Get split information from adata
    split_values = adata.obs[args.split_column].values
    
    # Check if embedding file exists
    if not embedding_path.exists():
        print(f"ERROR: Embedding file {embedding_path} not found. Make sure Agg_data=True was executed first.", flush=True)
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    print(f"Loading embeddings from {embedding_path}...", flush=True)
    with h5py.File(str(embedding_path), 'r') as h5f:
        # Eager loading - load entire dataset into memory at once
        X_scimilarity = h5f["X_scimilarity"][:]
        print(f"Loaded embeddings with shape: {X_scimilarity.shape}", flush=True)
        
        # Process train and test splits separately
        # Only process splits specified in splits_to_process
        splits_to_process_filtered = [s for s in splits_to_process if s in ['train', 'test']]
        if not splits_to_process_filtered:
            print(f"Warning: No valid splits to process. Valid splits are 'train' and 'test'.", flush=True)
            print(f"Requested splits: {splits_to_process}", flush=True)
        else:
            print(f"Processing splits: {splits_to_process_filtered}", flush=True)
            for split_name in splits_to_process_filtered:
                print(f"\n{'='*60}")
                print(f"Processing {split_name.upper()} split")
                print(f"{'='*60}")
                
                # Filter adata by split
                split_mask = (split_values == split_name)
                split_indices = np.where(split_mask)[0]
                
                if len(split_indices) == 0:
                    print(f"No cells found in {split_name} split, skipping...")
                    continue
                
                # Create a view of adata for this split (using indices)
                # We need to work with the full adata but filter operations by split
                # Instead, we'll filter bootstrap_indices after calling bootstrap_cells on split-filtered data
                adata_split = adata[split_indices].to_memory()
                
                # Call bootstrap_cells with appropriate augmentation setting
                # Enable augmentation for both train and test to ensure sufficient samples for PCA
                # This ensures perturbations below baseline are bootstrapped to meet minimum requirements
                enable_aug = True
                # Enable cell-line matching if cell_line column exists (backward compatible)
                use_cell_line_matching = 'cell_line' in adata_split.obs.columns
                print(f"Calling bootstrap_cells with enable_augmentation={enable_aug}, match_within_cell_line={use_cell_line_matching} for {split_name} split", flush=True)
                bootstrap_indices_split = bootstrap_cells(
                    adata_split, 
                    data_name, 
                    random_state=1, 
                    enable_augmentation=enable_aug,
                    match_within_cell_line=use_cell_line_matching
                )
                print(f'Finished cell bootstrapping for {split_name} split',flush=True)
                
                # Map indices from split-filtered adata back to original adata indices
                # The bootstrap_indices_split contains indices relative to adata_split
                # We need to map them back to original adata indices
                split_indices_map = {i: orig_idx for i, orig_idx in enumerate(split_indices)}
                
                # Check if cell_line column exists for backward compatibility
                has_cell_line = 'cell_line' in adata.obs.columns
                if has_cell_line:
                    cell_line_values = adata.obs['cell_line'].astype('string').values
                    print(f"Cell-line-aware OT matching enabled (found 'cell_line' column)", flush=True)
                else:
                    print(f"Cell-line column not found - using legacy grouping (no cell-line stratification)", flush=True)
                
                # Group perturbations by (pert_type, cell_line) for cell-line-aware matching
                # bootstrap_indices_split structure: {batch_id: {pert_tag: (pert_idx, ctrl_idx)}}
                # New format: pert_tag may include ||cell_line=... suffix
                perturbation_groups = defaultdict(lambda: {'pert_idx': [], 'ctrl_idx': []})
                
                # Flatten bootstrap_indices across all batches (if any)
                for batch_id in bootstrap_indices_split.keys():
                    for pert_tag in bootstrap_indices_split[batch_id].keys():
                        # Parse cell_line from tag if present (new format: {pert_tag}||cell_line={cell_line})
                        base_tag = pert_tag
                        cell_line = None
                        if '||cell_line=' in pert_tag:
                            base_tag, cell_line_suffix = pert_tag.split('||cell_line=', 1)
                            cell_line = cell_line_suffix
                        
                        # Extract perturbation type from base_tag
                        # Format: {pert}-{data_name} or {pert}-{data_name}-batch_{batch_id}
                        if f'-{data_name}-batch_' in base_tag:
                            pert_type = base_tag.split(f'-{data_name}-batch_')[0]
                        elif f'-{data_name}' in base_tag:
                            pert_type = base_tag.split(f'-{data_name}')[0]
                        else:
                            # If format doesn't match, use the tag as-is
                            pert_type = base_tag
                        
                        # Create grouping key: (pert_type, cell_line) or just pert_type for backward compatibility
                        group_key = (pert_type, cell_line) if cell_line is not None else (pert_type, None)
                        
                        indices_pair = bootstrap_indices_split[batch_id][pert_tag]
                        
                        # Map indices from split-filtered adata back to original adata indices
                        pert_idx_mapped = [split_indices_map[idx] for idx in indices_pair[0]]
                        ctrl_idx_mapped = [split_indices_map[idx] for idx in indices_pair[1]]
                        
                        if len(pert_idx_mapped) > 0 and len(ctrl_idx_mapped) > 0:
                            perturbation_groups[group_key]['pert_idx'].extend(pert_idx_mapped)
                            perturbation_groups[group_key]['ctrl_idx'].extend(ctrl_idx_mapped)
                
                # Pre-allocate lists for better memory management
                ctrl_indices_split = []
                pert_indices_split = []
                perturbation_info_split = []
                cell_line_info_split = []  # Track cell_line for each group
                
                # Process each (pert_type, cell_line) group for this split
                for group_key in tqdm(perturbation_groups.keys(), desc=f"Processing {split_name} perturbation groups"):
                    pert_type, cell_line = group_key
                    
                    pert_idx = np.array(perturbation_groups[group_key]['pert_idx'], dtype=int)
                    ctrl_idx = np.array(perturbation_groups[group_key]['ctrl_idx'], dtype=int)
                    
                    # Skip if no data for this group
                    if len(pert_idx) == 0 or len(ctrl_idx) == 0:
                        group_label = f"{pert_type} (cell_line={cell_line})" if cell_line else pert_type
                        print(f"Skipping {group_label} in {split_name} - no data available")
                        continue
                    
                    # Verify all indices are in the correct split
                    assert np.all(split_values[pert_idx] == split_name), f"Perturbation indices not all in {split_name} split"
                    assert np.all(split_values[ctrl_idx] == split_name), f"Control indices not all in {split_name} split"
                    
                    # Verify cell_line consistency if cell-line matching is enabled
                    if has_cell_line and cell_line is not None:
                        pert_cell_lines = cell_line_values[pert_idx]
                        ctrl_cell_lines = cell_line_values[ctrl_idx]
                        unique_pert_cl = pd.unique(pert_cell_lines)
                        unique_ctrl_cl = pd.unique(ctrl_cell_lines)
                        assert len(unique_pert_cl) == 1 and unique_pert_cl[0] == cell_line, \
                            f"Perturbation cell lines don't match expected {cell_line}: found {unique_pert_cl}"
                        assert len(unique_ctrl_cl) == 1 and unique_ctrl_cl[0] == cell_line, \
                            f"Control cell lines don't match expected {cell_line}: found {unique_ctrl_cl}"
                        assert unique_pert_cl[0] == unique_ctrl_cl[0], \
                            f"Perturbation and control cell lines don't match: pert={unique_pert_cl[0]}, ctrl={unique_ctrl_cl[0]}"
                    
                    X1_orig = X_scimilarity[ctrl_idx]  # control
                    X2_orig = X_scimilarity[pert_idx]  # perturbation

                    # Combine control and perturbation data for PCA fitting
                    X_combined = np.concatenate([X1_orig, X2_orig], axis=0).copy()
                    
                    # Apply PCA to reduce dimension
                    # Dynamically adjust n_components based on sample size to avoid errors
                    n_samples = X_combined.shape[0]
                    n_components = min(50, n_samples - 1)  # Ensure n_components < n_samples
                    if n_components < 1:
                        group_label = f"{pert_type} (cell_line={cell_line})" if cell_line else pert_type
                        print(f"Warning: Skipping {group_label} in {split_name} - insufficient samples (n_samples={n_samples})", flush=True)
                        continue
                    group_label = f"{pert_type} (cell_line={cell_line})" if cell_line else pert_type
                    print(f"Applying PCA to reduce dimension from {X_combined.shape[1]} to {n_components} for {group_label} ({split_name}) (n_samples={n_samples})", flush=True)
                    pca = PCA(n_components=n_components, random_state=42)
                    X_combined_pca = pca.fit_transform(X_combined)
                    
                    # Split back into control and perturbation groups
                    X1_latent = X_combined_pca[:X1_orig.shape[0]]
                    X2_latent = X_combined_pca[X1_orig.shape[0]:]
                    
                    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}", flush=True)

                    X = torch.from_numpy(X1_latent).type(torch.DoubleTensor).to(device)
                    Y = torch.from_numpy(X2_latent).type(torch.DoubleTensor).to(device)
                    
                    n = X1_orig.shape[0]
                    rank_schedule = rank_annealing.optimal_rank_schedule(n, hierarchy_depth=6, max_Q=max(2**15, n), max_rank=2**10)
                    null_out = open(os.devnull, 'w')
                    if n < 2**10:
                        C = torch.cdist(X, Y)
                        hrot = HR_OT.HierarchicalRefinementOT(C, rank_schedule, base_rank=1, device=device)
                    else:
                        hrot = HR_OT.HierarchicalRefinementOT.init_from_point_clouds(
                            X, Y, rank_schedule, base_rank=1, device=device
                        )
                    F = hrot.run(return_as_coupling=False)
                    null_out.close()

                    couple_array = np.array([
                        (a.cpu().item(), b.cpu().item()) for a, b in F
                    ])
                    
                    ctrl_indices_split.append(ctrl_idx[couple_array[:, 0].astype(int)])
                    pert_indices_split.append(pert_idx[couple_array[:, 1].astype(int)])
                    
                    matched_pert_indices = pert_idx[couple_array[:, 1].astype(int)]
                    perturbation_info_split.append(adata.obs.iloc[matched_pert_indices]['curated_perturbation'].values)
                    
                    # Track cell_line for this group (repeat for each matched pair)
                    n_matched = len(matched_pert_indices)
                    if has_cell_line and cell_line is not None:
                        cell_line_info_split.append([cell_line] * n_matched)
                    else:
                        # For backward compatibility: use None or 'unknown' when cell_line not available
                        cell_line_info_split.append([None] * n_matched)
                    
                    # Clear GPU memory after each iteration
                    del X, Y, hrot, F, couple_array, X_combined, X_combined_pca, pca
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                # Concatenate results for this split
                if len(ctrl_indices_split) > 0:
                    ctrl_indices_split = np.concatenate(ctrl_indices_split, axis=0)
                    pert_indices_split = np.concatenate(pert_indices_split, axis=0)
                    datanames_split = [data_name] * ctrl_indices_split.shape[0]
                    perturbation_info_split = np.concatenate(perturbation_info_split, axis=0)
                    cell_line_info_split = np.concatenate(cell_line_info_split, axis=0) if cell_line_info_split else None
                    
                    # Build DataFrame with cell_line column if available
                    df_dict = {
                        'perturbation': perturbation_info_split,
                        'dataname': datanames_split,
                        'control_index': ctrl_indices_split,
                        'perturbation_index': pert_indices_split
                    }
                    if cell_line_info_split is not None:
                        df_dict['cell_line'] = cell_line_info_split
                    
                    data_info_split = pd.DataFrame(df_dict)
                    
                    # Permute dataframe
                    data_info_split = data_info_split.sample(frac=1).reset_index(drop=True)
                    
                    # Save to ot_ids directory under DATA_ROOT (portable)
                    ot_ids_dir = data_dir / "ot_ids" / args.ot_ids_tag / data_name
                    ot_ids_dir.mkdir(parents=True, exist_ok=True)
                    output_file = ot_ids_dir / f"{data_name}_ot_match.{split_name}.idx.csv"
                    data_info_split.to_csv(output_file, index=False)
                    print(f"Saved {split_name} OT matching results: {len(data_info_split)} pairs to {output_file}")
                else:
                    print(f"No OT matches found for {split_name} split")

# %%       
