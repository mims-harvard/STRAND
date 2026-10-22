import pandas as pd
import pickle
import numpy as np
import anndata
import torch
from scipy import sparse as sp
import os
import scanpy as sc
import copy
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import cdist
import biolord
import wandb
from tqdm import tqdm
import gc
from collections import defaultdict
import argparse
import warnings
import random
torch.set_float32_matmul_precision('high')

# --- PREPROCESSING (Unchanged) ---
def preprocess_adata(adata_path, gene_path, mapping_path, mask_path):
    print(f"[MEMORY DEBUG] preprocess_adata: Loading h5ad file from {adata_path}", flush=True)
    adata = sc.read_h5ad(adata_path)
    
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var.index

    # USE CUSTOM GENES
    print(f"[MEMORY DEBUG] preprocess_adata: Loading gene list from {gene_path}", flush=True)
    genes = list(pd.read_csv(gene_path, header=None)[0])
    
    mapping = pd.read_csv(mapping_path, sep='\t')
    
    mask = []
    with open(mask_path, 'r') as f:
        for line in f:
            mask.append(float(line))
    genes_subset = [g for g, m in zip(genes, mask) if m == 1]
    indices = []
    missing = []
    for gene in genes_subset:
        matches = mapping.index[mapping['ensembl_gene_id'] == gene].tolist()
        if not matches:
            missing.append(gene)
            continue
        
        pick = None
        official_idxs = [i for i in matches if str(mapping.loc[i, 'map_type']).lower().startswith('official')]
        if official_idxs:
            pick = official_idxs[0]
        if pick is None:
            pick = matches[0]
        indices.append(pick)
    indices = list(dict.fromkeys(indices))
    
    adata = adata[:, indices].copy()
    print(sum(mask), len(missing), adata.X.shape)

    perturbs = []
    for perturbation in adata.obs['curated_perturbation']:
        perturbation = perturbation.rsplit('-', 2)[0]
        if perturbation == 'control':
            perturbs.append('ctrl')
        elif '_' not in perturbation:
            perturbs.append(perturbation + '+ctrl')
        else:
            combo = perturbation.split('_')
            perturbs.append(combo[0] + '+' + combo[1])

    adata.obs['perturbation'] = perturbs
    adata.obs.rename(columns={'perturbation': 'condition', 'celltype': 'cell_type'}, inplace=True)
    adata.obs['ind'] = adata.obs.index
    adata.obs['condition'] = adata.obs['condition'].to_numpy()
    
    return adata

def preprocess_adata_with_hgnc(adata_path, gene_path, mapping_path, mask_path):
    print(f"[MEMORY DEBUG] preprocess_adata_with_hgnc: Loading h5ad file from {adata_path}", flush=True)
    adata = sc.read_h5ad(adata_path)
    
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var.index
    
    genes = list(pd.read_csv(gene_path, header=None)[0])
    mapping = pd.read_csv(mapping_path, sep='\t', low_memory=False)
    
    mask = []
    with open(mask_path, 'r') as f:
        for line in f:
            mask.append(float(line))
    genes_subset = [g for g, m in zip(genes, mask) if m == 1]
    
    indices = []
    missing = []
    
    for gene in genes_subset:
        matches = mapping[mapping['ensembl_gene_id'] == gene]
        if len(matches) == 0:
            missing.append(gene)
            continue
        
        if 'map_type' in matches.columns:
            official_matches = matches[matches['map_type'].astype(str).str.lower().str.startswith('official')]
            if len(official_matches) > 0:
                selected_row = official_matches.iloc[0]
            else:
                selected_row = matches.iloc[0]
        else:
            selected_row = matches.iloc[0]
        
        if 'symbol' not in selected_row.index:
            missing.append(gene)
            continue
            
        gene_symbol = str(selected_row['symbol'])
        
        adata_var_matches = adata.var[adata.var['gene_name'] == gene_symbol]
        if len(adata_var_matches) > 0:
            adata_idx = adata_var_matches.index[0]
            adata_pos = adata.var.index.get_loc(adata_idx)
            indices.append(adata_pos)
        elif gene_symbol in adata.var.index:
            adata_pos = adata.var.index.get_loc(gene_symbol)
            indices.append(adata_pos)
        else:
            missing.append(gene)
    
    indices = list(dict.fromkeys(indices))
    indices = sorted(indices)
    
    if len(indices) == 0:
        raise ValueError(f"No genes could be matched between mapping file and adata. Missing: {len(missing)} genes.")
    
    adata = adata[:, indices].copy()
    print(f"HVG mask: {sum(mask)} genes, Matched: {len(indices)} genes, Missing: {len(missing)} genes, Final shape: {adata.X.shape}")

    perturbs = []
    for perturbation in adata.obs['curated_perturbation']:
        perturbation = perturbation.rsplit('-', 2)[0]
        if perturbation == 'control':
            perturbs.append('ctrl')
        elif '_' not in perturbation:
            perturbs.append(perturbation + '+ctrl')
        else:
            combo = perturbation.split('_')
            perturbs.append(combo[0] + '+' + combo[1])

    adata.obs['perturbation'] = perturbs
    adata.obs.rename(columns={'perturbation': 'condition', 'celltype': 'cell_type'}, inplace=True)
    adata.obs['ind'] = adata.obs.index
    adata.obs['condition'] = adata.obs['condition'].to_numpy()
    
    return adata

def _detect_mapping_format(mapping_path):
    try:
        mapping = pd.read_csv(mapping_path, sep='\t', nrows=5, low_memory=False)
        if 'var_index' in mapping.columns:
            return 'original'
        elif 'symbol' in mapping.columns and 'ensembl_gene_id' in mapping.columns:
            return 'hgnc'
        else:
            return 'unknown'
    except Exception as e:
        print(f"Warning: Could not detect mapping format: {e}")
        return 'unknown'

def preprocess_adata_auto(adata_path, gene_path, mapping_path, mask_path, go_graph_path=None, go_gene_path=None):
    mapping_format = _detect_mapping_format(mapping_path)
    
    if mapping_format == 'hgnc':
        print("Detected HGNC mapping format, using HGNC-based preprocessing")
        adata = preprocess_adata_with_hgnc(adata_path, gene_path, mapping_path, mask_path)
    elif mapping_format == 'original':
        print("Detected original mapping format, using original preprocessing")
        adata = preprocess_adata(adata_path, gene_path, mapping_path, mask_path)
    else:
        print(f"Warning: Unknown mapping format, defaulting to original preprocessing")
        adata = preprocess_adata(adata_path, gene_path, mapping_path, mask_path)
    
    if go_graph_path and go_gene_path:
        adata = add_go_features(adata, go_graph_path, go_gene_path)
    
    return adata

def add_go_features(adata, go_graph_path, go_gene_path, perturbation_col="condition"):
    print(f"[MEMORY DEBUG] add_go_features: Loading GO graph from {go_graph_path}", flush=True)
    df = pd.read_csv(go_graph_path)
    df = df.groupby('target').apply(lambda x: x.nlargest(20 + 1, ['importance'])).reset_index(drop=True)
    
    print(f"[MEMORY DEBUG] add_go_features: Loading gene list from {go_gene_path}", flush=True)
    with open(go_gene_path, 'rb') as f:
        gene_list = pickle.load(f)
    
    df = df[df["source"].isin(gene_list)]
    adata_perts = list(adata.obs[perturbation_col].unique())
    
    def get_map(pert):
        tmp = pd.DataFrame(np.zeros(len(gene_list)), index=gene_list)
        pert_data = df[df.target == pert]
        
        if len(pert_data) == 0 and '+' in pert:
            clean_pert = pert.split('+')[0]
            pert_data = df[df.target == clean_pert]
        
        if len(pert_data) > 0:
            tmp.loc[pert_data.source.values, :] = pert_data.importance.values[:, np.newaxis]
            return tmp.values.flatten()
        else:
            return tmp.values.flatten()
    
    pert2neighbor = {}
    for pert in adata_perts:
        pert_map = get_map(pert)
        pert2neighbor[pert] = pert_map
    
    adata.uns["pert2neighbor"] = pert2neighbor
    
    pert2neighbor_array = np.asarray([val for val in adata.uns["pert2neighbor"].values()])
    keep_idx = pert2neighbor_array.sum(0) > 0
    keep_idx1 = pert2neighbor_array.sum(0) > 1
    keep_idx2 = pert2neighbor_array.sum(0) > 2
    keep_idx3 = pert2neighbor_array.sum(0) > 3
    
    adata.uns["go_keep_idx"] = keep_idx
    adata.uns["go_keep_idx1"] = keep_idx1
    adata.uns["go_keep_idx2"] = keep_idx2
    adata.uns["go_keep_idx3"] = keep_idx3
    
    return adata

# --- SUBSET CREATION (Simpler now) ---

def create_split_subset_adata(adata, split_value, subset_name="subset", split_column_name="split"):
    """
    Create subset of adata based on split column value.
    
    Parameters:
    -----------
    adata : AnnData
        Input AnnData object
    split_value : str
        Value in split column to filter by ('train' or 'test')
    subset_name : str
        Name for the subset (for logging)
    split_column_name : str
        Name of the column in adata.obs to use for splitting (default: "split")
    """
    if split_column_name not in adata.obs.columns:
        raise ValueError(f"'{split_column_name}' column not found in adata.obs. Cannot create split-based subset.")
    
    # Filter by split value (case-insensitive)
    split_mask = adata.obs[split_column_name].astype(str).str.lower() == split_value.lower()
    split_indices = np.where(split_mask)[0]
    
    if len(split_indices) == 0:
        raise ValueError(f"No cells found with {split_column_name}='{split_value}'")
    
    print(f"[MEMORY DEBUG] create_split_subset_adata: About to copy subset with {len(split_indices)} cells, original shape: {adata.X.shape}", flush=True)
    if torch.cuda.is_available():
        print(f"[MEMORY DEBUG] create_split_subset_adata: GPU memory before copy: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    
    subset_adata = adata[split_indices].copy()
    
    print(f"[MEMORY DEBUG] create_split_subset_adata: After copy, subset shape: {subset_adata.X.shape}", flush=True)
    if torch.cuda.is_available():
        print(f"[MEMORY DEBUG] create_split_subset_adata: GPU memory after copy: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    
    subset_adata.obs['original_index'] = split_indices
    subset_adata.obs['subset_index'] = range(len(split_indices))
    
    print(f"Created {subset_name} subset: {len(split_indices)} cells from {split_column_name}='{split_value}'")
    
    return subset_adata

# --- HELPERS ---

def _infer_ordered_key_and_keep_idx(train_adata):
    """Return (ordered_key, keep_idx) used during training, or (None, None)."""
    ordered_candidates = [
        "perturbation_neighbors1", "perturbation_neighbors2",
        "perturbation_neighbors3", "perturbation_neighbors"
    ]
    ordered_key = next((k for k in ordered_candidates if k in train_adata.obsm), None)

    go_keep_for = {
        "perturbation_neighbors":  "go_keep_idx",
        "perturbation_neighbors1": "go_keep_idx1",
        "perturbation_neighbors2": "go_keep_idx2",
        "perturbation_neighbors3": "go_keep_idx3",
    }
    keep_idx = train_adata.uns.get(go_keep_for.get(ordered_key, ""), None)
    return ordered_key, keep_idx

def _seen_categories(series):
    """Set of seen categories for a pandas Series (respects categorical dtype)."""
    import pandas as pd
    if series is None or len(series) == 0:
        return set()
    if pd.api.types.is_categorical_dtype(series):
        return set(series.cat.categories)
    return set(series.astype(str).unique())

def _target_ordered_embedding(train_adata, test_adata, pert_row_idx, ordered_key, keep_idx, perturbation_col="condition"):
    import numpy as np
    # If test_adata already has the ordered embeddings, just copy that row's vector
    if ordered_key is not None and ordered_key in getattr(test_adata, "obsm", {}):
        vec = test_adata[pert_row_idx:pert_row_idx+1].obsm[ordered_key]
        return np.asarray(vec, dtype=np.float32)

    # Otherwise, build from the global GO mapping on the training object
    if "pert2neighbor" in train_adata.uns and ordered_key is not None and keep_idx is not None:
        cond = str(test_adata[pert_row_idx].obs.get(perturbation_col, "")) \
               if perturbation_col in test_adata.obs.columns else \
               str(test_adata[pert_row_idx].obs.get("perturbation", ""))
        base = train_adata.uns["pert2neighbor"].get(cond, None)
        if base is not None:
            v = np.asarray(base)[keep_idx].reshape(1, -1).astype(np.float32)
            return v

        # fallback: zero vector if we have no GO vector for this pert
        n_feats = int(keep_idx.sum())
        return np.zeros((1, n_feats), dtype=np.float32)

    return None

def setup_biolord_model(adata, varying_args=None, perturbation_col="condition"):
    default_params = {
        "seed": 1,
        "unknown_attribute_noise_param": 0.2,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "step_size_lr": 45,
        "attribute_dropout_rate": 0,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
        "n_latent": 32,
        "n_latent_attribute_ordered": 32,
        "reconstruction_penalty": 10000.0,
        "attribute_nn_width": 64,
        "attribute_nn_depth": 2,
        "attribute_nn_lr": 0.001,
        "attribute_nn_wd": 4e-8,
        "latent_lr": 0.01,
        "latent_wd": 0.00001,
        "decoder_width": 32,
        "decoder_depth": 2,
        "decoder_activation": True,
        "attribute_nn_activation": True,
        "unknown_attributes": False,
        "decoder_lr": 0.01,
        "decoder_wd": 0.01,
        "max_epochs": 200,
        "early_stopping_patience": 200,
        "ordered_attributes_key": "perturbation_neighbors1",
        "n_latent_attribute_categorical": 16,
        "validation_size": 0.0,
        "train_size": 1.0
    }
    
    if varying_args:
        default_params.update(varying_args)
    
    ordered_attributes_key = default_params.get("ordered_attributes_key", "perturbation_neighbors1")
    
    categorical_attributes_map = {}
    # unique_conditions = adata.obs[perturbation_col].unique()
    # categorical_attributes_map[perturbation_col] = {
    #     condition: idx for idx, condition in enumerate(unique_conditions)
    # }
    
    if 'cell_type' in adata.obs.columns:
        unique_cell_types = adata.obs['cell_type'].unique()
        categorical_attributes_map['cell_type'] = {
            cell_type: idx for idx, cell_type in enumerate(unique_cell_types)
        }
    
    ordered_attributes_map = None
    if 'pert2neighbor' in adata.uns and adata.uns['pert2neighbor']:
        go_key_map = {
            "perturbation_neighbors": "go_keep_idx",
            "perturbation_neighbors1": "go_keep_idx1", 
            "perturbation_neighbors2": "go_keep_idx2",
            "perturbation_neighbors3": "go_keep_idx3"
        }
        
        if ordered_attributes_key and ordered_attributes_key in go_key_map:
            keep_idx_key = go_key_map[ordered_attributes_key]
            
            if keep_idx_key in adata.uns:
                keep_idx = adata.uns[keep_idx_key]
                n_ordered_features = int(keep_idx.sum())
                
                if n_ordered_features > 0:
                    ordered_attributes_map = {
                        ordered_attributes_key: n_ordered_features
                    }
                    
                    if ordered_attributes_key not in adata.obsm:
                        print(f"[MEMORY DEBUG] setup_biolord_model: Creating GO embeddings array for {len(adata.obs)} cells, {n_ordered_features} features", flush=True)
                        if torch.cuda.is_available():
                            print(f"[MEMORY DEBUG] setup_biolord_model: GPU memory before GO embeddings: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
                        go_embeddings = []
                        for condition in adata.obs[perturbation_col]:
                            if condition in adata.uns['pert2neighbor']:
                                embedding = adata.uns['pert2neighbor'][condition][keep_idx]
                            else:
                                embedding = np.zeros(n_ordered_features)
                            go_embeddings.append(embedding)
                        
                        adata.obsm[ordered_attributes_key] = np.array(go_embeddings)
                        if torch.cuda.is_available():
                            print(f"[MEMORY DEBUG] setup_biolord_model: GPU memory after GO embeddings: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)

    if sp.issparse(adata.X):
        data = adata.X.data
        if np.isnan(data).any() or np.isinf(data).any():
            cleaned = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            adata.X.data[:] = cleaned
    else:
        X = adata.X
        if np.isnan(X).any() or np.isinf(X).any():
            adata.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if ordered_attributes_map:
        for key in ordered_attributes_map.keys():
            if key in adata.obsm:
                obsm_data = adata.obsm[key]
                obsm_nan = np.isnan(obsm_data).sum()
                obsm_inf = np.isinf(obsm_data).sum()
                
                if obsm_nan > 0 or obsm_inf > 0:
                    adata.obsm[key] = np.nan_to_num(obsm_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    categorical_attributes_keys = list(categorical_attributes_map.keys()) if categorical_attributes_map else []
    ordered_attributes_keys = list(ordered_attributes_map.keys()) if ordered_attributes_map else []
    
    biolord.Biolord.setup_anndata(
        adata=adata,
        categorical_attributes_keys=categorical_attributes_keys,
        ordered_attributes_keys=ordered_attributes_keys
    )
    
    model_params = {}
    training_params = {}
    
    biolord_model_param_keys = [
        'n_latent', 'model_name', 'train_classifiers', 'split_key', 
        'train_split', 'valid_split', 'test_split'
    ]
    
    module_param_keys = [
        'n_latent_attribute_categorical', 'n_latent_attribute_ordered',
        'reconstruction_penalty', 'unknown_attribute_penalty', 'use_batch_norm', 
        'use_layer_norm', 'unknown_attribute_noise_param', 'unknown_attributes',
        'attribute_dropout_rate', 'decoder_width', 'decoder_depth', 
        'decoder_activation', 'attribute_nn_width', 'attribute_nn_depth',
        'attribute_nn_activation', 'eval_r2_ordered', 'decoder_dropout_rate'
    ]
    
    for key in biolord_model_param_keys:
        if key in default_params:
            model_params[key] = default_params[key]
    
    module_params = {}
    for key in module_param_keys:
        if key in default_params:
            module_params[key] = default_params[key]
    
    training_param_keys = [
        'max_epochs', 'early_stopping_patience', 'batch_size', 'train_size',
        'validation_size', 'early_stopping', 'accelerator', 'device'
    ]
    
    plan_param_keys = [
        'latent_lr', 'latent_wd', 'decoder_lr', 'decoder_wd', 
        'attribute_nn_lr', 'attribute_nn_wd', 'cosine_scheduler', 
        'scheduler_final_lr', 'step_size_lr'
    ]
    
    for key in training_param_keys:
        if key in default_params:
            training_params[key] = default_params[key]
    
    plan_params = {}
    for key in plan_param_keys:
        if key in default_params:
            plan_params[key] = default_params[key]
    
    model = biolord.Biolord(
        adata=adata,
        module_params=module_params if module_params else None,
        **model_params
    )
    
    model.training_params = training_params
    model.plan_params = plan_params
    
    return model

def get_plan_kwargs(custom_plan_kwargs=None):
    default_plan_kwargs = {
        'latent_lr': 0.001,
        'latent_wd': 1e-6,
        'decoder_lr': 0.001,
        'decoder_wd': 1e-6,
        'attribute_nn_lr': 0.001,
        'attribute_nn_wd': 1e-6,
        'cosine_scheduler': False,
        'scheduler_final_lr': 1e-6,
        'step_size_lr': 50,
    }
    
    if custom_plan_kwargs:
        default_plan_kwargs.update(custom_plan_kwargs)
    
    return default_plan_kwargs

def setup_biolord_with_split_column(adata, varying_args=None, perturbation_col="condition", control_label="ctrl", split_column_name="split"):
    """
    Setup BioLord model using split column from adata.obs without pairing.
    
    Parameters:
    -----------
    adata : AnnData
        Input AnnData object
    varying_args : dict, optional
        Model configuration parameters
    perturbation_col : str
        Name of the perturbation column
    control_label : str
        Label for control cells
    split_column_name : str
        Name of the column in adata.obs to use for splitting (default: "split")
    """
    # Create train and test subsets
    train_adata = create_split_subset_adata(
        adata, split_value='train', subset_name="training", split_column_name=split_column_name
    )
    
    test_adata = create_split_subset_adata(
        adata, split_value='test', subset_name="testing", split_column_name=split_column_name
    )
    
    # Setup model with training data (UNPAIRED)
    model = setup_biolord_model(train_adata, varying_args=varying_args, perturbation_col=perturbation_col)
    
    # Attach test data to model for evaluation convenience
    model.test_adata = test_adata
    
    print(f'\nTrain Data Shape: {train_adata.X.shape}')
    print(f'Test Data Shape: {test_adata.X.shape}')
    
    return model, train_adata, test_adata

def train_biolord_model(model, use_wandb=True, wandb_project='biolord-benchmark', wandb_run_name=None, **override_params):
    import torch.nn as nn
    
    # Monkey-patch BatchNorm1d to handle batch_size=1
    if not hasattr(nn.BatchNorm1d, '_original_forward'):
        nn.BatchNorm1d._original_forward = nn.BatchNorm1d.forward
        
        def _safe_bn_forward(self, input):
            if self.training and input.size(0) == 1:
                # Use eval mode for single-sample batches (uses running stats)
                self.eval()
                result = nn.BatchNorm1d._original_forward(self, input)
                self.train()
                return result
            return nn.BatchNorm1d._original_forward(self, input)
        
        nn.BatchNorm1d.forward = _safe_bn_forward
        print("Applied BatchNorm1d monkey-patch for batch_size=1 compatibility")
    
    if not hasattr(model, 'training_params'):
        training_params = {
            'max_epochs': 200,
            'early_stopping_patience': 200
        }
        plan_params = {}
    else:
        training_params = model.training_params.copy()
        plan_params = getattr(model, 'plan_params', {}).copy()

    training_params.update(override_params)
    plan_kwargs = get_plan_kwargs(plan_params)

    training_kwargs = {
        'max_epochs': training_params.get('max_epochs', 200),
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'plan_kwargs': plan_kwargs,
        'enable_checkpointing': False,
        'enable_progress_bar': True,
        'enable_model_summary': False,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': None,
        'validation_size': 0.0,
        'train_size': 1.0,
    }

    if use_wandb:
        wandb_logger = create_wandb_logger(wandb_project, wandb_run_name)
        if wandb_logger is not None:
            training_kwargs['logger'] = wandb_logger
            print("Using PyTorch Lightning WandB logger")
        else:
            training_kwargs['logger'] = False
            print("No WandB logger available, continuing without logging")
    else:
        training_kwargs['logger'] = False

    excluded_params = ['max_epochs', 'accelerator', 'devices', 'plan_kwargs',
                       'validation_size', 'train_size']
    for key, value in training_params.items():
        if key not in excluded_params:
            training_kwargs[key] = value

    print("Starting training (Standard Unpaired Strategy)...")
    print(f"[MEMORY DEBUG] train_biolord_model: About to start training", flush=True)
    if torch.cuda.is_available():
        print(f"[MEMORY DEBUG] train_biolord_model: GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    model.train(**training_kwargs)
    print(f"[MEMORY DEBUG] train_biolord_model: Training completed", flush=True)
    if torch.cuda.is_available():
        print(f"[MEMORY DEBUG] train_biolord_model: GPU memory after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)

    return model

def run_biolord_pipeline(adata_path, train_csv_path=None, test_csv_path=None, gene_path=None, mapping_path=None, mask_path=None, 
                         go_graph_path=None, go_gene_path=None, custom_params=None, use_split_column="split",
                         use_wandb=True, wandb_project="biolord-benchmark", wandb_run_name=None, max_pairs_per_pert=None,
                         perturbation_col="condition", control_label="ctrl"):
    """
    Run BioLord pipeline with split column mode.
    
    Parameters:
    -----------
    use_split_column : str
        Column name in adata.obs to use for train/test split (default: "split")
        Can be "split" or "split_few_shot"
    """
    # Set up default parameters
    default_params = {
        "seed": 1,
        "n_latent": 32,
        "max_epochs": 200,
        # ... (rest of defaults implicitly handled in setup_biolord_model)
    }
    
    if custom_params:
        default_params.update(custom_params)
    
    # Load and preprocess adata
    print(f"Loading adata for split-column mode (using column: {use_split_column})...")
    print(f"[MEMORY DEBUG] run_biolord_pipeline: Loading h5ad file from {adata_path}", flush=True)
    if torch.cuda.is_available():
        print(f"[MEMORY DEBUG] run_biolord_pipeline: GPU memory before load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    adata = sc.read_h5ad(adata_path)
    print(f"[MEMORY DEBUG] run_biolord_pipeline: Loaded adata shape: {adata.X.shape}, sparse: {sp.issparse(adata.X)}", flush=True)
    
    if use_split_column not in adata.obs.columns:
        raise ValueError(f"'{use_split_column}' column not found in adata.obs. Cannot use split-column mode.")
    if perturbation_col not in adata.obs.columns:
        raise ValueError(f"'{perturbation_col}' column not found in adata.obs. Required for BioLord.")
    
    if 'gene_name' not in adata.var.columns:
        print("Warning: 'gene_name' not found in adata.var. Using var.index as gene_name.")
        adata.var['gene_name'] = adata.var.index
    if 'ind' not in adata.obs.columns:
        adata.obs['ind'] = adata.obs.index
    
    if go_graph_path and go_gene_path:
        print("Adding GO features...")
        adata = add_go_features(adata, go_graph_path, go_gene_path, perturbation_col=perturbation_col)
    
    # Setup model (UNPAIRED)
    model, train_adata, test_adata = setup_biolord_with_split_column(
        adata=adata,
        varying_args=default_params,
        perturbation_col=perturbation_col,
        control_label=control_label,
        split_column_name=use_split_column
    )
    
    # Train model
    trained_model = train_biolord_model(
        model,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        **{k: v for k, v in default_params.items() if k in ['max_epochs']}
    )
        
    return trained_model, train_adata, test_adata

def create_wandb_logger(project_name, run_name=None):
    try:
        from pytorch_lightning.loggers import WandbLogger
        return WandbLogger(
            project=project_name,
            name=run_name,
            log_model=False,
            save_dir='./biolord_wandb_logs'
        )
    except ImportError:
        print("Warning: pytorch_lightning WandbLogger not available, proceeding without logger")
        return None

def sample_random_control_cells(adata, n_samples=256, control_label='non-targeting', return_all=False, perturbation_col='condition'):
    # (Kept identical to previous turn logic)
    control_mask = None
    if perturbation_col in adata.obs.columns:
        mask_exact = adata.obs[perturbation_col] == control_label
        if mask_exact.sum() > 0:
            control_mask = mask_exact
        else:
            mask_contains = adata.obs[perturbation_col].astype(str).str.contains(control_label, case=False, na=False)
            if mask_contains.sum() > 0:
                control_mask = mask_contains

    if control_mask is None or control_mask.sum() == 0:
        raise ValueError(f"No control cells found with pattern '{control_label}'")
    
    control_indices = np.where(control_mask)[0]
    
    if return_all:
        return control_indices
    
    if len(control_indices) >= n_samples:
        sampled_indices = np.random.choice(control_indices, size=n_samples, replace=False)
    else:
        sampled_indices = np.random.choice(control_indices, size=n_samples, replace=True)
    
    return sampled_indices

def evaluate_batch_efficient(model, batch_size=64, as_sparse=False, pert_col: str = "gene", control_label="non-targeting", include_controls=True, perturbation_col="condition", n_control_samples=256):
    """
    Batch-efficient evaluation. 
    NOTE: Removed dependency on test_pairs, as training is now unpaired.
    """
    import pandas as pd
    import numpy as np
    from scipy import sparse as sp
    import torch, gc
    from tqdm import tqdm

    print(f"[MEMORY DEBUG] evaluate_batch_efficient: Starting batch evaluation", flush=True)
    
    test_adata = model.test_adata
    train_adata = model.adata

    print(f"Preparing batch evaluation for test set with {test_adata.n_obs} cells...")

    # Discover ordered key & keep mask used during training
    ordered_key, keep_idx = _infer_ordered_key_and_keep_idx(train_adata)
    seen_conditions = _seen_categories(train_adata.obs.get(perturbation_col))

    # Get all control cells for final output
    print("Getting all control cells for final output...")
    all_control_indices = sample_random_control_cells(
        test_adata,
        n_samples=None,
        control_label=control_label,
        return_all=True,
        perturbation_col=perturbation_col
    )
    
    if sp.issparse(test_adata.X):
        all_control_cells = test_adata.X[all_control_indices]
    else:
        all_control_cells = test_adata.X[all_control_indices]
    
    print(f"Found {len(all_control_indices)} total control cells")

    # Get all unique perturbations (excluding control)
    if perturbation_col in test_adata.obs.columns:
        pert_mask = test_adata.obs[perturbation_col] != control_label
        unique_perts = test_adata.obs[perturbation_col][pert_mask].unique()
    else:
        pert_mask = test_adata.obs[pert_col] != control_label
        unique_perts = test_adata.obs[pert_col][pert_mask].unique()
    
    print(f"Found {len(unique_perts)} unique perturbations")

    # Group perturbed cells by perturbation type
    pert_cells_by_type = defaultdict(list)
    pert_indices = np.where(pert_mask)[0]
    for idx in pert_indices:
        if perturbation_col in test_adata.obs.columns:
            pert_name = str(test_adata.obs[perturbation_col].iloc[idx])
        else:
            pert_name = str(test_adata.obs[pert_col].iloc[idx])
        pert_cells_by_type[pert_name].append(idx)

    total_predictions = len(unique_perts) * n_control_samples
    total_ground_truths = sum(len(pert_cells_by_type[pert]) for pert in unique_perts)
    gene_dim = test_adata.n_vars
    
    all_predictions = np.zeros((total_predictions, gene_dim), dtype=np.float32)
    all_ground_truths = np.zeros((total_ground_truths, gene_dim), dtype=np.float32)
    
    pred_idx = 0
    gt_idx = 0
    all_pert_labels = []
    pred_labels_list = []
    
    for pert_name in tqdm(unique_perts, desc="Processing perturbations"):
        # Sample control cells for this perturbation type
        control_indices = sample_random_control_cells(
            test_adata,
            n_samples=n_control_samples,
            control_label=control_label,
            return_all=False,
            perturbation_col=perturbation_col
        )
        
        if sp.issparse(test_adata.X):
            control_expr = test_adata.X[control_indices].toarray()
        else:
            control_expr = test_adata.X[control_indices]
            
        n_controls = len(control_expr)
        
        pert_cell_indices = pert_cells_by_type[pert_name]
        
        # Get GO embedding
        if len(pert_cell_indices) > 0:
            sample_pert_idx = pert_cell_indices[0]
            emb = _target_ordered_embedding(
                train_adata, test_adata, sample_pert_idx, ordered_key, keep_idx, perturbation_col=perturbation_col
            )
            if emb is None:
                n_feats = int(keep_idx.sum()) if keep_idx is not None else 0
                emb = np.zeros((1, n_feats))
        else:
            n_feats = int(keep_idx.sum()) if keep_idx is not None else 0
            emb = np.zeros((1, n_feats))
        
        n_controls = len(control_expr)
        batch_adata = train_adata[:n_controls].copy()
        
        batch_adata.X = control_expr
        
        if perturbation_col in train_adata.obs.columns:
            new_perts = [pert_name] * n_controls
            # if pert_name in seen_conditions:
            #     batch_adata.obs[perturbation_col] = [pert_name] * n_controls
            # else:
            #     fallback_cond = train_adata.obs[perturbation_col].iloc[0]
            #     batch_adata.obs[perturbation_col] = [fallback_cond] * n_controls
            batch_adata.obs[perturbation_col] = new_perts
        
        if pert_col in train_adata.obs.columns:
            batch_adata.obs[pert_col] = [pert_name] * n_controls
        
        batch_adata.var = train_adata.var
        
        if ordered_key is not None and emb is not None:
            batch_adata.obsm[ordered_key] = np.repeat(emb, n_controls, axis=0)
        
        with torch.no_grad():
            pred_out = model.predict(batch_adata)
            pred_adata_batch = pred_out[0] if isinstance(pred_out, (list, tuple)) else pred_out
            pred_expr_batch = pred_adata_batch.X
            if sp.issparse(pred_expr_batch):
                pred_expr_batch = pred_expr_batch.toarray()
            
        all_predictions[pred_idx:pred_idx + n_controls] = pred_expr_batch
        pred_labels_list.extend([pert_name] * n_controls)
        pred_idx += n_controls
        
        if len(pert_cell_indices) > 0:
            pert_exprs_sparse = test_adata.X[pert_cell_indices]
            if sp.issparse(pert_exprs_sparse):
                pert_exprs_dense = pert_exprs_sparse.toarray()
            else:
                pert_exprs_dense = pert_exprs_sparse
            
            n_gt_cells = len(pert_cell_indices)
            all_ground_truths[gt_idx:gt_idx + n_gt_cells] = pert_exprs_dense
            all_pert_labels.extend([pert_name] * n_gt_cells)
            gt_idx += n_gt_cells
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sp.issparse(all_control_cells):
        all_control_cells_dense = all_control_cells.toarray()
    else:
        all_control_cells_dense = all_control_cells
    
    truth_mat = np.vstack([all_control_cells_dense, all_ground_truths])
    pred_mat = np.vstack([all_control_cells_dense, all_predictions])
    
    control_labels = [control_label] * len(all_control_indices)
    pred_labels = control_labels + pred_labels_list
    truth_labels = control_labels + all_pert_labels

    if getattr(model, "adata", None) is not None and getattr(model.adata, "var", None) is not None \
       and model.adata.var.shape[0] == pred_mat.shape[1]:
        var = model.adata.var.copy()
    else:
        var_names = pd.Index([f"gene_{i}" for i in range(pred_mat.shape[1])])
        var = pd.DataFrame(index=var_names)

    X_truth = sp.csr_matrix(truth_mat.astype(np.float32, copy=False)) if as_sparse else truth_mat.astype(np.float32, copy=False)
    X_pred = sp.csr_matrix(pred_mat.astype(np.float32, copy=False)) if as_sparse else pred_mat.astype(np.float32, copy=False)

    return X_truth, X_pred, truth_labels, pred_labels, var


def process_labels(labels, control_label):
    """
    Process labels to extract gene names and standardize control labels.
    Ensures control labels are always 'non-targeting' for cell-eval compatibility.
    """
    processed = []
    for label in labels:
        label_str = str(label)
        
        # Check if this is a control label (case-insensitive)
        is_control = (
            label == control_label or 
            control_label.lower() in label_str.lower() or
            'non-targeting' in label_str.lower() or
            'control' in label_str.lower() or
            label_str.lower() == 'ctrl'
        )
        
        if is_control:
            # Always use 'non-targeting' for cell-eval compatibility
            processed.append('non-targeting')
        else:
            # Extract gene name (first part before separator)
            for sep in ['-', '+', '_']:
                if sep in label_str:
                    processed.append(label_str.split(sep)[0])
                    break
            else:
                processed.append(label_str)
    return processed


def generate_groundtruth_adata(adata, output_path, perturbation_col="gene", control_label="non-targeting"):
    """
    Generates a groundtruth AnnData object aligning with BioLord / cell-eval format.

    Structure: [All Control Cells] stacked on top of [All Real Perturbed Cells],
    with .obs["target_gene"] standardized by process_labels().
    """
    print("\n--- Generating Ground Truth AnnData ---")

    if perturbation_col not in adata.obs.columns:
        raise ValueError(f"Column '{perturbation_col}' not found in adata.obs")

    # Standardize perturbation labels -> 'target_gene'
    raw_labels = adata.obs[perturbation_col].tolist()
    processed = process_labels(raw_labels, control_label)
    adata = adata.copy()
    adata.obs["target_gene"] = processed

    # Identify controls vs perturbed (using standardized target_gene)
    ctrl_mask = (adata.obs["target_gene"] == "non-targeting").values
    if ctrl_mask.sum() == 0:
        print("WARNING: No 'non-targeting' cells found for ground truth generation.")
    
    adata_ctrl = adata[ctrl_mask].copy()
    adata_pert = adata[~ctrl_mask].copy()

    # Sort perturbed cells by target_gene for reproducible ordering
    if adata_pert.n_obs > 0:
        sort_idx = np.argsort(adata_pert.obs["target_gene"].values.astype(str))
        adata_pert = adata_pert[sort_idx].copy()

    print(f"Stats (ground truth): {adata_ctrl.n_obs} control cells, {adata_pert.n_obs} perturbed cells")

    # Concatenate: controls first, then perturbed
    adata_gt = anndata.concat([adata_ctrl, adata_pert])

    # Optional: annotate "cell_type" to mark these as real / ground truth
    adata_gt.obs["cell_type"] = "real"

    # Regenerate obs_names for clarity
    n_ctrl = adata_ctrl.n_obs
    n_real = adata_pert.n_obs
    new_index = [f"Ctrl_{i}" for i in range(n_ctrl)] + [f"Real_{i}" for i in range(n_real)]
    adata_gt.obs_names = new_index

    print(f"Saving Ground Truth to {output_path}...")
    try:
        adata_gt.write_h5ad(output_path)
        print("Successfully saved Ground Truth AnnData.")
    except Exception as e:
        print(f"ERROR: Failed to save ground truth file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biolord Training & Inference Pipeline')
    
    parser.add_argument("--adata_path", type=str, required=True, help='Path to input .h5ad file')
    # Column name to use for train/test split: "split" or "split_few_shot"
    parser.add_argument("--use_split_column", type=str, default="split", choices=["split", "split_few_shot"],
                        help='Column name in adata.obs to use for train/test split')
    parser.add_argument("--perturbation_col", type=str, default="gene")
    parser.add_argument("--control_label", type=str, default="non-targeting")
    
    # Optional args
    parser.add_argument("--go_graph_path", type=str, default=None)
    parser.add_argument("--go_gene_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="biolord-benchmark")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--ground_truths_path", type=str, required=True)
    
    # Preprocessing args (optional)
    parser.add_argument("--gene_path", type=str, default=None)
    parser.add_argument("--mapping_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)

    # Legacy args (kept for parser compatibility but ignored by logic)
    parser.add_argument("--train_csv_path", type=str, default=None)
    parser.add_argument("--test_csv_path", type=str, default=None)
    parser.add_argument("--max_pairs_per_pert", type=int, default=None)
    parser.add_argument("--gen_groundtruth", action="store_true", default=False,
                    help="Generate a separate ground truth file from raw AnnData (no training)")
    parser.add_argument("--gen_prediction", action="store_true", default=False,
                        help="Train BioLord and save prediction + ground truth files")

    args = parser.parse_args()
    
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    
    if not args.gen_prediction and not args.gen_groundtruth:
        print("WARNING: Neither --gen_prediction nor --gen_groundtruth was specified.")
        print("No output files will be written. Use --gen_prediction and/or --gen_groundtruth.")
        sys.exit(0)

    # Check if adata file exists
    if not os.path.exists(args.adata_path):
        print(f"ERROR: Input file does not exist: {args.adata_path}")
        sys.exit(1)

    print(f"Reading {args.adata_path}...")
    try:
        adata_full = anndata.read_h5ad(args.adata_path)
        print(f"Successfully loaded AnnData: {adata_full.shape[0]} cells, {adata_full.shape[1]} genes")
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        sys.exit(1)

    print(f"Running Standard Unpaired BioLord Pipeline using split column: {args.use_split_column}")
    print(f"Perturbation Column: {args.perturbation_col}")
    
    if args.gen_prediction:
        print(f"Running Standard Unpaired BioLord Pipeline using split column: {args.use_split_column}")
        print(f"Perturbation Column: {args.perturbation_col}")

        custom_params = {
            "max_epochs": args.max_epochs,
            "n_latent": args.n_latent,
        }

        # Run the pipeline (UNPAIRED). This call re-reads from args.adata_path internally;
        # that's fine, or you can refactor run_biolord_pipeline to accept adata directly.
        trained_model, train_adata, test_adata = run_biolord_pipeline(
            adata_path=args.adata_path,
            train_csv_path=None,
            test_csv_path=None,
            custom_params=custom_params,
            gene_path=args.gene_path,
            mapping_path=args.mapping_path,
            mask_path=args.mask_path,
            go_graph_path=args.go_graph_path,
            go_gene_path=args.go_gene_path,
            use_split_column=args.use_split_column,
            use_wandb=True,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            perturbation_col=args.perturbation_col,
            control_label=args.control_label
        )

        print(f"Saving model to {args.save_path}")
        trained_model.save(args.save_path, save_anndata=True, overwrite=True)

        print("Starting Evaluation...")
        X_truth, X_pred, truth_labels, pred_labels, var = evaluate_batch_efficient(
            trained_model,
            batch_size=64,
            perturbation_col=args.perturbation_col,
            control_label=args.control_label,
            n_control_samples=256,
        )

        # Use the same label processing for truth and pred
        processed_truth_labels = process_labels(truth_labels, args.control_label)
        processed_pred_labels = process_labels(pred_labels, args.control_label)

        obs_real = pd.DataFrame(
            {"target_gene": processed_truth_labels},
            index=[str(i) for i in range(len(processed_truth_labels))]
        )
        obs_pred = pd.DataFrame(
            {"target_gene": processed_pred_labels},
            index=[str(i) for i in range(len(processed_pred_labels))]
        )

        adata_real = anndata.AnnData(X=sp.csr_matrix(X_truth), obs=obs_real.copy(), var=var.copy())
        adata_pred = anndata.AnnData(X=sp.csr_matrix(X_pred), obs=obs_pred.copy(), var=var.copy())

        def validate_celleval_format(adata, expected_pert_col='target_gene', expected_control='non-targeting'):
            """Validate that AnnData has the correct format for cell-eval."""
            if expected_pert_col not in adata.obs.columns:
                raise ValueError(f"Missing required column '{expected_pert_col}' in output AnnData")
            
            unique_labels = adata.obs[expected_pert_col].unique()
            if expected_control not in unique_labels:
                control_like = [
                    l for l in unique_labels
                    if 'control' in str(l).lower()
                    or 'non-targeting' in str(l).lower()
                    or str(l).lower() == 'ctrl'
                ]
                if control_like:
                    print(f"WARNING: Found control-like labels that may not be standardized: {control_like}")
                else:
                    print(f"WARNING: No '{expected_control}' label found. Example labels: {unique_labels[:10]}")
            
            print(f"✓ Output validated: {expected_pert_col} column present, {len(unique_labels)} unique perturbations")

        # Validate before saving
        validate_celleval_format(adata_real, expected_pert_col='target_gene', expected_control='non-targeting')
        validate_celleval_format(adata_pred, expected_pert_col='target_gene', expected_control='non-targeting')

        adata_real.write_h5ad(args.ground_truths_path)
        adata_pred.write_h5ad(args.predictions_path)
        print("Prediction pipeline finished.")

    # ---------------------
    # MODE 2: GROUND TRUTH
    # ---------------------
    if args.gen_groundtruth and not args.gen_prediction:
        # Optionally restrict to test split only
        if args.use_split_column in adata_full.obs.columns:
            test_mask = adata_full.obs[args.use_split_column].astype(str).str.lower() == "test"
            if test_mask.sum() == 0:
                print(f"WARNING: '{args.use_split_column}' column present but no 'test' cells found. Using all cells for ground truth.")
                adata_gt_src = adata_full
            else:
                adata_gt_src = adata_full[test_mask].copy()
                print(f"Using only test split for ground truth: {adata_gt_src.n_obs} cells")
        else:
            adata_gt_src = adata_full
            print(f"No '{args.use_split_column}' column found. Using all cells for ground truth.")

        generate_groundtruth_adata(
            adata=adata_gt_src,
            output_path=args.ground_truths_path,
            perturbation_col=args.perturbation_col,
            control_label=args.control_label
        )

        print("Standalone ground truth generation finished.")
