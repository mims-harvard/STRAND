import numpy as np
import pandas as pd
import os
import sys
import scanpy as sc

# Path setup: local GEARS folder first (Mapping.py, utils.py), then GEARS repo (gears, data_utils).
# GEARS_REPO_PATH = directory containing gears/ and data_utils (official GEARS clone).
_script_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
proj_root = os.environ.get("GEARS_REPO_PATH", os.path.join(_script_dir, "official", "GEARS"))
proj_root = os.path.abspath(proj_root)
if not os.path.isdir(proj_root):
    raise RuntimeError(
        f"GEARS repo not found at {proj_root}. "
        "Clone the official GEARS repo and set GEARS_REPO_PATH to its root, or place it at benchmark/GEARS/official/GEARS."
    )
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from gears import PertData, GEARS
from scipy import sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
from tqdm import tqdm
import anndata
from Mapping import OptimalTransportData
import argparse
from utils import tar_data_download_wrapper

from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.append(os.path.join(_script_dir, ".."))
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def predict_with_test_controls(model, test_ctrl_adata, pert_genes, n_samples=256):
    """
    Make predictions using TEST control cells instead of training controls.
    Returns n_samples DIFFERENT predictions, one per test control cell.
    
    Args:
        model: Trained GEARS model object
        test_ctrl_adata: AnnData containing ALL TEST control cells (GEARS will sample internally)
        pert_genes: List of gene names to perturb, e.g., ['BRCA1'] or ['BRCA1', 'TP53']
        n_samples: Number of control cells to sample
    
    Returns:
        predictions: numpy array of shape (n_samples, num_genes)
    """
    device = model.device
    gene_list = list(model.gene_list)  # GEARS stores gene names here
    
    # Validate perturbation genes are in the model
    for gene in pert_genes:
        if gene not in gene_list:
            print(f"Warning: Gene {gene} not in model's gene list, skipping...")
            return None
    
    # Check we have control cells available
    n_available = len(test_ctrl_adata)
    if n_available == 0:
        return None
    
    # Determine actual number of samples (GEARS samples with replacement if needed)
    actual_samples = n_samples
        
    # Let GEARS handle the sampling internally via create_cell_graph_dataset_for_prediction
    # This function samples num_samples cells from ctrl_adata using np.random.randint
    cell_graphs = create_cell_graph_dataset_for_prediction(
        pert_genes,                # List of gene names to perturb
        test_ctrl_adata,           # Pass FULL test control adata - GEARS samples internally
        gene_list,                 # All gene names
        device,
        num_samples=actual_samples # GEARS will sample this many cells from test_ctrl_adata
    )
    
    # Run inference
    loader = DataLoader(cell_graphs, batch_size=actual_samples, shuffle=False)
    
    # Get the actual model (GNN)
    gnn_model = model.best_model if hasattr(model, 'best_model') else model.model
    gnn_model.eval()
    
    predictions_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = gnn_model(batch)  # Forward pass through GNN
            predictions_list.append(preds.cpu().numpy())
    
    predictions = np.vstack(predictions_list)
    return predictions  # Shape: (n_samples, num_genes)

def safe_series_nonzero(self):
    """Safe nonzero method for pandas Series that converts to numpy array first"""
    return np.array(self).nonzero()

def safe_toarray(X):
    """
    Safely convert input to dense array format.
    Handles both sparse and dense arrays.
    """
    if sp.issparse(X):
        return X.toarray()
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise TypeError("Input must be either a sparse matrix or numpy array")

class RobustGEARS(GEARS):
    def __init__(self, pert_data, control_label='ctrl', **kwargs):
        # Store the actual control label
        self.actual_control_label = control_label
        
        # Fix condition column type if needed
        if "condition" in pert_data.adata.obs:
            pert_data.adata.obs["condition"] = pert_data.adata.obs["condition"].to_numpy()
        
        # CRITICAL FIX: Temporarily rename control label to 'ctrl' for parent class
        # if control_label != 'ctrl':
        #     original_condition = pert_data.adata.obs["condition"].copy()
        #     mask = (pert_data.adata.obs["condition"] == control_label)
        #     pert_data.adata.obs.loc[mask, "condition"] = 'ctrl'
        
        super().__init__(pert_data, **kwargs)
        self.pert_data = pert_data 
        
        # if control_label != 'ctrl':
        #     pert_data.adata.obs["condition"] = original_condition
        
        mask = (self.adata.obs["condition"] == 'ctrl').to_numpy()
        if mask.sum() == 0:
            raise ValueError(f"No control cells found with label '{control_label}'")
        
        if sp.issparse(self.adata.X):
            ctrl_mean = np.asarray(self.adata.X[mask].mean(axis=0)).ravel()
        else:
            ctrl_mean = np.mean(self.adata.X[mask], axis=0)
        self.ctrl_expression = torch.tensor(ctrl_mean, dtype=torch.float32).to(self.device)
    
    def model_initialize(self, hidden_size=64, **kwargs):
        """
        Override model_initialize to:
        1. Pre-load the GO edge_list
        2. Ensure edge_list has 'importance' column
        """
        # Pre-load the edge_list (mimicking parent behavior)
        # super().model_initialize(hidden_size=hidden_size, **kwargs)
        from utils import get_similarity_network
        
        # Load edge_list using the same logic as parent class
        edge_list = get_similarity_network(
            network_type='go',
            adata=self.adata,
            threshold=0.4,  # default value
            k=20,  # default value
            data_path=self.pert_data.data_path,
            data_name=self.pert_data.dataset_name,
            split=self.pert_data.split,
            seed=self.pert_data.seed,
            train_gene_set_size=self.pert_data.train_gene_set_size,
            set2conditions=self.set2conditions,
            default_pert_graph=self.pert_data.default_pert_graph,
            pert_list=None
        )
        
        # FIX: Add 'importance' column if missing
        if 'importance' not in edge_list.columns:
            print("Warning: edge_list missing 'importance' column. Adding default value 1.0")
            edge_list['importance'] = 1.0
        
        # Store it so parent uses our patched version
        self.edge_list = edge_list
        
        # Now call parent (which will use self.edge_list if it exists)
        return super().model_initialize(hidden_size=hidden_size, **kwargs)
    
if not hasattr(pd.Series, 'nonzero'):
    pd.Series.nonzero = safe_series_nonzero

def create_split_subset_adata(adata, split_value, split_column='split'):
    mask = adata.obs[split_column].astype(str).str.lower() == split_value.lower()
    return adata[mask].copy()

def process_labels(labels, control_label):
    processed = []
    for label in labels:
        label_str = str(label)
        if control_label.lower() in label_str.lower() or control_label in label_str.lower():
            processed.append(control_label)
        else:
            for sep in ['-', '+', '_']:
                if sep in label_str:
                    processed.append(label_str.split(sep)[0])
                    break
            else:
                processed.append(label_str)
    return processed



def evaluate_256_controls_cell_line_aware(model, test_adata, n_control_samples=256, 
                                         perturbation_col="condition", control_label='non-targeting',
                                         cell_line_col='cell_line'):
    """
    Evaluate GEARS model with cell-line-aware control sampling.
    If multiple cell lines exist, processes each separately and returns separate results.
    """
    rng = np.random.default_rng(42)
    
    # Check for multiple cell lines
    has_cell_line = cell_line_col in test_adata.obs.columns
    if has_cell_line:
        unique_cell_lines = test_adata.obs[cell_line_col].unique()
        n_cell_lines = len(unique_cell_lines)
        print(f"Found {n_cell_lines} cell lines in test set: {unique_cell_lines}", flush=True)
        
        if n_cell_lines > 1:
            # MULTIPLE CELL LINES PATH - process each separately
            results = {}
            for cl_name in unique_cell_lines:
                print(f"\n{'='*40}\nProcessing Cell Line: {cl_name}\n{'='*40}", flush=True)
                
                # Filter to current cell line (case-insensitive)
                cl_mask = test_adata.obs[cell_line_col].astype(str).str.lower() == str(cl_name).lower()
                test_adata_cl = test_adata[cl_mask].copy()
                
                if test_adata_cl.n_obs == 0:
                    print(f"  WARNING: No cells found for cell line {cl_name}. Skipping.", flush=True)
                    continue
                
                # Process this cell line
                X_pred_cl, X_truth_cl, pred_labels_cl, truth_labels_cl = _evaluate_single_cell_line(
                    model, test_adata_cl, n_control_samples, perturbation_col, control_label, rng
                )
                
                results[cl_name] = {
                    'X_pred': X_pred_cl,
                    'X_truth': X_truth_cl,
                    'pred_labels': pred_labels_cl,
                    'truth_labels': truth_labels_cl
                }
            
            return results
        else:
            # Single cell line - use standard path
            test_adata_cl = test_adata
    else:
        # No cell line column - use standard path
        test_adata_cl = test_adata
    
    # SINGLE CELL LINE PATH (or no cell line column)
    X_pred, X_truth, pred_labels, truth_labels = _evaluate_single_cell_line(
        model, test_adata_cl, n_control_samples, perturbation_col, control_label, rng
    )
    
    return X_pred, X_truth, pred_labels, truth_labels


def _evaluate_single_cell_line(model, test_adata, n_control_samples, 
                                perturbation_col, control_label, rng):
    """
    CORRECTED: Uses test control cells for cell-specific predictions.
    """
    test_pert_vals = test_adata.obs[perturbation_col].astype(str).values
    is_control = (test_pert_vals == str(control_label))
    all_ctrl_indices = np.where(is_control)[0]
    
    if len(all_ctrl_indices) == 0:
        raise ValueError(f"No control cells found with label '{control_label}'")
    
    # Get test control subset
    test_ctrl_adata = test_adata[all_ctrl_indices]
    
    unique_perts = np.unique(test_pert_vals)
    unique_perts = unique_perts[unique_perts != str(control_label)]
    
    all_predictions = []
    all_ground_truths = []
    pred_labels = []
    truth_labels = []
    
    # Get valid graph nodes for filtering
    valid_graph_nodes = set(model.gene_list) if hasattr(model, 'gene_list') else set()
    
    successful_predictions = 0
    
    for pert_name in tqdm(unique_perts, desc="Evaluating perturbations"):
        # Extract genes
        if '+' in pert_name:
            genes = [g.strip() for g in pert_name.split('+') if g.strip() not in ['ctrl', '']]
        else:
            genes = [pert_name] if pert_name not in ['ctrl', ''] else []
        
        if not genes:
            continue
            
        # Check if genes are in graph
        if valid_graph_nodes and not all(g in valid_graph_nodes for g in genes):
            continue
        
        # Get ground truth
        gt_mask = (test_pert_vals == pert_name)
        gt_indices = np.where(gt_mask)[0]
        if len(gt_indices) == 0:
            continue
        
        # Use test control cells for cell-specific predictions
        try:
            pert_preds = predict_with_test_controls(
                model, 
                test_ctrl_adata, 
                genes, 
                n_samples=n_control_samples
            )
            
            if pert_preds is None:
                continue
                
            # pert_preds shape: (n_control_samples, num_genes) - DIFFERENT predictions!
            all_predictions.append(pert_preds)
            pred_labels.extend([pert_name] * len(pert_preds))
            successful_predictions += 1
            
        except Exception as e:
            print(f"  Error predicting {pert_name}: {e}")
            continue
        
        # Ground truth
        gt_expr = test_adata.X[gt_indices]
        gt_expr = gt_expr.toarray() if sp.issparse(gt_expr) else np.asarray(gt_expr)
        all_ground_truths.append(gt_expr)
        truth_labels.extend([pert_name] * len(gt_indices))
    
    print(f"Successfully predicted {successful_predictions}/{len(unique_perts)} perturbations")
    
    # Add control cells
    ctrl_expr = test_adata.X[all_ctrl_indices]
    ctrl_expr = ctrl_expr.toarray() if sp.issparse(ctrl_expr) else np.asarray(ctrl_expr)
    
    if len(all_predictions) > 0:
        X_pred = np.vstack([ctrl_expr] + all_predictions)
    else:
        X_pred = ctrl_expr
        
    if len(all_ground_truths) > 0:
        X_truth = np.vstack([ctrl_expr] + all_ground_truths)
    else:
        X_truth = ctrl_expr
    
    ctrl_labels = [control_label] * len(all_ctrl_indices)
    pred_labels = ctrl_labels + pred_labels
    truth_labels = ctrl_labels + truth_labels
    
    return X_pred, X_truth, pred_labels, truth_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GEARS training/testing pipeline")

    parser.add_argument("--adata_path", type=str, required=True)
    parser.add_argument("--pert_data_path", type=str, required=True)
    parser.add_argument("--pert_dataset_name", type=str, default="dataset")
    parser.add_argument("--wandb_project_name", type=str, default="Benchmark")
    parser.add_argument("--wandb_exp_name", type=str, default="GEARS")
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--ground_truths_path", type=str, required=True)
    parser.add_argument("--perturbation_col", type=str, default="condition")
    parser.add_argument("--control_label", type=str, default="ctrl")
    parser.add_argument("--split_column", type=str, default="split", 
                       help="Column name in adata.obs for train/test split")

    args = parser.parse_args()
    control_label = args.control_label
    
    # --- FIX: REMOVE CORRUPT METADATA ---
    # Since data is already log-normed, we delete the metadata key causing the read crash.
    import h5py
    try:
        with h5py.File(args.adata_path, 'r+') as f:
            if 'uns' in f and 'log1p' in f['uns']:
                del f['uns']['log1p']
    except Exception:
        pass # Ignore if file is locked or key missing
    # ------------------------------------
    
    adata = sc.read_h5ad(args.adata_path)
    
    if args.split_column not in adata.obs.columns:
        raise ValueError(f"'{args.split_column}' column not found in adata.obs")
    if args.perturbation_col in adata.obs.columns:
        print(f"Mapping '{args.perturbation_col}' to 'condition' for GEARS compatibility.")
        adata.obs['condition'] = adata.obs[args.perturbation_col].astype(str)  # Convert to string to avoid Categorical issues
        
    if args.control_label != 'ctrl':
        print(f"Renaming control label '{args.control_label}' -> 'ctrl' for GEARS compatibility")
        mask = adata.obs['condition'] == args.control_label
        adata.obs.loc[mask, 'condition'] = 'ctrl'
        # Store original label for output processing
        original_control_label = args.control_label
        args.control_label = 'ctrl'  # Use 'ctrl' for all GEARS operations
    
    # FIX: Reformat single perturbations to GEARS format ('GeneA' -> 'GeneA+ctrl')
    print("Reformatting perturbation names to GEARS format...")
    def format_condition(cond):
        if cond == 'ctrl':
            return 'ctrl'
        elif '+' not in cond:
            return f'{cond}+ctrl'  # Single pert format
        return cond
    adata.obs['condition'] = adata.obs['condition'].apply(format_condition)
        
    if 'condition' not in adata.obs.columns:
         raise ValueError("Column 'condition' not found. Please ensure perturbation_col is set correctly.")
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var.index
        
    if 'cell_line' in adata.obs.columns:
        print("Mapping 'cell_line' to 'cell_type' for GEARS compatibility.")
        print(adata.obs['cell_line'].unique())
        adata.obs['cell_type'] = adata.obs['cell_line']
    elif 'cell_type' not in adata.obs.columns:
        # Fallback if neither exists
        raise ValueError("Could not find 'cell_line' or 'cell_type' in adata.obs")
    
    train_adata = create_split_subset_adata(adata, 'train', args.split_column)
    test_adata = create_split_subset_adata(adata, 'test', args.split_column)
    
    os.makedirs(args.pert_data_path, exist_ok=True)

    # --- FIX: Pre-download GO essential graph file ---
    go_dir = os.path.join(args.pert_data_path, 'go_essential_all')
    go_file = os.path.join(go_dir, 'go_essential_all.csv')

    if not os.path.exists(go_file):
        print(f"[GO Graph] Downloading GO essential graph from Harvard Dataverse...")
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
        tar_data_download_wrapper(server_path, go_dir, args.pert_data_path)
        if os.path.exists(go_file):
            print(f"[GO Graph] Successfully downloaded to: {go_file}")
        else:
            raise RuntimeError(f"Failed to download GO graph file to {go_file}")
    else:
        print(f"[GO Graph] Using existing file: {go_file}")
    # ------------------------------------------------

    pert_data = OptimalTransportData(args.pert_data_path)
    pert_data.ot_pairs = False
    pert_data.new_data_process(dataset_name=args.pert_dataset_name, 
                               adata=train_adata, 
                               control_label=args.control_label,)
    pert_data.prepare_split(split='no_test', seed=42)
    pert_data.get_dataloader(batch_size=64)
    
    # In run_gears.py, around line 210
    model = RobustGEARS(pert_data, device='cuda', 
                        control_label=args.control_label,  # Add this parameter
                        weight_bias_track=True, 
                        proj_name=args.wandb_project_name, 
                        exp_name=args.wandb_exp_name)
    
    model.model_initialize(hidden_size=64)
    model.train(epochs=20, lr=1e-3)
    
    os.makedirs(args.model_save_path, exist_ok=True)
    model.save_model(args.model_save_path)
    
    
    test_perts_raw = test_adata.obs['condition'].unique()
    test_perts = [p for p in test_perts_raw if p != args.control_label and p != 'ctrl']
    
    pert_set = set(model.pert_list) if hasattr(model, 'pert_list') else set()
    gene_set = set(model.gene_list) if hasattr(model, 'gene_list') else set()

    valid_test_perts = []
    skipped_perts = []

    for p in test_perts:
        # Extract gene name(s) from formatted perturbation string
        # Handle formats: 'GeneA+ctrl', 'GeneA+GeneB', 'GeneA', etc.
        if '+ctrl' in p:
            genes = [p.replace('+ctrl', '')]  # Single: 'NOMO3+ctrl' -> ['NOMO3']
        elif '+' in p:
            genes = [g.strip() for g in p.split('+') if g.strip() != 'ctrl']  # Combinatorial: 'GeneA+GeneB' -> ['GeneA', 'GeneB']
        else:
            genes = [p]  # Already just gene name
        
        # Check if all genes are in the GO graph (pert_list) or gene_list
        if pert_set:
            # Use pert_list (GO graph genes) if available
            if all(g in pert_set for g in genes):
                valid_test_perts.append(p)
            else:
                skipped_perts.append(p)
        elif gene_set:
            # Fallback to gene_list
            if all(g in gene_set for g in genes):
                valid_test_perts.append(p)
            else:
                skipped_perts.append(p)
        else:
            # No filtering available - use all
            print("Warning: Cannot determine valid perturbations, using all")
            valid_test_perts = test_perts
            break
    
    
    
    print(f"\n=== Zero-Shot Evaluation Setup ===")
    print(f"Test perturbations: {len(test_perts)}")
    print(f"Valid (in GO graph): {len(valid_test_perts)}")
    print(f"Skipped (not in GO graph): {len(skipped_perts)}")
    if skipped_perts[:5]:
        print(f"  Examples skipped: {skipped_perts[:5]}")
    
    # Use model.predict() for zero-shot predictions
    # predict() expects list of lists: [['GENE1'], ['GENE2'], ...]
    print("\n=== Running Zero-Shot Predictions ===")
    all_predictions = {}
    all_logvars = {}
    
    for pert_gene in valid_test_perts:
        try:
            if '+ctrl' in pert_gene:
                genes = [pert_gene.replace('+ctrl', '')]  # Single pert: 'GeneA+ctrl' -> ['GeneA']
            elif '+' in pert_gene:
                # genes = pert_gene.split('+')  # Combinatorial: 'GeneA+GeneB' -> ['GeneA', 'GeneB']
                genes = [g.strip() for g in pert_gene.split('+') if g.strip() and g.strip() != 'ctrl']
            else:
                genes = [pert_gene]
            # predict() returns dict of {pert: prediction_array}
            # pred_result = model.predict([[pert_gene]])
            pred_result = model.predict([genes])
            key = '_'.join(genes)
            all_predictions[pert_gene] = pred_result[key]
            # all_predictions[pert_gene] = pred_result[pert_gene]
            if hasattr(model, 'config') and model.config.get('uncertainty', False):
                all_logvars[pert_gene] = model.saved_logvar_sum.get(pert_gene, None)
        except Exception as e:
            print(f"  Warning: Failed to predict {pert_gene}: {e}")
    
    print(f"Successfully predicted {len(all_predictions)} perturbations")
    
    # ========== COMPARE TO GROUND TRUTH ==========
    # Get ground truth from test_adata
    print("\n=== Collecting Ground Truth ===")
    
    # Get control cells mean expression (from training data, used by predict())
    ctrl_expr = model.ctrl_expression.cpu().numpy()
    
    # Collect ground truth from test_adata
    ground_truths = {}
    for pert_gene in valid_test_perts:
        mask = test_adata.obs['condition'] == pert_gene
        if mask.sum() > 0:
            gt_cells = test_adata[mask].X
            if hasattr(gt_cells, 'toarray'):
                gt_cells = gt_cells.toarray()
            ground_truths[pert_gene] = gt_cells.mean(axis=0)  # Mean expression per perturbation
    
    print(f"Ground truth collected for {len(ground_truths)} perturbations")
    
    # ========== SAVE RESULTS ==========
    import anndata as ad
    from scipy.sparse import csr_matrix
    
    # Get gene names from training data (with virtual nodes) - 5225 genes
    train_gene_names = pert_data.adata.var['gene_name'].values
    
    # Get gene names from test data (original, no virtual nodes) - 4760 genes
    test_gene_names = test_adata.var['gene_name'].values if 'gene_name' in test_adata.var.columns else test_adata.var_names.values
    
    # Build mappings for alignment
    train_gene_to_idx = {g: i for i, g in enumerate(train_gene_names)}
    test_gene_to_idx = {g: i for i, g in enumerate(test_gene_names)}
    
    # Find common genes that exist in BOTH training and test
    common_genes = [g for g in test_gene_names if g in train_gene_to_idx]
    
    # Get indices in each space for the common genes
    train_indices = [train_gene_to_idx[g] for g in common_genes]  # For slicing predictions
    test_indices = [test_gene_to_idx[g] for g in common_genes]    # For slicing ground truths
    
    print(f"Gene space alignment:")
    print(f"  Training genes (with virtual nodes): {len(train_gene_names)}")
    print(f"  Test genes (original): {len(test_gene_names)}")
    print(f"  Common genes for evaluation: {len(common_genes)}")
    
    # Predictions AnnData - slice to common genes using train indices
    pred_labels = []
    pred_matrix = []
    for pert_gene, pred in all_predictions.items():
        pred_labels.append(pert_gene)
        pred_matrix.append(pred[train_indices])  # Slice using training indices
    
    if pred_matrix:
        pred_matrix = np.vstack(pred_matrix)
        adata_pred = ad.AnnData(
            X=csr_matrix(pred_matrix),
            obs=pd.DataFrame({'perturbation': pred_labels}, index=[f"pred_{i}" for i in range(len(pred_labels))]),
            var=pd.DataFrame(index=common_genes)  # Use common_genes
        )
        adata_pred.write_h5ad(args.predictions_path)
        print(f"Saved predictions to {args.predictions_path} (shape: {pred_matrix.shape})")
    
    # Ground truths AnnData - slice to common genes using test indices
    gt_labels = []
    gt_matrix = []
    for pert_gene, gt in ground_truths.items():
        gt_labels.append(pert_gene)
        gt_matrix.append(gt[test_indices])  # Slice using test indices
    
    if gt_matrix:
        gt_matrix = np.vstack(gt_matrix)
        adata_gt = ad.AnnData(
            X=csr_matrix(gt_matrix),
            obs=pd.DataFrame({'perturbation': gt_labels}, index=[f"gt_{i}" for i in range(len(gt_labels))]),
            var=pd.DataFrame(index=common_genes)  # Use same common_genes
        )
        adata_gt.write_h5ad(args.ground_truths_path)
        print(f"Saved ground truths to {args.ground_truths_path} (shape: {gt_matrix.shape})")
    
    print("\n=== Zero-Shot Evaluation Complete ===")
