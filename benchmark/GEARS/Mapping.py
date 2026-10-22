"""
Mapping.py: data/mapping wrapper for GEARS (renamed from OptimalTransport.py).
No optimal transport logic is required for this pipeline; the class OptimalTransportData
subclasses PertData for format handling. Paths are set by run_gears.py (local utils + GEARS_REPO_PATH).
"""
import sys
import os
from gears import PertData, GEARS
from scipy import sparse as sp
import random
from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import scanpy as sc
from tqdm import tqdm
from scipy.sparse import csr_array, csr_matrix, issparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from utils import print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts, tar_data_download_wrapper

class OptimalTransportData(PertData):
    def __init__(self, data_path, gene_set_path=None, default_pert_graph=True, ot_csv=None):
        super().__init__(data_path)
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}
        self.ot_csv = ot_csv
        self.ot_pairs = False
        self.pairs = {}  # Initialize pairs attribute

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
            
            
    def create_cell_graph(self, X, y, de_idx, pert_category, pert_idx):
        """
        Create a PyG Data object for a single cell.
        
        GEARS format: each GENE is a node with 1 feature [expression]
        Shape: x = (num_genes, 1), y = (num_genes,)
        """
        # Ensure 2D array with shape (1, num_genes) for proper transpose
        X = np.array(X).flatten()  # (num_genes,)
        y = np.array(y).flatten()  # (num_genes,)
        
        # Reshape to (1, num_genes) then transpose to (num_genes, 1)
        X = X.reshape(1, -1)  # (1, num_genes)
        feature_mat = torch.Tensor(X).T  # (num_genes, 1) - GEARS expected format
        
        # Get perturbation gene indices
        if '+' in pert_category:
            pert_genes = [p.strip() for p in pert_category.split('+') if p.strip() != 'ctrl']
        else:
            pert_genes = [pert_category] if pert_category != 'ctrl' else []
        
        # Map to indices using node_map_pert (same as official get_pert_idx)
        pert_idx_list = []
        for gene in pert_genes:
            if gene in self.node_map_pert:
                pert_idx_list.append(self.node_map_pert[gene])
            elif hasattr(self, 'node_map_pert_lower') and gene.lower() in self.node_map_pert_lower:
                _, idx = self.node_map_pert_lower[gene.lower()]
                pert_idx_list.append(idx)
        
        if not pert_idx_list:
            pert_idx_list = [-1]
        
        # Create the Data object matching GEARS format exactly
        data = Data(
            x=feature_mat,  # (num_genes, 1)
            y=torch.Tensor(y),  # (num_genes,)
            de_idx=torch.tensor(de_idx, dtype=torch.long) if not isinstance(de_idx, torch.Tensor) else de_idx,
            pert=pert_category,
            pert_idx=pert_idx_list  # List of integers, not tensor
        )
        
        return data
    
    def new_data_process(self, dataset_name,
                         adata=None,
                         skip_calc_de=False,
                         control_label='ctrl',
                         additional_genes=None):
        """
        Process new dataset with "Virtual Node" support for missing perturbation genes.
        """
        # 1. Basic Column Setup
        print(adata.obs.columns.values)
        adata = self._create_required_columns(adata, dataset_name)
        
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        # ---------------------------------------------------------
        # CRITICAL FIX: Pad adata with missing perturbation genes (Virtual Nodes)
        # This ensures graph propagation works for unmeasured genes without crashing.
        # ---------------------------------------------------------
        print_sys("Checking for perturbation genes missing from feature space...")
        
        # Identify all unique perturbations (handling single and combo 'A+B')
        all_perts = set()
        for p in adata.obs['condition'].unique():
            if p != control_label:
                if '+' in p:
                    for sub_p in p.split('+'):
                        all_perts.add(sub_p.strip())
                else:
                    all_perts.add(p)
                    
        # NEW: Add additional genes (e.g., test set perturbations) to vocabulary
        if additional_genes is not None:
            print_sys(f"Adding {len(additional_genes)} additional genes to vocabulary check (e.g., zero-shot test genes).")
            for g in additional_genes:
                all_perts.add(g)
        
        # Compare against existing features
        existing_genes = set(adata.var['gene_name'])
        missing_perts = list(all_perts - existing_genes)
        
        if len(missing_perts) > 0:
            print_sys(f"Found {len(missing_perts)} perturbations not in feature space (Virtual Nodes).")
            # print_sys(f"Examples: {missing_perts[:5]}")
            print_sys("Padding adata with zero-expression vectors...")
            
            # Create zero matrix for missing genes
            padding = csr_matrix((adata.shape[0], len(missing_perts)))
            
            # Create temporary AnnData for the padding
            padding_var = pd.DataFrame(index=missing_perts)
            padding_var['gene_name'] = missing_perts
            
            # Fill other var columns with defaults to match schema
            for col in adata.var.columns:
                if col != 'gene_name':
                    if pd.api.types.is_numeric_dtype(adata.var[col]):
                        padding_var[col] = 0
                    elif pd.api.types.is_bool_dtype(adata.var[col]):
                        padding_var[col] = False
                    else:
                        padding_var[col] = "virtual_node"
            
            adata_padding = sc.AnnData(X=padding, obs=adata.obs, var=padding_var)
            
            # Concatenate: Original + Virtual Nodes
            adata = sc.concat([adata, adata_padding], axis=1, join='outer', merge='unique')
            
            # Verify uniqueness of gene_name just in case
            if not adata.var['gene_name'].is_unique:
                 print_sys("Warning: gene_name is not unique after padding. Deduplicating...")
                 adata.var_names_make_unique()
            
            print_sys(f"New adata shape with virtual nodes: {adata.shape}")
        else:
            print_sys("All perturbation genes are present in feature space.")
        # ---------------------------------------------------------

        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        
        self.control_label = control_label
        
        # Calculate DE genes (Virtual nodes will naturally have 0 DE, which is correct)
        self.adata = get_DE_genes(adata, skip_calc_de, control_label=control_label)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata, control_label=control_label)
        
        # Process CSV pairs
        if self.ot_csv is not None and len(self.ot_csv) > 0:
            print_sys("Processing optimal transport pairs...")
            self._store_original_cell_mappings()
            self._process_ot_pairs_with_original_indices()
            self.verify_csv_pairs_integrity()
        elif self.ot_csv is not None:
            print_sys("CSV is empty, skipping OT pair processing")
        
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        
        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == control_label]
        
        if len(self.ctrl_adata) == 0:
            print_sys(f"WARNING: No control cells found with label '{control_label}'.")

        self.gene_names = self.adata.var.gene_name
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        
        if os.path.isfile(dataset_fname):
            print_sys("Local pyg dataset detected. Reusing cached cell_graphs.pkl ...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))
            print_sys("Cache loaded; skipping graph reconstruction.")
        else:
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            self.analyze_perturbation_coverage()
            print_sys("Saving new dataset pyg object at " + dataset_fname)
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
            print_sys("Done!")
    
    def _create_required_columns(self, adata, dataset_name):
        """
        Automatically create required columns based on dataset name.
        
        Currently implemented for HCT116 datasets only.
        For other datasets, logs "currently not implemented" and passes.
        
        For HCT116 datasets:
        - Creates cell_type column with "colorectal_carcinoma"
        - Creates condition column from curated_perturbation.rsplit('-', 2)[0]
        
        Parameters
        ----------
        adata: AnnData object
            AnnData object to modify
        dataset_name: str
            Name of dataset (case-insensitive check for "HCT116")
            
        Returns
        -------
        adata: AnnData object
            Modified AnnData object with created columns
        """
        dataset_name_lower = dataset_name.lower()
        
        if 'hct116' in dataset_name_lower:
            print_sys(f"Auto-creating columns for HCT116 dataset: {dataset_name}")
            
            # Create cell_type column if it doesn't exist
            if 'cell_type' not in adata.obs.columns.values:
                adata.obs['cell_type'] = 'colorectal_carcinoma'
                print_sys("Created cell_type column with value 'colorectal_carcinoma'")
            
            # Create condition column from curated_perturbation if it doesn't exist
            if 'condition' not in adata.obs.columns.values:
                if 'curated_perturbation' in adata.obs.columns.values:
                    # Create condition from curated_perturbation using rsplit('-', 2)[0]
                    adata.obs['condition'] = adata.obs['curated_perturbation'].apply(
                        lambda x: str(x).rsplit('-', 2)[0] if pd.notna(x) else x
                    )
                    print_sys("Created condition column from curated_perturbation using rsplit('-', 2)[0]")
                else:
                    print_sys("WARNING: curated_perturbation column not found. Cannot create condition column.")
        else:
            print_sys(f"Column auto-creation currently not implemented for dataset: {dataset_name}")
        
        return adata
    
    def _store_original_cell_mappings(self):
        """
        Store original cell ID to index mappings before any filtering
        """
        self.original_cell_mapping = {}
        if 'ind' in self.adata.obs.columns:
            for idx, cell_id in enumerate(self.adata.obs['ind']):
                self.original_cell_mapping[cell_id] = idx
        print_sys(f"Stored mappings for {len(self.original_cell_mapping)} cells")
    
    def _process_ot_pairs_with_original_indices(self):
        """
        Process optimal transport pairs using original CSV indices directly
        """
        if self.ot_csv is None or len(self.ot_csv) == 0:
            return
            
        print_sys("=== Processing OT Pairs with Original Indices ===")
        print_sys(f"CSV shape: {self.ot_csv.shape}")
        
        # Map cell IDs to original indices to validate pairs
        id_to_idx = {}
        if 'control_id' in self.ot_csv.columns and 'pert_id' in self.ot_csv.columns:
            # Use cell IDs if available
            for idx, cell_id in enumerate(self.adata.obs['ind']):
                id_to_idx[cell_id] = idx
        
        # Process pairs directly using CSV indices
        self.ot = self.ot_csv.copy()
        valid_pairs = 0
        invalid_pairs = 0
        
        # Validate and store pairs
        for idx, row in self.ot_csv.iterrows():
            ctrl_idx = int(row['control_index'])
            pert_idx = int(row['perturbation_index'])
            
            # Check if indices are valid for current adata size
            if 0 <= ctrl_idx < len(self.adata) and 0 <= pert_idx < len(self.adata):
                valid_pairs += 1
            else:
                invalid_pairs += 1
                # Keep the pair but mark for debugging
                print_sys(f"Invalid pair found: ctrl_idx={ctrl_idx}, pert_idx={pert_idx}, adata_size={len(self.adata)}")
        
        print_sys(f"Pair validation: {valid_pairs} valid, {invalid_pairs} invalid out of {len(self.ot_csv)} total")
        
        # Use original indices directly (no mapping needed)
        self.ot['ctrl_new'] = self.ot['control_index'].astype(int)
        self.ot['pert_new'] = self.ot['perturbation_index'].astype(int)
        
        # Add perturbation labels using original indices
        self.ot['pert'] = [self.adata.obs['condition'].iloc[i] for i in self.ot['perturbation_index']]
        
        # Create pairs dictionary
        self.pairs = {}
        for p, g in self.ot.groupby('pert'):
            # Only include valid pairs
            valid_group_pairs = []
            for _, row in g.iterrows():
                ctrl_idx = int(row['ctrl_new'])
                pert_idx = int(row['pert_new'])
                if 0 <= ctrl_idx < len(self.adata) and 0 <= pert_idx < len(self.adata):
                    valid_group_pairs.append((ctrl_idx, pert_idx))
            
            if valid_group_pairs:
                self.pairs[p] = valid_group_pairs
        
        # Set the ot_pairs flag
        self.ot_pairs = True
        
        # Report final results
        total_pairs = sum(len(pairs) for pairs in self.pairs.values())
        print_sys(f"Final pairs created:")
        print_sys(f"Total valid pairs from CSV: {total_pairs}")
        print_sys(f"Unique perturbations with pairs: {len(self.pairs)}")
        
        # Show examples
        for pert, pairs in list(self.pairs.items())[:5]:
            print_sys(f"  {pert}: {len(pairs)} pairs")
        if len(self.pairs) > 5:
            print_sys(f"  ... and {len(self.pairs) - 5} more perturbations")
        
        if total_pairs == 0:
            print_sys("WARNING: No valid pairs were created from CSV. Check CSV indices.")
            self.ot_pairs = False
            self.pairs = {}
        
        print_sys("==============================================")
    
    def verify_csv_pairs_integrity(self):
        """
        Verify that CSV pairs are correctly preserved and match expectations
        """
        if not self.ot_pairs or not hasattr(self, 'pairs') or not self.pairs:
            print_sys("No CSV pairs to verify")
            return False
            
        print_sys("=== Verifying CSV Pairs Integrity ===")
        
        # Count original CSV pairs
        original_pairs = len(self.ot_csv) if self.ot_csv is not None else 0
        processed_pairs = sum(len(pairs) for pairs in self.pairs.values())
        
        print_sys(f"Original CSV pairs: {original_pairs}")
        print_sys(f"Processed pairs: {processed_pairs}")
        
        if original_pairs != processed_pairs:
            print_sys(f"⚠ WARNING: Pair count mismatch!")
            return False
        
        # Verify each pair is valid
        total_valid = 0
        total_invalid = 0
        
        for pert_name, pairs in self.pairs.items():
            valid_count = 0
            invalid_count = 0
            
            for ctrl_idx, pert_idx in pairs:
                # Check index bounds
                if 0 <= ctrl_idx < len(self.adata) and 0 <= pert_idx < len(self.adata):
                    # Check perturbation matches
                    actual_pert = self.adata.obs['condition'].iloc[pert_idx]
                    if actual_pert == pert_name:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        print_sys(f"Perturbation mismatch: {pert_name} vs {actual_pert} at {pert_idx}")
                else:
                    invalid_count += 1
                    print_sys(f"Index out of bounds: ctrl={ctrl_idx}, pert={pert_idx}")
            
            total_valid += valid_count
            total_invalid += invalid_count
            
            if invalid_count > 0:
                print_sys(f"{pert_name}: {valid_count} valid, {invalid_count} invalid pairs")
        
        print_sys(f"Overall: {total_valid} valid, {total_invalid} invalid pairs")
        
        if total_invalid == 0:
            print_sys("✓ All CSV pairs are valid and correctly preserved")
            return True
        else:
            print_sys(f"⚠ Found {total_invalid} invalid pairs")
            return False
    
    def _process_ot_pairs(self):
        """
        Process optimal transport pairs from CSV
        """
        if self.ot_csv is None:
            return
            
        # Check if CSV is empty after index update
        if len(self.ot_csv) == 0:
            # print_sys("WARNING: CSV is empty after index update")
            # print_sys("Falling back to default behavior")
            self.ot_pairs = False
            self.pairs = {}
            return
            
        # print_sys("=== Processing OT Pairs ===")
        # print_sys(f"CSV shape: {self.ot_csv.shape}")
        
        # Since CSV indices are already valid positional indices, use them directly
        self.ot = self.ot_csv.copy()
        
        # The CSV indices are already positional indices, so we can use them directly
        # Just ensure they are integers
        self.ot['ctrl_new'] = self.ot['control_index'].astype(int)
        self.ot['pert_new'] = self.ot['perturbation_index'].astype(int)
        
        # Debug mapping results
        # print_sys(f"Mapping results:")
        # print_sys(f"  - Original control indices: {self.ot['control_index'].nunique()}")
        # print_sys(f"  - Mapped control indices: {self.ot['ctrl_new'].nunique()}")
        # print_sys(f"  - Original pert indices: {self.ot['perturbation_index'].nunique()}")
        # print_sys(f"  - Mapped pert indices: {self.ot['pert_new'].nunique()}")
        
        # Add perturbation labels
        self.ot['pert'] = [self.adata.obs['condition'].iloc[i] for i in self.ot['perturbation_index']]
        
        # Debug: Check what perturbations we found
        unique_perts = self.ot['pert'].unique()
        # print_sys(f"Found perturbations in CSV: {unique_perts}")
        
        self.pairs = {p: list(zip(g['ctrl_new'].values,
             g['pert_new'].values))
        for p, g in self.ot.groupby('pert')}
        
        # Set the ot_pairs flag to True since we have pairs
        self.ot_pairs = True
        
        # Debug final pairs
        print_sys(f"Final pairs created:")
        total_pairs = sum(len(pairs) for pairs in self.pairs.values())
        print_sys(f"Total pairs from CSV: {total_pairs}")
        print_sys(f"Unique perturbations with pairs: {len(self.pairs)}")
        
        # Show some examples
        for pert, pairs in list(self.pairs.items())[:5]:
            print_sys(f"  {pert}: {len(pairs)} pairs")
        if len(self.pairs) > 5:
            print_sys(f"  ... and {len(self.pairs) - 5} more perturbations")
        
        # If no pairs were created, fall back to default behavior
        if total_pairs == 0:
            print_sys("WARNING: No pairs were created from CSV. Falling back to default behavior.")
            self.ot_pairs = False
            self.pairs = {}
        
        # print_sys("==========================")
    
    def _filter_adata_to_csv_perts(self):
        """
        Filter adata to only include perturbations specified in CSV files.
        This ensures the model only trains/evaluates on the exact perturbations
        specified in the CSV files.
        
        IMPORTANT: Always includes control cells ('ctrl') even if not explicitly
        in CSV, since control cells are needed for the model to work.
        """
        if not self.ot_pairs or not hasattr(self, 'pairs') or not self.pairs:
            print_sys("No CSV pairs available, skipping adata filtering")
            return
            
        # Get unique perturbations from CSV pairs
        csv_perts = set(self.pairs.keys())
        
        # CRITICAL: Always include control cells, even if not explicitly in CSV
        # This is because:
        # 1. Control cells are referenced by control_index in CSV pairs
        # 2. GEARS model needs control cells to compute baseline expression
        # 3. Control cells are used as input in control->perturbation pairs
        control_label = getattr(self, 'control_label', 'ctrl')
        csv_perts_with_ctrl = csv_perts.copy()
        csv_perts_with_ctrl.add(control_label)
        
        print_sys(f"=== Filtering adata to CSV perturbations ===")
        print_sys(f"CSV perturbations: {len(csv_perts)}")
        print_sys(f"CSV perturbations + ctrl: {len(csv_perts_with_ctrl)}")
        print_sys(f"Original adata perturbations: {len(self.adata.obs['condition'].unique())}")
        
        # Filter adata to only include cells with perturbations in CSV (including ctrl)
        mask = self.adata.obs['condition'].isin(csv_perts_with_ctrl)
        original_size = self.adata.n_obs
        self.adata = self.adata[mask].copy()
        
        print_sys(f"Filtered adata size: {original_size} -> {self.adata.n_obs} cells")
        print_sys(f"Remaining perturbations: {len(self.adata.obs['condition'].unique())}")
        
        # Check control cells
        ctrl_count = (self.adata.obs['condition'] == control_label).sum()
        print_sys(f"Control cells remaining: {ctrl_count}")
        
        if ctrl_count == 0:
            print_sys("⚠ WARNING: No control cells found after filtering!")
            print_sys("  This will cause the GEARS model to fail.")
            print_sys("  The CSV should include control cells referenced by control_index.")
        else:
            print_sys("✓ Control cells preserved for model training")
        
        # Verify all remaining perturbations are in CSV (or ctrl)
        remaining_perts = set(self.adata.obs['condition'].unique())
        if remaining_perts.issubset(csv_perts_with_ctrl):
            print_sys("✓ All remaining perturbations are in CSV or are control cells")
        else:
            extra_perts = remaining_perts - csv_perts_with_ctrl
            print_sys(f"⚠ WARNING: Found extra perturbations not in CSV: {extra_perts}")
        
        print_sys("==========================================")
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph. 
        
        CRITICAL: Maps genes strictly to adata.var indices. 
        Since new_data_process now adds virtual nodes, this handles ALL genes safely.
        """
        
        # 1. Load the knowledge graph structure (GO / Essential Genes)
        if self.gene_set_path is not None:
            path_ = self.gene_set_path
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
        elif self.default_pert_graph is False:
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
        else:
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path, 'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)

        # 2. Filter gene2go to relevant genes
        # Note: We keep the gene2go mapping for the GNN structure, but we 
        # MUST derive indices from our actual adata, not this list.
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}
        
        # 3. STRICT INDEX ALIGNMENT
        # We override pert_names to be exactly the columns of our adata.
        # This ensures that index i in pert_names corresponds to column i in adata.X
        self.pert_names = self.adata.var['gene_name'].values
        
        # Create mapping: Gene Name -> Integer Index in adata.X
        self.node_map_pert = {x: i for i, x in enumerate(self.pert_names)}
        
        # Create case-insensitive mapping for safer lookups
        self.node_map_pert_lower = {str(x).lower(): (x, i) for i, x in enumerate(self.pert_names)}
        
        print_sys(f"Perturbation genes set. Mapped {len(self.pert_names)} genes (including virtual nodes) aligned to adata.")
        
    def load(self, data_name = None, data_path = None, ot_csv=None):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path

        Parameters
        ----------
        data_name: str
            Name of dataset
        data_path: str
            Path to dataset

        Returns
        -------
        None

        """
        
        if data_name in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either norman, adamson, dixit "
                             "replogle_k562 or replogle_rpe1 "
                             "or a path to an h5ad file")
        
        self.set_pert_genes()
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)
        
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]

        self.adata = self.adata[filter_go.index.values, :]
        
        # Update CSV indices to match the filtered adata
        if self.ot_csv is not None:
            print_sys("Updating CSV indices to match filtered adata...")
            self.update_csv_indices()
        
        # Create mapping from original indices to new indices
        # Convert categorical to string if needed for mapping
        if self.adata.obs['ind'].dtype.name == 'category':
            id2idx = dict(zip(self.adata.obs['ind'].astype(str), np.arange(self.adata.n_obs)))
        else:
            id2idx = dict(zip(self.adata.obs['ind'], np.arange(self.adata.n_obs)))
        
        
        # Only process pairs if they haven't been processed yet
        if not hasattr(self, 'pairs') or self.pairs is None:
            self.ot = self.ot_csv.copy()
            # Since CSV indices are already valid positional indices, use them directly
            self.ot['ctrl_new'] = self.ot['control_index'].astype(int)
            self.ot['pert_new'] = self.ot['perturbation_index'].astype(int)
            
            # Debug: Check mapping results
            print_sys(f"Mapping results:")
            print_sys(f"  - Original control indices: {self.ot['control_index'].nunique()}")
            print_sys(f"  - Mapped control indices: {self.ot['ctrl_new'].nunique()}")
            print_sys(f"  - Original pert indices: {self.ot['perturbation_index'].nunique()}")
            print_sys(f"  - Mapped pert indices: {self.ot['pert_new'].nunique()}")
            
            # Add perturbation labels
            self.ot['pert'] = [self.adata.obs['condition'].iloc[i] for i in self.ot['perturbation_index']]
            
            # Debug: Check what perturbations we found
            unique_perts = self.ot['pert'].unique()
            print_sys(f"Found perturbations in CSV: {unique_perts}")
            
            self.pairs = {p: list(zip(g['ctrl_new'].values,
                 g['pert_new'].values))
            for p, g in self.ot.groupby('pert')}
            
            # Set the ot_pairs flag to True since we have pairs
            self.ot_pairs = True
            
            # Debug: Check final pairs
            print_sys(f"Final pairs created:")
            for pert, pairs in self.pairs.items():
                print_sys(f"  {pert}: {len(pairs)} pairs")
            
            # If no pairs were created, fall back to default behavior
            total_pairs = sum(len(pairs) for pairs in self.pairs.values())
            if total_pairs == 0:
                print_sys("WARNING: No pairs were created from CSV. Falling back to default behavior.")
                self.ot_pairs = False
                self.pairs = {}
        else:
            print_sys("Pairs already processed during new_data_process")
            # Make sure ot_pairs is set
            self.ot_pairs = True
            
        # CRITICAL: Filter adata to only include perturbations specified in CSV
        if self.ot_pairs and hasattr(self, 'pairs') and self.pairs:
            self._filter_adata_to_csv_perts()
            
        print_sys("==========================")
        
        if not issparse(self.adata.X):
            self.adata.X = csr_matrix(self.adata.X)

        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            control_label = getattr(self, 'control_label', 'ctrl')
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == control_label]
            self.gene_names = self.adata.var.gene_name
            
            
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
            
    def create_cell_graph_dataset(self, split_adata, pert_category, num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        MEMORY-OPTIMIZED: Avoids storing Xs/ys lists.
        """
        import gc
        
        num_de_genes = 20        
        # Get indices for this perturbation
        # NOTE: slicing adata is fast (view), accessing .X can be slow/memory heavy depending on format
        de_idx = []
        if 'rank_genes_groups_cov_all' in split_adata.uns:
            try:
                de_genes = split_adata.uns['rank_genes_groups_cov_all']['names'][pert_category][:num_de_genes]
                for gene in de_genes:
                    if gene in self.node_map_pert:
                        de_idx.append(self.node_map_pert[gene])
            except (KeyError, TypeError):
                pass
        
        # Pad with -1 if not enough DE genes found
        while len(de_idx) < num_de_genes:
            de_idx.append(-1)
        
        de_idx = de_idx[:num_de_genes]  # Ensure exactly num_de_genes

        cell_graphs = []
        
        # 1. OPTIMAL TRANSPORT MODE
        if self.ot_pairs and hasattr(self, 'pairs') and self.pairs:
            if pert_category in self.pairs and len(self.pairs[pert_category]) > 0:
                
                # Process pairs one by one
                for c_idx, p_idx in self.pairs[pert_category]:
                    if c_idx < len(self.adata) and p_idx < len(self.adata):
                        
                        # 1. Extract Single Sparse Vectors (Low Memory)
                        X_sparse = self.adata.X[c_idx]
                        y_sparse = self.adata.X[p_idx]
                        
                        # 2. Convert to Dense ONLY for Graph Creation
                        X_dense = X_sparse.toarray() if sp.issparse(X_sparse) else np.asarray(X_sparse)
                        y_dense = y_sparse.toarray() if sp.issparse(y_sparse) else np.asarray(y_sparse)
                        
                        # 3. Create Graph
                        try:
                            # Assuming create_cell_graph returns a PyG Data object
                            cell_graph = self.create_cell_graph(
                                X_dense, y_dense, de_idx, pert_category, None
                            )
                            cell_graphs.append(cell_graph)
                        except Exception as e:
                            print_sys(f"Error creating graph: {e}")
                        
                        # 4. CRITICAL: Cleanup immediate dense arrays
                        del X_dense, y_dense, X_sparse, y_sparse
                        
                # GC after processing all pairs for this perturbation
                # gc.collect()
                
        # 2. RANDOM SAMPLING MODE (Fallback)
        else:
            # Get indices of perturbation cells
            pert_mask = (split_adata.obs['condition'] == pert_category)
            pert_indices = np.where(pert_mask)[0]
            
            if len(self.ctrl_adata) == 0:
                return []

            for cell_idx in pert_indices:
                # 1. Extract Target (Perturbed)
                cell_z = split_adata.X[cell_idx]
                y_dense = cell_z.toarray() if sp.issparse(cell_z) else np.asarray(cell_z)
                
                # 2. Sample Controls
                ctrl_indices = np.random.randint(0, len(self.ctrl_adata), num_samples)
                
                for c_idx in ctrl_indices:
                    # Extract Source (Control)
                    c = self.ctrl_adata.X[c_idx]
                    X_dense = c.toarray() if sp.issparse(c) else np.asarray(c)
                    
                    try:
                        cell_graph = self.create_cell_graph(
                            X_dense, y_dense, de_idx, pert_category, None
                        )
                        cell_graphs.append(cell_graph)
                    except Exception as e:
                        pass
                    
                    del X_dense
                
                del y_dense
                
            gc.collect()

        return cell_graphs
    
    
    def verify_pair_usage(self):
        """
        Verify and print statistics about the pairs being used
        """
        print_sys("=== Pair Usage Verification ===")
        print_sys(f"ot_pairs flag: {self.ot_pairs}")
        print_sys(f"Has pairs attribute: {hasattr(self, 'pairs')}")
        if hasattr(self, 'pairs'):
            print_sys(f"Pairs is None: {self.pairs is None}")
            print_sys(f"Pairs is empty: {len(self.pairs) == 0 if self.pairs is not None else 'N/A'}")
        
        if not self.ot_pairs:
            print_sys("ot_pairs is False, using all available pairs")
            return
            
        total_pairs = 0
        for pert, pairs in self.pairs.items():
            print_sys(f"Perturbation {pert}: {len(pairs)} pairs")
            total_pairs += len(pairs)
        print_sys(f"Total pairs: {total_pairs}")
        
        # Check if all pairs from CSV are being used
        if hasattr(self, 'ot') and self.ot is not None:
            csv_pairs = len(self.ot)
            print_sys(f"Pairs in CSV: {csv_pairs}")
            if total_pairs != csv_pairs:
                print_sys(f"WARNING: Mismatch between CSV pairs ({csv_pairs}) and used pairs ({total_pairs})")
        print_sys("================================")

    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition.
        Optimized to minimize intermediate object storage.
        """
        import gc
        print_sys("Creating dataset file...")
        
        # Initialize dictionary
        self.dataset_processed = {}
        
        # Path setup
        pyg_path = os.path.join(self.dataset_path, 'data_pyg')
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        
        if self.ot_pairs and hasattr(self, 'pairs') and self.pairs:
            total_pairs_used = 0
            csv_perturbations = sorted(list(set(self.pairs.keys()))) # Sort for consistent tqdm
            
            print_sys(f"Processing {len(csv_perturbations)} perturbations from CSV pairs")
            
            for i, p in enumerate(tqdm(csv_perturbations)):
                if p in self.pairs and len(self.pairs[p]) > 0:
                    # This calls the OPTIMIZED function below
                    dataset = self.create_cell_graph_dataset(self.adata, p)
                    self.dataset_processed[p] = dataset
                    total_pairs_used += len(self.pairs[p])
                else:
                    self.dataset_processed[p] = []
                
                # Aggressive GC every 50 perturbations to prevent memory fragmentation
                if i % 50 == 0:
                    gc.collect()
            
            print_sys(f"Total samples created: {sum(len(d) for d in self.dataset_processed.values())}")
            
        else:
            # Fallback mode
            print_sys("No pairs available, creating data for all perturbations")
            unique_perts = sorted(list(self.adata.obs['condition'].unique()))
            
            for i, p in enumerate(tqdm(unique_perts)):
                self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
                if i % 50 == 0:
                    gc.collect()

        # Save once at the end
        print_sys(f"Saving dataset to {dataset_fname}...")
        try:
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
        except MemoryError:
            print_sys("CRITICAL: OOM during pickle dump. Try using joblib instead.")
            # Fallback to joblib if available (better for large numpy data)
            try:
                import joblib
                joblib.dump(self.dataset_processed, dataset_fname)
            except ImportError:
                raise
                
        print_sys("Done!")
    
    # Clear memory immediately after saving if you don't need it in this class anymore
    # (Optional, depends on if GEARS needs self.dataset_processed accessible later)
    # del self.dataset_processed
    # gc.collect()

    def verify_dataloader_integrity(self, train_loader, test_loader=None):
        """
        Verify that the loaded data in trainloader/testloader is correct
        
        Parameters
        ----------
        train_loader: DataLoader
            Training data loader
        test_loader: DataLoader, optional
            Test data loader
        """
        print_sys("=== Dataloader Integrity Verification ===")
        
        # Verify training data
        print_sys("Verifying training data...")
        train_stats = self._verify_loader(train_loader, "train")
        
        # Verify test data if provided
        if test_loader is not None:
            print_sys("Verifying test data...")
            test_stats = self._verify_loader(test_loader, "test")
        else:
            test_stats = None
            
        # Summary
        print_sys("=== Summary ===")
        print_sys(f"Training: {train_stats['total_samples']} samples, {train_stats['unique_perts']} unique perturbations")
        if test_stats:
            print_sys(f"Test: {test_stats['total_samples']} samples, {test_stats['unique_perts']} unique perturbations")
        
        # Check if all expected perturbations are present
        expected_perts = set(self.adata.obs['condition'].unique())
        train_perts = set(train_stats['perturbations'])
        missing_train = expected_perts - train_perts
        if missing_train:
            print_sys(f"WARNING: Missing perturbations in training: {missing_train}")
        else:
            print_sys("✓ All expected perturbations present in training")
            
        if test_stats:
            test_perts = set(test_stats['perturbations'])
            missing_test = expected_perts - test_perts
            if missing_test:
                print_sys(f"WARNING: Missing perturbations in test: {missing_test}")
            else:
                print_sys("✓ All expected perturbations present in test")
        
        print_sys("================================")
        return train_stats, test_stats
    
    def _verify_loader(self, loader, name):
        """
        Verify a single dataloader
        
        Parameters
        ----------
        loader: DataLoader
            Data loader to verify
        name: str
            Name of the loader (train/test)
            
        Returns
        -------
        dict
            Statistics about the loader
        """
        total_samples = 0
        perturbations = []
        batch_sizes = []
        feature_dims = []
        target_dims = []
        
        print_sys(f"  Checking {name} loader...")
        
        for batch_idx, batch in enumerate(loader):
            # Count samples
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
            total_samples += batch_size
            batch_sizes.append(batch_size)
            
            # Check perturbations
            if hasattr(batch, 'pert') and batch.pert is not None:
                perts = batch.pert.tolist() if hasattr(batch.pert, 'tolist') else batch.pert
                perturbations.extend(perts)
            
            # Check feature dimensions
            if hasattr(batch, 'x') and batch.x is not None:
                feature_dims.append(batch.x.shape)
            
            # Check target dimensions
            if hasattr(batch, 'y') and batch.y is not None:
                target_dims.append(batch.y.shape)
            
            # Check first few batches in detail
            if batch_idx < 3:
                print_sys(f"    Batch {batch_idx}: {batch_size} samples")
                if hasattr(batch, 'pert') and batch.pert is not None:
                    unique_perts = set(batch.pert.tolist() if hasattr(batch.pert, 'tolist') else batch.pert)
                    print_sys(f"      Perturbations: {unique_perts}")
                if hasattr(batch, 'x') and batch.x is not None:
                    print_sys(f"      Feature shape: {batch.x.shape}")
                if hasattr(batch, 'y') and batch.y is not None:
                    print_sys(f"      Target shape: {batch.y.shape}")
        
        unique_perts = len(set(perturbations))
        
        print_sys(f"  {name} loader summary:")
        print_sys(f"    Total samples: {total_samples}")
        print_sys(f"    Unique perturbations: {unique_perts}")
        print_sys(f"    Number of batches: {len(batch_sizes)}")
        print_sys(f"    Average batch size: {np.mean(batch_sizes):.1f}")
        
        if feature_dims:
            print_sys(f"    Feature dimensions: {feature_dims[0]}")
        if target_dims:
            print_sys(f"    Target dimensions: {target_dims[0]}")
        
        return {
            'total_samples': total_samples,
            'unique_perts': unique_perts,
            'perturbations': perturbations,
            'batch_sizes': batch_sizes,
            'feature_dims': feature_dims,
            'target_dims': target_dims
        }

    def verify_data_matches(self, train_loader, num_samples=5):
        """
        Verify that the data matches are correct by checking sample pairs
        
        Parameters
        ----------
        train_loader: DataLoader
            Training data loader
        num_samples: int
            Number of sample pairs to check
        """
        print_sys("=== Data Match Verification ===")
        
        sample_count = 0
        for batch_idx, batch in enumerate(train_loader):
            if sample_count >= num_samples:
                break
                
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
            
            for i in range(min(batch_size, num_samples - sample_count)):
                print_sys(f"Sample {sample_count + 1}:")
                
                # Get the sample data
                if hasattr(batch, 'x') and batch.x is not None:
                    x = batch.x[i] if batch.x.dim() > 1 else batch.x
                    print_sys(f"  Input features shape: {x.shape}")
                    print_sys(f"  Input features range: [{x.min():.3f}, {x.max():.3f}]")
                
                if hasattr(batch, 'y') and batch.y is not None:
                    y = batch.y[i] if batch.y.dim() > 1 else batch.y
                    print_sys(f"  Target features shape: {y.shape}")
                    print_sys(f"  Target features range: [{y.min():.3f}, {y.max():.3f}]")
                
                if hasattr(batch, 'pert') and batch.pert is not None:
                    pert = batch.pert[i] if hasattr(batch.pert, '__len__') and len(batch.pert) > 1 else batch.pert
                    print_sys(f"  Perturbation: {pert}")
                
                if hasattr(batch, 'pert_idx') and batch.pert_idx is not None:
                    pert_idx = batch.pert_idx[i] if hasattr(batch.pert_idx, '__len__') and len(batch.pert_idx) > 1 else batch.pert_idx
                    print_sys(f"  Perturbation index: {pert_idx}")
                
                # Check if input and target are different (as expected for perturbation data)
                if hasattr(batch, 'x') and hasattr(batch, 'y') and batch.x is not None and batch.y is not None:
                    x_sample = batch.x[i] if batch.x.dim() > 1 else batch.x
                    y_sample = batch.y[i] if batch.y.dim() > 1 else batch.y
                    if x_sample.shape == y_sample.shape:
                        diff = torch.abs(x_sample - y_sample).mean().item()
                        print_sys(f"  Mean difference between input and target: {diff:.4f}")
                        if diff < 1e-6:
                            print_sys(f"  WARNING: Input and target are nearly identical!")
                        else:
                            print_sys(f"  ✓ Input and target are different (as expected)")
                
                print_sys("")
                sample_count += 1
        
        print_sys("================================")

    def verify_perturbation_distribution(self, train_loader, test_loader=None):
        """
        Verify the perturbation distribution in the dataloaders
        
        Parameters
        ----------
        train_loader: DataLoader
            Training data loader
        test_loader: DataLoader, optional
            Test data loader
        """
        print_sys("=== Perturbation Distribution Verification ===")
        
        # Analyze training data
        train_pert_counts = {}
        train_total = 0
        
        for batch in train_loader:
            if hasattr(batch, 'pert') and batch.pert is not None:
                perts = batch.pert.tolist() if hasattr(batch.pert, 'tolist') else batch.pert
                for pert in perts:
                    train_pert_counts[pert] = train_pert_counts.get(pert, 0) + 1
                    train_total += 1
        
        print_sys("Training perturbation distribution:")
        for pert, count in sorted(train_pert_counts.items()):
            percentage = (count / train_total) * 100
            print_sys(f"  {pert}: {count} samples ({percentage:.1f}%)")
        
        # Analyze test data if provided
        if test_loader is not None:
            test_pert_counts = {}
            test_total = 0
            
            for batch in test_loader:
                if hasattr(batch, 'pert') and batch.pert is not None:
                    perts = batch.pert.tolist() if hasattr(batch.pert, 'tolist') else batch.pert
                    for pert in perts:
                        test_pert_counts[pert] = test_pert_counts.get(pert, 0) + 1
                        test_total += 1
            
            print_sys("Test perturbation distribution:")
            for pert, count in sorted(test_pert_counts.items()):
                percentage = (count / test_total) * 100
                print_sys(f"  {pert}: {count} samples ({percentage:.1f}%)")
        
        # Check for potential issues
        print_sys("Distribution analysis:")
        
        # Check for highly imbalanced perturbations
        if train_pert_counts:
            max_count = max(train_pert_counts.values())
            min_count = min(train_pert_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print_sys(f"  Max/min sample ratio: {imbalance_ratio:.1f}")
            if imbalance_ratio > 10:
                print_sys(f"  WARNING: High imbalance detected!")
            else:
                print_sys(f"  ✓ Sample distribution looks reasonable")
        
        # Check for perturbations with very few samples
        low_count_perts = [pert for pert, count in train_pert_counts.items() if count < 10]
        if low_count_perts:
            print_sys(f"  WARNING: Perturbations with <10 samples: {low_count_perts}")
        else:
            print_sys(f"  ✓ All perturbations have sufficient samples")
        
        # Check for control perturbation
        if 'ctrl' in train_pert_counts:
            ctrl_percentage = (train_pert_counts['ctrl'] / train_total) * 100
            print_sys(f"  Control samples: {ctrl_percentage:.1f}%")
            if ctrl_percentage < 10:
                print_sys(f"  WARNING: Low control sample percentage!")
            elif ctrl_percentage > 50:
                print_sys(f"  WARNING: High control sample percentage!")
            else:
                print_sys(f"  ✓ Control sample percentage looks reasonable")
        
        print_sys("================================")

    def verify_csv_pairs_in_dataloader(self, train_loader, test_loader=None):
        """
        Verify that all pairs from the CSV are present in the dataloaders
        
        Parameters
        ----------
        train_loader: DataLoader
            Training data loader
        test_loader: DataLoader, optional
            Test data loader
        """
        print_sys("=== CSV Pairs Verification ===")
        
        if not hasattr(self, 'pairs') or not self.pairs:
            print_sys("No CSV pairs available to verify")
            return
        
        # Collect all pairs from dataloaders
        train_pairs = set()
        test_pairs = set()
        
        # Extract pairs from training data
        for batch in train_loader:
            if hasattr(batch, 'x') and hasattr(batch, 'y') and batch.x is not None and batch.y is not None:
                # For each sample in the batch, create a pair identifier
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
                for i in range(batch_size):
                    x_sample = batch.x[i] if batch.x.dim() > 1 else batch.x
                    y_sample = batch.y[i] if batch.y.dim() > 1 else batch.y
                    # Create a hash of the pair for comparison
                    pair_hash = hash((tuple(x_sample.flatten().tolist()), tuple(y_sample.flatten().tolist())))
                    train_pairs.add(pair_hash)
        
        # Extract pairs from test data if available
        if test_loader is not None:
            for batch in test_loader:
                if hasattr(batch, 'x') and hasattr(batch, 'y') and batch.x is not None and batch.y is not None:
                    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
                    for i in range(batch_size):
                        x_sample = batch.x[i] if batch.x.dim() > 1 else batch.x
                        y_sample = batch.y[i] if batch.y.dim() > 1 else batch.y
                        pair_hash = hash((tuple(x_sample.flatten().tolist()), tuple(y_sample.flatten().tolist())))
                        test_pairs.add(pair_hash)
        
        # Create expected pairs from CSV
        expected_pairs = set()
        if hasattr(self, 'ot') and self.ot is not None:
            for _, row in self.ot.iterrows():
                if not pd.isna(row['ctrl_new']) and not pd.isna(row['pert_new']):
                    ctrl_idx = int(row['ctrl_new'])
                    pert_idx = int(row['pert_new'])
                    if ctrl_idx < len(self.adata) and pert_idx < len(self.adata):
                        # Get the actual data for this pair
                        ctrl_data = self.adata.X[ctrl_idx].toarray().flatten()
                        pert_data = self.adata.X[pert_idx].toarray().flatten()
                        pair_hash = hash((tuple(ctrl_data.tolist()), tuple(pert_data.tolist())))
                        expected_pairs.add(pair_hash)
        
        # Compare
        all_dataloader_pairs = train_pairs.union(test_pairs)
        missing_pairs = expected_pairs - all_dataloader_pairs
        extra_pairs = all_dataloader_pairs - expected_pairs
        
        print_sys(f"CSV pairs analysis:")
        print_sys(f"  Expected pairs from CSV: {len(expected_pairs)}")
        print_sys(f"  Pairs in training: {len(train_pairs)}")
        print_sys(f"  Pairs in test: {len(test_pairs)}")
        print_sys(f"  Total pairs in dataloaders: {len(all_dataloader_pairs)}")
        print_sys(f"  Missing pairs: {len(missing_pairs)}")
        print_sys(f"  Extra pairs: {len(extra_pairs)}")
        
        if len(missing_pairs) == 0:
            print_sys("  ✓ All CSV pairs are present in dataloaders")
        else:
            print_sys(f"  WARNING: {len(missing_pairs)} CSV pairs are missing from dataloaders")
        
        if len(extra_pairs) > 0:
            print_sys(f"  NOTE: {len(extra_pairs)} additional pairs in dataloaders (may be from default behavior)")
        
        # Check coverage by perturbation
        if hasattr(self, 'ot') and self.ot is not None:
            print_sys(f"Coverage by perturbation:")
            for pert in self.ot['pert'].unique():
                pert_pairs = self.ot[self.ot['pert'] == pert]
                pert_expected = len(pert_pairs)
                print_sys(f"  {pert}: {pert_expected} expected pairs")
        
        print_sys("================================")

    def update_csv_indices(self):
        """
        Update CSV indices to match the current adata after processing
        This handles cases where adata filtering changes the indices
        """
        if self.ot_csv is None:
            print_sys("No CSV data available to update")
            return
            
        print_sys("=== Updating CSV Indices ===")
        print_sys(f"Original CSV shape: {self.ot_csv.shape}")
        
        if not hasattr(self, 'adata') or self.adata is None:
            print_sys("No adata available for index update")
            return
        
        # Create mapping from original cell IDs to new indices
        if 'ind' in self.adata.obs.columns:
            # Create mapping from original cell IDs to new positions
            id_to_new_idx = {}
            for new_idx, original_id in enumerate(self.adata.obs['ind']):
                id_to_new_idx[original_id] = new_idx
            
            print_sys(f"Created mapping for {len(id_to_new_idx)} cells")
            
            # Update CSV indices using the mapping
            updated_csv = self.ot_csv.copy()
            original_pairs = len(self.ot_csv)
            valid_pairs = 0
            
            for idx, row in self.ot_csv.iterrows():
                ctrl_id = row['control_id'] if 'control_id' in row else None
                pert_id = row['pert_id'] if 'pert_id' in row else None
                
                if ctrl_id in id_to_new_idx and pert_id in id_to_new_idx:
                    updated_csv.loc[idx, 'control_index'] = id_to_new_idx[ctrl_id]
                    updated_csv.loc[idx, 'perturbation_index'] = id_to_new_idx[pert_id]
                    valid_pairs += 1
                else:
                    # Mark invalid pairs
                    updated_csv.loc[idx, 'control_index'] = -1
                    updated_csv.loc[idx, 'perturbation_index'] = -1
            
            # Filter out invalid pairs
            valid_mask = (updated_csv['control_index'] >= 0) & (updated_csv['perturbation_index'] >= 0)
            self.ot_csv = updated_csv[valid_mask].copy()
            
            print_sys(f"Updated CSV indices:")
            print_sys(f"  Original pairs: {original_pairs}")
            print_sys(f"  Valid pairs: {len(self.ot_csv)}")
            print_sys(f"  Lost pairs: {original_pairs - len(self.ot_csv)}")
            print_sys("✓ CSV indices updated successfully")
        else:
            print_sys("WARNING: No 'ind' column in current adata")
            print_sys("Cannot update CSV indices - using original indices")

    def reprocess_with_updated_indices(self):
        """
        Reprocess the data with updated CSV indices
        This is useful if the CSV indices were updated manually
        """
        print_sys("=== Reprocessing with Updated Indices ===")
        
        if self.ot_csv is None:
            print_sys("No CSV data available")
            return
        
        # Clear existing pairs
        if hasattr(self, 'pairs'):
            delattr(self, 'pairs')
        if hasattr(self, 'ot'):
            delattr(self, 'ot')
        
        # Reset ot_pairs flag
        self.ot_pairs = False
        
        # Update indices
        self.update_csv_indices()
        
        # Reprocess pairs
        if self.ot_csv is not None and len(self.ot_csv) > 0:
            print_sys("Reprocessing optimal transport pairs...")
            self._process_ot_pairs()
            
            # Verify the results
            if hasattr(self, 'pairs') and self.pairs:
                total_pairs = sum(len(pairs) for pairs in self.pairs.values())
                print_sys(f"✓ Successfully created {total_pairs} pairs from {len(self.ot_csv)} CSV rows")
            else:
                print_sys("WARNING: Still no pairs created after index update")
        else:
            print_sys("WARNING: No valid pairs in CSV after index update")
        
        print_sys("================================")

    def debug_index_mapping(self):
        """
        Debug the index mapping issue by showing original vs current indices
        """
        print_sys("=== Index Mapping Debug ===")
        
        if self.ot_csv is None:
            print_sys("No CSV data available")
            return
            
        print_sys(f"CSV shape: {self.ot_csv.shape}")
        print_sys(f"CSV columns: {self.ot_csv.columns.tolist()}")
        
        # Show CSV index ranges
        print_sys(f"CSV index ranges:")
        print_sys(f"  control_index: {self.ot_csv['control_index'].min()} to {self.ot_csv['control_index'].max()}")
        print_sys(f"  perturbation_index: {self.ot_csv['perturbation_index'].min()} to {self.ot_csv['perturbation_index'].max()}")
        
        # Show sample CSV indices
        print_sys(f"Sample CSV indices:")
        print_sys(f"  control_index: {self.ot_csv['control_index'].head(10).tolist()}")
        print_sys(f"  perturbation_index: {self.ot_csv['perturbation_index'].head(10).tolist()}")
        
        if hasattr(self, 'adata') and self.adata is not None:
            print_sys(f"Current adata shape: {self.adata.shape}")
            print_sys(f"Current adata index range: 0 to {len(self.adata)-1}")
            
            if 'ind' in self.adata.obs.columns:
                current_indices = self.adata.obs['ind'].values
                print_sys(f"Current adata 'ind' column:")
                print_sys(f"  Type: {type(current_indices)}")
                print_sys(f"  Shape: {current_indices.shape}")
                print_sys(f"  Sample values: {current_indices[:10].tolist()}")
                
                # Check if CSV indices are in current adata indices
                csv_control_set = set(self.ot_csv['control_index'])
                csv_pert_set = set(self.ot_csv['perturbation_index'])
                current_set = set(current_indices)
                
                control_overlap = len(csv_control_set.intersection(current_set))
                pert_overlap = len(csv_pert_set.intersection(current_set))
                
                print_sys(f"Index overlap analysis:")
                print_sys(f"  CSV control indices: {len(csv_control_set)}")
                print_sys(f"  CSV pert indices: {len(csv_pert_set)}")
                print_sys(f"  Current adata indices: {len(current_set)}")
                
                # Fix division by zero
                control_pct = (control_overlap/len(csv_control_set)*100) if len(csv_control_set) > 0 else 0
                pert_pct = (pert_overlap/len(csv_pert_set)*100) if len(csv_pert_set) > 0 else 0
                
                print_sys(f"  Control overlap: {control_overlap} ({control_pct:.1f}%)")
                print_sys(f"  Pert overlap: {pert_overlap} ({pert_pct:.1f}%)")
                
                if control_overlap == 0 and pert_overlap == 0:
                    print_sys("WARNING: No overlap between CSV indices and current adata indices!")
                    print_sys("This suggests the CSV was created from a different version of the adata")
                else:
                    print_sys("✓ Some overlap found between CSV and current adata indices")
            else:
                print_sys("WARNING: No 'ind' column in current adata")
        else:
            print_sys("WARNING: No adata available")
        
        print_sys("==========================")

    def debug_original_adata_indices(self, original_adata):
        """
        Debug the original adata indices to understand the mapping issue
        
        Parameters
        ----------
        original_adata: AnnData
            The original adata before processing
        """
        print_sys("=== Original Adata Index Debug ===")
        
        if original_adata is None:
            print_sys("No original adata provided")
            return
            
        print_sys(f"Original adata shape: {original_adata.shape}")
        
        if 'ind' in original_adata.obs.columns:
            original_indices = original_adata.obs['ind'].values
            print_sys(f"Original adata 'ind' column:")
            print_sys(f"  Type: {type(original_indices)}")
            print_sys(f"  Shape: {original_indices.shape}")
            print_sys(f"  Sample values: {original_indices[:10].tolist()}")
            
            if self.ot_csv is not None and len(self.ot_csv) > 0:
                # Check overlap with original adata
                csv_control_set = set(self.ot_csv['control_index'])
                csv_pert_set = set(self.ot_csv['perturbation_index'])
                original_set = set(original_indices)
                
                control_overlap = len(csv_control_set.intersection(original_set))
                pert_overlap = len(csv_pert_set.intersection(original_set))
                
                print_sys(f"Original adata overlap analysis:")
                print_sys(f"  CSV control indices: {len(csv_control_set)}")
                print_sys(f"  CSV pert indices: {len(csv_pert_set)}")
                print_sys(f"  Original adata indices: {len(original_set)}")
                
                control_pct = (control_overlap/len(csv_control_set)*100) if len(csv_control_set) > 0 else 0
                pert_pct = (pert_overlap/len(csv_pert_set)*100) if len(csv_pert_set) > 0 else 0
                
                print_sys(f"  Control overlap: {control_overlap} ({control_pct:.1f}%)")
                print_sys(f"  Pert overlap: {pert_overlap} ({pert_pct:.1f}%)")
                
                if control_overlap > 0 or pert_overlap > 0:
                    print_sys("✓ CSV indices match original adata indices")
                    print_sys("The issue is that the adata was filtered during processing")
                else:
                    print_sys("WARNING: CSV indices don't match original adata indices")
        else:
            print_sys("WARNING: No 'ind' column in original adata")
        
        print_sys("==========================")

    def analyze_csv_indices(self):
        """
        Analyze what the CSV indices represent and provide guidance for mapping
        """
        print_sys("=== CSV Index Analysis ===")
        
        if self.ot_csv is None or len(self.ot_csv) == 0:
            print_sys("No CSV data available")
            return
            
        print_sys(f"CSV shape: {self.ot_csv.shape}")
        
        # Analyze the index ranges
        control_min, control_max = self.ot_csv['control_index'].min(), self.ot_csv['control_index'].max()
        pert_min, pert_max = self.ot_csv['perturbation_index'].min(), self.ot_csv['perturbation_index'].max()
        
        print_sys(f"Index ranges:")
        print_sys(f"  control_index: {control_min} to {control_max}")
        print_sys(f"  perturbation_index: {pert_min} to {pert_max}")
        
        # Check if indices are 0-based
        if control_min == 0 and pert_min == 0:
            print_sys("✓ Indices appear to be 0-based (start from 0)")
        else:
            print_sys("⚠ Indices don't start from 0")
        
        # Check if indices are within reasonable range for adata
        if hasattr(self, 'adata') and self.adata is not None:
            adata_size = len(self.adata)
            print_sys(f"Current adata size: {adata_size}")
            
            if control_max < adata_size and pert_max < adata_size:
                print_sys("✓ All CSV indices are within current adata range")
                print_sys("This suggests the CSV indices are positional indices for the current adata")
                return True
            else:
                print_sys("⚠ Some CSV indices exceed current adata range")
                print_sys("This suggests the CSV was created from a different/larger adata")
                return False
        
        return False

    def filter_set2conditions_for_available_data(self):
        """
        Filter set2conditions to only include perturbations that have data in dataset_processed
        This prevents KeyError when get_dataloader tries to access non-existent perturbations
        """
        if not hasattr(self, 'set2conditions'):
            print_sys("No set2conditions found, skipping filtering")
            return
            
        print_sys("=== Filtering set2conditions for available data ===")
        
        # Get all available perturbations in dataset_processed
        available_perts = set(self.dataset_processed.keys())
        
        # Filter each split
        for split_name, pert_list in self.set2conditions.items():
            original_count = len(pert_list)
            filtered_perts = [p for p in pert_list if p in available_perts]
            removed_perts = [p for p in pert_list if p not in available_perts]
            
            if removed_perts:
                print_sys(f"Split '{split_name}': Removed {len(removed_perts)} perturbations without data:")
                for pert in removed_perts:
                    print_sys(f"  - {pert}")
            
            self.set2conditions[split_name] = filtered_perts
            print_sys(f"Split '{split_name}': {original_count} -> {len(filtered_perts)} perturbations")
        
        print_sys("================================")

    def analyze_perturbation_coverage(self):
        """
        Analyze the coverage of perturbations between CSV pairs and dataset
        """
        print_sys("=== Perturbation Coverage Analysis ===")
        
        # Get all perturbations in the dataset
        all_dataset_perts = set(self.adata.obs['condition'].unique())
        
        # Get perturbations with pairs in CSV
        if hasattr(self, 'pairs') and self.pairs:
            csv_perts = set(self.pairs.keys())
        else:
            csv_perts = set()
        
        # Get perturbations that will be processed
        if hasattr(self, 'dataset_processed'):
            processed_perts = set(self.dataset_processed.keys())
        else:
            processed_perts = set()
        
        print_sys(f"Dataset perturbations: {len(all_dataset_perts)}")
        print_sys(f"CSV perturbations: {len(csv_perts)}")
        print_sys(f"Processed perturbations: {len(processed_perts)}")
        
        # Analyze single vs combo perturbations
        single_perts = {p for p in all_dataset_perts if p != 'ctrl' and p.count('+') == 1}
        combo_perts = {p for p in all_dataset_perts if p != 'ctrl' and p.count('+') > 1}
        
        print_sys(f"Single perturbations: {len(single_perts)}")
        print_sys(f"Combo perturbations: {len(combo_perts)}")
        
        # Verify filtering worked correctly
        if csv_perts:
            csv_single = {p for p in csv_perts if p != 'ctrl' and p.count('+') == 1}
            csv_combo = {p for p in csv_perts if p != 'ctrl' and p.count('+') > 1}
            print_sys(f"CSV covers {len(csv_single)}/{len(single_perts)} single perturbations")
            print_sys(f"CSV covers {len(csv_combo)}/{len(combo_perts)} combo perturbations")
            
            # Check if dataset was properly filtered (including control cells)
            csv_perts_with_ctrl = csv_perts.copy()
            csv_perts_with_ctrl.add('ctrl')
            
            if all_dataset_perts.issubset(csv_perts_with_ctrl):
                print_sys("✓ Dataset properly filtered to CSV perturbations + control cells")
            else:
                extra_perts = all_dataset_perts - csv_perts_with_ctrl
                print_sys(f"⚠ Dataset contains {len(extra_perts)} perturbations not in CSV")
                print_sys("  This suggests filtering may not have worked correctly")
        
        # Check CSV coverage
        csv_single_coverage = csv_perts.intersection(single_perts)
        csv_combo_coverage = csv_perts.intersection(combo_perts)
        
        print_sys(f"CSV covers {len(csv_single_coverage)}/{len(single_perts)} single perturbations")
        print_sys(f"CSV covers {len(csv_combo_coverage)}/{len(combo_perts)} combo perturbations")
        
        # Show missing perturbations
        missing_single = single_perts - csv_perts
        missing_combo = combo_perts - csv_perts
        
        if missing_single:
            print_sys(f"Missing single perturbations ({len(missing_single)}):")
            for pert in sorted(missing_single)[:10]:  # Show first 10
                print_sys(f"  - {pert}")
            if len(missing_single) > 10:
                print_sys(f"  ... and {len(missing_single) - 10} more")
        
        if missing_combo:
            print_sys(f"Missing combo perturbations ({len(missing_combo)}):")
            for pert in sorted(missing_combo)[:10]:  # Show first 10
                print_sys(f"  - {pert}")
            if len(missing_combo) > 10:
                print_sys(f"  ... and {len(missing_combo) - 10} more")
        
        print_sys("================================")

    def get_dataloader(self, batch_size, test_batch_size=None):
        """
        Override get_dataloader to:
        1. Filter set2conditions for available data
        2. Ensure all required splits exist (train/val/test)
        """
        # Filter set2conditions to only include perturbations with available data
        self.filter_set2conditions_for_available_data()
        
        # Ensure all required splits exist (GEARS expects train, val, test)
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if split not in self.set2conditions:
                if split == 'val' and 'train' in self.set2conditions:
                    # Create val from train (10% for val, 90% for train as user specified)
                    train_perts = list(self.set2conditions['train'])
                    n_val = max(1, int(len(train_perts) * 0.1))  # 10% for validation
                    random.seed(self.seed if self.seed else 42)
                    random.shuffle(train_perts)
                    self.set2conditions['val'] = train_perts[:n_val]
                    self.set2conditions['train'] = train_perts[n_val:]
                    print_sys(f"Created val split from train: {n_val} perturbations (10%)")
                    print_sys(f"Remaining train: {len(self.set2conditions['train'])} perturbations (90%)")
                else:
                    # Create empty split for test (we'll evaluate on separate test_adata)
                    self.set2conditions[split] = []
                    print_sys(f"Created empty {split} split")
        
        # Call parent method
        return super().get_dataloader(batch_size, test_batch_size)