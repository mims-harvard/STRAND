"""
Gene utilities: Gene annotation and perturbation tag curation.

This module contains:
- curate_perturb_tag: Curate perturbation tags for AnnData objects
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import scanpy as sc


def curate_perturb_tag(adata, data_name):
    '''
    Adds a curated perturbation label to adata.obs['curated_perturbation']
    
    Each label is:
        - 'control-<dataset_name>' for negative controls
        - '<gene>' or 'gene1+gene2+...' for gene perturbations
    Also returns:
        - perturb_dict: dict mapping curated tags to list of raw entries
    '''

    perturb_dict = defaultdict(list)
    perturb_to_label = {}
    adata = adata[adata.obs['perturbation'].notna()].copy()
    perturb_type = set(adata.obs['perturbation'].values)
    if 'gemgroup' not in adata.obs.columns:
        adata.obs['gemgroup'] = '0'
    if data_name == 'AdamsonWeissman2016_GSM2406675_10X001':
        for entry in perturb_type:
            if pd.isna(entry):
                continue
            if '62(mod)' in entry:
                label = f'control-{data_name}-batch_0'
            else:
                if '_' in entry:
                    gene_name = entry.split('_')[0]
                    label = f'{gene_name}-{data_name}-batch_0'
                else:
                    print(f"Error parsing entry: {entry}")
                    label = f'unknown-{data_name}-batch_0'
            entry += f'+batch_0'
            perturb_dict[label].append(entry)
            perturb_to_label[entry] = label
        adata.obs['perturbation-batch'] = adata.obs['perturbation'].astype(str) + '+batch_0'

    elif data_name == 'NormanWeissman2019_filtered':
        batch_ids = adata.obs['gemgroup'].unique()
        # batch_ids = [0]
        for bathch_id in batch_ids:
            for entry in perturb_type:
            
                entry = entry.replace('_', '+')
                label = f'{entry}-{data_name}-batch_{bathch_id}'
                entry += f'+batch_{bathch_id}'
                # label = label.replace('_', '+')
                perturb_dict[label].append(entry)
                perturb_to_label[entry] = label
        adata.obs['perturbation-batch'] = adata.obs['perturbation'].astype(str) + '+batch_' + adata.obs['gemgroup'].astype(str)

    elif data_name == 'DixitRegev2016':
        for entry in perturb_type:
            components = entry.split('_')
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
                label = f'control-{data_name}-batch_0'
            else:
                label = '+'.join(sorted(genes)) + f'-{data_name}-batch_0'
            entry += f'+batch_0'
            perturb_dict[label].append(entry)
            perturb_to_label[entry] = label
        
        adata.obs['perturbation-batch'] = adata.obs['perturbation'].astype(str) + '+batch_0'

    else:
        raise ValueError(f"Dataset: {data_name} QC is not implemented")

    # Add curated label to AnnData
    adata.obs['curated_perturbation'] = adata.obs['perturbation-batch'].map(perturb_to_label)

    return perturb_dict, adata
