"""
Data preprocessing utilities: gene name conversion, normalization, and transformations.

This module contains:
- convert_gene_name: Convert gene names to official symbols
- convert_to_ensembl: Convert gene names to Ensembl IDs
- gaussian_copula_transform: Transform data to approximate Gaussian
- debug_conversion_reduction: Debug gene name conversion
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from typing import List, Union, Optional
import os

# Import config for default paths
try:
    from ..config.defaults import HGNC_COMPLETE_SET_PATH
except ImportError:
    # Fallback when the config package cannot be imported
    HGNC_COMPLETE_SET_PATH = os.environ.get("HGNC_COMPLETE_SET_PATH", "")


def debug_conversion_reduction(adata, cq):
    """
    Debug function to check gene name conversion and overlap reduction.
    
    Args:
        adata: AnnData object with gene names in var.index
        cq: CellQuery object with gene_order attribute
    """
    # Get original genes
    adata_genes = adata.var.index.tolist()
    cq_genes = cq.gene_order
    
    # Check original overlap
    orig_overlap = set(adata_genes) & set(cq_genes)
    print(f"Original overlap: {len(orig_overlap)} genes")
    print(f"Examples: {list(orig_overlap)[:10]}")
    
    # Convert
    adata_converted = convert_gene_name(adata_genes, suffix_token=0)
    cq_converted = convert_gene_name(cq_genes, suffix_token=0)
    
    # Check converted overlap
    conv_overlap = set(adata_converted) & set(cq_converted)
    print(f"\nConverted overlap: {len(conv_overlap)} genes")
    print(f"Examples: {list(conv_overlap)[:10]}")
    
    # Find genes that matched before but not after
    lost_matches = orig_overlap - conv_overlap
    print(f"\nLost {len(lost_matches)} matches after conversion!")
    
    # Analyze lost matches
    for gene in list(lost_matches)[:10]:
        idx_adata = adata_genes.index(gene)
        idx_cq = cq_genes.index(gene)
        print(f"\nGene: {gene}")
        print(f"  adata: {gene} -> {adata_converted[idx_adata]}")
        print(f"  cq: {gene} -> {cq_converted[idx_cq]}")


def gaussian_copula_transform(X: np.ndarray) -> np.ndarray:
    """
    Transform non-Gaussian data to approximately standard Gaussian using rank transform.
    
    Args:
        X: (N, D) non-Gaussian data
        
    Returns:
        Z: (N, D) approximately standard Gaussian
    """
    N, D = X.shape
    Z = np.zeros_like(X)
    for d in range(D):
        ranks = rankdata(X[:, d], method='average')  # rank transform
        u = (ranks - 0.5) / N                        # uniform in (0,1)
        Z[:, d] = norm.ppf(u)                        # Gaussianize
    return Z


def convert_gene_name(gene_names: Union[List[str], np.ndarray], suffix_token: int = 2) -> np.ndarray:
    """
    Convert gene names to official HGNC symbols.
    
    Handles:
    - Ensembl IDs (ENSG...)
    - Alias symbols
    - Previous symbols
    - Official symbols (unchanged)
    
    Args:
        gene_names: List or array of gene names to convert
        suffix_token: Token position for suffix handling (0=no suffix, 2=with suffix)
        
    Returns:
        Array of converted gene names (official symbols)
    """
    # Load HGNC gene annotation
    hgnc = pd.read_csv(
        HGNC_COMPLETE_SET_PATH,
        sep='\t',
        dtype=str
    )
    
    # Build multiple mapping dictionaries
    alias_map = {}
    ensembl_map = {}
    officials = set(hgnc['symbol']
                   .str.strip()
                   .str.upper()
               )
    for idx, row in hgnc.iterrows():
        official = (row['symbol'].strip().upper())
        
        # Map Ensembl ID to official symbol
        if 'ensembl_gene_id' in row and pd.notna(row['ensembl_gene_id']):
            ensembl_map[row['ensembl_gene_id'].strip()] = official
        alias_map[official] = official
        # Map aliases to official symbol
        for col in ['alias_symbol', 'prev_symbol']:
            if col in row and pd.notna(row[col]):
                for alias in row[col].split('|'):
                    alias_clean = alias.strip().upper()
                    if not alias_clean or alias_clean in officials:
                        continue
                    if alias_clean:
                        alias_map[alias_clean] = official
        
    
    print(f"Built alias map with {len(alias_map)} entries")
    print(f"Built Ensembl map with {len(ensembl_map)} entries")
    
    # Convert gene names
    converted_genes = []
    ensembl_converted = 0
    alias_converted = 0
    unchanged = 0
    
    for full_gene_name in gene_names:
        full_gene_name = str(full_gene_name).strip()
        
        if suffix_token == 0:
            # Check if it's an Ensembl ID
            if full_gene_name.startswith('ENSG'):
                if full_gene_name in ensembl_map:
                    converted_genes.append(ensembl_map[full_gene_name])
                    ensembl_converted += 1
                else:
                    converted_genes.append(full_gene_name)  # Keep original if not found
                    unchanged += 1
            else:
                # Handle regular gene symbols
                gene_upper = full_gene_name.upper()
                if gene_upper in alias_map:
                    converted_genes.append(alias_map[gene_upper])
                    if alias_map[gene_upper] != gene_upper:
                        alias_converted += 1
                else:
                    converted_genes.append(gene_upper)
                    unchanged += 1
        else:
            # Handle suffix case (similar logic)...
            # For now, just convert the base name
            gene_upper = full_gene_name.upper()
            if gene_upper in alias_map:
                converted_genes.append(alias_map[gene_upper])
                if alias_map[gene_upper] != gene_upper:
                    alias_converted += 1
            else:
                converted_genes.append(gene_upper)
                unchanged += 1
    
    print(f"Converted: {ensembl_converted} Ensembl IDs, {alias_converted} aliases, {unchanged} unchanged")
    return np.array(converted_genes)


def convert_to_ensembl(gene_names: Union[List[str], np.ndarray], suffix_token: int = 0, verbose: bool = False) -> np.ndarray:
    """
    Convert gene symbols to ENSEMBL IDs, preserving unique IDs for aliases.
    
    Args:
        gene_names: List or array of gene names to convert
        suffix_token: Token position for suffix handling (0=no suffix, 2=with suffix)
        verbose: Whether to print conversion statistics
        
    Returns:
        Array of Ensembl IDs (or original names if conversion fails)
    """
    # Load HGNC gene annotation
    hgnc = pd.read_csv(
        HGNC_COMPLETE_SET_PATH,
        sep='\t',
        dtype=str
    )
    
    # Build mapping dictionaries
    symbol_to_ensembl = {}  # Maps symbols to their own ENSEMBL IDs
    ensembl_map = {}  # Maps ensembl to itself
    
    # First pass: Map each symbol to its own ENSEMBL ID
    for idx, row in hgnc.iterrows():
        if pd.notna(row['ensembl_gene_id']) and pd.notna(row['symbol']):
            ensembl_id = row['ensembl_gene_id'].strip()
            official = row['symbol'].strip().upper()
            
            # Map ENSEMBL ID to itself
            ensembl_map[ensembl_id] = ensembl_id
            
            # Map official symbol to its ENSEMBL ID
            symbol_to_ensembl[official] = ensembl_id
    
    # Second pass: Map aliases to their own ENSEMBL ID if they have one
    for idx, row in hgnc.iterrows():
        if pd.notna(row['ensembl_gene_id']):
            ensembl_id = row['ensembl_gene_id'].strip()
            
            # Map aliases - but only if they don't already have their own mapping
            for col in ['alias_symbol', 'prev_symbol']:
                if col in row and pd.notna(row[col]):
                    for alias in row[col].split('|'):
                        alias_clean = alias.strip().upper()
                        if alias_clean and alias_clean not in symbol_to_ensembl:
                            # This alias doesn't have its own ENSEMBL ID, 
                            # so map it to the official symbol's ID
                            symbol_to_ensembl[alias_clean] = ensembl_id
    
    # Convert gene names
    converted_genes = []
    ensembl_converted = 0
    symbol_converted = 0
    unchanged = 0
    unchanged_list = []
    
    for full_gene_name in gene_names:
        full_gene_name = str(full_gene_name).strip()
        
        if suffix_token == 0:
            # Check if it's an Ensembl ID
            if full_gene_name.startswith('ENSG'):
                if full_gene_name in ensembl_map:
                    converted_genes.append(ensembl_map[full_gene_name])
                    ensembl_converted += 1
                else:
                    converted_genes.append(full_gene_name)
                    unchanged += 1
                    unchanged_list.append(full_gene_name)
            else:
                # Handle regular gene symbols
                gene_upper = full_gene_name.upper()
                if gene_upper in symbol_to_ensembl:
                    converted_genes.append(symbol_to_ensembl[gene_upper])
                    symbol_converted += 1
                else:
                    converted_genes.append(full_gene_name)
                    unchanged += 1
                    unchanged_list.append(full_gene_name)
        else:
            # Handle suffix case - extract base gene name
            parts = full_gene_name.split('.')
            if len(parts) >= suffix_token:
                base_gene = '.'.join(parts[:suffix_token])
            else:
                base_gene = full_gene_name
            
            # Check if base is an Ensembl ID
            if base_gene.startswith('ENSG'):
                if base_gene in ensembl_map:
                    if len(parts) > suffix_token:
                        suffix = '.'.join(parts[suffix_token:])
                        converted_genes.append(f"{ensembl_map[base_gene]}.{suffix}")
                    else:
                        converted_genes.append(ensembl_map[base_gene])
                    ensembl_converted += 1
                else:
                    converted_genes.append(full_gene_name)
                    unchanged += 1
                    unchanged_list.append(full_gene_name)
            else:
                gene_upper = base_gene.upper()
                if gene_upper in symbol_to_ensembl:
                    if len(parts) > suffix_token:
                        suffix = '.'.join(parts[suffix_token:])
                        converted_genes.append(f"{symbol_to_ensembl[gene_upper]}.{suffix}")
                    else:
                        converted_genes.append(symbol_to_ensembl[gene_upper])
                    symbol_converted += 1
                else:
                    converted_genes.append(full_gene_name)
                    unchanged += 1
                    unchanged_list.append(full_gene_name)
    
    print(f"Converted: {ensembl_converted} Ensembl IDs, {symbol_converted} symbols, {unchanged} unchanged")
    if verbose:
        unchanged_list = set(unchanged_list)
        print(f"Unchanged list: {list(unchanged_list)[:20]}")
    return np.array(converted_genes)

