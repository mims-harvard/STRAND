"""
DNA utilities: Sequence manipulation, annotation loading, and perturbation generation.

This module contains:
- DNA annotation and reference genome loading
- Sequence manipulation (reverse complement, PAM search)
- Perturbation generation (targeted and batch)
- Gene TSS sequence generation
"""

import numpy as np
import pandas as pd
import random
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import os
import json
import warnings
import gzip
from typing import List, Dict, Optional, Tuple, Union

# Import config for default paths
try:
    from ..config.defaults import (
        GENE_ANNOTATION_PATH, DNA_REFERENCE_PATH, CCRE_BED_PATH
    )
except ImportError:
    # Fallback when the config package cannot be imported
    GENE_ANNOTATION_PATH = os.environ.get("GENE_ANNOTATION_PATH", "")
    DNA_REFERENCE_PATH = os.environ.get("DNA_REFERENCE_PATH", "")
    CCRE_BED_PATH = os.environ.get("CCRE_BED_PATH", "")


def get_DNA_annot(annotPath: str | None = None):
    '''
    Get DNA annotation file
    '''
    if annotPath is None:
        annotPath = GENE_ANNOTATION_PATH
    if not annotPath:
        raise ValueError(
            "Gene annotation path not set. Provide `annotPath=` or set env var GENE_ANNOTATION_PATH."
        )
    
    all_annots_transcript = pd.read_csv(annotPath,sep='\t',comment='#',header=None)
    if all_annots_transcript.shape[1] == 9:
        # Stock GENCODE GTF: add the lookup column (gene symbol) as a 10th column.
        all_annots_transcript[9] = all_annots_transcript[8].str.extract(r'gene_name "([^"]+)"', expand=False)
    all_annots_transcript = all_annots_transcript.rename(columns={0: 'chrom', 2: 'feature', 3: 'start', 4: 'end', 6: 'strand', 8: 'attr',9:'gene_id'})
    all_annots_transcript = dict(tuple(all_annots_transcript.groupby('gene_id')))
    return all_annots_transcript

def get_dna_record(DNAPath: str | None = None):
    '''
    Get all chromosome records from the reference genome.
    '''
    if DNAPath is None:
        DNAPath = DNA_REFERENCE_PATH
    if not DNAPath:
        raise ValueError(
            "DNA reference path not set. Provide `DNAPath=` or set env var DNA_REFERENCE_PATH."
        )
    records = {}
    for chrom in range(1, 23):
        record = next(SeqIO.parse(f'{DNAPath}/chr{chrom}.fasta', 'fasta'))
        records[f'chr{chrom}'] = (record)
    for chrom in ['X', 'Y', 'M']:
        record = next(SeqIO.parse(f'{DNAPath}/chr{chrom}.fasta', 'fasta'))
        records[f'chr{chrom}'] = (record)
    return records

def load_ccres_bed(bed_path, merge_gap=100):
    """
    Load cCRE intervals from a BED file (supports gzipped files).
    
    Args:
        bed_path: Path to BED file (can be .gz)
        merge_gap: Merge intervals within this gap (bp). Set to 0 to disable merging.
    
    Returns:
        dict: {chrom: [(start, end), ...]} where coordinates are 1-based inclusive
              Returns None if bed_path is None or empty
    """
    if bed_path is None or bed_path == '':
        return None
    
    if not os.path.exists(bed_path):
        warnings.warn(f"cCRE BED file not found: {bed_path}, proceeding without cCREs")
        return None
    
    try:
        # Use pandas for efficient reading (handles gzip automatically)
        # Read only first 3 columns (chrom, start, end)
        df = pd.read_csv(
            bed_path,
            sep='\t',
            header=None,
            comment='#',
            usecols=[0, 1, 2],
            names=['chrom', 'start', 'end'],
            dtype={'chrom': str, 'start': int, 'end': int},
            engine='c',  # Use C engine for speed
            na_filter=False,  # Don't convert strings to NaN
        )
        
        if len(df) == 0:
            return None
        
        # Convert 0-based start (exclusive end) to 1-based inclusive
        df['start'] = df['start'] + 1
        
        # Group by chromosome and convert to list of tuples
        ccre_dict = {}
        for chrom, group in df.groupby('chrom'):
            intervals = [(row['start'], row['end']) for _, row in group.iterrows()]
            intervals.sort(key=lambda x: x[0])  # Sort by start
            
            # Merge overlapping/adjacent intervals if requested
            if merge_gap > 0 and len(intervals) > 0:
                merged = []
                current_start, current_end = intervals[0]
                
                for start, end in intervals[1:]:
                    if start <= current_end + merge_gap:
                        # Merge: extend current_end
                        current_end = max(current_end, end)
                    else:
                        # Gap too large: save current and start new
                        merged.append((current_start, current_end))
                        current_start, current_end = start, end
                
                merged.append((current_start, current_end))
                intervals = merged
            
            ccre_dict[chrom] = intervals
        
        return ccre_dict
    
    except Exception as e:
        warnings.warn(f"Error loading cCRE BED file {bed_path}: {e}, proceeding without cCREs")
        return None

def get_exons(annot_exon, gene_list):
    '''
    gene_list: a list of gene names
    annot: annotation file containing exon information
    '''
    filtered_annot_exon = annot_exon[annot_exon['genes_name'].isin(gene_list)]
    exon_lists = list(zip(filtered_annot_exon.iloc[:,3],filtered_annot_exon.iloc[:,4]))
    return exon_lists


def random_dna_sequence(length):
    """Generate a random DNA sequence of given length."""
    return ''.join(random.choices("ACGT", k=length))

def perturb_sequence(long_seq, gstart, gend, max_indel_length=5, mutation_prob=1., mutation_type=None, mutation_types=["insertion", "deletion", "substitution", "duplication", "scrambling"], return_mut_type=False):
    """
    Apply a random perturbation (insertion, deletion, substitution, duplication, scrambling) 
    to a given DNA sequence.
    
    Args:
        seq (str): Input DNA sequence.
        
        start (int): Start position of the gene region.
        end (int): End position of the gene region.
        max_indel_length (int): Maximum length of inserted or deleted sequences.
        mutation_prob (float): Probability of applying a perturbation (to control how often mutations happen).

    Returns:
        str: Perturbed DNA sequence.
    """
    # seq = long_seq[gstart:gend] ## target gene region
    
    if random.random() > mutation_prob:  # Sometimes, return the original sequence unchanged
        return long_seq
    if mutation_type is None:
        mutation_type = random.choice(mutation_types)
    assert mutation_type in mutation_types, f"Invalid mutation {mutation_type}"
    seq_length = len(long_seq)

    # Choose a random site for perturbation
    pos = random.randint(gstart, gend - 1) if seq_length > 1 else 0
    mut_length = random.randint(1, max_indel_length)
    print(f'introduced {mutation_type}, with mutate lenth: {mut_length}')
    if mutation_type == "insertion":
        # Insert a random sequence at a random position
        insert_seq = random_dna_sequence(mut_length)
        perturbed_seq = long_seq[:pos] + insert_seq + long_seq[pos:]

    elif mutation_type == "deletion":
        # Delete a random-length subsequence
        end_pos = min(pos + mut_length, seq_length)
        perturbed_seq = long_seq[:pos] + long_seq[end_pos:]

    elif mutation_type == "substitution":
        # Replace a random-length subsequence with a new random sequence
        end_pos = min(pos + mut_length, seq_length)
        substitute_seq = random_dna_sequence(end_pos - pos)
        perturbed_seq = long_seq[:pos] + substitute_seq + long_seq[end_pos:]

    elif mutation_type == "duplication":
        # Duplicate a random-length subsequence
        end_pos = min(pos + mut_length, seq_length)
        duplicated_seq = long_seq[pos:end_pos]  # Copy the segment
        perturbed_seq = long_seq[:pos] + duplicated_seq + long_seq[pos:]

    elif mutation_type == "scrambling":
        # Shuffle a small segment within the sequence
        end_pos = min(pos + mut_length, seq_length)
        scrambled_seq = list(long_seq[pos:end_pos])
        random.shuffle(scrambled_seq)
        scrambled_seq = "".join(scrambled_seq)
        perturbed_seq = long_seq[:pos] + scrambled_seq + long_seq[end_pos:]

    # perturbed_seq = long_seq[:gstart] + perturbed_seq + long_seq[gend:] ## return the original long sequence
    if return_mut_type==True:
        return perturbed_seq,mutation_type,mut_length
    else:
        return perturbed_seq



def perturb_sequence_exon(long_seq, abs_start, exon_intervals, mutation_prob=1.,perturbed_region=False,get_perturb_len=False,mutation_types=["insertion", "deletion", "substitution", "duplication", "scrambling"]):
    """
    Apply a random perturbation (insertion, deletion, substitution, duplication, scrambling) 
    to a given DNA sequence only in target exon intervals.
    
    Args:
        seq (str): Input DNA sequence.
        abs_start (int): input sequence physical starting point -- used to get relative interval
        max_indel_length (int): Maximum length of inserted or deleted sequences.
        mutation_prob (float): Probability of applying a perturbation (to control how often mutations happen).

    Returns:
        str: Perturbed DNA sequence.
    """
    
    if random.random() > mutation_prob:
        ## no perturbation under certain chance
        return long_seq
    
    start, end = random.choice(exon_intervals) ## absolute start and end of the exon interval
    # print(f'absolute exon interval is [{start}, {end}]')
    start, end = start-abs_start, end-abs_start + 1 ## relative start and end of the exon interval; no need to minus  1 since both star and abs_start are 1 based. 
    # print(f'relative exon interval is [{start}, {end}]')
    # Defensive check in case intervals are invalid

    # seq = long_seq[start:end]
    
    perturbed_seq, mutation_type, mut_length = perturb_sequence(long_seq,start,end, return_mut_type=True,mutation_types=mutation_types)
    if get_perturb_len:
        perturb_len = len(perturbed_seq) - len(long_seq)
    
    # perturbed_seq = long_seq[:start] + perturbed_seq + long_seq[end:] ## return the original long sequence -- deprecated after revising perturb_sequence function
    # print(f'perturbed seq length: {len(perturbed_seq)}')
    if perturbed_region:
        if get_perturb_len:
            return perturbed_seq, mutation_type, mut_length, start, end, perturb_len
        else:
            return perturbed_seq, mutation_type, mut_length, start, end
    else:
        return perturbed_seq, mutation_type
    
    
def searchPAM(long_seq, gRNA_end):
    '''
    Search for the PAM region: 3 bp upstream of the NGG sequence 
    Input:
        long_seq: The original sequence
        gRNA_end: the guide-RNA end position
    Return:
        edit_region: an absolute index that 3 bp upstream of the PAM (NGG) region
    '''
    
    pam_pattern = re.compile(r'[ATCG]GG')
    match = pam_pattern.search(long_seq, gRNA_end)
    if match:
        pam_start = match.start()  # start index of the PAM in long_seq
        edit_region = pam_start - 3  # cut site: 3 bp upstream of the PAM
        return edit_region
    else:
        return None



def gen_perturb_DNA(gene_name, targetType, savePath, 
                    NumPerturb=50,
                    DNAPath=None,
                    targetPath=None,
                    FIX_WIN=524288):
    if DNAPath is None:
        DNAPath = DNA_REFERENCE_PATH
    if not DNAPath:
        raise ValueError("DNAPath not set. Provide DNAPath or set env DNA_REFERENCE_PATH.")
    if targetPath is None:
        targetPath = os.environ.get("PROTEIN_CODING_GENES_PATH", "")
    if not targetPath:
        raise ValueError(
            "targetPath not set. Provide targetPath or set env PROTEIN_CODING_GENES_PATH."
        )
    ## load the exon annotation file
    exon_gtf = os.environ.get("PROTEIN_EXON_GTF_PATH", "")
    if not exon_gtf:
        raise ValueError(
            "Protein exon annotation not set. Set env PROTEIN_EXON_GTF_PATH to a protein-coding exon GTF."
        )
    all_annots_protein_exon = pd.read_csv(exon_gtf, sep='\t')
    ## DNA sequence path (contain fasta file from different chromosomes)
    
    
    annot_df=pd.read_csv(targetPath, sep='\t')
    gene_annot_entry = annot_df[annot_df.geneinfo.isin([gene_name])] ## get the entry of target gene to be queried
    if targetType == 'gene':
        chrom = gene_annot_entry.chr.values[0]
        pstart = int(gene_annot_entry.pstart.values[0])
        pend = int(gene_annot_entry.pend.values[0])
        geneinfo = gene_annot_entry.geneinfo.values[0]
        assert geneinfo == gene_name,f'gene names donnot match, {geneinfo} vs {gene_name}'
    else:
        raise Exception(f"targetType: {targetType} is currently not implemented")
    
    record = next(SeqIO.parse(f'{DNAPath}/chr{chrom}.fasta', 'fasta'))
    DNA_data=[]
    default_proxy_len=10000
    if pend - pstart + 1 < FIX_WIN:
        gene_len = pend - pstart + 1 ## if start is 1 and end is 2, then the sequence length should be 2 rather than 1
        start_proxy_len = (FIX_WIN - gene_len) // 2
        end_proxy_len = FIX_WIN - gene_len - start_proxy_len
        proxy_start = max(1, pstart - start_proxy_len)
        proxy_end   = min(len(record.seq), pend + end_proxy_len)
        if pstart - start_proxy_len<=0:
            proxy_end += start_proxy_len-pstart + 1 ## if hit the start, then shifting the window to the right
        elif pend + end_proxy_len > len(record.seq):
            proxy_start -= pend + end_proxy_len-len(record.seq) ## If hit the end, then shifting the window to the left
        
        borzoi_sequences = str(record.seq[(proxy_start-1):proxy_end]).upper() 
        assert len(borzoi_sequences) == FIX_WIN, f'actual length is {len(borzoi_sequences)}; expected length is {FIX_WIN}'
    
    else: # pend - pstart + 1 > FIX_WIN
        # The gene segment is longer than FIX_WIN. Select a window.
        print(f"Gene length {pend - pstart + 1} >= FIX_WIN {FIX_WIN}. Taking first {FIX_WIN} bases.") # Debug print
        proxy_start = max(1, pstart - default_proxy_len) ## plus 10k upstream
        proxy_end = min(len(record.seq), pend + default_proxy_len) ## Ensure 1-based inclusive end
        
        borzoi_sequences = str(record.seq[(proxy_start-1):proxy_end]).upper() 
    
    abs_start = proxy_start
    # Some logic for generating 'sequences_perturb'
    
    # Convert to BioPython Seq objects
    borzoi_sequences_bio = Seq(borzoi_sequences)
    
    
    # Optionally store as SeqRecords
    # (useful if you want to keep track of IDs or descriptions)
    borzoi_seq_rec = SeqRecord(
        borzoi_sequences_bio, 
        id=f"{geneinfo}.0", 
    )
    
    
    annotations_dict = {
        "gene_rel_region": [pstart-proxy_start,pend-proxy_start],
        "gene_abs_region": [pstart,pend],
        "gene_name": geneinfo
    }
    
    borzoi_seq_rec.description = json.dumps(annotations_dict) 
    DNA_data.append(borzoi_seq_rec)
    
    
    if targetType == 'gene':
        intervals = get_exons(all_annots_protein_exon, [geneinfo]) ## get the exon regions of the target gene
    elif targetType == 'sequence':
        edit_location = searchPAM(str(record.seq[:]),gRNA_end)
        intervals = [(edit_location, edit_location)] ## edit location is very specific
        
    for perturb_id in range(1,NumPerturb+1):
        
        
        sequences_perturb, mutation_type, mut_length, perturb_start, perturb_end, perturb_len = perturb_sequence_exon(borzoi_sequences,abs_start=abs_start, exon_intervals=intervals,perturbed_region=True,get_perturb_len=True,mutation_types=["insertion", "deletion"])
        # perturb_len record the sequence length change
        
        # Slice or pad the perturbed sequence as you showed
        
        borzoi_sequences_perturb = sequences_perturb[:FIX_WIN]  # in case the mutated seq is longer
        
        
        if len(borzoi_sequences_perturb) < FIX_WIN:
            borzoi_sequences_perturb += 'N'*(FIX_WIN - len(borzoi_sequences_perturb))

        assert len(borzoi_sequences_perturb) == FIX_WIN, f'actual length is {len(borzoi_sequences_perturb)}; expected length is {FIX_WIN}'
        borzoi_sequences_perturb_bio = Seq(borzoi_sequences_perturb)
        borzoi_seq_perturb_rec = SeqRecord(
            borzoi_sequences_perturb_bio, 
            id=f"{geneinfo}.{perturb_id}", 
        )
        
        
        annotations_dict = {
            "gene_rel_region": [pstart-proxy_start,pend-proxy_start+perturb_len],
            "gene_abs_region": [pstart,pend+perturb_len],
            "perturb_rel_region": [perturb_start,perturb_end],
            "gene_name": geneinfo,
            "mutation_type": mutation_type,
            "mutation_len": mut_length
        }
        
        borzoi_seq_perturb_rec.description = json.dumps(annotations_dict) 
        # DNA_data.append(borzoi_seq_perturb_rec)

        DNA_data.append(borzoi_seq_perturb_rec)
    
    # DNA_data.append([sequences,geneinfo, 0, int(pstart), int(pend), int(proxy)]) ## 0 original sequence
    
    os.makedirs(savePath, exist_ok=True)
    SeqIO.write(DNA_data, f"{savePath}/{geneinfo}_sequence.fasta", "fasta")
    
    

def gen_gene_TSS_sequence(gene_ID,
                          DNAPath=None,
                          targetPath=None,
                          FIX_WIN=524288,
                          dna_record=None,
                          all_annots_transcript=None,
                          jitter_window=0,
                          ccre_intervals_dict=None):
    '''
    Use the first TSS of the gene as the target region to expand
    Let's now assume the all_annots_transcript is a dict to store the gene name and the corresponding annotation entry -- speed up the process
    jitter_window: shifting range of the window
    '''
    tag_categories = ["MANE_Select", "Ensembl_canonical", "appris_principal_1", "GENCODE_Primary", "basic"]
    tag_counts = {category: 0 for category in tag_categories}
    
    if DNAPath is None:
        DNAPath = DNA_REFERENCE_PATH
    if targetPath is None:
        targetPath = GENE_ANNOTATION_PATH

    if all_annots_transcript is None:
        ## need to change to dict
        all_annots_transcript = pd.read_csv(targetPath, sep='\t', comment='#', header=None)
        all_annots_transcript = all_annots_transcript.rename(columns={0: 'chrom', 2: 'feature', 3: 'start', 4: 'end', 6: 'strand', 8: 'attr',9:'gene_name',10:'gene_id'})
        all_annots_transcript = dict(tuple(all_annots_transcript.groupby('gene_id')))
    
    # Get the entry of the target gene to be queried
    # gene_annot_entry = all_annots_transcript[all_annots_transcript.iloc[:, -1].str.upper().isin([gene_name])] ## newly changed to upper case to match the gene name in the annotation file
    gene_annot_entry = all_annots_transcript.get(gene_ID)
    
    # print(f'Gene_name: {gene_name}')
    # print(f'gene_annot_entry: {gene_annot_entry}')
    if gene_annot_entry is None:
        warnings.warn(f'Gene {gene_ID} not found in the annotation file.')
        return None
        # raise ValueError(f'Gene {gene_ID} not found in the annotation file.')
    chrom = gene_annot_entry.iloc[0, 0]
    # print(f'chromosome: {chrom}')
    direction = gene_annot_entry.iloc[0, 6]
    
    # Identify the canonical entry
    # canonical_entry = gene_annot_entry[gene_annot_entry.iloc[:, 8].str.contains('canonical', na=False)]
    canonical_entry = None
    for tag in tag_categories:
        candidates = gene_annot_entry[gene_annot_entry['attr'].str.contains(tag, na=False)]
        if not candidates.empty:
            canonical_entry = candidates.iloc[[0]]
            break
    if canonical_entry is None:
        canonical_entry = gene_annot_entry.iloc[[0]]

    if canonical_entry.shape[0] > 1:
        canonical_entry = canonical_entry.iloc[0]
    elif canonical_entry.shape[0] == 0:
        raise ValueError(f'No canonical entry found for {gene_ID}')
        
    combined_entries = canonical_entry.copy()
    
    Entryinfo = combined_entries.iloc[:,8]
    # print(f'Entryinfo: {Entryinfo}')
    # Determine TSS based on the combined entries
    if direction == '+':
        TSS = combined_entries.iloc[:, 3].values
        TES = combined_entries.iloc[:, 4].values
    else:
        TSS = combined_entries.iloc[:, 4].values
        TES = combined_entries.iloc[:, 3].values
        
    
    
    TSS_TES_list = combined_entries.iloc[:, [3, 4]].values.tolist()
    
    if dna_record is not None:
        record = dna_record[chrom]
    else:
        record = next(SeqIO.parse(f'{DNAPath}/{chrom}.fasta', 'fasta'))
    
    results = []
    
    for tid, tss in enumerate(TSS):
        left_proxy_len = (FIX_WIN) // 2 
        right_proxy_len = FIX_WIN - left_proxy_len - 1 
        
        if tss - left_proxy_len <= 0:
            residual = left_proxy_len - tss + 1
            left_proxy_len = tss - 1
            right_proxy_len += residual
        elif tss + right_proxy_len > len(record.seq):
            residual = tss + right_proxy_len - len(record.seq)
            right_proxy_len = len(record.seq) - tss
            left_proxy_len += residual
        
        assert left_proxy_len + right_proxy_len + 1 == FIX_WIN, f'actual length is {left_proxy_len + right_proxy_len + 1}; expected length is {FIX_WIN}'
        
        min_shift = 1 - (tss - left_proxy_len)
        # maximum positive shift so that proxy_end ≤ len(record)
        max_shift = len(record.seq) - (tss + right_proxy_len)
        
        # intersect with your desired jitter_window
        shift_low  = max(-jitter_window, min_shift)
        shift_high = min( jitter_window, max_shift)
        
        if jitter_window > 0:
            shifting = random.randint(shift_low, shift_high)
        else:
            shifting = 0
        
        
        
        proxy_start = tss - left_proxy_len + shifting 
        proxy_end = tss + right_proxy_len + shifting
        borzoi_sequences = str(record.seq[(proxy_start-1):proxy_end]).upper()
        
        assert proxy_end - proxy_start + 1 == FIX_WIN, f'actual length is {proxy_end - proxy_start + 1}; expected length is {FIX_WIN}'
        
        TSS_rel = tss - proxy_start
        TES_rel = TES[tid] - proxy_start
        
        # Find overlapping cCREs for this window
        ccre_intervals_rel = []
        if ccre_intervals_dict is not None and chrom in ccre_intervals_dict:
            for ccre_start, ccre_end in ccre_intervals_dict[chrom]:
                # Check if cCRE overlaps with window [proxy_start, proxy_end]
                if ccre_end >= proxy_start and ccre_start <= proxy_end:
                    # Convert to window-relative coordinates
                    rel_ccre_start = max(0, ccre_start - proxy_start)
                    rel_ccre_end = min(FIX_WIN - 1, ccre_end - proxy_start)
                    if rel_ccre_end >= rel_ccre_start:  # Valid interval
                        ccre_intervals_rel.append((int(rel_ccre_start), int(rel_ccre_end)))
        
        results.append({
            'seq': borzoi_sequences,
            'chrom': chrom,
            'proxy_start': proxy_start,
            'proxy_end': proxy_end,
            'TSS': tss,
            'TSS_rel': TSS_rel,
            'TES_rel': TES_rel,
            'TSS_TES_list': TSS_TES_list,
            'Direction': direction,
            'Info': Entryinfo.iloc[tid],
            'ccre_intervals_rel': ccre_intervals_rel,  # List of (rel_start, rel_end) tuples
        })
    
    return results


def aggregate_perturbation(metadata):
    '''
    This function aims to set up a protocal that works for all perturbation types
    '''
    # seq = list(seq)
    # print('metadata seqinfo keys:', metadata['seqinfo']['Ensembl_canonical'].keys())
    # seq = list(metadata['seqinfo']['Ensembl_canonical']['seq'])
    seq = list(metadata['seq']) ## seq is a list of characters
    seq_orig = seq.copy()
    perturb_type = metadata.get('perturb_type')
    if perturb_type == 'CRISPR-i':
        TSS = metadata.get('TSS_rel')
        promotor_len = 200
        # print(f'promotor length: {promotor_len}')
        # if direction == '-':
        seq[TSS-promotor_len:TSS+promotor_len] = ['N'] * (2* promotor_len) ## mask the promotor region
        # elif direction == '+':
        #     seq[TSS-promotor_len+1:TSS+1] = ['N'] * promotor_len
    elif perturb_type == 'CRISPR-KO':
        TSS = metadata.get('TSS_rel')
        TES = metadata.get('TES_rel')
        direction = metadata.get('Direction')
        if direction == '+':
            seq[TSS:TES] = ['N'] * (TES-TSS) ## mask the region
        elif direction == '-':
            seq[TES:TSS] = ['N'] * (TSS-TES) ## mask the region
    else:
        raise ValueError(f"perturbation type: {perturb_type} not implemented")
        
    return ''.join(seq), ''.join(seq_orig)



def aggregate_mask_perturbation(metadata):
    '''
    Instead of masking the DNA as "N" as aggregate_perturbation, we create a mask channel to indicate the masked region
    '''
    # seq = list(seq)
    # print('metadata seqinfo keys:', metadata['seqinfo']['Ensembl_canonical'].keys())
    # seq = list(metadata['seqinfo']['Ensembl_canonical']['seq'])
    seq = list(metadata['seq']) ## seq is a list of characters
    mask_seq = [0] * len(seq)
    perturb_type = metadata.get('perturb_type')
    if perturb_type == 'CRISPR-i':
        TSS = metadata.get('TSS_rel')
        promotor_len = 200
        # print(f'promotor length: {promotor_len}')
        # if direction == '-':
        mask_seq[TSS-promotor_len:TSS+promotor_len] = [1] * (2* promotor_len) ## mask the promotor region
        # elif direction == '+':
        #     seq[TSS-promotor_len+1:TSS+1] = ['N'] * promotor_len
    elif perturb_type == 'CRISPR-KO':
        TSS = metadata.get('TSS_rel')
        TES = metadata.get('TES_rel')
        direction = metadata.get('Direction')
        if direction == '+':
            mask_seq[TSS:TES] = [1] * (TES-TSS) ## mask the region
        elif direction == '-':
            mask_seq[TES:TSS] = [1] * (TSS-TES) ## mask the region
    else:
        raise ValueError(f"perturbation type: {perturb_type} not implemented")
        
    return ''.join(seq), np.array(mask_seq).reshape(1, -1)

import re
from typing import List, Tuple

BASE_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
def rc(s: str) -> str:
    return s.translate(BASE_COMP)[::-1]

def _c(s: str) -> str:
    return s.translate(BASE_COMP)

def pam_to_forward(pam: str) -> str:
    """Return forward-strand regex for PAM on the *reverse* strand."""
    return rc(pam).replace('N', '[ACGT]')

def build_regex(gRNA: str, direction: str, pam_list: List[str]) -> re.Pattern:
    """
    Build a forward-strand regex for (protospacer, PAM) with correct order.
    """
    if direction == '+':                        # gRNA binds reverse strand
        protospacer = rc(gRNA)                 # put it on fwd
        pams = "|".join(pam.replace('N', '[ACGT]') for pam in pam_list)
        # print(f'pams : {pams}')
        return re.compile(fr"{protospacer}({pams})")   # gRNA_rc + PAM
    else:                                       # gRNA binds forward strand
        protospacer = (gRNA)
        # PAMs are on reverse strand Ã complement on fwd Ã precede gRNA
        pams_fwd = "|".join(pam_to_forward(p) for p in pam_list)
        return re.compile(fr"({pams_fwd}){protospacer}")   # PAM + gRNA

def find_site(seq_fwd: str,
              gRNA: str,
              direction: str,
              pam_list: List[str]) -> Tuple[int, int]:
    """
    Return (start,end) of the protospacer in forward-strand coords.
    Orientation denoted the direction of the gRNA 
    direction denoted the direction of the TSS direction of the DNA
    """
    orientations = ['+', '-'] 
    for orientation in orientations:
        pat = build_regex(gRNA, orientation, pam_list)
        m = pat.search(seq_fwd)
        if m:
            break
    else:
        return None
    
    # protospacer group is always the *last* group
    if direction == '+':
        return (m.end() - len(gRNA), m.end())  # protospacer at end
    else:
        return (m.start() + len(m.group(1)), m.start() + len(m.group(1)) + len(gRNA))  # skip PAM

def targeted_perturbation(seq_fwd: str, meta: dict,
                          pam_list = [""], mask_ratio = .9) -> str:
    if meta["perturb_type"] != "CRISPR-i":
        raise NotImplementedError

    TSS_abs = meta['TSS']
    
    TSS, direction, prom = meta["TSS_rel"], meta["Direction"], meta["promoter_len"]
    # print(f'TSS_rel: {TSS}')
    gRNA = meta["gRNA"].upper()

    # window: 50&+300 or user-specified prom_len
    win_start = max(0, TSS - prom)
    win_end   = min(len(seq_fwd), TSS + prom)
    

    # print(f'window_start: {win_start}')
    # print(f'window_end: {win_end}')

    # print(f'window start abs: {win_start + TSS_abs - TSS}')
    # print(f'window end abs: {win_end + TSS_abs - TSS}')

    window    = seq_fwd[win_start:win_end]
    # print(f'seqfwd window seq is {window}')

    interval = find_site(window, gRNA, direction, pam_list)

    if not interval:
        return None
    s, e = interval 
    abs_s, abs_e = win_start + s, win_start + e
    seq_list = list(seq_fwd)
    mask_indices = list(range(abs_s, abs_e))

    # Sample a subset of indices to mask
    num_to_mask = int(mask_ratio * len(mask_indices))
    subset_to_mask = random.sample(mask_indices, num_to_mask)

    # Apply masking
    for idx in subset_to_mask:
        seq_list[idx] = 'N'
    
    return ''.join(seq_list)


def target_perturbation_batch(metalist,gRNA_seq, gene_name):
    '''
    metalist: a list of dict
    '''
    for meta in metalist:
        meta['gene_name']=gene_name
        meta['promoter_len'] = 1000
        meta['gRNA'] = gRNA_seq[1:]
        # stats['gRNA'] = 'GGCAGCCTCGGGATACTCC'
        meta["perturb_type"] = 'CRISPR-i'
        seq = meta['seq']
        # print(f'seq: {seq}')
        # print(f'meta: {meta}')
        perturbed_seq = targeted_perturbation(seq, meta)
        if not perturbed_seq is None:
            return (perturbed_seq, seq, meta)
    print(f'Cannot find match for {gene_name} with gRNA: {gRNA_seq}; return as proxy')

    perturbed_seq, ref_seq = aggregate_perturbation(metalist[0])
    return perturbed_seq, ref_seq, metalist[0]



from collections import defaultdict
import pandas as pd
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
