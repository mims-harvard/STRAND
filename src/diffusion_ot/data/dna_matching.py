"""
DNA embedding matching: Functions to match gene labels and intervals to DNA embeddings.

This module contains:
- match_gene_dna_embeddings: Match gene labels to DNA embeddings
- match_gene_pert_mask_dna_embeddings: Match gene labels with perturbation masks to DNA embeddings
- match_interval_pert_mask_dna_embeddings: Match interval labels to DNA embeddings
- Helper functions for sequence generation and mask aggregation
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import random
from typing import List, Dict, Optional, Tuple, Union
import warnings
try:
    from baskerville import dna  # calico/baskerville
except ImportError:  # vendored copy of dna_1hot
    from . import _dna_onehot as dna

# Import config defaults for portable reference paths
try:
    from ..config.defaults import DNA_REFERENCE_PATH
except Exception:
    import os
    DNA_REFERENCE_PATH = os.environ.get("DNA_REFERENCE_PATH", "")

# Import utils functions from the refactored package (no legacy fallback)
from ..utils import (
    gen_gene_TSS_sequence, aggregate_perturbation, 
    targeted_perturbation, target_perturbation_batch,
    aggregate_mask_perturbation
)

# Import cache
from .cache import LRUCache
# Import gene name translation helper
from .datasets import _translate_gene_name


def _as_BLD(x: torch.Tensor) -> torch.Tensor:
    """
    Coerce a tensor to (B, L, D). Accepts shapes:
      (B, D) -> (B, 1, D)
      (B, 1, D) -> (B, 1, D)
      (B, L, D) -> (B, L, D)
      (L, D) -> (1, L, D)
      (D,) -> (1, 1, D)
    """
    if x.dim() == 3:
        # (B, L, D)
        return x
    if x.dim() == 2:
        # could be (B, D) or (L, D); assume batch if first is "B"? We detect B when it's 1 and caller context
        B_or_L, D = x.shape
        if B_or_L == 1:
            # ambiguous; treat as batch of size 1 → (1, L=1, D) if already (1,D)
            return x.unsqueeze(1)  # (1,1,D)
        else:
            # treat as (L, D) → (1, L, D)
            return x.unsqueeze(0)
    if x.dim() == 1:
        # (D,) -> (1,1,D)
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"_as_BLD: unsupported shape {tuple(x.shape)}")


def _to_LD_per_sample(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize any per-sample output to (L, D).
    Accepts shapes: (1,D), (1,1,D), (1,L,D), (L,D), (D,)
    """
    if x.dim() == 3:
        # (B, L, D) with B expected 1 here
        if x.size(0) != 1:
            raise ValueError(f"_to_LD_per_sample expects batch=1, got shape {tuple(x.shape)}")
        return x[0]  # (L, D)
    if x.dim() == 2:
        # (L, D) or (1, D)
        return x if x.size(0) > 1 else x  # (L,D) where L may be 1
    if x.dim() == 1:
        # (D,) -> (1, D)
        return x.unsqueeze(0)
    raise ValueError(f"_to_LD_per_sample: unsupported shape {tuple(x.shape)}")


def match_gene_dna_embeddings(gene_labels, 
                              model,
                              cache=None,
                              dna_record=None,
                              all_annots_transcript=None,
                              promoter_len=200,
                              perturb_type='CRISPR-i',
                              gRNAs=None,
                              return_seq_info=False,
                              jitter_window=0):
    
    '''
    If gRNAs is None, this function get a rough perturbation effect of the gene.
    Arguments:
        Input:
            gene_labels: list of gene names -- convert to ensembl gene ID
            model: Borzoi model
            gRNAs: dict: {gene_name: gRNA_sequence} for each gene, if None, guess the approximate perturbation effect
        Output:
            gene_embeddings: list of gene embeddings
    '''
    if 'ENSG' not in gene_labels[0]:
        raise ValueError(f'Gene labels must be in the format of ENSG00000000000, got {gene_labels[0]}')
    DNA_delta_embeddings = []
    DNA_ref_embeddings = []
    
    for gene_label in gene_labels:
        DNA_ref_embedding = []
        if gene_label == 'control':
            DNA_null = torch.zeros_like(next(iter(cache.cache.values()))[0])
            DNA_ref_embeddings.append(DNA_null)
            continue
        
        # Apply gene name translation (AC118549.1 -> ZZZ3)
        gene_label = _translate_gene_name(str(gene_label))
        
        # gene_names = gene_label.split('-')[0] ## gene name excluding the dataset name
        # Handle both 2-part (gene-dataset) and 3-part (gene-batch-dataset) label formats
        # Gene names can contain hyphens (e.g., "RBM14-RBM4"), so we need to check for batch pattern
        parts = gene_label.rsplit('-', 2)
        if len(parts) == 3 and parts[1].startswith('batch_'):
            # Format: gene-batch-dataset (e.g., "RBM14-RBM4-batch_0-rpe1")
            # parts[1] starts with "batch_", so it's a batch identifier
            gene_names, batch_name, dataset_name = parts
        elif len(parts) == 2:
            # Format: gene-dataset (e.g., "RBM14-RBM4-rpe1" or "SETD1A-rpe1")
            gene_names, dataset_name = parts
            batch_name = '0'  # Default batch name for datasets without batch info
        elif len(parts) == 3:
            # Has 3 parts but second part doesn't start with "batch_" - treat as 2-part
            # This handles cases like "RBM14-RBM4-rpe1" where gene name has a hyphen
            gene_names = '-'.join(parts[:2])  # Join first two parts as gene name
            dataset_name = parts[2]
            batch_name = '0'
        else:
            raise ValueError(f"Unexpected token format: {gene_label}. Expected format: 'gene-dataset' or 'gene-batch-dataset'")
        gene_names_list = gene_names.split('+')  # In case of multiple genes
        if len(gene_names_list) > 1:
            print(f'Multiple genes: {gene_names_list}')
            raise ValueError("Multiple genes are not supported yet")
        # Uppercase gene_names to match cache keys (cache stores genes in uppercase)
        gene_names = gene_names.upper()
        
        if cache is not None and gene_names in cache:
            # DNA_delta_embeddings.append(cache[gene_name][0])
            DNA_ref_embedding.append(cache[gene_names][0])
        else:
            seq_info_all = []
            for gene_name in gene_names_list:
                
                # print(f'Processing gene: {gene_name} in cache')
                # print('start gen_gene_TSS_sequence for gene:', gene_name)
                seq_info = gen_gene_TSS_sequence(gene_name,dna_record=dna_record, all_annots_transcript=all_annots_transcript,jitter_window=jitter_window)
                    
                direction = seq_info[0]['Direction']
                if gRNAs.get(gene_name) is not None:
                    # print(f'Use gRNAs {gRNAs[gene_name]} for gene {gene_name}')
                    DNA_seq_alt,DNA_seq_ref, coords = target_perturbation_batch(seq_info,gRNAs[gene_name], gene_name)
                    # DNA_seq_alt,DNA_seq_ref, coords = targeted_perturbation(seq_info,gRNAs[gene_name], gene_name)
                    # seq_info['coords'] = coords
                else:
                    # print(f'No gRNAs provided for gene {gene_name}, using the approximate perturbation region')
                    seq_info[0]['promoter_len'] = promoter_len
                    seq_info[0]['perturb_type'] = perturb_type
                    # print("!!! gRNAs info is not provided, using the approximated perturbation region +-200 bp")
                    # raise ValueError("gRNAs must be provided to get the perturbation effect in the lastest version")
                    DNA_seq_alt, DNA_seq_ref = aggregate_perturbation(seq_info[0])
                one_hot_ref = dna.dna_1hot(DNA_seq_ref).T 
                # one_hot_alt = dna.dna_1hot(DNA_seq_alt).T
                device = model.DNAmodel.device
                with torch.autocast("cuda"):
                    one_hot_ref = torch.tensor(one_hot_ref,dtype=torch.float32).to(device)
                    # one_hot_alt = torch.tensor(one_hot_alt,dtype=torch.float32).to(device)
                    dna_ref_emb = model(one_hot_ref[None,...])
                    if direction == '-':
                    #     print(f'direction is -, but currently dont flip for debug purpose')
                    #     # dna_delta_emb = dna_delta_emb.flip(1) ## reverse transcription
                        dna_ref_emb = dna_ref_emb.flip(1)
                
                
                # DNA_delta_embeddings.append(dna_delta_emb)
                DNA_ref_embedding.append(dna_ref_emb)
                
                seq_info_all.append(seq_info)
            
        DNA_ref_embedding = torch.cat(DNA_ref_embedding, dim=1)  # Concatenate embeddings for multiple genes
        if cache is not None:
            # cache[gene_name] = [dna_delta_emb, dna_ref_emb]
            cache[gene_names] = [DNA_ref_embedding]
        DNA_ref_embeddings.append(DNA_ref_embedding)
    # DNA_delta_embeddings = torch.cat(DNA_delta_embeddings, dim=0)
    dna_embed_list = [emb.squeeze(0) for emb in DNA_ref_embeddings]
    # DNA_ref_embeddings = torch.cat(DNA_ref_embeddings, dim=0)
    DNA_ref_embeddings = pad_sequence(dna_embed_list, batch_first=True)
    if return_seq_info:
        return DNA_ref_embeddings, seq_info_all
    else:
        return DNA_ref_embeddings


def match_gene_pert_mask_dna_embeddings(gene_labels, 
                                        model,
                                        cache=None,
                                        dna_record=None,
                                        all_annots_transcript=None,
                                        promoter_len=200, ## match with the match_gene_dna_embeddings function parameter
                                        perturb_type='CRISPR-i',
                                        gRNAs=None,
                                        return_seq_info=False,
                                        jitter_window=0,
                                        flip_seq_prob=0.0, 
                                        seed=0,
                                        max_batch_size=32,
                                        DNA_emb_dim=64,
                                        DNA_emb_len: int = 1,
                                        drop_mask_prob=0.0,
                                        flip_mask_sign=False,
                                        VERBOSE=False,
                                        ccre_intervals_dict=None,
                                        z_ctrl=None):
    from torch.nn.utils.rnn import pad_sequence
    import torch
    import random
    import time
    import numpy as np
    

    device = next(model.parameters()).device

    assert flip_seq_prob < 0.5, "flip_seq_prob must be less than 0.5"

    # Preallocate to maintain correct ordering matching gene_labels
    DNA_ref_embeddings = [None] * len(gene_labels)
    seq_info_all = []
    failed_gene_labels = []

    batch_tensors = []
    batch_labels = []
    batch_indices = []
    # RC_INDEX = torch.tensor([3, 2, 1, 0])
    RC_INDEX = torch.tensor([3, 2, 1, 0], device=device)

    # Timing variables
    total_seq_gen_time = 0
    total_agg_time = 0
    total_tensor_time = 0
    total_model_time = 0
    total_pad_time = 0
    sample_count = 0
    ccre_count = []  # Track cCRE counts for logging
    
    # Cache hit/miss timing
    total_cache_hit_time = 0
    total_cache_miss_time = 0
    cache_hit_count = 0
    cache_miss_count = 0
    total_cache_lookup_time = 0
    
    LUT = torch.full((256,), 4, dtype=torch.long, device=device)  # default = 4 ("other")
    for base, idx in (('A',0), ('C',1), ('G',2), ('T',3)):
        LUT[ord(base)] = idx
        LUT[ord(base.lower())] = idx          # handle lowercase too

    def flush_batch():
        nonlocal batch_tensors, batch_labels, DNA_ref_embeddings, total_model_time, cache, DNA_emb_len
        if len(batch_tensors) == 0:
            return

        time_start = time.time()
        # Enable gradients to allow adaptor parameters to be updated during training
        grad_ctx = torch.enable_grad()
        
        # Batch processing: collect all samples and process together
        with grad_ctx, torch.autocast(device_type='cuda'):
            # Pad variable-length sequences
            from torch.nn.utils.rnn import pad_sequence
            batched_input = pad_sequence(batch_tensors, batch_first=True)  # (B, 5, L_max)
            
            # Extract batched metadata
            # Uppercase cache keys to match cache storage format (cache stores genes in uppercase)
            cache_keys = [label.rsplit('-', 2)[0].upper() for label in batch_labels]
            ccre_intervals_list = []
            for i in range(len(batch_labels)):
                global_idx = batch_indices[i]  # Use global index to correctly access seq_info_all
                if seq_info_all and len(seq_info_all) > global_idx:
                    si = seq_info_all[global_idx]
                    if si and isinstance(si, list) and len(si) > 0:
                        si_dict = si[0] if isinstance(si[0], dict) else si
                        ccre_intervals_list.append(si_dict.get('ccre_intervals_rel', None))
                    else:
                        ccre_intervals_list.append(None)
                else:
                    ccre_intervals_list.append(None)
            
            # Stack z_ctrl if available
            if z_ctrl is not None:
                assert z_ctrl.dim() == 2, "z_ctrl must be a 2D tensor"
                batched_z_ctrl = torch.stack([z_ctrl[batch_indices[i]] for i in range(len(batch_labels))])  # (B, cell_dim)
            else:
                batched_z_ctrl = None
            
            # Single batched model call
            batched_output = model(
                batched_input,
                verbose=VERBOSE,
                cache=cache,
                cache_keys=cache_keys,
                ccre_intervals_rel=ccre_intervals_list,
                jitter_window=jitter_window,
                z_ctrl=batched_z_ctrl
            )  # (B, 1, out_dim)
            
            # Extract per-sample outputs and assign by index to maintain ordering
            for i in range(batched_output.shape[0]):
                global_idx = batch_indices[i]
                out_LD = _to_LD_per_sample(batched_output[i:i+1])  # (L, D)
                
                # track inferred L if we haven't yet established a control length
                if DNA_emb_len == 1 and out_LD.size(0) > 1:
                    DNA_emb_len = int(out_LD.size(0))
                
                # Keep gradients enabled for both cache hits and misses to allow adaptor training
                DNA_ref_embeddings[global_idx] = out_LD
            
            del batched_input, batched_output, batched_z_ctrl
            # Removed torch.cuda.empty_cache() - it's a synchronous operation that adds ~100-500ms per call
            # Only call if memory is actually an issue, and much less frequently (e.g., every 100 batches)

        total_model_time += time.time() - time_start
        batch_tensors.clear()
        batch_labels.clear()
        batch_indices.clear()
        seq_info_batch.clear()
    
    def flush_cache_hit_batch():
        nonlocal cache_hit_inputs, cache_hit_keys, cache_hit_ccre, cache_hit_z, cache_hit_indices
        nonlocal DNA_ref_embeddings, total_cache_hit_time, cache_hit_count, sample_count, DNA_emb_len
        if len(cache_hit_inputs) == 0:
            return
        
        t_cache_hit_batch_start = time.time()
        # Enable gradients to allow adaptor parameters to be updated during training
        grad_ctx = torch.enable_grad()
        
        # Batch processing: collect all cache hits and process together
        with grad_ctx, torch.autocast(device_type='cuda'):
            # Stack inputs (all have same shape from dummy tensors)
            batched_input = torch.stack(cache_hit_inputs, dim=0)  # (B, 5, L)
            
            # Stack z_ctrl if available
            if cache_hit_z[0] is not None:
                batched_z_ctrl = torch.stack(cache_hit_z, dim=0)  # (B, cell_dim)
            else:
                batched_z_ctrl = None
            
            # Single batched model call for all cache hits
            batched_output = model(
                batched_input,
                verbose=VERBOSE,
                cache=cache,
                cache_keys=cache_hit_keys,
                ccre_intervals_rel=cache_hit_ccre,
                jitter_window=jitter_window,
                z_ctrl=batched_z_ctrl
            )  # (B, 1, out_dim)
            
            # Extract per-sample outputs and assign by index to maintain ordering
            for i in range(batched_output.shape[0]):
                idx = cache_hit_indices[i]
                out_LD = _to_LD_per_sample(batched_output[i:i+1])  # (L, D)
                
                # track inferred L if we haven't yet established a control length
                if DNA_emb_len == 1 and out_LD.size(0) > 1:
                    DNA_emb_len = int(out_LD.size(0))
                
                # Keep gradients enabled to allow adaptor training
                DNA_ref_embeddings[idx] = out_LD
            
            del batched_input, batched_output, batched_z_ctrl
        
        # Track timing per-batch
        t_cache_hit_batch_end = time.time()
        batch_time = t_cache_hit_batch_end - t_cache_hit_batch_start
        total_cache_hit_time += batch_time
        cache_hit_count += len(cache_hit_inputs)
        sample_count += len(cache_hit_inputs)
        
        # Clear batch lists
        cache_hit_inputs.clear()
        cache_hit_keys.clear()
        cache_hit_ccre.clear()
        cache_hit_z.clear()
        cache_hit_indices.clear()
    
    
    def gpu_onehot(seq: str, device: torch.device, dtype=torch.bfloat16) -> torch.Tensor:
        """
        Return a (4, L) float32 tensor on *device*.
        Non‑ACGT characters produce an all‑zero column.
        """
        # 1) bytes → LongTensor on GPU
        seq_bytes  = seq.encode('ascii')
        bytes_gpu  = torch.frombuffer(seq_bytes, dtype=torch.uint8).to(device)
        # bytes_gpu = torch.tensor(list(seq.encode('ascii')),
        #                         dtype=torch.long,
        #                         device=device)

        # 2) lookup indices 0..4
        idx = LUT[bytes_gpu.long()]                    # shape (L,)
        L   = idx.size(0)

        # 3) allocate zeros
        one_hot = torch.zeros((4, L), dtype=dtype, device=device)

        # 4) scatter only the valid bases
        mask = idx < 4                          # True where A/C/G/T
        if mask.any():
            valid_idx = idx[mask]               # values 0..3
            # F.one_hot returns (n_valid, 4), dtype long → cast to float
            one_hot[:, mask] = F.one_hot(valid_idx, num_classes=4
                                        ).T.to(dtype=dtype)

        return one_hot
    
    # Optimized batch tensor processing
    def process_batch_tensors(dna_sequences, mask_sequences, directions, flip_flags):
        """Process all tensors in a batch efficiently"""
        batch_size = len(dna_sequences)
        
        # Pre-allocate lists for batch processing
        one_hot_tensors = []
        
        # Process one-hot encodings in batch
        for dna_seq in dna_sequences:
            # one_hot_np = fast_dna_onehot(dna_seq)
            # one_hot_tensors.append(torch.from_numpy(one_hot_np))
            
            one_hot_gpu = gpu_onehot(dna_seq, device, dtype=torch.float32)
            # one_hot_gpu = gpu_onehot(dna_seq, device, dtype=torch.bfloat16)
            # one_hot_vanilla = torch.tensor(dna.dna_1hot(dna_seq).T, dtype=torch.float32, device=device)
            # print(f'......... check if dna_1hot and gpu_onehot produce the same tensors.........')
            # assert torch.equal(one_hot_gpu, one_hot_vanilla), "dna_1hot vs gpu_onehot produce different tensors" 
            
            one_hot_tensors.append(one_hot_gpu)
            
            
        
        # Apply flipping and concatenation efficiently
        combined_tensors = []
        for i in range(batch_size):
            one_hot = one_hot_tensors[i]
            mask_seq = mask_sequences[i]
            direction = directions[i]
            do_flip = flip_flags[i]

            mask = torch.as_tensor(mask_seq,
                           dtype=torch.float32,
                           device=device)  # ( L)

            # print(f'mask has ones: {mask.sum()}')
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # (1, L)
            
            # Apply flipping if needed
            if (direction == '+' and do_flip) or (direction == '-' and not do_flip):
            #     print(f'direction is -, but currently dont flip RC for debug purpose')
                one_hot = one_hot.flip(1).index_select(0, RC_INDEX)
                mask = mask.flip(1)  # Flip along sequence dimension
            
            
            # Concatenate efficiently using torch.cat
            if VERBOSE:
                print(f'one_hot summary: {one_hot.sum(axis=1)}, mask sum: {mask.sum()}')
            combined = torch.cat([one_hot, mask], dim=0)
            combined_tensors.append(combined)
        
        return combined_tensors
    
    # Pre-generate random flip decisions for the entire batch
    flip_decisions = torch.rand(len(gene_labels)) < flip_seq_prob
    
    # Batch storage for efficient processing
    dna_sequences_batch = []
    mask_sequences_batch = []
    directions_batch = []
    flip_flags_batch = []
    seq_info_batch = []
    
    # Batch storage for cache hits
    cache_hit_inputs = []
    cache_hit_keys = []
    cache_hit_ccre = []
    cache_hit_z = []
    cache_hit_indices = []
    
    # Initialize cache miss timer (will be set when first cache miss occurs)
    t_cache_miss_start = None
    
    for idx, gene_label in enumerate(gene_labels):
        if gene_label == 'control':
            # Dummy tensor for control – keep on CPU (no transfer needed)
            dummy_tensor = torch.zeros((1, DNA_emb_dim), dtype=torch.bfloat16)  # Remove device=device
            DNA_ref_embeddings[idx] = dummy_tensor
            continue

        # Apply gene name translation (AC118549.1 -> ZZZ3)
        gene_label = _translate_gene_name(str(gene_label))

        # Handle both 2-part (gene-dataset) and 3-part (gene-batch-dataset) label formats
        # Gene names can contain hyphens (e.g., "RBM14-RBM4"), so we need to check for batch pattern
        parts = gene_label.rsplit('-', 2)
        if len(parts) == 3 and parts[1].startswith('batch_'):
            # Format: gene-batch-dataset (e.g., "RBM14-RBM4-batch_0-rpe1")
            # parts[1] starts with "batch_", so it's a batch identifier
            gene_names, batch_name, dataset_name = parts
        elif len(parts) == 2:
            # Format: gene-dataset (e.g., "RBM14-RBM4-rpe1" or "SETD1A-rpe1")
            gene_names, dataset_name = parts
            batch_name = '0'  # Default batch name for datasets without batch info
        elif len(parts) == 3:
            # Has 3 parts but second part doesn't start with "batch_" - treat as 2-part
            # This handles cases like "RBM14-RBM4-rpe1" where gene name has a hyphen
            gene_names = '-'.join(parts[:2])  # Join first two parts as gene name
            dataset_name = parts[2]
            batch_name = '0'
        else:
            raise ValueError(f"Unexpected token format: {gene_label}. Expected format: 'gene-dataset' or 'gene-batch-dataset'")

        gene_names_list = gene_names.split('+')
        if len(gene_names_list) > 1:
            raise ValueError(f"Multiple genes not supported yet: {gene_names_list}")
        gene_name = gene_names_list[0]
        # Uppercase gene_name to match cache keys (cache stores genes in uppercase)
        gene_name = gene_name.upper()

        # Use cache if available (old format: final embeddings)
        # Note: New format (Borzoi features) is handled by model internally
        t_cache_lookup_start = time.time()
        cache_hit = cache is not None and gene_name in cache
        t_cache_lookup_end = time.time()
        total_cache_lookup_time += (t_cache_lookup_end - t_cache_lookup_start)
        
        if cache_hit:
            cached = cache[gene_name]
            
            # Check if it's the new format (tuple) or old format (tensor)
            if isinstance(cached, tuple):
                # New format: tuple of (feat_full, mask_full, ccre_intervals_cached)
                # Model will handle caching internally via cache_key parameter
                # Skip ALL sequence generation, mask generation, and one-hot conversion
                try:
                    # Extract ccre_intervals_rel directly from cache (already there!)
                    feat_full, mask_full, ccre_intervals_rel = cached
                    
                    # Get minimal info only if return_seq_info is needed
                    if return_seq_info:
                        # Skip expensive sequence generation - create minimal seq_info from cached data
                        seq_info = [{
                            'gene_name': gene_name,
                            'ccre_intervals_rel': ccre_intervals_rel,  # Already from cache
                        }]
                        seq_info_all.append(seq_info)
                    
                    if z_ctrl is not None:
                        assert z_ctrl.dim() == 2
                        z_ctrl_i = z_ctrl[idx:idx+1]  # shape (1, D)
                    else:
                        z_ctrl_i = None
                
                    # Create minimal dummy input tensor (model doesn't use it on cache hit, just for signature)
                    # Use a small fixed size since model will use cached feat_full anyway
                    dummy_L = 4096  # Fixed size, model will use cached feat_full anyway
                    dummy_dna = torch.zeros((4, dummy_L), dtype=torch.float32, device=device)
                    dummy_mask = torch.zeros((1, dummy_L), dtype=torch.float32, device=device)
                    combined = torch.cat([dummy_dna, dummy_mask], dim=0)  # (5, L) - no unsqueeze to match cache miss format
                    
                    # Collect into batch lists instead of processing immediately
                    cache_hit_inputs.append(combined)
                    cache_hit_keys.append(gene_name)
                    cache_hit_ccre.append(ccre_intervals_rel)
                    cache_hit_z.append(z_ctrl_i)
                    cache_hit_indices.append(idx)
                    
                    # Flush batch if it's full (use same effective batch size as cache misses)
                    effective_batch_size = min(max_batch_size, 16)
                    if len(cache_hit_inputs) >= effective_batch_size:
                        flush_cache_hit_batch()
                    
                    continue  # Skip normal batching flow
                except Exception as e:
                    # Fall back to normal flow if optimization fails
                    if VERBOSE:
                        print(f"Cache optimization failed for {gene_name}: {e}, falling back to normal flow")
                        import traceback
                        traceback.print_exc()
                    pass  # Continue to normal flow
            else:
                # Old format: final embedding tensor - use it directly
                cached_LD = cached if torch.is_tensor(cached) else torch.as_tensor(cached)
                DNA_ref_embeddings[idx] = cached_LD.to(device, non_blocking=True)
                # Also update DNA_emb_len if still unknown
                if DNA_emb_len == 1 and cached_LD.size(0) > 1:
                    DNA_emb_len = int(cached_LD.size(0))
                continue

        # Sequence generation (cache miss path)
        if t_cache_miss_start is None:
            t_cache_miss_start = time.time()
        time_start = time.time()
        seq_info = gen_gene_TSS_sequence(gene_name, dna_record=dna_record,
                                         all_annots_transcript=all_annots_transcript,
                                         jitter_window=jitter_window,
                                         ccre_intervals_dict=ccre_intervals_dict)
        time_end = time.time()
        seq_time = time_end - time_start
        total_seq_gen_time += seq_time
        # print(f"Time taken for sequence generation: {seq_time:.4f} seconds")
        
        if seq_info is None:
            failed_gene_labels.append(gene_label)
            dummy_tensor = torch.zeros(
                (1, DNA_emb_dim),
                dtype=torch.bfloat16,   # <- critical: match autocast output
                device=device
            )
            DNA_ref_embeddings[idx] = dummy_tensor  # placeholder
            continue

        seq_info[0]['promoter_len'] = promoter_len
        seq_info[0]['perturb_type'] = perturb_type
        direction = seq_info[0]['Direction']

        if gRNAs is not None and gRNAs.get(gene_name) is not None:
            raise NotImplementedError("gRNAs is not supported yet")

        time_start = time.time()
        DNA_seq_ref, DNA_seq_mask = aggregate_mask_perturbation(seq_info[0])
        # Extract cCRE intervals from seq_info
        ccre_intervals_rel = seq_info[0].get('ccre_intervals_rel', [])
        num_ccres = len(ccre_intervals_rel) if ccre_intervals_rel else 0
        if VERBOSE:
            print(f'Number of cCRE intervals in window for {gene_name}: {num_ccres}')
        # print(f'DNA_seq_mask has ones before flip: {DNA_seq_mask.sum()}')
        is_drop = random.random() < drop_mask_prob
        if is_drop:
            DNA_seq_mask = np.zeros_like(DNA_seq_mask)
        
        if flip_mask_sign:
            DNA_seq_mask = -DNA_seq_mask ## Flip positiv to negative
        
        time_end = time.time()
        agg_time = time_end - time_start
        total_agg_time += agg_time
        # print(f"Time taken for aggregate_mask_perturbation: {agg_time:.4f} seconds")
        
        # Store for batch processing instead of immediate tensor conversion
        dna_sequences_batch.append(DNA_seq_ref)
        mask_sequences_batch.append(DNA_seq_mask)
        directions_batch.append(direction)
        flip_flags_batch.append(flip_decisions[idx].item())
        
        batch_labels.append(gene_label)
        batch_indices.append(idx)
        seq_info_all.append(seq_info)
        seq_info_batch.append(seq_info)
        sample_count += 1
        ccre_count.append(num_ccres)

        # Flush if batch is full (use smaller batches for BorzoiPerturbAdapter)
        effective_batch_size = min(max_batch_size, 16)  # Increased from 8 to 16 for better GPU utilization
        if len(dna_sequences_batch) >= effective_batch_size:
            # Batch process all tensor operations at once
            time_start = time.time()
            
            # Use optimized batch processing
            combined_tensors = process_batch_tensors(
                dna_sequences_batch, mask_sequences_batch, 
                directions_batch, flip_flags_batch
            )
            
            # Add all tensors to batch at once
            batch_tensors.extend(combined_tensors)
            
            time_end = time.time()
            tensor_time = time_end - time_start
            total_tensor_time += tensor_time
            # print(f"Time taken for batch tensor operations: {tensor_time:.4f} seconds")
            
            # Process the batch
            flush_batch()
            
            # Track cache miss time for this batch
            t_cache_miss_end = time.time()
            cache_miss_time = t_cache_miss_end - t_cache_miss_start
            total_cache_miss_time += cache_miss_time
            cache_miss_count += len(dna_sequences_batch)
            
            # Clear batch storage
            dna_sequences_batch.clear()
            mask_sequences_batch.clear()
            directions_batch.clear()
            flip_flags_batch.clear()
            
            # Reset cache miss timer for next batch
            t_cache_miss_start = time.time()

    # Final flush for remaining sequences
    if dna_sequences_batch:
        time_start = time.time()
        
        # Use optimized batch processing for remaining sequences
        combined_tensors = process_batch_tensors(
            dna_sequences_batch, mask_sequences_batch, 
            directions_batch, flip_flags_batch
        )
        
        batch_tensors.extend(combined_tensors)
        
        time_end = time.time()
        tensor_time = time_end - time_start
        total_tensor_time += tensor_time
        # print(f"Time taken for final batch tensor operations: {tensor_time:.4f} seconds")
        
        flush_batch()
        
        # Track cache miss time for final batch
        if t_cache_miss_start is not None:
            t_cache_miss_end = time.time()
            cache_miss_time = t_cache_miss_end - t_cache_miss_start
            total_cache_miss_time += cache_miss_time
            cache_miss_count += len(dna_sequences_batch)

    # Final flush for remaining cache hits
    if cache_hit_inputs:
        flush_cache_hit_batch()

    # pad output - filter out None values before padding
    time_start = time.time()
    # Filter out None values (shouldn't happen, but handle gracefully)
    DNA_ref_embeddings_filtered = [e for e in DNA_ref_embeddings if e is not None]
    DNA_ref_embeddings = pad_sequence(DNA_ref_embeddings_filtered, batch_first=True)
    time_end = time.time()
    pad_time = time_end - time_start
    total_pad_time += pad_time
    # print(f"Time taken for pad_sequence: {pad_time:.4f} seconds")

    # Print summary statistics
    if VERBOSE:
        print(f"\n=== TIMING SUMMARY ===")
        print(f"Total samples processed: {sample_count}")
        if cache_hit_count > 0:
            print(f"Cache hits: {cache_hit_count} (avg {total_cache_hit_time/cache_hit_count:.4f}s each, total {total_cache_hit_time:.4f}s)")
        if cache_miss_count > 0:
            print(f"Cache misses: {cache_miss_count} (avg {total_cache_miss_time/cache_miss_count:.4f}s each, total {total_cache_miss_time:.4f}s)")
        if total_cache_lookup_time > 0:
            print(f"Cache lookup time: {total_cache_lookup_time:.4f}s")
        print(f"Sequence generation: {total_seq_gen_time:.4f}s ({total_seq_gen_time/sample_count:.4f}s per sample)")
        print(f"Aggregate perturbation: {total_agg_time:.4f}s ({total_agg_time/sample_count:.4f}s per sample)")
        print(f"Tensor operations: {total_tensor_time:.4f}s ({total_tensor_time/sample_count:.4f}s per sample)")
        print(f"Model inference: {total_model_time:.4f}s ({total_model_time/sample_count:.4f}s per sample)")
        print(f"Final padding: {total_pad_time:.4f}s")
        print(f"Total time: {total_seq_gen_time + total_agg_time + total_tensor_time + total_model_time + total_pad_time:.4f}s")
        if len(ccre_count) > 0:
            print(f"cCRE intervals: mean={np.mean(ccre_count):.2f}, min={min(ccre_count)}, max={max(ccre_count)}, total={sum(ccre_count)}")

    if return_seq_info:
        return DNA_ref_embeddings, seq_info_all, failed_gene_labels
    else:
        return DNA_ref_embeddings, failed_gene_labels


# dna_matching.py
import time
import torch
from torch.nn.utils.rnn import pad_sequence


def _canonical_key(label: str) -> str:
    """
    Turns a raw gene label into the key that the cache uses.
    This is the *only* string work we do per label.
    """
    label = _translate_gene_name(label).strip()
    parts = label.rsplit('-', 2)
    if len(parts) == 3 and parts[1].startswith('batch_'):
        gene_names = parts[0]
    elif len(parts) == 2:
        gene_names = parts[0]
    elif len(parts) == 3:
        # Handle "GENE1-GENE2-dataset" where gene name has hyphen
        gene_names = '-'.join(parts[:2])
    else:
        raise ValueError(f"Unexpected token format: {label}. Expected format: 'gene-dataset' or 'gene-batch-dataset'")
    return gene_names.split('+')[0].upper()


def match_gene_pert_mask_dna_embeddings_minimal(gene_labels,
                                                 cache=None,
                                                 disk_cache=None,
                                                 DNA_emb_dim=64,
                                                 VERBOSE=False):
    """
    Minimal test function that retrieves cached gene embeddings and applies average pooling.
    Skips ALL CPU work (no sequence generation, no mask generation, no one‑hot encoding)
    and ALL model inference (no BorzoiPerturbAdapter forward pass).

    This function is used to validate if CPU bottlenecks are the main issue.

    Args:
        gene_labels: List of gene labels (same format as main function)
        cache: LRUCache or dict containing cached embeddings (memory cache – checked first)
        disk_cache: DiskDNACache containing cached embeddings (checked if memory cache misses)
        DNA_emb_dim: Expected embedding dimension (for control cases)
        VERBOSE: Whether to print debug information (unused in optimized version)

    Returns:
        DNA_ref_embeddings: Padded tensor of embeddings (B, L_max, D)
        failed_gene_labels: List of gene labels that failed (cache misses) – empty list if errors are raised

    Raises:
        RuntimeError: If gene embedding is not found in cache or disk cache
    """
    # ------------------------------------------------------------------
    # 1️⃣  Choose device (detect from cache if available for DDP compatibility)
    # ------------------------------------------------------------------
    # Try to detect device from cache (for DDP: each rank has its own device)
    device = None
    if cache is not None and hasattr(cache, 'device') and cache.device is not None:
        device = cache.device
    elif disk_cache is not None and hasattr(disk_cache, 'device'):
        device = disk_cache.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if VERBOSE:
        print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # ------------------------------------------------------------------
    # 2️⃣  Pre-compute canonical keys (no extra Python objects later)
    # ------------------------------------------------------------------
    canonical_keys = [_canonical_key(lbl) for lbl in gene_labels]

    # ------------------------------------------------------------------
    # 3️⃣  Prepare containers
    # ------------------------------------------------------------------
    B = len(gene_labels)
    DNA_ref_embeddings = [None] * B
    raw_embeddings_for_pooling = []   # [(idx, tensor), ...]  tensor shape: (L, C) - already on GPU
    already_pooled_embeddings = []     # [(idx, tensor), ...]  tensor shape: (1, C) or (C,) - already on GPU

    # Timing accumulators
    total_main_loop_time = 0.0
    total_cache_lookup_time = 0.0
    total_processing_time = 0.0
    total_pooling_time = 0.0
    total_already_pooled_time = 0.0
    total_final_padding_time = 0.0

    # ------------------------------------------------------------------
    # 4️⃣  Main loop – work with GPU tensors directly from cache
    # ------------------------------------------------------------------
    if VERBOSE:
        t_main_loop_start = time.time()
    
    for idx, (gene_label, gene_name) in enumerate(zip(gene_labels, canonical_keys)):
        if gene_label == 'control':
            DNA_ref_embeddings[idx] = torch.zeros((1, DNA_emb_dim), dtype=torch.bfloat16, device=device)
            continue

        # ---- cache lookup (embeddings already on GPU) ----
        if VERBOSE:
            t_cache_lookup_start = time.time()
        cached_data = cache.get(gene_name) if cache else None
        if cached_data is None:
            cached_data = disk_cache.get(gene_name) if disk_cache else None
            if cached_data is None:
                raise RuntimeError(f"Gene '{gene_name}' not found in cache or disk cache. "
                                   f"Please precompute embeddings using DNA_diff_cache_prepare.py")
            # Populate memory cache for future calls (already on GPU)
            if cache is not None:
                cache[gene_name] = cached_data
        if VERBOSE:
            t_cache_lookup_end = time.time()
            total_cache_lookup_time += (t_cache_lookup_end - t_cache_lookup_start)

        # ---- process cached data (already on GPU, just normalize shape) ----
        if VERBOSE:
            t_processing_start = time.time()
        # Extract feature tensor (already on GPU from cache)
        if isinstance(cached_data, tuple):
            feat_full = cached_data[0]  # Already a GPU tensor
        else:
            feat_full = cached_data  # Already a GPU tensor
        
        # Ensure it's a tensor (should already be from cache)
        if not isinstance(feat_full, torch.Tensor):
            feat_full = torch.as_tensor(feat_full).to(device=device, dtype=torch.bfloat16)
        else:
            # Ensure correct device and dtype (should already be correct from cache)
            if feat_full.device != device:
                feat_full = feat_full.to(device=device, non_blocking=True)
            if feat_full.dtype != torch.bfloat16 and device.type == 'cuda':
                feat_full = feat_full.to(dtype=torch.bfloat16)

        # Normalize shape to (L, C) - work directly on GPU
        if feat_full.dim() == 3:
            feat_full = feat_full.squeeze(0) if feat_full.shape[0] == 1 else feat_full[0]
        if feat_full.dim() == 1:
            feat_full = feat_full.unsqueeze(0)

        if feat_full.shape[0] > 1:
            raw_embeddings_for_pooling.append((idx, feat_full))  # Already on GPU
        else:
            already_pooled_embeddings.append((idx, feat_full))  # Already on GPU
        if VERBOSE:
            t_processing_end = time.time()
            total_processing_time += (t_processing_end - t_processing_start)

    if VERBOSE:
        t_main_loop_end = time.time()
        total_main_loop_time = t_main_loop_end - t_main_loop_start

    # ------------------------------------------------------------------
    # 4️⃣  Batch pooling – all tensors already on GPU, no transfer needed
    # ------------------------------------------------------------------
    if raw_embeddings_for_pooling:
        if VERBOSE:
            t_pooling_start = time.time()
        indices, tensors = zip(*raw_embeddings_for_pooling)

        # OPTIMIZATION: Requires all tensors to be same length for efficient batching
        # Check if all have same shape
        first_shape = tensors[0].shape
        all_same_length = all(t.shape == first_shape for t in tensors)
        
        if not all_same_length:
            # Collect unique shapes for error message
            unique_shapes = {t.shape for t in tensors}
            raise ValueError(
                f"All embeddings must have the same length for optimized batching. "
                f"Found {len(unique_shapes)} different shapes: {unique_shapes}. "
                f"Expected all to match: {first_shape}"
            )
        
        # Stack all tensors directly on GPU (no CPU-GPU transfer)
        stacked_gpu = torch.stack(tensors, dim=0)  # (B, L, C) - already on GPU
        
        # Direct mean pooling (no padding needed, no masking needed)
        pooled_batch = stacked_gpu.mean(dim=1, keepdim=True)  # (B, 1, C) - already bfloat16 on GPU

        # Write back – no extra copy
        for i, orig_idx in enumerate(indices):
            DNA_ref_embeddings[orig_idx] = pooled_batch[i]   # (1, C) on GPU
        
        if VERBOSE:
            t_pooling_end = time.time()
            total_pooling_time = t_pooling_end - t_pooling_start
    else:
        if VERBOSE:
            total_pooling_time = 0.0

    # ------------------------------------------------------------------
    # 5️⃣  Already‑pooled embeddings – stack directly on GPU (no transfer)
    # ------------------------------------------------------------------
    if already_pooled_embeddings:
        if VERBOSE:
            t_already_pooled_start = time.time()
        indices, tensors = zip(*already_pooled_embeddings)
        # Stack directly on GPU (tensors already on GPU from cache)
        stacked = torch.stack(tensors, dim=0)  # (B, L, C) where L==1 or (B, C) - already on GPU
        if stacked.dim() == 2:  # (B, C) → (B, 1, C)
            stacked = stacked.unsqueeze(1)
        for i, orig_idx in enumerate(indices):
            DNA_ref_embeddings[orig_idx] = stacked[i]   # (1, C) on GPU
        
        if VERBOSE:
            t_already_pooled_end = time.time()
            total_already_pooled_time = t_already_pooled_end - t_already_pooled_start
    else:
        if VERBOSE:
            total_already_pooled_time = 0.0

    # ------------------------------------------------------------------
    # 6️⃣  Final padding – all tensors are already on GPU
    # ------------------------------------------------------------------
    if VERBOSE:
        t_final_padding_start = time.time()
    # All entries are (1, C) or (1, L, C) – we just need to make them (B, L_max, C)
    valid_embeddings = [e for e in DNA_ref_embeddings if e is not None]

    if len(valid_embeddings) == 0:
        DNA_ref_embeddings = torch.zeros((0, 1, DNA_emb_dim), dtype=torch.bfloat16, device=device)
    else:
        # Determine the maximum sequence length in the batch
        max_len = max(e.shape[1] if e.dim() == 3 else 1 for e in valid_embeddings)

        # Pad each tensor to (1, max_len, C) on GPU
        padded_list = []
        for e in valid_embeddings:
            if e.dim() == 2:                     # (1, C) → (1, 1, C)
                e = e.unsqueeze(1)
            elif e.dim() == 3 and e.shape[1] < max_len:
                # Pad the sequence dimension (the middle dim)
                e = F.pad(e, (0, 0, 0, max_len - e.shape[1]), mode='constant', value=0)
            padded_list.append(e)

        # Concatenate along the batch dimension
        DNA_ref_embeddings = torch.cat(padded_list, dim=0)   # (B, max_len, C)

    if VERBOSE:
        t_final_padding_end = time.time()
        total_final_padding_time = t_final_padding_end - t_final_padding_start

        # Print timing summary
        print(f"\n=== MINIMAL FUNCTION TIMING ===")
        print(f"Total main loop time: {total_main_loop_time:.4f} sec")
        print(f"  - Cache lookup time: {total_cache_lookup_time:.4f} sec")
        print(f"  - Processing time: {total_processing_time:.4f} sec")
        if raw_embeddings_for_pooling:
            print(f"Batch pooling time: {total_pooling_time:.4f} sec")
        if already_pooled_embeddings:
            print(f"Already-pooled embeddings time: {total_already_pooled_time:.4f} sec")
        print(f"Final padding time: {total_final_padding_time:.4f} sec")
        total_time = total_main_loop_time + total_pooling_time + total_already_pooled_time + total_final_padding_time
        print(f"Total time: {total_time:.4f} sec")
        print(f"\n=== MINIMAL FUNCTION SUMMARY ===")
        print(f"Total samples: {B}")
        print(f"Raw embeddings for pooling: {len(raw_embeddings_for_pooling)}")
        print(f"Already-pooled embeddings: {len(already_pooled_embeddings)}")
        print(f"Successful retrievals: {len(valid_embeddings)}")
        print(f"Note: Cache misses raise RuntimeError instead of returning dummy embeddings")

    return DNA_ref_embeddings, []


# =========================
# Interval-based DNA embedding functions (migrated from utils.py)
# =========================
# def match_gene_pert_mask_dna_embeddings_cached(gene_labels, 
#                                         model,
#                                         cache=None,
#                                         disk_cache=None,
#                                         dna_record=None,
#                                         all_annots_transcript=None,
#                                         promoter_len=200,
#                                         perturb_type='CRISPR-i',
#                                         gRNAs=None,
#                                         return_seq_info=False,
#                                         jitter_window=0,
#                                         flip_seq_prob=0.0, 
#                                         seed=0,
#                                         max_batch_size=32,
#                                         DNA_emb_dim=64,
#                                         DNA_emb_len: int = 1,
#                                         drop_mask_prob=0.0,
#                                         flip_mask_sign=False,
#                                         VERBOSE=False,
#                                         ccre_intervals_dict=None,
#                                         z_ctrl=None):
#     """
#     Cache-only version: loads Borzoi embeddings from cache (memory or disk) and passes to adapter.
#     No DNA sequence generation or one-hot encoding.
    
#     Optimized version with pre-computed keys, batched cache lookups, and comprehensive timing.
    
#     Args:
#         cache: Memory cache (LRUCache) - checked first
#         disk_cache: Disk cache (DiskDNACache) - checked if memory cache misses
#         model: BorzoiPerturbAdapter with load_backbone=False
#     """
#     from torch.nn.utils.rnn import pad_sequence
#     import torch
#     import time
#     import numpy as np
    
#     # ------------------------------------------------------------------
#     # 1️⃣  Get device (cached once)
#     # ------------------------------------------------------------------
#     try:
#         if hasattr(model, 'perturbation') and hasattr(model.perturbation, 'pos') and model.perturbation.pos is not None:
#             device = model.perturbation.pos.weight.device
#         else:
#             device = next(model.parameters()).device
#     except (AttributeError, StopIteration):
#         # Ultimate fallback: try to get device from any CUDA parameter, else CPU
#         params = list(model.parameters())
#         if params:
#             cuda_params = [p for p in params if p.device.type == 'cuda']
#             device = cuda_params[0].device if cuda_params else params[0].device
#         else:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     if VERBOSE:
#         print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
#     # ------------------------------------------------------------------
#     # 2️⃣  Pre-compute canonical keys and identify control cases
#     # ------------------------------------------------------------------
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_precompute_start = time.time()
    
#     B = len(gene_labels)
#     precomputed_data = []  # List of (idx, original_label, cache_key, is_control, translated_label)
#     control_indices = []
    
#     for idx, gene_label in enumerate(gene_labels):
#         if gene_label == 'control':
#             # Control case - use dummy key
#             dummy_key = f"__control_{idx}__"
#             precomputed_data.append((idx, gene_label, dummy_key, True, None))
#             control_indices.append(idx)
#         else:
#             # Apply gene name translation (AC118549.1 -> ZZZ3)
#             translated_label = _translate_gene_name(str(gene_label))
#             # Pre-compute canonical cache key using _canonical_key function
#             cache_key = _canonical_key(translated_label)
#             precomputed_data.append((idx, gene_label, cache_key, False, translated_label))
    
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_precompute_end = time.time()
#         total_precompute_time = t_precompute_end - t_precompute_start
#     else:
#         total_precompute_time = 0.0
    
#     # ------------------------------------------------------------------
#     # 3️⃣  Batch cache lookups (memory cache first, then disk cache)
#     # ------------------------------------------------------------------
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_cache_lookup_start = time.time()
    
#     # Collect all non-control cache keys
#     non_control_keys = [cache_key for _, _, cache_key, is_control, _ in precomputed_data if not is_control]
#     unique_keys = list(set(non_control_keys))  # Deduplicate
    
#     # Batch-check memory cache
#     memory_cache_hits = {}
#     memory_cache_misses = []
#     if cache is not None:
#         for key in unique_keys:
#             if key in cache:
#                 memory_cache_hits[key] = cache[key]
#             else:
#                 memory_cache_misses.append(key)
#     else:
#         memory_cache_misses = unique_keys
    
#     # Batch-check disk cache for memory misses
#     disk_cache_hits = {}
#     disk_cache_misses = []
#     if disk_cache is not None:
#         for key in memory_cache_misses:
#             if key in disk_cache:
#                 disk_cache_hits[key] = disk_cache[key]
#                 # Promote to memory cache
#                 if cache is not None:
#                     cache[key] = disk_cache[key]
#             else:
#                 disk_cache_misses.append(key)
#     else:
#         disk_cache_misses = memory_cache_misses
    
#     # Validate all keys are found (except controls which are handled separately)
#     if disk_cache_misses:
#         missing_keys_str = ', '.join(disk_cache_misses[:5])  # Show first 5
#         if len(disk_cache_misses) > 5:
#             missing_keys_str += f" ... and {len(disk_cache_misses) - 5} more"
#         raise RuntimeError(f"Genes not found in cache or disk cache: {missing_keys_str}. "
#                           f"Please precompute embeddings using DNA_diff_cache_prepare.py")
    
#     # Build lookup dictionary for fast access
#     cache_data_dict = {}
#     cache_data_dict.update(memory_cache_hits)
#     cache_data_dict.update(disk_cache_hits)
    
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_cache_lookup_end = time.time()
#         total_cache_lookup_time = t_cache_lookup_end - t_cache_lookup_start
#         print(f"Cache lookup: {len(memory_cache_hits)} memory hits, {len(disk_cache_hits)} disk hits, "
#               f"{len(disk_cache_misses)} misses")
#     else:
#         total_cache_lookup_time = 0.0
    
#     # ------------------------------------------------------------------
#     # 4️⃣  Batch-create control dummy cache entries
#     # ------------------------------------------------------------------
#     if control_indices:
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_control_setup_start = time.time()
        
#         for idx in control_indices:
#             dummy_key = f"__control_{idx}__"
#             dummy_feat = torch.zeros((1, 1, 1536), dtype=torch.float32, device=device)  # (1, L=1, C=1536)
#             dummy_mask = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)  # (1, 1, L=1)
#             # Store in cache temporarily so adapter can find it
#             if cache is not None:
#                 cache[dummy_key] = (dummy_feat, dummy_mask, None)
#             elif disk_cache is not None:
#                 disk_cache[dummy_key] = (dummy_feat.cpu(), dummy_mask.cpu(), None)
        
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_control_setup_end = time.time()
#             total_control_setup_time = t_control_setup_end - t_control_setup_start
#         else:
#             total_control_setup_time = 0.0
#     else:
#         total_control_setup_time = 0.0
    
#     # ------------------------------------------------------------------
#     # 5️⃣  Prepare data structures and timing
#     # ------------------------------------------------------------------
#     DNA_ref_embeddings = []
#     seq_info_all = []
#     failed_gene_labels = []
    
#     # Batch storage: (idx, cache_key, cached_data, ccre_info, original_label)
#     batch_data = []
    
#     # Timing accumulators
#     total_batch_prep_time = 0.0
#     total_model_time = 0.0
#     total_output_extraction_time = 0.0
#     total_final_padding_time = 0.0
#     cache_hit_count = 0
#     cache_miss_count = len(disk_cache_misses)
    
#     # Dummy tensor for adapter signature (reused across batch)
#     dummy_L = 128
#     dummy_combined = torch.zeros((5, dummy_L), dtype=torch.float32, device=device)
    
#     # ------------------------------------------------------------------
#     # 6️⃣  Optimized flush_batch function
#     # ------------------------------------------------------------------
#     def flush_batch():
#         nonlocal DNA_ref_embeddings, total_model_time, total_output_extraction_time, DNA_emb_len
#         if len(batch_data) == 0:
#             return
        
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_batch_prep_start = time.time()
        
#         # Extract batch components (cache keys are already pre-computed)
#         batch_indices = [item[0] for item in batch_data]
#         batch_cache_keys = [item[1] for item in batch_data]
        
#         # Optimize z_ctrl stacking using tensor indexing
#         if z_ctrl is not None:
#             assert z_ctrl.dim() == 2, "z_ctrl must be a 2D tensor"
#             # Use tensor indexing for efficiency
#             indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=z_ctrl.device)
#             batched_z_ctrl = z_ctrl[indices_tensor]  # More efficient than stack + list comprehension
#             # Ensure z_ctrl is on the same device as the model
#             if batched_z_ctrl.device != device:
#                 batched_z_ctrl = batched_z_ctrl.to(device, non_blocking=True)
#         else:
#             batched_z_ctrl = None
        
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_batch_prep_end = time.time()
#             total_batch_prep_time += (t_batch_prep_end - t_batch_prep_start)
        
#         # Model inference
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_model_start = time.time()
        
#         with torch.autocast(device_type='cuda'):
#             # Construct dummy input (adapter won't use it, just for signature)
#             B = len(batch_data)
#             dummy_input = dummy_combined.unsqueeze(0).expand(B, -1, -1)  # (B, 5, dummy_L)
            
#             # Call adapter with cache - it will look up cached (feat, mask, ccre) tuples using cache_keys
#             batched_output = model(
#                 dummy_input,
#                 verbose=VERBOSE,
#                 cache=cache,
#                 cache_keys=batch_cache_keys,
#                 jitter_window=jitter_window,
#                 z_ctrl=batched_z_ctrl
#             )  # (B, 1, out_dim)
        
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_model_end = time.time()
#             total_model_time += (t_model_end - t_model_start)
        
#         # Extract per-sample outputs
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_output_extraction_start = time.time()
        
#         for i in range(batched_output.shape[0]):
#             out_LD = _to_LD_per_sample(batched_output[i:i+1])  # (L, D)
            
#             if DNA_emb_len == 1 and out_LD.size(0) > 1:
#                 DNA_emb_len = int(out_LD.size(0))
            
#             DNA_ref_embeddings.append(out_LD)
        
#         if VERBOSE:
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_output_extraction_end = time.time()
#             total_output_extraction_time += (t_output_extraction_end - t_output_extraction_start)
        
#         # Cleanup
#         del dummy_input, batched_output, batched_z_ctrl
#         batch_data.clear()
    
#     # ------------------------------------------------------------------
#     # 7️⃣  Main processing loop (using pre-computed data)
#     # ------------------------------------------------------------------
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_main_loop_start = time.time()
#     else:
#         t_main_loop_start = None
    
#     for idx, original_label, cache_key, is_control, translated_label in precomputed_data:
#         if is_control:
#             # Control case - add to batch with dummy key
#             batch_data.append((idx, cache_key, None, None, original_label))
#             cache_hit_count += 1
#             if return_seq_info:
#                 seq_info_all.append([{'gene_name': 'control', 'ccre_intervals_rel': None}])
#         else:
#             # Get cached data (already validated to exist)
#             cached_data = cache_data_dict[cache_key]
            
#             # Handle different cache formats
#             if isinstance(cached_data, tuple) and len(cached_data) == 3:
#                 # New format: (feat, mask, ccre_intervals)
#                 _, _, ccre_intervals_rel = cached_data
#                 batch_data.append((idx, cache_key, cached_data, ccre_intervals_rel, original_label))
                
#                 cache_hit_count += 1
                
#                 # Build seq_info if requested
#                 if return_seq_info:
#                     # Extract gene name from cache_key (it's already uppercase)
#                     gene_name = cache_key
#                     seq_info = [{
#                         'gene_name': gene_name,
#                         'ccre_intervals_rel': ccre_intervals_rel,
#                     }]
#                     seq_info_all.append(seq_info)
#             else:
#                 # Old format: final embedding tensor (already processed by adapter)
#                 # Convert to new format and update cache
#                 cached_LD = cached_data if torch.is_tensor(cached_data) else torch.as_tensor(cached_data)
#                 L_ = cached_LD.shape[0]
#                 mask_dummy = torch.zeros(1, 1, L_, device=device, dtype=torch.float32)
#                 new_cached_data = (cached_LD.to(device, non_blocking=True), mask_dummy, None)
                
#                 # Update cache with new format
#                 if cache is not None:
#                     cache[cache_key] = new_cached_data
#                 elif disk_cache is not None:
#                     disk_cache[cache_key] = (cached_LD.cpu(), mask_dummy.cpu(), None)
                
#                 batch_data.append((idx, cache_key, new_cached_data, None, original_label))
#                 cache_hit_count += 1
                
#                 if return_seq_info:
#                     gene_name = cache_key
#                     seq_info_all.append([{'gene_name': gene_name, 'ccre_intervals_rel': None}])
        
#         # Flush if batch is full
#         if len(batch_data) >= min(max_batch_size, 16):
#             flush_batch()
    
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_main_loop_end = time.time()
#         total_main_loop_time = t_main_loop_end - t_main_loop_start
    
#     # ------------------------------------------------------------------
#     # 8️⃣  Final flush
#     # ------------------------------------------------------------------
#     if batch_data:
#         flush_batch()
    
#     # ------------------------------------------------------------------
#     # 9️⃣  Final padding
#     # ------------------------------------------------------------------
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_final_padding_start = time.time()
    
#     DNA_ref_embeddings = pad_sequence(DNA_ref_embeddings, batch_first=True)
    
#     if VERBOSE:
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_final_padding_end = time.time()
#         total_final_padding_time = t_final_padding_end - t_final_padding_start
    
#     # ------------------------------------------------------------------
#     # 🔟  Print comprehensive timing summary
#     # ------------------------------------------------------------------
#     if VERBOSE:
#         print(f"\n=== CACHED MATCHING SUMMARY ===")
#         print(f"Total samples: {B}")
#         print(f"Cache hits: {cache_hit_count}")
#         print(f"Cache misses: {cache_miss_count}")
#         print(f"\n=== TIMING BREAKDOWN ===")
#         print(f"Pre-computation time: {total_precompute_time:.4f} sec")
#         print(f"Cache lookup time: {total_cache_lookup_time:.4f} sec")
#         if control_indices:
#             print(f"Control setup time: {total_control_setup_time:.4f} sec")
#         print(f"Main loop time: {total_main_loop_time:.4f} sec")
#         print(f"  - Batch preparation time: {total_batch_prep_time:.4f} sec")
#         print(f"  - Model inference time: {total_model_time:.4f} sec")
#         print(f"  - Output extraction time: {total_output_extraction_time:.4f} sec")
#         print(f"Final padding time: {total_final_padding_time:.4f} sec")
#         total_time = total_precompute_time + \
#                      total_cache_lookup_time + \
#                      total_control_setup_time + \
#                      total_main_loop_time + \
#                      total_final_padding_time
#         print(f"Total time: {total_time:.4f} sec")
    
#     if return_seq_info:
#         return DNA_ref_embeddings, seq_info_all, failed_gene_labels
#     else:
#         return DNA_ref_embeddings, failed_gene_labels


def match_gene_pert_mask_dna_embeddings_cached(gene_labels, 
                                        model,
                                        cache=None,
                                        disk_cache=None,
                                        dna_record=None,
                                        all_annots_transcript=None,
                                        promoter_len=200,
                                        perturb_type='CRISPR-i',
                                        gRNAs=None,
                                        return_seq_info=False,
                                        jitter_window=0,
                                        flip_seq_prob=0.0, 
                                        seed=0,
                                        max_batch_size=32,
                                        DNA_emb_dim=64,
                                        DNA_emb_len: int = 1,
                                        drop_mask_prob=0.0,
                                        flip_mask_sign=False,
                                        VERBOSE=False,
                                        ccre_intervals_dict=None,
                                        z_ctrl=None):
    """
    Cache-only version: loads Borzoi embeddings from cache (assumes GPU residence) and passes to adapter.
    
    Optimized for GPU-resident cache:
    1. Ensures all cache promotions (Disk -> Memory) move to GPU immediately.
    2. Generates Control/Dummy keys directly on GPU.
    3. Performs padding and stacking on GPU.
    """
    from torch.nn.utils.rnn import pad_sequence
    import torch
    import time
    
    # ------------------------------------------------------------------
    # 1️⃣  Get device (High Priority)
    # ------------------------------------------------------------------
    # We prioritize the perturbation module as it is the likely interaction point
    if hasattr(model, 'perturbation') and hasattr(model.perturbation, 'pos') and model.perturbation.pos is not None:
        device = model.perturbation.pos.weight.device
    else:
        # Fallback to the first parameter
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure z_ctrl is on the correct device immediately to avoid transfers later
    if z_ctrl is not None and z_ctrl.device != device:
        z_ctrl = z_ctrl.to(device, non_blocking=True)

    if VERBOSE:
        print(f"Using device: {device}")

    # Helper to move tuple data to device efficiently
    def _to_device_tuple(data_tuple, target_device):
        feat, mask, info = data_tuple
        # Only move tensors; pass 'info' (likely dict/None) through
        return (
            feat.to(target_device, non_blocking=True) if isinstance(feat, torch.Tensor) else feat,
            mask.to(target_device, non_blocking=True) if isinstance(mask, torch.Tensor) else mask,
            info
        )

    # ------------------------------------------------------------------
    # 2️⃣  Pre-compute canonical keys and identify control cases
    # ------------------------------------------------------------------
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_precompute_start = time.time()
    
    B = len(gene_labels)
    precomputed_data = [] 
    control_indices = []
    
    for idx, gene_label in enumerate(gene_labels):
        if gene_label == 'control':
            dummy_key = f"__control_{idx}__"
            precomputed_data.append((idx, gene_label, dummy_key, True, None))
            control_indices.append(idx)
        else:
            translated_label = _translate_gene_name(str(gene_label))
            cache_key = _canonical_key(translated_label)
            precomputed_data.append((idx, gene_label, cache_key, False, translated_label))
    
    # ------------------------------------------------------------------
    # 3️⃣  Batch cache lookups (Optimize Disk -> GPU flow)
    # ------------------------------------------------------------------
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_cache_lookup_start = time.time()
    
    non_control_keys = [k for _, _, k, is_ctrl, _ in precomputed_data if not is_ctrl]
    unique_keys = list(set(non_control_keys))
    
    memory_cache_hits = {}
    memory_cache_misses = []
    
    # Check Memory Cache (Expectation: Already on GPU)
    if cache is not None:
        for key in unique_keys:
            if key in cache:
                memory_cache_hits[key] = cache[key]
            else:
                memory_cache_misses.append(key)
    else:
        memory_cache_misses = unique_keys
    
    # Check Disk Cache (Expectation: CPU -> Move to GPU immediately)
    disk_cache_hits = {}
    disk_cache_misses = []
    
    if disk_cache is not None and memory_cache_misses:
        for key in memory_cache_misses:
            if key in disk_cache:
                # LOAD AND MOVE TO GPU IMMEDIATELY
                # This prevents the loop later from dealing with CPU tensors
                cpu_data = disk_cache[key]
                gpu_data = _to_device_tuple(cpu_data, device)
                
                disk_cache_hits[key] = gpu_data
                
                # Update memory cache with the GPU-resident tensor
                if cache is not None:
                    cache[key] = gpu_data 
            else:
                disk_cache_misses.append(key)
    else:
        disk_cache_misses = memory_cache_misses
    
    if disk_cache_misses:
        raise RuntimeError(f"Genes not found in cache: {disk_cache_misses[:5]}...")
    
    # Merge dicts (all values are now guaranteed on GPU)
    cache_data_dict = {**memory_cache_hits, **disk_cache_hits}
    
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_cache_lookup_end = time.time()
        print(f"Cache lookup finished. Memory hits: {len(memory_cache_hits)}, Disk hits: {len(disk_cache_hits)}")

    # ------------------------------------------------------------------
    # 4️⃣  Batch-create control dummy cache entries (Pure GPU)
    # ------------------------------------------------------------------
    if control_indices:
        # Create one dummy template on GPU
        dummy_feat = torch.zeros((1, 1, 1536), dtype=torch.float32, device=device)
        dummy_mask = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
        
        for idx in control_indices:
            dummy_key = f"__control_{idx}__"
            # Assign without .cpu() conversion - keep it hot
            gpu_tuple = (dummy_feat, dummy_mask, None)
            
            if cache is not None:
                cache[dummy_key] = gpu_tuple
            
            # Note: We do NOT write to disk_cache here to avoid unnecessary I/O 
            # or CPU transfers during the hot path.
            
            # Add to local lookup so the main loop finds it
            cache_data_dict[dummy_key] = gpu_tuple
            
    # ------------------------------------------------------------------
    # 5️⃣  Prepare data structures
    # ------------------------------------------------------------------
    DNA_ref_embeddings = []
    seq_info_all = []
    failed_gene_labels = []
    batch_data = [] # List of keys
    batch_indices = [] # List of original indices
    
    # Dummy tensor for adapter signature (reused across batch, stays on GPU)
    dummy_L = 128
    dummy_combined = torch.zeros((5, dummy_L), dtype=torch.float32, device=device)
    
    # ------------------------------------------------------------------
    # 6️⃣  Optimized flush_batch function
    # ------------------------------------------------------------------
    def flush_batch():
        nonlocal DNA_ref_embeddings, DNA_emb_len, batch_data, batch_indices
        if not batch_data:
            return
        
        # 1. Prepare Control Latents (z_ctrl)
        # Use vectorized indexing on GPU, avoiding CPU loops/stacking
        batched_z_ctrl = None
        if z_ctrl is not None:
            # batch_indices is a python list, convert to tensor on device for indexing
            idx_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
            batched_z_ctrl = z_ctrl[idx_tensor] # Slicing happens on GPU
        
        # 2. Model Inference
        # We pass keys; model retrieves tensors. 
        # Since we ensured cache_data_dict and cache[] hold GPU tensors, no transfer occurs here.
        with torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            B_curr = len(batch_data)
            dummy_input = dummy_combined.unsqueeze(0).expand(B_curr, -1, -1)
            
            batched_output = model(
                dummy_input,
                verbose=VERBOSE,
                cache=cache,
                cache_keys=batch_data, # Pass keys
                jitter_window=jitter_window,
                z_ctrl=batched_z_ctrl
            )
        
        # 3. Output Extraction (Keep on GPU)
        # Assuming batched_output is (B, 1, out_dim)
        # We iterate and append. Since batched_output is on GPU, slices are on GPU.
        for i in range(batched_output.shape[0]):
            # _to_LD_per_sample MUST preserve device. 
            # If it doesn't, we should enforce it: .to(device)
            out_LD = _to_LD_per_sample(batched_output[i:i+1]) 
            
            # Dynamic length update
            if DNA_emb_len == 1 and out_LD.size(0) > 1:
                DNA_emb_len = int(out_LD.size(0))
            
            DNA_ref_embeddings.append(out_LD)
        
        # Cleanup
        batch_data = []
        batch_indices = []
    
    # ------------------------------------------------------------------
    # 7️⃣  Main processing loop (Fast iteration)
    # ------------------------------------------------------------------
    for idx, original_label, cache_key, is_control, translated_label in precomputed_data:
        
        # Data existence check (already done in step 3, but we need the specific objects for logic)
        # Note: We don't need to grab the data here, just the key, because the model looks it up.
        
        # For sequence info logic
        cached_tuple = cache_data_dict.get(cache_key)
        
        # Compatibility check for old format (tensor only) vs new format (tuple)
        # If it's old format, we fix it on the fly and ensure GPU residence
        if isinstance(cached_tuple, torch.Tensor):
            # It's just a tensor, wrap it
            cached_LD = cached_tuple.to(device, non_blocking=True)
            mask_dummy = torch.zeros(1, 1, cached_LD.shape[0], device=device)
            cached_tuple = (cached_LD, mask_dummy, None)
            
            # Update caches with the tuple format
            if cache is not None: cache[cache_key] = cached_tuple
            cache_data_dict[cache_key] = cached_tuple

        # Add to batch
        batch_data.append(cache_key)
        batch_indices.append(idx)
        
        # Handle metadata (CPU side, lightweight)
        if return_seq_info:
            ccre_intervals_rel = cached_tuple[2] if (cached_tuple and len(cached_tuple) > 2) else None
            seq_info_all.append([{
                'gene_name': 'control' if is_control else cache_key, 
                'ccre_intervals_rel': ccre_intervals_rel
            }])

        if len(batch_data) >= max_batch_size:
            flush_batch()
    
    # ------------------------------------------------------------------
    # 8️⃣  Final flush & Padding
    # ------------------------------------------------------------------
    if batch_data:
        flush_batch()
    
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_pad_start = time.time()
        
    # Pad sequence works on a list of GPU tensors and returns a GPU tensor
    # No CPU communication happens here.
    DNA_ref_embeddings = pad_sequence(DNA_ref_embeddings, batch_first=True)
    
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print(f"Final padding time: {time.time() - t_pad_start:.4f} sec")
    
    if return_seq_info:
        return DNA_ref_embeddings, seq_info_all, failed_gene_labels
    else:
        return DNA_ref_embeddings, failed_gene_labels


def _extract_borzoi_features_and_mask(model, dna_1hot_tensor, mask_full_tensor, device):
    """
    Extract Borzoi features and downsample mask to match cache format.
    
    Args:
        model: BorzoiPerturbAdapter instance
        dna_1hot_tensor: (1, 4, 524288) float32 tensor on device
        mask_full_tensor: (1, 1, 524288) float32 tensor on device
        device: torch.device
    
    Returns:
        feat: (L128, 1536) float32 tensor (batch dimension removed)
        mask_128: (1, L128) float32 tensor (batch dimension removed)
    """
    # Access Borzoi backbone
    borzoi_model = model.borzoi
    if borzoi_model is None:
        raise ValueError("Borzoi backbone not available in model (load_backbone=False)")
    
    with torch.no_grad():
        # Extract Borzoi features following borzoi_forward_full pattern
        # conv + pool
        feat = borzoi_model.conv_dna(dna_1hot_tensor)
        feat = borzoi_model._max_pool(feat)
        # residual tower
        for block in borzoi_model.res_tower:
            feat = block(feat)
        # unet1 -> (1, L128, C)
        feat = borzoi_model.unet1(feat).transpose(1, 2)
        # transformer in bf16 for speed
        if device.type == 'cuda':
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                feat = feat.to(torch.bfloat16)
                feat = borzoi_model.transformer(feat)
            feat = feat.to(torch.float32)
        else:
            feat = borzoi_model.transformer(feat)
            feat = feat.to(torch.float32)
    
    # Downsample mask to match L128 resolution
    L128 = feat.shape[1]
    mask_128 = F.interpolate(mask_full_tensor, size=L128, mode='nearest')  # (1, 1, L128)
    
    # Remove batch dimension to match cache format
    feat = feat[0]  # (L128, 1536)
    mask_128 = mask_128[0]  # (1, L128)
    
    return feat, mask_128


def match_interval_pert_mask_dna_embeddings(interval_labels, 
                                            model,
                                            cache=None,
                                            dna_record=None,
                                            FIX_WIN=524_288,
                                            jitter_window=0,
                                            return_seq_info=False,
                                            flip_seq_prob=0.0, 
                                            seed=None,
                                            max_batch_size=32,
                                            DNA_emb_dim=64,
                                            drop_mask_prob=0.0,
                                            flip_mask_sign=False, ## distinguish between positive (CRIPSR-a) and negative (CRIPSR-i) mask
                                            VERBOSE=False,
                                            DNAPath=None,
                                            ccre_intervals_dict=None,
                                            z_ctrl=None):
    """
    Like your match_gene_pert_mask_dna_embeddings but driven by intervals.
    interval_labels may include "control" tokens (handled as zeros).
    Accepts labels such as:
      - 'chr1:12345-12567'
      - 'chr1:12345-12567(+)-mybatch-myds'
    """
    from torch.nn.utils.rnn import pad_sequence
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    if DNAPath is None:
        DNAPath = DNA_REFERENCE_PATH
    if not DNAPath and dna_record is None:
        raise ValueError("DNAPath not set and dna_record not provided. Set env DNA_REFERENCE_PATH or pass DNAPath.")


    device = next(model.parameters()).device
    assert flip_seq_prob < 0.5, "flip_seq_prob must be less than 0.5"

    DNA_ref_embeddings = []
    seq_info_all = []
    failed_labels = []

    batch_tensors = []
    batch_labels = []
    batch_indices = []
    
    RC_INDEX = torch.tensor([3, 2, 1, 0], device=device)

    # Timing
    total_seq_gen_time = 0
    total_agg_time = 0
    total_tensor_time = 0
    total_model_time = 0
    total_pad_time = 0
    sample_count = 0

    # ASCII LUT on GPU (A,C,G,T → 0..3 ; other → 4 which we ignore)
    LUT = torch.full((256,), 4, dtype=torch.long, device=device)
    for base, idx in (('A',0), ('C',1), ('G',2), ('T',3)):
        LUT[ord(base)] = idx
        LUT[ord(base.lower())] = idx

    def flush_batch():
        nonlocal batch_tensors, batch_labels, DNA_ref_embeddings, total_model_time, cache, seq_info_all
        if len(batch_tensors) == 0:
            return

        t0 = time.time()
        grad_ctx = torch.no_grad() if cache is not None else torch.enable_grad()
        
        # Batch processing: collect all samples and process together
        with grad_ctx, torch.autocast(device_type='cuda'):
            # Pad variable-length sequences
            from torch.nn.utils.rnn import pad_sequence
            batched_input = pad_sequence(batch_tensors, batch_first=True)  # (B, 5, L_max)
            
            # Extract batched metadata
            # Uppercase cache keys to match cache storage format (cache stores genes in uppercase)
            cache_keys = [label.rsplit('-', 2)[0].upper() for label in batch_labels]
            ccre_intervals_list = []
            for i in range(len(batch_labels)):
                si = seq_info_batch[i] if i < len(seq_info_batch) else None
                if isinstance(si, dict):
                    ccre_intervals_list.append(si.get('ccre_intervals_rel', None))
                elif isinstance(si, list) and len(si) > 0:
                    si_dict = si[0] if isinstance(si[0], dict) else si
                    ccre_intervals_list.append(si_dict.get('ccre_intervals_rel', None))
                else:
                    ccre_intervals_list.append(None)
            
            # Stack z_ctrl if available
            if z_ctrl is not None:
                assert z_ctrl.dim() == 2, "z_ctrl must be a 2D tensor"
                batched_z_ctrl = torch.stack([z_ctrl[batch_indices[i]] for i in range(len(batch_labels))])  # (B, cell_dim)
            else:
                batched_z_ctrl = None
            
            # Single batched model call
            batched_output = model(
                batched_input,
                verbose=VERBOSE,
                cache=cache,
                cache_keys=cache_keys,
                ccre_intervals_rel=ccre_intervals_list,
                jitter_window=jitter_window,
                z_ctrl=batched_z_ctrl
            )  # (B, 1, out_dim)
            
            # Extract per-sample outputs
            for i in range(batched_output.shape[0]):
                out = batched_output[i:i+1].squeeze()  # (out_dim,)
                out = batched_output[i:i+1].reshape(1, -1)  # (1, out_dim)
                
                if cache is not None:
                    DNA_ref_embeddings.append(out.detach())
                else:
                    DNA_ref_embeddings.append(out)
            
            del batched_input, batched_output, batched_z_ctrl
            # Removed torch.cuda.empty_cache() - it's a synchronous operation that adds ~100-500ms per call
            # Only call if memory is actually an issue, and much less frequently (e.g., every 100 batches)

        total_model_time += (time.time() - t0)
        batch_tensors.clear()
        batch_labels.clear()
        batch_indices.clear()
        seq_info_batch.clear()
        
    def gpu_onehot(seq: str, device: torch.device, dtype=torch.bfloat16) -> torch.Tensor:
        seq_bytes  = seq.encode('ascii')
        bytes_gpu  = torch.frombuffer(seq_bytes, dtype=torch.uint8).to(device)
        idx = LUT[bytes_gpu.long()]   # (L,)
        L   = idx.numel()
        one_hot = torch.zeros((4, L), dtype=dtype, device=device)
        mask = (idx < 4)
        if mask.any():
            valid_idx = idx[mask]
            one_hot[:, mask] = F.one_hot(valid_idx, num_classes=4).T.to(dtype=dtype)
        return one_hot

    def process_batch_tensors(dna_sequences, mask_sequences, directions, flip_flags):
        combined_tensors = []
        # build one-hots
        one_hots = [gpu_onehot(s, device, dtype=torch.float32) for s in dna_sequences]
        for i in range(len(one_hots)):
            oh = one_hots[i]
            m  = mask_sequences[i]
            direction = directions[i]
            do_flip   = flip_flags[i]

            mask = torch.as_tensor(m, dtype=torch.float32, device=device).unsqueeze(0)  # (1,L)

            # apply RC flip dependent on strand and sampling
            if (direction == '+' and do_flip) or (direction == '-' and not do_flip):
                oh = oh.flip(1).index_select(0, RC_INDEX)
                mask = mask.flip(1)

            combined = torch.cat([oh, mask], dim=0)  # (5, L)
            combined_tensors.append(combined)
        return combined_tensors

    # Pre-generate flip decisions
    flip_decisions = torch.rand(len(interval_labels), device=device) < flip_seq_prob

    dna_sequences_batch, mask_sequences_batch = [], []
    directions_batch,  flip_flags_batch = [], []
    seq_info_batch = []
    ccre_count = []  # Track cCRE counts for logging

    for idx, label in enumerate(interval_labels):
        if label == 'control':
            with torch.no_grad():
                dummy = torch.zeros((1, DNA_emb_dim), dtype=torch.bfloat16, device=device)
            DNA_ref_embeddings.append(dummy)
            continue

        try:
            # print(f'try label: {label}')
            chrom, start, end, strand, batch_name, dataset_name = parse_interval_label(label)
            # print(f'Processing {chrom}:{start}-{end}({strand})')
        except Exception as e:
            failed_labels.append(label)
            # keep shape contract
            dummy = torch.zeros((1, DNA_emb_dim), dtype=torch.bfloat16, device=device)
            DNA_ref_embeddings.append(dummy)
            continue

        # cache?
        core_key = f"{chrom}:{start}-{end}({strand})"
        if cache is not None and core_key in cache:
            cached = cache[core_key]
            # Check if it's the new format (tuple) or old format (tensor)
            if isinstance(cached, tuple):
                # New format: tuple of (feat_full, mask_full, ccre_intervals_cached)
                # Model will handle caching internally via cache_key parameter
                # Skip ALL sequence generation, mask generation, and one-hot conversion
                try:
                    # Extract ccre_intervals_rel directly from cache (already there!)
                    feat_full, mask_full, ccre_intervals_rel = cached
                    
                    # Get minimal info only if return_seq_info is needed
                    if return_seq_info:
                        # Skip expensive sequence generation - create minimal seq_info from cached data
                        seq_info = {
                            'chrom': chrom,
                            'interval_start': start,
                            'interval_end': end,
                            'ccre_intervals_rel': ccre_intervals_rel,  # Already from cache
                            'Direction': strand,
                            'Info': f'{chrom}:{start}-{end}({strand})'
                        }
                        seq_info_all.append(seq_info)
                    else:
                        seq_info = None
                    
                    # Track cCRE count for logging
                    num_ccres = len(ccre_intervals_rel) if ccre_intervals_rel else 0
                    ccre_count.append(num_ccres)
                    
                    # Create minimal dummy input tensor (model doesn't use it on cache hit, just for signature)
                    # Use a small fixed size since model will use cached feat_full anyway
                    dummy_L = 4096  # Fixed size, model will use cached feat_full anyway
                    dummy_dna = torch.zeros((4, dummy_L), dtype=torch.float32, device=device)
                    dummy_mask = torch.zeros((1, dummy_L), dtype=torch.float32, device=device)
                    combined = torch.cat([dummy_dna, dummy_mask], dim=0).unsqueeze(0)  # (1, 5, L)
                    
                    if z_ctrl is not None:
                        assert z_ctrl.dim() == 2
                        z_ctrl_i = z_ctrl[idx:idx+1]  # shape (1, D)
                    else:
                        z_ctrl_i = None
                        
                    # Process immediately via model with cache - model handles everything internally
                    with torch.no_grad(), torch.autocast(device_type='cuda'):
                        single_output = model(combined, verbose=VERBOSE, cache=cache, cache_keys=[core_key], 
                                            ccre_intervals_rel=[ccre_intervals_rel], jitter_window=jitter_window, z_ctrl=z_ctrl_i)
                        out = single_output.squeeze()
                        out = single_output.reshape(1, -1)
                        DNA_ref_embeddings.append(out.detach())
                    
                    sample_count += 1
                    
                    del combined, dummy_dna, dummy_mask, single_output, out
                    # Removed torch.cuda.empty_cache() - it's a synchronous operation that adds ~100-500ms per call
                    continue  # Skip normal batching flow
                except Exception as e:
                    # Fall back to normal flow if optimization fails
                    if VERBOSE:
                        print(f"Cache optimization failed for {core_key}: {e}, falling back to normal flow")
                        import traceback
                        traceback.print_exc()
                    pass  # Continue to normal flow
            else:
                # Old format: use cached embedding directly
                DNA_ref_embeddings.append(cached.to(device, non_blocking=True))
                continue

        # sequence gen
        t0 = time.time()
        try:
            if VERBOSE:
                print(f'Generating sequence for {label} with jitter {jitter_window}; chrom: {chrom}, start: {start}, end: {end}, strand: {strand}')
            seq_info = gen_interval_centered_sequence(chrom, start, end, strand=strand,
                                                      DNAPath=DNAPath, FIX_WIN=FIX_WIN,
                                                      dna_record=dna_record,
                                                      jitter_window=jitter_window,
                                                      ccre_intervals_dict=ccre_intervals_dict)
            # print(f'seq info: {seq_info}')
        except Exception as e:
            failed_labels.append(label)
            dummy = torch.zeros((1, DNA_emb_dim), dtype=torch.bfloat16, device=device)
            DNA_ref_embeddings.append(dummy)
            continue
        total_seq_gen_time += (time.time() - t0)

        # mask aggregation
        t1 = time.time()
        DNA_seq_ref, DNA_seq_mask = aggregate_mask_from_interval(seq_info, FIX_WIN=FIX_WIN)
        ccre_intervals_rel = seq_info.get('ccre_intervals_rel', [])
        num_ccres = len(ccre_intervals_rel)
        if VERBOSE:
            print(f'Original seq: {DNA_seq_ref[:50]}...{DNA_seq_ref[-50:]}')
            print(f'Mask number of 1s: {DNA_seq_mask.sum()}')
            print(f'Number of cCRE intervals in window: {num_ccres}')
            if num_ccres == 0:
                warnings.warn(f'No cCRE intervals found for {label}')

        # optional drop/flip-mask-sign
        if random.random() < drop_mask_prob:
            DNA_seq_mask = np.zeros_like(DNA_seq_mask)
        if flip_mask_sign:
            DNA_seq_mask = -DNA_seq_mask

        total_agg_time += (time.time() - t1)

        # Extract Borzoi features and downsampled mask for cache-compatible format
        try:
            # Convert sequence to one-hot encoding
            dna_1hot = dna.dna_1hot(DNA_seq_ref).T  # (4, 524288) numpy array
            dna_1hot_tensor = torch.tensor(dna_1hot, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, 524288)
            
            # Convert mask to tensor
            mask_full_tensor = torch.tensor(DNA_seq_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 524288)
            
            # Extract Borzoi features and downsampled mask
            feat, mask_128 = _extract_borzoi_features_and_mask(model, dna_1hot_tensor, mask_full_tensor, device)
            
            # Cache the extracted features if cache is provided
            if cache is not None:
                cache[core_key] = (feat.detach().cpu(), mask_128.detach().cpu(), ccre_intervals_rel)
                if VERBOSE:
                    print(f'Cached Borzoi features for {core_key}: feat shape {feat.shape}, mask shape {mask_128.shape}')
            
            # Clean up GPU tensors
            del dna_1hot_tensor, mask_full_tensor, feat, mask_128
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            # If feature extraction fails, log but continue with normal flow
            if VERBOSE:
                print(f'Warning: Failed to extract Borzoi features for {core_key}: {e}')
                import traceback
                traceback.print_exc()

        # batch storage
        dna_sequences_batch.append(DNA_seq_ref)
        mask_sequences_batch.append(DNA_seq_mask)
        directions_batch.append(seq_info['Direction'])
        flip_flags_batch.append(bool(flip_decisions[idx].item()))
        batch_labels.append(label)
        batch_indices.append(idx)
        seq_info_all.append(seq_info)
        sample_count += 1
        
        # Track cCRE count for logging
        if 'ccre_count' not in locals():
            ccre_count = []
        ccre_count.append(num_ccres)

        # memory-savvy micro-batching (same heuristic as your original)
        effective_bs = min(max_batch_size, 16)  # Increased from 8 to 16 for better GPU utilization
        if len(dna_sequences_batch) >= effective_bs:
            t2 = time.time()
            combined = process_batch_tensors(dna_sequences_batch, mask_sequences_batch,
                                             directions_batch, flip_flags_batch)
            batch_tensors.extend(combined)
            total_tensor_time += (time.time() - t2)

            flush_batch()
            dna_sequences_batch.clear()
            mask_sequences_batch.clear()
            directions_batch.clear()
            flip_flags_batch.clear()
            batch_indices.clear()
            
    # final flush
    if dna_sequences_batch:
        t2 = time.time()
        combined = process_batch_tensors(dna_sequences_batch, mask_sequences_batch,
                                         directions_batch, flip_flags_batch)
        batch_tensors.extend(combined)
        total_tensor_time += (time.time() - t2)
        flush_batch()

    # pad output
    t3 = time.time()
    from torch.nn.utils.rnn import pad_sequence
    DNA_ref_embeddings = pad_sequence(DNA_ref_embeddings, batch_first=True)
    if VERBOSE:
        print(f'Final DNA_ref_embeddings is: {DNA_ref_embeddings}')
    total_pad_time += (time.time() - t3)

    if VERBOSE:
        print(f"\n=== TIMING (interval) ===")
        if sample_count > 0:
            print(f"samples: {sample_count}")
            print(f"sequence gen: {total_seq_gen_time:.4f}s ({total_seq_gen_time/max(sample_count,1):.4f}/sample)")
            print(f"mask agg:     {total_agg_time:.4f}s ({total_agg_time/max(sample_count,1):.4f}/sample)")
            print(f"tensor ops:   {total_tensor_time:.4f}s ({total_tensor_time/max(sample_count,1):.4f}/sample)")
            print(f"model infer:  {total_model_time:.4f}s ({total_model_time/max(sample_count,1):.4f}/sample)")
            print(f"pad:          {total_pad_time:.4f}s")
            tot = total_seq_gen_time + total_agg_time + total_tensor_time + total_model_time + total_pad_time
            print(f"total:        {tot:.4f}s")
            if len(ccre_count) > 0:
                print(f"cCRE intervals: mean={np.mean(ccre_count):.2f}, min={min(ccre_count)}, max={max(ccre_count)}, total={sum(ccre_count)}")

    return (DNA_ref_embeddings, seq_info_all, failed_labels) if return_seq_info else (DNA_ref_embeddings, failed_labels)



def match_interval_pert_mask_dna_embeddings_cached(interval_labels, 
                                            model,
                                            cache=None,
                                            disk_cache=None,
                                            dna_record=None,
                                            FIX_WIN=524_288,
                                            jitter_window=0,
                                            return_seq_info=False,
                                            flip_seq_prob=0.0, 
                                            seed=None,
                                            max_batch_size=32,
                                            DNA_emb_dim=64,
                                            drop_mask_prob=0.0,
                                            flip_mask_sign=False,
                                            VERBOSE=False,
                                            DNAPath=None,
                                            ccre_intervals_dict=None,
                                            z_ctrl=None):
    """
    Cache-only version for intervals: loads Borzoi embeddings from cache (memory or disk) and passes to adapter.
    No DNA sequence generation or one-hot encoding.
    
    Args:
        cache: Memory cache (LRUCache) - checked first
        disk_cache: Disk cache (DiskDNACache) - checked if memory cache misses
        model: BorzoiPerturbAdapter with load_backbone=False
    """
    from torch.nn.utils.rnn import pad_sequence
    import torch
    import time
    import numpy as np

    if DNAPath is None:
        DNAPath = DNA_REFERENCE_PATH
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = next(model.parameters()).device
    
    DNA_ref_embeddings = []
    seq_info_all = []
    failed_labels = []
    
    # Batch storage for cache keys and indices only (model looks up data from cache)
    batch_labels = []
    batch_indices = []
    
    # Timing
    total_cache_time = 0
    total_model_time = 0
    cache_hit_count = 0
    cache_miss_count = 0
    
    # Dummy tensor for adapter signature (reused across batch)
    dummy_L = 128
    dummy_combined = torch.zeros((5, dummy_L), dtype=torch.float32, device=device)
    
    def flush_batch():
        nonlocal DNA_ref_embeddings, total_model_time
        if len(batch_labels) == 0:
            return
        
        if VERBOSE:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_flush_start = time.time()
        
        t0 = time.time()
        with torch.autocast(device_type='cuda'):
            # Construct dummy input (adapter won't use it, just for signature)
            B = len(batch_labels)
            dummy_input = dummy_combined.unsqueeze(0).expand(B, -1, -1)  # (B, 5, dummy_L)
            
            # Extract cache keys (use core interval format, model will look up cached data using these keys)
            cache_keys = []
            for label in batch_labels:
                # Handle dummy keys for control/failed cases
                if isinstance(label, str) and (label.startswith("__control_") or label.startswith("__failed_")):
                    cache_keys.append(label)  # Use dummy key directly
                else:
                    try:
                        chrom, start, end, strand, _, _ = parse_interval_label(label)
                        core_key = f"{chrom}:{start}-{end}({strand})"
                        cache_keys.append(core_key)
                    except:
                        cache_keys.append(None)
            
            # Stack z_ctrl if available
            if z_ctrl is not None:
                assert z_ctrl.dim() == 2, "z_ctrl must be a 2D tensor"
                batched_z_ctrl = torch.stack([z_ctrl[batch_indices[i]] for i in range(len(batch_labels))])
            else:
                batched_z_ctrl = None
            
            # Call adapter with cache - it will look up cached (feat, mask, ccre) tuples using cache_keys
            # Note: ccre_intervals_rel parameter is removed - model uses ccre from cached tuple
            batched_output = model(
                dummy_input,
                verbose=VERBOSE,
                cache=cache,
                cache_keys=cache_keys,
                jitter_window=jitter_window,
                z_ctrl=batched_z_ctrl
            )  # (B, 1, out_dim)
            
            # Extract per-sample outputs
            for i in range(batched_output.shape[0]):
                out = batched_output[i:i+1].reshape(1, -1)  # (1, out_dim)
                DNA_ref_embeddings.append(out)
            
            del dummy_input, batched_output, batched_z_ctrl
            # Removed torch.cuda.empty_cache() - it's a synchronous operation that adds ~100-500ms per call
            # Only call if memory is actually an issue, and much less frequently (e.g., every 100 batches)
        
        total_model_time += (time.time() - t0)
        
        if VERBOSE:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_flush_end = time.time()
            print(f"  flush_batch() time: {t_flush_end - t_flush_start:.4f} sec (batch size: {len(batch_labels)})")
        
        batch_labels.clear()
        batch_indices.clear()
    
    for idx, label in enumerate(interval_labels):
        if label == 'control':
            # Route 'control' through adapter to keep DNA_model parameters in graph
            # Create a minimal dummy feature tuple and add to batch
            dummy_key = f"__control_{idx}__"
            dummy_feat = torch.zeros((1, 1, 1536), dtype=torch.float32, device=device)  # (1, L=1, C=1536)
            dummy_mask = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)  # (1, 1, L=1)
            # Store in cache temporarily so adapter can find it
            if cache is not None:
                cache[dummy_key] = (dummy_feat, dummy_mask, None)
            elif disk_cache is not None:
                disk_cache[dummy_key] = (dummy_feat.cpu(), dummy_mask.cpu(), None)
            batch_labels.append(dummy_key)  # Use dummy key for cache lookup
            batch_indices.append(idx)
            if len(batch_labels) >= min(max_batch_size, 16):  # Increased from 8 to 16 for better GPU utilization
                flush_batch()
            continue
        
        try:
            chrom, start, end, strand, batch_name, dataset_name = parse_interval_label(label)
        except Exception as e:
            failed_labels.append(label)
            # Route failed labels through adapter (same as control)
            dummy_key = f"__failed_{idx}__"
            dummy_feat = torch.zeros((1, 1, 1536), dtype=torch.float32, device=device)
            dummy_mask = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
            if cache is not None:
                cache[dummy_key] = (dummy_feat, dummy_mask, None)
            elif disk_cache is not None:
                disk_cache[dummy_key] = (dummy_feat.cpu(), dummy_mask.cpu(), None)
            batch_labels.append(dummy_key)
            batch_indices.append(idx)
            if len(batch_labels) >= min(max_batch_size, 16):  # Increased from 8 to 16 for better GPU utilization
                flush_batch()
            continue
        
        # Two-tier cache lookup: memory cache first, then disk cache
        core_key = f"{chrom}:{start}-{end}({strand})"
        t0 = time.time()
        cached_data = None
        
        # Check memory cache first
        if cache is not None and core_key in cache:
            cached_data = cache[core_key]
            if VERBOSE:
                print(f"Memory cache hit for {core_key}")
        # Check disk cache if memory miss
        elif disk_cache is not None and core_key in disk_cache:
            cached_data = disk_cache[core_key]
            # Promote to memory cache if available
            if cache is not None:
                cache[core_key] = cached_data
            if VERBOSE:
                print(f"Disk cache hit for {core_key}")
        
        total_cache_time += (time.time() - t0)
        
        if cached_data is None:
            # Cache miss - raise error in cache-only mode
            raise RuntimeError(f"Interval '{core_key}' not found in cache or disk cache. "
                             f"Please precompute embeddings using DNA_diff_cache_prepare.py")
        
        # Handle different cache formats
        if isinstance(cached_data, tuple) and len(cached_data) == 3:
            # New format: (feat, mask, ccre_intervals)
            # Model will look up this tuple from cache using cache_keys, so we don't need to store it here
            _, _, ccre_intervals_rel = cached_data
            
            # Add to batch (only need labels and indices for cache key lookup and z_ctrl indexing)
            batch_labels.append(label)
            batch_indices.append(idx)
            
            cache_hit_count += 1
            
            # Build seq_info if requested (extract ccre from cached tuple)
            if return_seq_info:
                seq_info = {
                    'chrom': chrom,
                    'interval_start': start,
                    'interval_end': end,
                    'ccre_intervals_rel': ccre_intervals_rel,
                    'Direction': strand,
                    'Info': f'{chrom}:{start}-{end}({strand})'
                }
                seq_info_all.append(seq_info)
            
            # Flush if batch is full
            if len(batch_labels) >= min(max_batch_size, 16):  # Increased from 8 to 16 for better GPU utilization
                flush_batch()
        else:
            # Old format: final embedding tensor (already processed by adapter)
            # Repackage as tuple so adapter still runs (ensures DNA_model parameters receive grads)
            cached_LD = cached_data if torch.is_tensor(cached_data) else torch.as_tensor(cached_data)
            # Ensure it's on the right device
            if cached_LD.device != device:
                cached_LD = cached_LD.to(device, non_blocking=True)
            
            # Determine shape: could be (L, D) or (1, L, D) or (D,)
            if cached_LD.dim() == 1:
                # (D,) -> treat as single token embedding
                L_ = 1
                cached_LD = cached_LD.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            elif cached_LD.dim() == 2:
                # (L, D) -> add batch dimension
                L_ = cached_LD.shape[0]
                cached_LD = cached_LD.unsqueeze(0)  # (1, L, D)
            else:
                # (1, L, D) or (B, L, D) -> use as is
                L_ = cached_LD.shape[1] if cached_LD.dim() == 3 else cached_LD.shape[0]
                if cached_LD.dim() == 3 and cached_LD.shape[0] > 1:
                    # Multiple batches, take first
                    cached_LD = cached_LD[0:1]
            
            # Create dummy mask matching the feature length
            mask_dummy = torch.zeros(1, 1, L_, device=device, dtype=torch.float32)
            
            # Repackage and store back in cache so adapter can process it
            if cache is not None:
                cache[core_key] = (cached_LD.to(device, non_blocking=True), mask_dummy, None)
            elif disk_cache is not None:
                cache[core_key] = (cached_LD.cpu(), mask_dummy.cpu(), None)
            
            # Add to batch so flush_batch() runs the adapter
            batch_labels.append(label)
            batch_indices.append(idx)
            cache_hit_count += 1
            
            if return_seq_info:
                seq_info = {
                    'chrom': chrom,
                    'interval_start': start,
                    'interval_end': end,
                    'ccre_intervals_rel': None,  # Old format doesn't have cCRE info
                    'Direction': strand,
                    'Info': f'{chrom}:{start}-{end}({strand})'
                }
                seq_info_all.append(seq_info)
            
            # Flush if batch is full
            if len(batch_labels) >= min(max_batch_size, 16):  # Increased from 8 to 16 for better GPU utilization
                flush_batch()
    
    # Final flush
    if batch_labels:
        flush_batch()
    
    # Time padding
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_padding_start = time.time()
    
    # Pad output
    DNA_ref_embeddings = pad_sequence(DNA_ref_embeddings, batch_first=True)
    
    if VERBOSE:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_padding_end = time.time()
        print(f"\n=== CACHED MATCHING TIMING (interval) ===")
        print(f"Total cache lookup time: {total_cache_time:.4f} sec")
        print(f"Total model inference time (flush_batch): {total_model_time:.4f} sec")
        print(f"Padding time: {t_padding_end - t_padding_start:.4f} sec")
        print(f"\n=== CACHED MATCHING SUMMARY (interval) ===")
        print(f"Total samples: {len(interval_labels)}")
        print(f"Cache hits: {cache_hit_count}")
        print(f"Cache misses: {cache_miss_count}")
    
    return (DNA_ref_embeddings, seq_info_all, failed_labels) if return_seq_info else (DNA_ref_embeddings, failed_labels)



def parse_interval_label(label: str):
    """
    Accepts:
      - 'chr1:12345-12567'
      - 'chr1:12345-12567(+)'
      - 'chr1:12345-12567(-)'
      - 'chr1:12345-12567-batch-dataset'
      - 'chr1:12345-12567(+)-batch-dataset'
      - 'chr1:12345-12567(-)-batch-dataset'

    Returns:
      chrom, start, end, strand, batch_name, dataset_name
    """
    import re

    # Updated regex to capture optional strand in parentheses: (+), (-), or nothing
    m = re.match(
        r"^(chr[0-9A-Za-z]+):(\d[\d,]*)-(\d[\d,]*)(?:\(([+-])\))?(?:-([^-]+)-([^-]+))?$",
        label,
    )
    if not m:
        raise ValueError(f"Interval label parsing failed for '{label}'")

    chrom = m.group(1)
    start = int(m.group(2).replace(",", ""))
    end = int(m.group(3).replace(",", ""))
    strand = m.group(4) if m.group(4) else "+"  # Extract strand or default to "+"
    batch_name = m.group(5)
    dataset_name = m.group(6)

    if end < start:
        raise ValueError(f"Interval has end < start in '{label}'")

    return chrom, start, end, strand, batch_name, dataset_name




def gen_interval_centered_sequence(chrom, start, end,
                                   strand='+',
                                   DNAPath=None,
                                   FIX_WIN=524_288,
                                   dna_record=None,
                                   jitter_window=0,
                                   ccre_intervals_dict=None):
    """
    Center a FIX_WIN window at mid = floor((start+end)/2),
    apply optional jitter (bounded to stay in-chrom),
    return sequence + bookkeeping for mapping the interval into window coordinates.

    Args:
        ccre_intervals_dict: dict {chrom: [(start, end), ...]} from load_ccres_bed()
    
    Returns dict with keys:
      seq, chrom, proxy_start, proxy_end,
      interval_start, interval_end,
      interval_rel_start, interval_rel_end,
      ccre_intervals_rel, Direction, Info
    """
    if DNAPath is None:
        DNAPath = DNA_REFERENCE_PATH
    if dna_record is not None:
        record = dna_record[chrom]
    else:
        if not DNAPath:
            raise ValueError("DNAPath not set and dna_record not provided. Set env DNA_REFERENCE_PATH or pass DNAPath.")
        record = next(SeqIO.parse(f'{DNAPath}/{chrom}.fasta', 'fasta'))

    Lgen = len(record.seq)
    mid = (start + end) // 2

    left_len  = FIX_WIN // 2
    right_len = FIX_WIN - left_len - 1  # keep total FIX_WIN with a central base

    # adjust near chromosome edges
    if mid - left_len <= 0:
        residual = left_len - mid + 1
        left_len  = mid - 1
        right_len += residual
    elif mid + right_len > Lgen:
        residual  = mid + right_len - Lgen
        right_len = Lgen - mid
        left_len += residual

    assert left_len + right_len + 1 == FIX_WIN, \
        f'actual {left_len + right_len + 1} vs expected {FIX_WIN}'

    # jitter bounds so the window remains valid
    min_shift = 1 - (mid - left_len)
    max_shift = Lgen - (mid + right_len)
    shift_low  = max(-jitter_window, min_shift)
    shift_high = min( jitter_window, max_shift)
    shifting = random.randint(shift_low, shift_high) if jitter_window > 0 else 0

    proxy_start = mid - left_len + shifting
    proxy_end   = mid + right_len + shifting
    seq = str(record.seq[(proxy_start - 1):proxy_end]).upper()

    assert proxy_end - proxy_start + 1 == FIX_WIN, \
        f'actual {proxy_end - proxy_start + 1} vs expected {FIX_WIN}'

    # map interval into window-relative coordinates
    rel_start = max(0, start - proxy_start)
    rel_end   = min(FIX_WIN - 1, end - proxy_start)

    # It's possible the provided interval partially falls outside the extracted window
    # after jitter/edge corrections; clamp and warn if so:
    if start < proxy_start or end > proxy_end:
        warnings.warn(f"Interval [{start},{end}] extends beyond window "
                      f"[{proxy_start},{proxy_end}] on {chrom}; mask will be clipped.")

    # Find overlapping cCREs
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

    return {
        'seq': seq,
        'chrom': chrom,
        'proxy_start': proxy_start,
        'proxy_end': proxy_end,
        'interval_start': start,
        'interval_end': end,
        'interval_rel_start': int(rel_start),
        'interval_rel_end': int(rel_end),
        'ccre_intervals_rel': ccre_intervals_rel,  # List of (rel_start, rel_end) tuples
        'Direction': strand,
        'Info': f'{chrom}:{start}-{end}({strand})'
    }


def aggregate_mask_from_interval(seq_info, FIX_WIN=524_288):
    """
    Construct a (FIX_WIN,) mask with 1s for [rel_start, rel_end] inclusive.
    """
    mask = np.zeros((FIX_WIN,), dtype=np.float32)
    a = int(seq_info['interval_rel_start'])
    b = int(seq_info['interval_rel_end'])
    if b >= 0 and a <= (FIX_WIN - 1):
        a_clamp = max(0, a)
        b_clamp = min(FIX_WIN - 1, b)
        if b_clamp >= a_clamp:
            mask[a_clamp:b_clamp + 1] = 1.0
    return seq_info['seq'], mask