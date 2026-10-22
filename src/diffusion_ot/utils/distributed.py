"""
Distributed training utilities: DDP-aware tensor and list gathering.

This module contains:
- gather_and_concatenate_tensors: Gather and concatenate tensors across DDP ranks
- gather_and_concatenate_list: Gather and concatenate lists across DDP ranks
"""

import torch
import torch.distributed as dist
from typing import List, Union


def gather_and_concatenate_tensors(tensor_list):
    """
    Gathers lists of tensors from all processes and concatenates them.
    Handles uneven tensor sizes across processes.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return torch.cat(tensor_list, dim=0)

    gathered_lists = [None] * world_size

    dist.all_gather_object(gathered_lists, tensor_list)

    all_tensors = [tensor for sublist in gathered_lists for tensor in sublist]
    
    return torch.cat(all_tensors, dim=0)


def gather_and_concatenate_list(data_list):
    """
    Gathers lists of data from all processes and concatenates them.
    Handles uneven data sizes across processes.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return data_list

    gathered_lists = [None] * world_size

    dist.all_gather_object(gathered_lists, data_list)

    all_data = [item for sublist in gathered_lists for item in sublist]
    
    return all_data




# NOTE: parse_interval_label, gen_interval_centered_sequence, aggregate_mask_from_interval, 
# and match_interval_pert_mask_dna_embeddings have been migrated to diff_helper.py
