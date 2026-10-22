"""
DDP setup and utilities: Distributed Data Parallel initialization and helpers.

This module contains:
- setup_ddp: Initialize DDP process group
- cleanup: Cleanup DDP process group
- find_free_port: Find available port for DDP
- gather_numpy_arrays_across_ranks: Gather numpy arrays across DDP ranks
"""

import socket
import os
import torch
import torch.distributed as dist
import numpy as np
from typing import List


def setup_ddp(rank, world_size):
    """
    Initialize DDP using IPv4 loopback for single-node multi-GPU to avoid IPv6/localhost issues.
    Falls back to env:// if MASTER_ADDR/PORT are already properly set by the launcher (e.g., torchrun).
    """
    torch.cuda.set_device(rank)
    
    # Check if process group is already initialized (e.g., by torchrun)
    if dist.is_initialized():
        return
    
    # Check if torchrun environment variables are set
    # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    # mp.spawn only sets MASTER_ADDR/MASTER_PORT (we set them in main()), but NOT RANK
    has_rank_env = os.environ.get("RANK") is not None
    has_world_size_env = os.environ.get("WORLD_SIZE") is not None
    
    if has_rank_env and has_world_size_env:
        # Use environment variables (torchrun sets these automatically)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    else:
        # Manual initialization for mp.spawn fallback
        # mp.spawn doesn't set RANK/WORLD_SIZE, so we use explicit TCP init with provided rank/world_size
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )


def cleanup():
    dist.destroy_process_group()


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def ddp_sum(x):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def gather_numpy_arrays_across_ranks(arrays_list, device):
    """Gather list of numpy arrays from all ranks and concatenate them."""
    if not arrays_list:
        return np.array([])
    
    # Concatenate arrays from this rank
    concatenated = np.concatenate(arrays_list, axis=0)
    
    # Convert to tensor for gathering
    tensor = torch.from_numpy(concatenated).to(device)
    
    # Gather from all ranks (requires gather_tensor_across_ranks to be defined elsewhere)
    # For now, we'll use a simple all_gather approach
    gathered_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_list, tensor)
    
    # Concatenate and convert back to numpy
    gathered_tensor = torch.cat(gathered_list, dim=0)
    return gathered_tensor.cpu().numpy()


def toggle_grad(params, enable: bool = True):
    """Enable/disable grads in-place (fast: no new tensors are created)."""
    for p in params:
        p.requires_grad_(enable)


def flatten_dict(d, parent_key='', sep='/'):
    """Flatten nested dicts like {'train': {'lr': 1e-3}} -> {'train/lr': 1e-3}"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

