"""
Cache utilities: LRU cache for DNA embeddings and other cached data.

This module contains:
- LRUCache: Least Recently Used cache with statistics tracking
"""

from collections import OrderedDict
from typing import Optional, Dict, Any
import os
import pickle
import torch
import numpy as np
import hashlib
import re


class LRUCache:
    """
    Least Recently Used (LRU) Cache with statistics tracking.
    
    Tracks cache hits, misses, inserts, updates, and evictions for monitoring.
    Supports GPU-native storage when device is specified.
    """
    def __init__(self, max_size=256, device=None):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.device = device  # Target device for GPU-native storage (None = CPU)
        # Cumulative counters
        self.hits = 0
        self.misses = 0
        self.inserts = 0
        self.updates = 0
        self.evictions = 0
        # Step counters (reset each logging step)
        self.step_hits = 0
        self.step_misses = 0
        self.step_inserts = 0
        self.step_updates = 0
        self.step_evictions = 0

    def _ensure_gpu_format(self, value):
        """
        Ensure tensor is in correct GPU format (bfloat16, on target device).
        Returns value unchanged if not a tensor or device is None.
        """
        if self.device is None or not isinstance(value, torch.Tensor):
            return value
        
        # Convert to GPU if on CPU
        # Move to target device if not already on it (handles CPU->GPU and GPU->GPU transfers)
        if value.device != self.device:
            value = value.to(self.device, non_blocking=True)
        
        # Ensure correct dtype (bfloat16 for GPU)
        if value.device.type == 'cuda' and value.dtype != torch.bfloat16:
            value = value.to(torch.bfloat16)
        
        return value

    def get(self, key):
        """Get value from cache, updating access order."""
        if key not in self.cache:
            # miss
            self.misses += 1
            self.step_misses += 1
            return None
        # Move key to end to mark it as recently used
        self.cache.move_to_end(key)
        # hit
        self.hits += 1
        self.step_hits += 1
        value = self.cache[key]
        # Ensure GPU format if device is specified
        return self._ensure_gpu_format(value)

    def set(self, key, value):
        """Set value in cache, updating access order."""
        existed = key in self.cache
        if existed:
            # Update and mark as recently used
            self.cache.move_to_end(key)
            self.updates += 1
            self.step_updates += 1
        else:
            self.inserts += 1
            self.step_inserts += 1
        # Ensure GPU format before storing
        self.cache[key] = self._ensure_gpu_format(value)
        if len(self.cache) > self.max_size:
            # Remove least recently used item
            try:
                self.cache.popitem(last=False)
            finally:
                self.evictions += 1
                self.step_evictions += 1

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    @property
    def size(self):
        """Current cache size."""
        return len(self.cache)

    def reset_step_counters(self):
        """Reset step-level counters (for periodic logging)."""
        self.step_hits = 0
        self.step_misses = 0
        self.step_inserts = 0
        self.step_updates = 0
        self.step_evictions = 0

    def stats(self, reset_step: bool = False) -> Dict[str, Any]:
        """
        Return counters for monitoring/logging.
        
        Args:
            reset_step: If True, reset step counters after returning stats
            
        Returns:
            Dictionary with cache statistics
        """
        totals = {
            "hits_total": self.hits,
            "misses_total": self.misses,
            "inserts_total": self.inserts,
            "updates_total": self.updates,
            "evictions_total": self.evictions,
        }
        step = {
            "hits_step": self.step_hits,
            "misses_step": self.step_misses,
            "inserts_step": self.step_inserts,
            "updates_step": self.step_updates,
            "evictions_step": self.step_evictions,
        }
        out = {
            **totals,
            **step,
            "size": len(self.cache),
            "hit_rate_step": (self.step_hits / (self.step_hits + self.step_misses)) if (self.step_hits + self.step_misses) > 0 else float("nan"),
            "hit_rate_total": (self.hits / (self.hits + self.misses)) if (self.hits + self.misses) > 0 else float("nan"),
        }
        if reset_step:
            self.reset_step_counters()
        return out


def _sanitize_filename(key):
    """
    Sanitize a key (gene symbol) to be filesystem-safe.
    Replaces invalid characters with underscores and ensures the filename is not too long.
    """
    # Replace invalid filesystem characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(key))
    # Remove leading/trailing dots and spaces (Windows doesn't allow these)
    sanitized = sanitized.strip('. ')
    # Limit length to avoid filesystem issues (max 255 chars for most filesystems)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unknown"
    return sanitized


def _pool_embeddings_4chunks(tensor, k=8, device=None):
    """
    Average pool embeddings by grouping every 4 consecutive chunks.
    
    Args:
        tensor: Tensor of shape (L, C) where L is sequence length, C is embedding dim
        device: Optional device hint (for GPU operations, inferred from tensor if None)
    
    Returns:
        Pooled tensor of shape (ceil(L/4), C)
    
    Examples:
        L=8: (8, C) -> (2, C)  [8//4 = 2]
        L=10: (10, C) -> (3, C)  [10//4 = 2, remainder 2 -> 1, total 3]
        L=3: (3, C) -> (1, C)  [3 < 4, average all]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    
    L, C = tensor.shape
    
    # Edge case: empty tensor
    if L == 0:
        return tensor
    
    # Edge case: L < 4, just average all chunks
    if L < k:
        return tensor.mean(dim=0, keepdim=True)  # (1, C)
    
    # Main part: process chunks divisible by 4
    main_len = (L // k) * k
    
    if main_len > 0:
        main_part = tensor[:main_len]  # (main_len, C)
        # Reshape: (main_len, C) -> (main_len // 4, 4, C)
        main_reshaped = main_part.view(main_len // k, k, C)
        # Average over dim=1 (the 4 chunks)
        main_pooled = main_reshaped.mean(dim=1)  # (main_len // 4, C)
    else:
        main_pooled = None
    
    # Remainder: handle remaining chunks if L % 4 != 0
    remainder_len = L % k
    if remainder_len > 0:
        remainder_part = tensor[main_len:]  # (remainder_len, C)
        # Average remaining chunks
        remainder_pooled = remainder_part.mean(dim=0, keepdim=True)  # (1, C)
    else:
        remainder_pooled = None
    
    # Concatenate main and remainder if both exist
    if main_pooled is not None and remainder_pooled is not None:
        pooled = torch.cat([main_pooled, remainder_pooled], dim=0)  # (main_len // k + 1, C)
    elif main_pooled is not None:
        pooled = main_pooled  # (main_len // 4, C)
    elif remainder_pooled is not None:
        pooled = remainder_pooled  # (1, C)
    else:
        # Should not happen, but handle gracefully
        pooled = tensor.mean(dim=0, keepdim=True)  # (1, C)
    
    return pooled


class DiskDNACache:
    """
    Disk-backed cache for DNA embeddings using numpy memmap for large tensors.
    Stores metadata (shapes, ccre info) in a pickle file and tensors in .npy files.
    Mimics LRUCache interface.
    """
    def __init__(self, cache_dir, mode='r', max_size=0, device='cpu', k=8):
        self.cache_dir = cache_dir
        self.mode = mode  # 'r' for read-only (training), 'w+' for precompute/write
        self.max_size = max_size  # ignored, for compatibility
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.meta_path = os.path.join(cache_dir, 'metadata.pkl')
        self.metadata = {}
        self.k = k
        # Counters for stats
        self.hits = 0
        self.misses = 0
        self.step_hits = 0
        self.step_misses = 0
        
        if mode == 'w+':
            os.makedirs(cache_dir, exist_ok=True)
        elif not os.path.exists(cache_dir) and mode == 'r':
            print(f"Warning: DiskDNACache directory {cache_dir} does not exist. Cache misses expected.")
            
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded DiskDNACache metadata with {len(self.metadata)} entries.")
            except Exception as e:
                print(f"Failed to load DiskDNACache metadata: {e}")

    def get(self, key):
        if key not in self.metadata:
            self.misses += 1
            self.step_misses += 1
            return None
            
        meta = self.metadata[key]
        try:
            feat_path = os.path.join(self.cache_dir, meta['feat_file'])
            mask_path = os.path.join(self.cache_dir, meta['mask_file'])
            
            # Load as memmap in read-only mode to save RAM, then move directly to GPU
            feat_np = np.load(feat_path, mmap_mode='r')
            mask_np = np.load(mask_path, mmap_mode='r')
            
            # Convert to tensor and move to GPU with correct dtype immediately
            # Use non_blocking=True for async transfer if device is GPU
            if self.device.type == 'cuda':
                feat = torch.from_numpy(feat_np).to(self.device, dtype=torch.bfloat16, non_blocking=True)
                mask = torch.from_numpy(mask_np).to(self.device, dtype=torch.bfloat16, non_blocking=True)
            else:
                # CPU fallback
                feat = torch.from_numpy(feat_np).to(self.device)
                mask = torch.from_numpy(mask_np).to(self.device)
            
            # Apply average pooling (every 4 chunks) to reduce memory usage
            # Pool on GPU if device is GPU (more efficient)
            feat = _pool_embeddings_4chunks(feat, device=self.device, k=self.k)
            mask = _pool_embeddings_4chunks(mask, device=self.device, k=self.k)
            
            ccre = meta.get('ccre', [])
            
            self.hits += 1
            self.step_hits += 1
            return (feat, mask, ccre)
        except Exception as e:
            print(f"Error loading {key} from DiskDNACache: {e}")
            self.misses += 1
            self.step_misses += 1
            return None

    def set(self, key, value):
        if self.mode != 'w+':
            return 
            
        # Value is expected to be (feat, mask, ccre)
        if not isinstance(value, tuple) or len(value) != 3:
            return 
            
        feat, mask, ccre = value
        
        # Convert to numpy
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        # Use sanitized gene name as filename (instead of hash)
        sanitized_key = _sanitize_filename(key)
        
        # Check for filename collision (if another key already uses this sanitized name)
        # If collision, fall back to hash for uniqueness
        feat_file_base = f"{sanitized_key}_feat.npy"
        mask_file_base = f"{sanitized_key}_mask.npy"
        
        # Check if files exist but belong to a different key
        feat_path = os.path.join(self.cache_dir, feat_file_base)
        if os.path.exists(feat_path):
            # Check if this file is already associated with this key
            existing_key = None
            for k, meta in self.metadata.items():
                if meta.get('feat_file') == feat_file_base and k != key:
                    existing_key = k
                    break
            if existing_key is not None:
                # Collision detected, use hash as fallback
                key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
                feat_file = f"{key_hash}_feat.npy"
                mask_file = f"{key_hash}_mask.npy"
            else:
                # Same key, reuse filename
                feat_file = feat_file_base
                mask_file = mask_file_base
        else:
            # No collision, use sanitized name
            feat_file = feat_file_base
            mask_file = mask_file_base
        
        np.save(os.path.join(self.cache_dir, feat_file), feat)
        np.save(os.path.join(self.cache_dir, mask_file), mask)
        
        self.metadata[key] = {
            'feat_file': feat_file,
            'mask_file': mask_file,
            'ccre': ccre,
            'feat_shape': feat.shape
        }

    def __contains__(self, key):
        return key in self.metadata

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)
        
    def save_metadata(self):
        if self.mode == 'w+':
            with open(self.meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                
    def update_device(self, device):
        """
        Update the target device for GPU-native storage.
        Useful for DDP scenarios where device is detected after initialization.
        
        Args:
            device: Target device (torch.device or string)
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
    
    def reset_step_counters(self):
        self.step_hits = 0
        self.step_misses = 0
        
    def stats(self, reset_step=False):
        stats = {
            'hits_total': self.hits,
            'misses_total': self.misses,
            'hits_step': self.step_hits,
            'misses_step': self.step_misses,
            'size': len(self.metadata)
        }
        if reset_step:
            self.reset_step_counters()
        return stats

