"""
Label factorization: Convert string labels to contiguous integer IDs in DDP.

This module contains:
- StringLabelFactorizer: DDP-safe string-to-ID mapping
"""

import torch
import torch.distributed as dist
from typing import List


class StringLabelFactorizer:
    def __init__(self):
        self.str2id, self.id2str = {}, []
        self._is_ddp = dist.is_available() and dist.is_initialized()
        self._rank   = dist.get_rank() if self._is_ddp else 0
        self._world  = dist.get_world_size() if self._is_ddp else 1

    @torch.no_grad()
    def encode(self, labels: list[str]) -> torch.LongTensor:
        if not labels:
            return torch.empty(0, dtype=torch.long)

        # 1) collect unseen labels on this rank
        new_here = [s for s in labels if s not in self.str2id]

        # 2) all-gather unseen across ranks (DDP)
        if self._is_ddp:
            gathered = [None] * self._world
            dist.all_gather_object(gathered, new_here)
            # flatten + uniq while preserving order
            seen, new_all = set(), []
            for lst in gathered:
                for s in lst:
                    if (s not in self.str2id) and (s not in seen):
                        seen.add(s); new_all.append(s)
        else:
            new_all = new_here

        # 3) rank 0 assigns IDs; then broadcast the list of (str, id)
        if self._rank == 0:
            updates = []
            for s in new_all:
                if s not in self.str2id:
                    i = len(self.id2str)
                    self.str2id[s] = i
                    self.id2str.append(s)
                    updates.append((s, i))
        else:
            updates = None

        # IMPORTANT: broadcast via a 1-element list and read back from [0]
        obj_list = [updates]
        if self._is_ddp:
            dist.broadcast_object_list(obj_list, src=0)
        updates = obj_list[0] or []  # now populated on all ranks

        # 4) apply updates on non-zero ranks (rank 0 already has them)
        if self._rank != 0:
            for s, i in updates:
                if s not in self.str2id:
                    self.str2id[s] = i
                    # ensure id2str length
                    if i == len(self.id2str):
                        self.id2str.append(s)
                    elif i < len(self.id2str):
                        self.id2str[i] = s
                    else:
                        self.id2str.extend([None] * (i - len(self.id2str)))
                        self.id2str.append(s)

        # 5) map current labels to ids (now guaranteed to exist)
        try:
            ids = torch.tensor([self.str2id[s] for s in labels], dtype=torch.long)
        except KeyError as e:
            # Rare timing edge-case: retry once with the missing label
            missing = str(e.args[0])
            # Force sync that single label
            self.encode([missing])  # will broadcast and install mapping
            ids = torch.tensor([self.str2id[s] for s in labels], dtype=torch.long)

        return ids
