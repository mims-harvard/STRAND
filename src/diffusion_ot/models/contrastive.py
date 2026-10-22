import torch

import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Union
from collections import defaultdict
class ProtoCLIPContrast(nn.Module):
    """
    Proto-CLIP with EMA-stabilized soft targets (RNA raw space),
    using *global* perturbation IDs for the EMA bank, and compact
    batch-local IDs only for grouping via index_add_.
    
    Uses raw RNA values (no projection) to avoid domain shifting.
    DNA embeddings are projected via a simple 2-layer MLP to match RNA dimension.

    forward() inputs:
      - pert_emb: (B, D) or (B, 1, D) DeepSets output per sample
      - x0: (B, G) raw perturb RNA (single-cell)
      - labels: List[str] | (B,) int (stable across batches & workers if tensor)
      - weights: Optional (B,) float per-sample weights for proto averaging
      - x1: (B, G) optional raw control RNA (single-cell); if provided, pseudo-bulk uses (pert - ctrl)
    """

    def __init__(
        self,
        deepset_dim: int,
        rna_dim: int,
        proj_dim: int = 128,
        lambda_weight: float = 1.,
        init_temp: float = 0.07,
        detach_rna: bool = True,
        # --- EMA knobs ---
        use_ema: bool = False,
        ema_momentum: float = 0.9,        # higher → slower target updates
        ema_mix_in_targets: float = 0.5,  # how much EMA to mix when forming S (0..1)
        track_dna_proto: bool = False,     # optional: also smooth DNA prototypes
        # --- Soft targets & context scaling ---
        use_soft_targets: bool = True,
        soft_temp: float = 0.1,
        use_context_scaling: bool = True,
        ctrl_dim: int = 128,               # dimension of z_ctrl for context scaler
        context_scaler_hidden: int = 64,
    ):
        super().__init__()
        self.lambda_weight = float(lambda_weight)
        self.detach_rna = bool(detach_rna)
        self.use_soft_targets = bool(use_soft_targets)
        self.soft_temp = float(soft_temp)
        self.use_context_scaling = bool(use_context_scaling)

        # Simplified DNA projection: 2-layer (Linear -> GELU -> Linear) projecting to rna_dim
        # No RNA projection - using raw RNA values to avoid domain shifting
        self.proj_dna = nn.Sequential(
            nn.Linear(deepset_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, rna_dim),
        )

        # learnable logit scale; clamped at use time
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

        # --- Context-dependent scaler (function of z_ctrl) ---
        if self.use_context_scaling:
            self.context_scaler = nn.Sequential(
                nn.Linear(ctrl_dim, context_scaler_hidden),
                nn.GELU(),
                nn.Linear(context_scaler_hidden, 1),
                nn.Softplus()  # ensures positive scaling
            )
        else:
            self.context_scaler = None

        # --- EMA state (dicts keyed by *global* label id) ---
        self.use_ema = use_ema
        self.ema_momentum = float(ema_momentum)
        self.ema_mix_in_targets = float(ema_mix_in_targets)
        self.track_dna_proto = bool(track_dna_proto)

        # Keep EMA vectors in float32 on the active device
        self._rna_proto_ema = {}                 # gid -> Tensor(G,)
        self._rna_counts    = defaultdict(int)   # gid -> int
        self._dna_proto_ema = {}                 # gid -> Tensor(D,)
        self._dna_counts    = defaultdict(int)   # gid -> int

        # Stable global ID mapping for string labels
        self._label2gid = {}   # str/int -> global int id
        self._next_gid  = 0

    # --- helpers ---

    def _labels_to_global_ids(self, labels: Union[List[str], torch.Tensor], device) -> torch.Tensor:
        """Return stable global ids (LongTensor[B]). If labels is tensor, assume already-stable global ints."""
        if isinstance(labels, torch.Tensor):
            return labels.to(device=device, dtype=torch.long)
        gids = []
        for lab in labels:
            if lab not in self._label2gid:
                self._label2gid[lab] = self._next_gid
                self._next_gid += 1
            gids.append(self._label2gid[lab])
        return torch.as_tensor(gids, device=device, dtype=torch.long)

    @staticmethod
    def _group_prototypes(X: torch.Tensor, ids_compact: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Group by *compact* ids: ids_compact in [0, P).
        Returns mean (or weighted mean) per compact id.
        """
        B, D = X.shape
        # Ensure ids (and weights) live on the same device as X
        ids_compact = ids_compact.to(X.device)
        if weights is not None:
            weights = weights.to(X.device)

        P = int(ids_compact.max().item()) + 1
        Xf = X.float()
        num = torch.zeros(P, D, device=X.device, dtype=torch.float32)
        den = torch.zeros(P, 1, device=X.device, dtype=torch.float32)

        if weights is None:
            num.index_add_(0, ids_compact, Xf)
            den.index_add_(0, ids_compact, torch.ones(B, 1, device=X.device))
        else:
            w = weights.view(B, 1).float()
            num.index_add_(0, ids_compact, w * Xf)
            den.index_add_(0, ids_compact, w)

        den = den.clamp_min(1e-12)
        return num / den

    def _ema_gather(self, gids_unique: torch.Tensor, bank: dict, device, dim: int) -> torch.Tensor:
        """Return stacked EMA vectors (float32, on device) for the given *global* ids. Initialize missing with zeros."""
        vecs = []
        for gid in gids_unique.tolist():
            if gid not in bank:
                bank[gid] = torch.zeros(dim, dtype=torch.float32, device=device)
            vecs.append(bank[gid])
        return torch.stack(vecs, dim=0)  # (P, D)

    def _ema_update(self, gids_unique: torch.Tensor, curr: torch.Tensor, bank: dict, count_bank: defaultdict):
        """EMA_{new} = m*EMA + (1-m)*curr; in-place update per *global* id."""
        m = self.ema_momentum
        for idx_local, gid in enumerate(gids_unique.tolist()):
            v = curr[idx_local].detach().float()
            if gid not in bank:
                bank[gid] = v.clone()
                _ = count_bank[gid]    # ensure key exists (0)
                count_bank[gid] += 1
            else:
                bank[gid].mul_(m).add_(v, alpha=(1.0 - m))
                _ = count_bank[gid]
                count_bank[gid] += 1

    # --- forward ---
    def forward(
        self,
        pert_emb: torch.Tensor,                  # (B, D) or (B, 1, D) or (B, L, D)
        x0: torch.Tensor,                        # (B, G) raw perturb RNA (single-cell or pseudo-bulk)
        labels: Union[List[str], torch.Tensor],  # (B,) stable global ids (tensor ok) or strings
        weights: Optional[torch.Tensor] = None,  # (B,)
        x1: torch.Tensor = None,                 # (B, G) raw control RNA (optional)
        z_ctrl: Optional[torch.Tensor] = None,    # (B, ctrl_dim) or (B, 1, ctrl_dim) control RNA embedding for context scaling
        verbose: bool = False,
    ):
        """
        Contrast current-batch DNA prototypes vs *EMA pseudo-bulk* RNA targets.
        Steps:
        1) Group current batch by compact ids (mapped from global ids).
        2) Pool DNA embeddings → dna_proto_cur (P, D).
        3) Pool RNA (x0 or x0 - x1) → rna_bulk_cur (P, G).
        4) Fetch *pre-update* EMA for those global ids → rna_ema_prev (P, G).
        5) Build logits between proj(DNA) and raw (EMA RNA) and do InfoNCE with diagonal targets.
           DNA is projected to rna_dim, RNA is used raw (no projection to avoid domain shifting).
        6) After loss, update EMA bank with rna_bulk_cur for those ids.
        """
        device = x0.device
        dtype  = x0.dtype

        # ---- map labels to stable global ids, then compact ids for this batch ----
        global_ids = self._labels_to_global_ids(labels, device)                     # (B,)
        gids_unique, compact_ids = torch.unique(global_ids, return_inverse=True)    # (P,), (B,)
        P = int(gids_unique.numel())
        if P < 2:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return {"loss": zero, "clip_loss_raw": zero, "P": torch.tensor(P, device=device)}

        # ---- normalize pert_emb to 2D and pool tokens if needed ----
        # Accept (B,D), (B,1,D) or (B,L,D) and reduce to (B,D)
        if pert_emb.dim() == 3:
            # If L>1, mean-pool tokens; if L==1, squeeze
            if pert_emb.size(1) == 1:
                pert_b = pert_emb.squeeze(1)                         # (B,D)
            else:
                pert_b = pert_emb.mean(dim=1)                        # (B,D)
        else:
            pert_b = pert_emb                                        # (B,D)

        # Ensure RNA tensors are (B,G)
        x0 = x0.squeeze(1) if x0.dim() == 3 else x0
        if x1 is not None:
            x1 = x1.squeeze(1) if x1.dim() == 3 else x1

        # ---- group-by-perturbation prototypes on CURRENT batch ----
        dna_proto_cur = self._group_prototypes(pert_b, compact_ids, weights)        # (P, D_dna)
        rna_bulk_pert = self._group_prototypes(x0,     compact_ids, weights)        # (P, G)
        if x1 is not None:
            rna_bulk_ctrl = self._group_prototypes(x1, compact_ids, weights)        # (P, G)
            rna_bulk_cur  = rna_bulk_pert - rna_bulk_ctrl                            # (P, G)
        else:
            rna_bulk_cur  = rna_bulk_pert                                           # (P, G)

        # ---- fetch *pre-update* EMA targets for these global ids ----
        if self.use_ema:
            rna_ema_prev = self._ema_gather(gids_unique, self._rna_proto_ema, device, rna_bulk_cur.size(1))  # (P,G)
            # targets are the *EMA* (teacher); do NOT mix with current if you want pure EMA targets
            rna_target = rna_ema_prev
        else:
            # fall back to current batch prototypes if EMA disabled
            rna_target = rna_bulk_cur

        # ---- context-dependent scaling (before projection) ----
        if self.use_context_scaling and z_ctrl is not None:
            # Handle z_ctrl shape: (B, ctrl_dim) or (B, 1, ctrl_dim)
            z_ctrl_flat = z_ctrl.squeeze(1) if z_ctrl.dim() == 3 else z_ctrl  # (B, ctrl_dim)
            # Pool z_ctrl by perturbation group (same grouping as DNA/RNA)
            z_ctrl_pooled = self._group_prototypes(z_ctrl_flat, compact_ids, weights)  # (P, ctrl_dim)
            # Compute context scaler per perturbation
            context_scale = self.context_scaler(z_ctrl_pooled)  # (P, 1)
            context_scale = context_scale.squeeze(-1)  # (P,)
            # Scale DNA embeddings BEFORE projection (preserves scale effect)
            dna_proto_cur = dna_proto_cur * context_scale.unsqueeze(-1)  # (P, D_dna)
        else:
            context_scale = None

        # ---- projections & logits (DNA vs EMA-RNA) ----
        # Project DNA embeddings (now with context-dependent scaling applied)
        z_dna = self.proj_dna(dna_proto_cur.to(dtype))  # (P, rna_dim)
        z_dna = F.normalize(z_dna, dim=-1)  # (P, rna_dim) - normalize AFTER projection
        # Use raw RNA values (no projection) to avoid domain shifting
        # if detach_rna=True, do not backprop into RNA targets
        rna_for_proj = rna_target.detach() if self.detach_rna else rna_target
        z_rna = F.normalize(rna_for_proj.to(dtype), dim=-1)  # (P, rna_dim) - raw RNA normalized

        # cosine logits with temperature clamp
        logit_scale = self.logit_scale.exp().clamp(5.0, 33.0)                       # τ in [~0.2, 0.03]
        logits = logit_scale * (z_dna.float() @ z_rna.float().t())                  # (P, P)

        # ---- soft targets based on RNA similarity (preserves perturbation similarity structure) ----
        if self.use_soft_targets:
            # Compute RNA similarity matrix (cosine similarity between RNA prototypes)
            rna_normalized = F.normalize(rna_target, dim=1)  # (P, G)
            rna_sim = rna_normalized @ rna_normalized.T  # (P, P) cosine similarity matrix
            # Ensure diagonal = 1.0 (perfect self-match)
            rna_sim.fill_diagonal_(1.0)
            # Convert to soft targets via temperature-scaled softmax
            soft_targets = torch.softmax(rna_sim / self.soft_temp, dim=1)  # (P, P), rows sum to 1
        else:
            # Fallback to hard diagonal targets
            soft_targets = torch.eye(P, device=logits.device, dtype=logits.dtype)  # (P, P)
        
        # stable log-softmax; do both directions
        logp_rows = torch.log_softmax(logits,    dim=1)         # DNA → EMA-RNA
        logp_cols = torch.log_softmax(logits.t(), dim=1)        # EMA-RNA → DNA
        # Cross-entropy with soft targets (preserves similarity structure)
        loss_rows = -(soft_targets * logp_rows).sum(dim=1).mean()
        loss_cols = -(soft_targets * logp_cols).sum(dim=1).mean()
        clip_loss_raw = 0.5 * (loss_rows + loss_cols)
        loss = self.lambda_weight * clip_loss_raw

        if verbose:
            with torch.no_grad():
                # cosine stats of targets (optional debugging)
                r = rna_target
                eps = 1e-8
                r = r / (r.norm(dim=1, keepdim=True).clamp_min(eps))
                S = (r @ r.t()).clamp(-1, 1)
                print(f"[CLIP] P={P} | logits mean={logits.mean().item():.3f} "
                    f"| S(min/mean/max)=({S.min().item():.3f}/{S.mean().item():.3f}/{S.max().item():.3f})")

        # ---- post-loss: update EMA ONLY for the ids present this step ----
        if self.use_ema:
            self._ema_update(gids_unique, rna_bulk_cur, self._rna_proto_ema, self._rna_counts)
            if self.track_dna_proto:
                # optional: keep an EMA of DNA prototypes too (not used in loss here)
                dna_ema_prev = self._ema_gather(gids_unique, self._dna_proto_ema, device, dna_proto_cur.size(1))
                self._ema_update(gids_unique, dna_proto_cur, self._dna_proto_ema, self._dna_counts)

        return {
            "loss": loss,
            "clip_loss_raw": clip_loss_raw.detach(),
            "P": torch.tensor(P, device=device),
        }

    def get_adaptive_margins(
        self,
        labels_orig: Union[List[str], torch.Tensor],
        labels_shuffled: Union[List[str], torch.Tensor],
        device: torch.device,
        min_margin: float = 0.01,
        use_l2: bool = True,
        fallback_margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Get adaptive margins for shuffling loss based on EMA pseudo-bulk RNA distances.
        
        Args:
            labels_orig: Original perturbation labels (B,)
            labels_shuffled: Shuffled perturbation labels (B,) - what each sample maps to
            device: Device for tensors
            min_margin: Minimum margin to avoid collapse
            use_l2: Whether to use L2 (True) or L1 (False) distance
            fallback_margin: Fixed margin to use when EMA not available
            
        Returns:
            margins: (B,) tensor of adaptive margins per sample
        """
        B = len(labels_orig) if isinstance(labels_orig, list) else labels_orig.size(0)
        margins = torch.full((B,), fallback_margin, device=device, dtype=torch.float32)
        
        # Map labels to global IDs
        gids_orig = self._labels_to_global_ids(labels_orig, device)
        gids_shuffled = self._labels_to_global_ids(labels_shuffled, device)
        
        # For each sample, compute distance between its original and shuffled perturbation
        for i in range(B):
            gid_orig = gids_orig[i].item()
            gid_shuffled = gids_shuffled[i].item()
            
            # Skip if same perturbation (shouldn't happen with proper shuffling, but safe)
            if gid_orig == gid_shuffled:
                margins[i] = fallback_margin
                continue
            
            # Get EMA embeddings for both perturbations
            # Use symmetric key: (min, max) for consistency
            key = (min(gid_orig, gid_shuffled), max(gid_orig, gid_shuffled))
            
            # Check if both EMA entries exist
            if gid_orig not in self._rna_proto_ema or gid_shuffled not in self._rna_proto_ema:
                # Fallback to fixed margin if EMA not available
                margins[i] = fallback_margin
                continue
            
            # Get EMA pseudo-bulk embeddings
            emb_orig = self._rna_proto_ema[gid_orig].to(device)
            emb_shuffled = self._rna_proto_ema[gid_shuffled].to(device)
            
            # Compute distance
            if use_l2:
                dist = torch.norm(emb_orig - emb_shuffled, p=2).item()
            else:
                dist = torch.abs(emb_orig - emb_shuffled).mean().item()
            
            # Clip to minimum margin and use as adaptive margin
            margins[i] = max(dist, min_margin)
        
        return margins

