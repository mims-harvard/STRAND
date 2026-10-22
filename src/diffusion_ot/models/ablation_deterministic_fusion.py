"""
Ablation "w/o Diffusion" (paper Fig. 4, right; App. J.2): ``DeterministicFusionAblation``.

Replaces the stochastic I2SB transport with a deterministic MLP on the concatenated control
latent and DNA perturbation embedding, z_pert = MLP([z_ctrl; z_dna]). The DNA Perturbation
Module, decoder and alignment loss are the same as in ``strand_model.STRANDModel``.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict
import time
import torch.distributed as dist

from .diffusion import scDiffusion1DStable, ddpm_sample_i2sb, i2sb_training_step
from .attention import MultiSequenceDeepSet, MultiSequenceDeepSetCond, timestep_embedding
from .heads import DNAFullPredictor
from .utils import draw_condition
import warnings
import re
# Import helper functions (will be refactored later)

from ..data import (
    match_gene_dna_embeddings, 
    match_gene_pert_mask_dna_embeddings,
    match_interval_pert_mask_dna_embeddings, 
    match_gene_pert_mask_dna_embeddings_cached,
    match_interval_pert_mask_dna_embeddings_cached,
    LRUCache
)
from ..utils import (
    gather_and_concatenate_tensors, 
    gather_and_concatenate_list,
    PerturbationWeighter, 
    StringLabelFactorizer
)
from ..losses import (
    HurdleGaussianLoss, 
    GaussianLoss, 
    ZINBMSELoss, 
    HurdleTruncGaussLoss,
    predict_unconditional_mean, 
    HurdleTruncGaussResidualLoss,
    predict_unconditional_mean_residual, 
    HurdleTruncGaussDeltaLoss,
    predict_unconditional_mean_delta,
    variance_floor_loss,
    cov_offdiag_loss,
    offdiag_cosine_stats,
)
from ..schedulers import symmetric_beta, get_official_betas, cosine_beta_schedule
from ..evaluation import compute_e_distance


# Import contrastive learning if present in the refactored package (no legacy fallback)

from .contrastive import ProtoCLIPContrast



_INTERVAL_LABEL_RE = re.compile(
    r"^chr[0-9A-Za-z]+:\d[\d,]*-\d[\d,]*(?:\(\s*[+-]\s*\))?(?:-[^-]+-[^-]+)?$"
)

def _looks_like_interval_label(label: str) -> bool:
    if not isinstance(label, str):
        return False
    # Avoid mistaking control-like tokens for intervals
    if label.lower().startswith("control"):
        return False
    return _INTERVAL_LABEL_RE.match(label) is not None



class DeterministicFusionAblation(nn.Module):
    def __init__(self,
                 DNA_model,
                 T_train=1000, # Kept for API compatibility, unused
                 dna_dim=1920,
                 ctrl_dim=128,
                 RNA_dim=978,
                 hidden_dim=128,
                 num_heads=4,
                 num_layers=4,
                 cond_dim=256,
                 embedding_dim=128,
                 dna_records=None,
                 gene_annot=None,
                 useFiLM=True,
                 use_scheduler=True,
                 use_adaptive_weight=True,
                 loss_metric='htg_nll',
                 train_mask_only=True,
                 verbose=False,
                 drop_mask_prob=0.0,
                 deepset_dim=128,
                 adapt_weight_type='uncertainty',
                 use_contrastive_loss=True,
                 use_VIC=False,
                 use_ctrl_to_dna=False,
                 embed_backend: str = 'gene',
                 cfg_dropout_prob: float = 0.1, 
                 DNA_emb_len=1,
                 w_clip: float = 1.0,
                 # scheduler and CM options (Unused in ablation but kept for init signature)
                 beta_schedule: str = 'linear_symmetric', 
                 use_cm: bool = False,
                 cm_weight: float = 0.1,
                 cm_dt: float = 0.05,
                 use_weighted_loss: bool = True,
                 dna_cache_dir: str = None, 
                 dna_cache_k: int = 8,
                 cell_line_nums: int = 1,
                 cell_line_map: dict = None,
                 ):
        super().__init__()
        
        # --- Standard Validations (Same as Original) ---
        assert embed_backend in {'gene', 'interval', 'auto'}, \
            f"embed_backend must be 'gene', 'interval', or 'auto', got: {embed_backend}"
        self.embed_backend = embed_backend
        assert adapt_weight_type in ['uncertainty', 'GradNorm'], f"Invalid adapt_weight_type: {adapt_weight_type}"
        self.adapt_weight_type = adapt_weight_type
        self.T_train = T_train
        self.use_ctrl_to_dna = use_ctrl_to_dna
        allowed_losses = {'htg_nll', 'nll', 'mse'}
        if loss_metric not in allowed_losses:
            raise ValueError(f"Invalid loss: {loss_metric}. Choose from {allowed_losses}.")
        
        self.loss_metric = loss_metric
        self.DNA_model = DNA_model
        self.ctrl_dim = ctrl_dim
        self.dna_dim = dna_dim
        self.dna_records = dna_records
        self.gene_annot = gene_annot
        self.DNA_emb_len = DNA_emb_len
        self.deepset_dim = deepset_dim # Added for clarity

        # Initialize disk cache if path provided
        self.disk_cache = None
        if dna_cache_dir is not None:
            from ..data.cache import DiskDNACache
            self.disk_cache = DiskDNACache(dna_cache_dir, mode='r', device='cpu', k=dna_cache_k)
            if verbose:
                print(f"Initialized DiskDNACache from {dna_cache_dir} with {len(self.disk_cache.metadata)} entries")
        
        # --- ABLATION CHANGE: Replaced scDiffusion1DStable with Concatenation MLP ---
        # Input: Control Embedding (ctrl_dim) + DNA Condition (deepset_dim)
        # Output: Perturbed Embedding (embedding_dim)
        # Architecture: Input -> ReLU -> Hidden -> ReLU -> Output
        self.concat_mlp = nn.Sequential(
            nn.Linear(ctrl_dim + deepset_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        print("Initialized Concatenation Ablation Model (MLP Baseline)")
        # --------------------------------------------------------------------------

        # RNA predictor head for x_ctrl + DNA -> x_pert (Same as Original)
        self.predictor_head = DNAFullPredictor(
            dna_dim=deepset_dim, 
            ctrl_dim=embedding_dim,
            RNA_dim=RNA_dim,
            hidden_dim=hidden_dim,
            num_heads=1,
            num_layers=num_layers,
            loss_metric=loss_metric,
            verbose=verbose,
        )
        
        self.use_adaptive_weight = use_adaptive_weight
        self.w_clip = float(w_clip)
        self.contrastive_loss = use_contrastive_loss
        self.use_weighted_loss = bool(use_weighted_loss)
        
        # CM params kept only to prevent breakage if accessed, but logically unused
        self.use_cm = False 

        # --- Adaptive Weights Setup ---
        if use_adaptive_weight and adapt_weight_type == 'GradNorm':
            # learnable raw weights
            if use_contrastive_loss:
                self.raw_w = nn.Parameter(torch.ones(2)) # Reduced from 3 to 2 (Diff removed) -> Actually we keep pred + clip
            else:
                self.raw_w = nn.Parameter(torch.ones(1))
            self.initial_losses = None  
        elif use_adaptive_weight and self.adapt_weight_type == 'uncertainty':
            # Allocate homoscedastic uncertainties for potential active losses:
            # [pred] + optional [clip] (No diffusion, No CM)
            n_unc = 1 + int(self.contrastive_loss)
            log_sigma2_init = torch.zeros(n_unc)
            self.log_sigma2 = nn.Parameter(log_sigma2_init)
            self._log_sigma2_min = math.log(1e-3)
            self._log_sigma2_max = math.log(1e3)

        self.use_scheduler = use_scheduler
        self.schedule_scaler = None
        self.verbose = verbose
        self.drop_mask_prob = drop_mask_prob
        self.use_VIC = use_VIC
        
        # DNA Encoders (Same as Original)
        self.shallow_encoder = nn.Sequential(
            nn.Linear(dna_dim, deepset_dim, bias=False),
        )
        self.shallow_diff_encoder = nn.Sequential(
            nn.Linear(dna_dim, deepset_dim, bias=False),
        )
        
        self.null_pred_ln = nn.LayerNorm(deepset_dim)
        self.pert_pred_ln = nn.LayerNorm(deepset_dim)
        self.null_ln = nn.LayerNorm(deepset_dim)
        self.pert_ln = nn.LayerNorm(deepset_dim)
        
        self.cell_line_nums = cell_line_nums
        self.cell_line_map = cell_line_map
        
        self.null_dna_emb = nn.Embedding(self.cell_line_nums, deepset_dim)       # for latent gen
        self.null_dna_pred_emb = nn.Embedding(self.cell_line_nums, deepset_dim)  # for predictor cond
        
        nn.init.normal_(self.null_dna_emb.weight, std=1e-2)
        nn.init.normal_(self.null_dna_pred_emb.weight, std=1e-2)
        
        self.project_off_null = True
        
        if self.contrastive_loss:
            self.CLIP_contrast = ProtoCLIPContrast(
                deepset_dim=deepset_dim,
                rna_dim=deepset_dim,
                proj_dim=128,
                lambda_weight=1.0,
                init_temp=0.07,
                detach_rna=True,
                use_soft_targets=True,
                soft_temp=0.1,
                use_context_scaling=True,
                ctrl_dim=ctrl_dim,
                context_scaler_hidden=64,
                use_ema=True, 
                ema_momentum=0.99,
            )
            
        # Register buffers for CLIP (Same as Original)
        self.register_buffer("z_mu",      torch.zeros(1, 1, ctrl_dim, dtype=torch.float32))
        self.register_buffer("x_mu",      torch.zeros(1,    RNA_dim, dtype=torch.float32))
        self.register_buffer("z_mu_ema",  torch.zeros(1, 1, ctrl_dim, dtype=torch.float32))
        self.register_buffer("x_mu_ema",  torch.zeros(1,    RNA_dim, dtype=torch.float32))
        self.register_buffer("z_mu_ctrl_ema",  torch.zeros(1, 1, ctrl_dim, dtype=torch.float32))
        self.register_buffer("x_mu_ctrl_ema",  torch.zeros(1,    RNA_dim, dtype=torch.float32))
        self.register_buffer("z_mu_step",         torch.zeros((), dtype=torch.long))
        self.register_buffer("x_mu_step",         torch.zeros((), dtype=torch.long))
        self.register_buffer("z_mu_ctrl_step",    torch.zeros((), dtype=torch.long))
        self.register_buffer("x_mu_ctrl_step",    torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_step",  torch.zeros((), dtype=torch.long))
        self.register_buffer("mu_count",  torch.zeros((), dtype=torch.long))
        self.ema_beta = 0.5 
        
        self.weighter = PerturbationWeighter(alpha=0.5, beta=None, ema_gamma=1e-3, w_cap=5.0, prior=10.0)
        self.string_factorizer = StringLabelFactorizer()

    def forward(self, batch, sgRNAs, betas, alphas, alphas_cumprod, cache_embeddings=None, weight=1., 
                activate_scheduler=True, teacher_forcing=False, epoch=None, total_epoch=None, 
                jitter_window=32, flip_prob=0.1, step=0, total_step=1000,
                flip_mask_sign_prob = 0.2, scaler=None, return_pert_emb=False, gene_mu=None,
                ccre_intervals_dict=None):
        
        device = next(self.parameters()).device
        self.learned_alpha = 1.0
        z_0 = batch["perturb"].unsqueeze(1).to(device)
        z_ctrl = batch["control"].unsqueeze(1).to(device)
        x_ctrl = batch["control_raw"].to(device)
        x_0 = batch["perturb_raw"].to(device)
        
        ### match cell line ids
        cell_line_ids = self._coerce_cell_line_ids(batch.get("cell_line_id", None), B=z_0.size(0), device=device)
        self.ema_beta = 0.99
        
        cond_type = draw_condition(step)
        
        # Synchronize cond_type 
        cond_map = {"p_to_p": 0, "unconditional": 1, "regular": 2}
        inv_map = {v: k for k, v in cond_map.items()}
        code = torch.tensor([cond_map.get(cond_type, 2)], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(code, src=0)
        cond_type = inv_map[int(code.item())]
        
        # Determine CLIP usage
        use_clip = self.contrastive_loss and (cond_type in ["p_to_p"])
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(use_clip)], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            use_clip = bool(flag.item())
        
        # # Noise augmentation (optional)
        # if self.beta_schedule != 'cosine':
        #     if self.training:
        #         if np.random.rand() < 0.5:
        #             z_ctrl = z_ctrl + torch.randn_like(z_ctrl) * 0.05
        
        step = min(step, total_step)
        labels = batch["label"]
        t0 = time.time()
        is_train = self.training
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # --- 1. Build DNA Embeddings ---
            collect_ccre_info = (step % 100 == 0) if step is not None else False
            build_result = self._build_dna_embeddings(
                labels,
                cache_embeddings=cache_embeddings,
                sgRNAs=sgRNAs,
                jitter_window=jitter_window,
                flip_prob=flip_prob,
                flip_mask_sign=False,
                return_seq_info=collect_ccre_info,
                DNA_emb_len=self.DNA_emb_len,
                ccre_intervals_dict=ccre_intervals_dict,
                z_ctrl=z_ctrl.squeeze(1),
            )
            
            if len(build_result) == 3:
                dna_ref_embeddings, _, failed_gene_labels = build_result
            else:
                dna_ref_embeddings, failed_gene_labels = build_result
            
            # ... (Masking logic for failed labels - kept same) ...
            dna_ref_embeddings = dna_ref_embeddings.to(device)
            if len(failed_gene_labels) > 0:
                mask = torch.tensor([lab not in failed_gene_labels for lab in labels], dtype=torch.bool, device=device)
                z_0 = z_0[mask]
                z_ctrl = z_ctrl[mask]
                x_ctrl = x_ctrl[mask]
                x_0 = x_0[mask]
                dna_ref_embeddings = dna_ref_embeddings[mask]
                cell_line_ids = cell_line_ids[mask]
                labels = [lab for lab, keep in zip(labels, mask.tolist()) if keep]
            
            B = z_0.size(0)
            z_src, z_tgt = z_ctrl, z_0
            x_src, x_tgt = x_ctrl, x_0
            
            # --- Adaptive Weighting ---
            if self.use_weighted_loss:
                ids = self.string_factorizer.encode(labels).to(device)
                weights = self.weighter.update_and_get_weights(ids, loss_device=device)
            else:
                weights = None
            
            # --- p_to_p Logic (kept for CLIP statistics tracking) ---
            if cond_type == "p_to_p":
                z_b = z_0.mean(dim=0, keepdim=True).detach()
                x_b = x_0.mean(dim=0, keepdim=True).detach()
                x_ctrl_b = x_ctrl.mean(dim=0, keepdim=True).detach()
                z_ctrl_b = z_ctrl.mean(dim=0, keepdim=True).detach()
                z_hat      = self._ema_update(self.z_mu_ema,      z_b,      self.ema_beta, self.z_mu_step)
                x_hat      = self._ema_update(self.x_mu_ema,      x_b,      self.ema_beta, self.x_mu_step)
                zc_hat     = self._ema_update(self.z_mu_ctrl_ema, z_ctrl_b,self.ema_beta, self.z_mu_ctrl_step)
                xc_hat     = self._ema_update(self.x_mu_ctrl_ema, x_ctrl_b,self.ema_beta, self.x_mu_ctrl_step)
                delta_z = (z_hat - zc_hat).detach()
                delta_x = (x_hat - xc_hat).detach()
                z_src = z_src + delta_z.expand(B, -1, -1)
                x_src = x_src + delta_x.expand(B, -1)

            # --- 2. Projection and Conditions ---
            # Project DNA embeddings
            pert_diff_emb_delta = self.pert_ln(self.shallow_diff_encoder(dna_ref_embeddings)) # (B, 1, D)
            pert_pred_emb_delta = self.pert_pred_ln(self.shallow_encoder(dna_ref_embeddings))
            
            L = dna_ref_embeddings.size(1)
            null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype) # (B, D)
            null_pred = self.null_pred_ln(self.null_dna_pred_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)
            null = null.unsqueeze(1).expand(B, L, -1)
            null_pred = null_pred.unsqueeze(1).expand(B, L, -1)
            
            if cond_type == "unconditional":
                # Zero out DNA info for unconditional testing
                pert_diff_emb = null + pert_diff_emb_delta * 0.0
                pert_pred_emb = null_pred + pert_pred_emb_delta * 0.0
            else:
                # Add null bias
                pert_diff_emb = pert_diff_emb_delta + null 
                pert_pred_emb = pert_pred_emb_delta + null_pred 
                
            diff_cond = pert_diff_emb
            pred_cond = pert_pred_emb

            # --- 3. CLIP / Contrastive Loss ---
            clip_dict = None
            if self.contrastive_loss:
                if use_clip:
                    clip_dict = self.CLIP_contrast(pert_diff_emb, z_tgt.squeeze(1), labels, x1=z_src.squeeze(1), z_ctrl=z_src.squeeze(1), verbose=self.verbose)
                else:
                    # Tax Collector logic
                    tau = getattr(self, 'clip_tax', 1e-8)
                    l2_tax = torch.zeros((), device=pert_diff_emb.device, dtype=pert_diff_emb.dtype)
                    for p in self.CLIP_contrast.parameters():
                        if p.requires_grad:
                            l2_tax = l2_tax + p.float().pow(2).sum()
                    clip_dict = {'loss': tau * l2_tax}

            # --- ABLATION CHANGE: CONCATENATION PREDICTION ---
            # Prepare inputs: z_src (Control) and diff_cond (DNA)
            # z_src: (B, 1, ctrl_dim) -> (B, ctrl_dim)
            # diff_cond: (B, 1, deepset_dim) -> (B, deepset_dim)
            
            z_c_input = z_src.squeeze(1)
            z_dna_input = diff_cond.squeeze(1)
            
            concat_input = torch.cat([z_c_input, z_dna_input], dim=-1) # (B, ctrl_dim + deepset_dim)
            
            # Predict
            z_0_pred_flat = self.concat_mlp(concat_input) # (B, embedding_dim)
            z_0_pred = z_0_pred_flat.unsqueeze(1) # (B, 1, embedding_dim)
            
            # Dummy diffusion loss (0.0) to maintain dict structure
            loss_diff = {'loss_diff': torch.tensor(0.0, device=device, requires_grad=True)}
            
            # -------------------------------------------------

            # --- 4. Decoder / Predictor Head ---
            # z_0_pred is now the output of the MLP, passed to the gene decoder
            loss_pred, predict_dict_orig = self.predictor_head(
                pred_cond, x_src, ctrl_emb=z_src, pert_emb=z_0_pred, x_pert=x_tgt, time_emb=None, weights=weights
            ) 

            loss_diff_all = loss_diff['loss_diff'] # Effectively 0
            
            # --- 5. Uncertainty and Total Loss ---
            uncertainty_metrics = None
            loss_cm_val = torch.tensor(float('nan'), device=device) # No CM in concatenation
            
            if self.use_adaptive_weight and self.training:
                if self.adapt_weight_type == 'uncertainty':
                    # Active losses: [pred] + [clip] (No diff)
                    pred_idx = 0
                    clip_idx = None
                    losses = [loss_pred]
                    loss_names = ['pred']
                    loss_indices = [pred_idx]
                    
                    if self.contrastive_loss and clip_dict is not None and ('loss' in clip_dict):
                         losses.append(clip_dict['loss'].to(loss_diff_all.dtype))
                         loss_names.append('clip')
                         clip_idx = 1
                         loss_indices.append(clip_idx)
                    
                    loss_vec = torch.stack(losses)
                    log_sigma2_eff = self.log_sigma2.clamp(self._log_sigma2_min, self._log_sigma2_max)
                    sigma2_eff = torch.exp(log_sigma2_eff)
                    
                    # Select relevant sigmas
                    sigma2_selected = sigma2_eff[loss_indices]
                    log_sigma2_selected = log_sigma2_eff[loss_indices]
                    
                    weighted = 0.5 * (loss_vec / sigma2_selected + log_sigma2_selected)
                    total_loss = weighted.sum()
                    
                    uncertainty_metrics = {
                        'names': loss_names,
                        'raw_losses': loss_vec.detach().cpu().float(),
                        'log_sigma2': log_sigma2_selected.detach().cpu().float(),
                        'sigma2': sigma2_selected.detach().cpu().float(),
                        'weights': (0.5 / sigma2_selected).detach().cpu().float(),
                    }
            else:
                # Non-adaptive path
                total_loss = weight * loss_pred 
                if self.contrastive_loss and clip_dict is not None and ('loss' in clip_dict):
                    total_loss = total_loss + self.w_clip * clip_dict['loss'].to(total_loss.dtype)

            # --- Zero-Batch Tax (Updated for Ablation) ---
            if self.training:
                tau_zero = getattr(self, 'zero_batch_tax', 1e-8)
                if tau_zero > 0:
                    zero_tax = torch.zeros((), device=device, dtype=total_loss.dtype)
                    critical = [
                        getattr(self, 'DNA_model', None),
                        getattr(self, 'concat_mlp', None), # Changed from diffusion_model
                        getattr(self, 'shallow_encoder', None),
                        getattr(self, 'shallow_diff_encoder', None),
                        getattr(self, 'predictor_head', None),
                        getattr(self, 'CLIP_contrast', None),
                    ]
                    for mod in critical:
                        if mod is not None:
                            for p in mod.parameters():
                                if p.requires_grad:
                                    zero_tax = zero_tax + p.float().pow(2).sum()
                    total_loss = total_loss + tau_zero * zero_tax
            
            # --- Output Construction ---
            if self.contrastive_loss and clip_dict is not None and 'loss' in clip_dict:
                loss_clip_val = clip_dict['loss'].detach()
            else:
                loss_clip_val = torch.tensor(float('nan'), device=device)

            out = {
                'loss_diff': loss_diff_all, # 0.0
                'loss_pred': loss_pred,
                'loss_clip': loss_clip_val,
                'loss_cm': loss_cm_val,     # NaN
                'total_loss': total_loss,
            }
            
            if uncertainty_metrics is not None:
                out['uncertainty'] = uncertainty_metrics
            
            # (Keeping cCRE logic same as original)
            if hasattr(self, '_last_seq_info') and self._last_seq_info:
                ccre_counts = []
                for si in self._last_seq_info:
                    if si and isinstance(si, list) and len(si) > 0:
                        si_dict = si[0] if isinstance(si[0], dict) else si
                        ccre_intervals_rel = si_dict.get('ccre_intervals_rel', [])
                        ccre_counts.append(len(ccre_intervals_rel) if ccre_intervals_rel else 0)
                if len(ccre_counts) > 0:
                    out['ccre_count'] = ccre_counts
            
            if return_pert_emb:
                out['pert_emb'] = pert_pred_emb_delta.mean(dim=1).detach().to('cpu')
                out['pert_diff_emb'] = pert_diff_emb_delta.mean(dim=1).detach().to('cpu')
                out['null_emb'] = null_pred.mean(dim=1).detach().to('cpu')
                out['labels'] = list(labels)

        return out
    
    @torch.no_grad()
    def predict(self, x_ctrl, z_ctrl, labels, gene_annot, 
                sgRNAs, dna_record, cell_line_ids=None, cache_embeddings=None, z_0=None, 
                flip_mask_sign=False, debug=False, return_mask=False):
        """
        Modified Predict for Concatenation Ablation.
        Directly uses MLP mapping instead of diffusion sampling.
        """
        self.eval()
        device = next(self.parameters()).device
        x_ctrl = x_ctrl.to(device)
        
        if cell_line_ids is not None:
            cell_line_ids = self._coerce_cell_line_ids(cell_line_ids, B=len(labels), device=device)
        
        if z_ctrl is not None:
            z_ctrl = z_ctrl.to(device) 
        if z_0 is not None:
            z_0 = z_0.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 1. Get DNA Embeddings
            dna_ref_embeddings, failed_gene_labels = self._build_dna_embeddings(
                labels, cache_embeddings=cache_embeddings, sgRNAs=sgRNAs,
                jitter_window=0, flip_prob=0.0, flip_mask_sign=flip_mask_sign,
                return_seq_info=False, DNA_emb_len=self.DNA_emb_len, z_ctrl=z_ctrl.squeeze(1),
            )
            
            # Handle masking (same as original)
            if len(failed_gene_labels) > 0:
                mask = torch.tensor([lab not in failed_gene_labels for lab in labels], dtype=torch.bool, device=device)
                z_ctrl = z_ctrl[mask]
                x_ctrl = x_ctrl[mask]
                dna_ref_embeddings = dna_ref_embeddings[mask]
                cell_line_ids = cell_line_ids[mask]
                labels = [lab for lab, keep in zip(labels, mask.tolist()) if keep]
            else:
                mask = torch.ones(len(labels), dtype=torch.bool, device=device)
            
            dna_ref_embeddings = dna_ref_embeddings.to(device)
            if dna_ref_embeddings.dim() == 2:
                dna_ref_embeddings = dna_ref_embeddings.unsqueeze(1)
            
            if z_ctrl.dim() == 2:
                z_ctrl = z_ctrl.unsqueeze(1)

            # 2. Get DNA Condition (deepset dim)
            pert_diff_emb = self.shallow_encoder(dna_ref_embeddings)
            pert_diff_emb = self.pert_pred_ln(pert_diff_emb)
            
            B = dna_ref_embeddings.size(0)
            null_pred = self.null_pred_ln(self.null_dna_pred_emb(cell_line_ids)).to(dna_ref_embeddings.dtype)
            null_pred = null_pred.unsqueeze(1).expand(B, dna_ref_embeddings.size(1), -1)
            
            pert_diff_emb = pert_diff_emb + null_pred
            
            # --- ABLATION PREDICTION ---
            # z_c: (B, ctrl_dim), z_dna: (B, deepset_dim)
            concat_input = torch.cat([z_ctrl.squeeze(1), pert_diff_emb.squeeze(1)], dim=-1)
            z_0_pred_flat = self.concat_mlp(concat_input)
            z_0_pred = z_0_pred_flat.unsqueeze(1)
            # ---------------------------
            
            # 3. Decode to Gene Expression
            pred_dict = self.predictor_head(
                None, x_ctrl, ctrl_emb=z_ctrl, pert_emb=z_0_pred,
            )
            
            if self.loss_metric == 'htg_nll':
                logits, mu, sigma, mu_ref = pred_dict['logits'], pred_dict['mu'], pred_dict['sigma'], pred_dict['mu_ref']
                x_0_pred = predict_unconditional_mean_residual(logits, mu, sigma, mu_ref)
            elif self.loss_metric == 'nll':
                x_0_pred = pred_dict['mu']
            elif self.loss_metric == 'mse':
                x_0_pred = pred_dict['mu']
                
        self.meta_data = pred_dict
        self.pert_emb = pert_diff_emb
        if return_mask:
            return x_0_pred, mask
        else:
            return x_0_pred

    @torch.no_grad()
    def eval_diffusion(self, val_loader,
                       cache_embeddings=None,
                       sgRNAs=None,
                       dna_record=None,
                       gene_annot=None,
                       flip_mask_sign=False,
                       # Arguments kept for compatibility, but ignored
                       Ts=(2, 6, 10, 20, 30, 50, 100), 
                       scale_list=(0.1, 1.0, 2.0, 5.0),
                       scaler=None, sample_limit=5000,
                       eval_mode='macro',):
        """
        Modified Eval for Concatenation Baseline.
        Since the model is deterministic, we do not sweep over T or scale.
        We run a single pass and compute e-distance.
        """
        self.eval()
        device = next(self.parameters()).device
        ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp else 0
        
        # We only run ONE configuration (deterministic)
        local_generated_samples = []
        local_ground_truths = []
        local_labels = []
        
        print("Running Concatenation Baseline Evaluation (Deterministic)...")
        
        for step, batch in enumerate(val_loader):
            bsz = batch["perturb"].size(0)
            cell_line_ids = self._coerce_cell_line_ids(batch.get("cell_line_id", None), B=bsz, device=device)
            z_0    = batch["perturb"].unsqueeze(1).to(device)
            z_ctrl = batch["control"].unsqueeze(1).to(device)
            labels = batch["label"]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                dna_ref_embeddings, _ = self._build_dna_embeddings(
                    labels, cache_embeddings=cache_embeddings, sgRNAs=sgRNAs,
                    jitter_window=0, flip_prob=0.0, flip_mask_sign=flip_mask_sign,
                    return_seq_info=False, DNA_emb_len=self.DNA_emb_len, z_ctrl=z_ctrl.squeeze(1),
                )
                if dna_ref_embeddings.device != device:
                    dna_ref_embeddings = dna_ref_embeddings.to(device)

                # DNA Encoding
                pert_emb = self.shallow_diff_encoder(dna_ref_embeddings)
                pert_emb = self.pert_ln(pert_emb)
                
                B = dna_ref_embeddings.size(0)
                L = dna_ref_embeddings.size(1)
                null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)
                null = null.unsqueeze(1).expand(B, L, -1)
                pert_emb = pert_emb + null
                
                # --- ABLATION PREDICTION ---
                concat_input = torch.cat([z_ctrl.squeeze(1), pert_emb.squeeze(1)], dim=-1)
                generated_sample = self.concat_mlp(concat_input).unsqueeze(1) # (B, 1, emb_dim)
                # ---------------------------

                local_generated_samples.append(generated_sample.squeeze(1).to('cpu', dtype=torch.float32))
                local_ground_truths.append(z_0.squeeze(1).to('cpu', dtype=torch.float32))
                local_labels.extend(labels)

        # Gather results
        if ddp:
            all_generated_samples = gather_and_concatenate_tensors(local_generated_samples).to(torch.float32).cpu()
            all_ground_truths     = gather_and_concatenate_tensors(local_ground_truths).to(torch.float32).cpu()
            all_labels = gather_and_concatenate_list(local_labels)
            dist.barrier()
        else:
            all_generated_samples = (torch.cat(local_generated_samples, 0) if local_generated_samples else torch.empty(0)).to(torch.float32).cpu()
            all_ground_truths = (torch.cat(local_ground_truths, 0) if local_ground_truths else torch.empty(0)).to(torch.float32).cpu()
            all_labels = local_labels

        if rank == 0 and all_generated_samples.numel() > 0:
            gen_np   = all_generated_samples.numpy()
            truth_np = all_ground_truths.numpy()
            loss = compute_e_distance(gen_np, truth_np, all_labels, average=eval_mode)
            print(f"[Eval Concatenation] e-distance={loss:.4f}")
            
            # Set dummy "best" params for API compatibility
            self.T_sample   = 0
            self.best_scale = 1.0
            self.best_eta   = 0
            self.best_beta = self.best_alphas = self.best_alphas_cumprod = None

    @torch.no_grad()
    def sample_latent(self, z_ctrl, labels, gene_annot, sgRNAs, dna_record, cell_line_ids=None, cache_embeddings=None, flip_mask_sign=False):
        """
        Modified Sample Latent for Concatenation Ablation.
        Returns deterministic MLP prediction.
        """
        self.eval()
        device = next(self.parameters()).device
        z_ctrl = z_ctrl.to(device)
        if z_ctrl.dim() == 2:
            z_ctrl = z_ctrl.unsqueeze(1)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            dna_ref_embeddings, _ = self._build_dna_embeddings(
                labels, cache_embeddings=cache_embeddings, sgRNAs=sgRNAs,
                jitter_window=0, flip_prob=0.0, flip_mask_sign=flip_mask_sign,
                return_seq_info=False, DNA_emb_len=self.DNA_emb_len, z_ctrl=z_ctrl.squeeze(1),
            )
            if dna_ref_embeddings.dim() == 2:
                dna_ref_embeddings = dna_ref_embeddings.unsqueeze(1)
            dna_ref_embeddings = dna_ref_embeddings.to(device)
            
            pert_emb = self.shallow_diff_encoder(dna_ref_embeddings)
            pert_emb = self.pert_ln(pert_emb)
            
            L = dna_ref_embeddings.size(1)
            B = dna_ref_embeddings.size(0)
            null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)
            null = null.unsqueeze(1).expand(B, L, -1)
            pert_emb = pert_emb + null
            
            # --- ABLATION PREDICTION ---
            concat_input = torch.cat([z_ctrl.squeeze(1), pert_emb.squeeze(1)], dim=-1)
            generated_sample = self.concat_mlp(concat_input).unsqueeze(1)
            # ---------------------------
            
        return generated_sample.squeeze(1).to(torch.float32)

    # ... (Helper methods _merge_mean, _ema_update, _build_dna_embeddings, _factorize_labels, etc. remain unchanged) ...
    # [Rest of the helper functions from the original file should be kept here unmodified]
    def _merge_mean(self, mu_buf, cnt_buf, batch_mean, b):
        bm = batch_mean.detach().to(mu_buf.dtype)
        cnt = cnt_buf.item()
        if cnt == 0:
            mu_buf.copy_(bm)
            cnt_buf.fill_(b)
            return
        cnt_new = cnt + b
        mu_buf.add_( (bm - mu_buf) * (b / float(cnt_new)) )
        cnt_buf.fill_(cnt_new)

    def _ema_update(self, ema_buf, batch_mean, beta, step_buf):
        bm = batch_mean.detach().to(device=ema_buf.device, dtype=ema_buf.dtype)
        step = step_buf.item() + 1
        if step == 1:
            ema_buf.copy_(bm)
        else:
            ema_buf.mul_(beta).add_(bm, alpha=(1.0 - beta))
        step_buf.fill_(step)
        bias_corr = 1.0 - (beta ** step)
        return ema_buf / bias_corr
    
    def _build_dna_embeddings(self, labels, cache_embeddings=None, sgRNAs=None, jitter_window=0, flip_prob=0.0, flip_mask_sign=False, return_seq_info=False, DNA_emb_len=1, ccre_intervals_dict=None, z_ctrl: torch.Tensor = None):
        backend = self.embed_backend
        if backend == 'auto':
            use_interval = any(_looks_like_interval_label(l) for l in labels if isinstance(l, str))
            backend = 'interval' if use_interval else 'gene'

        if backend == 'gene':
            call_verbose = self.verbose
            out = match_gene_pert_mask_dna_embeddings_cached(
                labels, self.DNA_model, cache=cache_embeddings, disk_cache=self.disk_cache,
                dna_record=self.dna_records, all_annots_transcript=self.gene_annot,
                promoter_len=200, perturb_type='CRISPR-i', gRNAs=sgRNAs,
                return_seq_info=return_seq_info, jitter_window=jitter_window, flip_seq_prob=flip_prob,
                DNA_emb_dim=self.dna_dim, drop_mask_prob=self.drop_mask_prob,
                flip_mask_sign=flip_mask_sign, VERBOSE=call_verbose, DNA_emb_len=DNA_emb_len,
                ccre_intervals_dict=ccre_intervals_dict, z_ctrl=z_ctrl,
            )
            if return_seq_info and len(out) >= 2:
                self._last_seq_info = out[1] if isinstance(out[1], list) else None
            return out
        else:
            call_verbose = self.verbose
            if return_seq_info:
                dna_ref_embeddings, seq_info_all, failed = match_interval_pert_mask_dna_embeddings_cached(
                    labels, self.DNA_model, cache=cache_embeddings, disk_cache=self.disk_cache,
                    dna_record=self.dna_records, FIX_WIN=524_288, jitter_window=jitter_window,
                    return_seq_info=True, flip_seq_prob=flip_prob, seed=0, max_batch_size=32,
                    DNA_emb_dim=self.dna_dim, drop_mask_prob=self.drop_mask_prob,
                    flip_mask_sign=flip_mask_sign, VERBOSE=call_verbose,
                    ccre_intervals_dict=ccre_intervals_dict, z_ctrl=z_ctrl,
                )
                self._last_seq_info = seq_info_all
                return dna_ref_embeddings, seq_info_all, failed
            else:
                dna_ref_embeddings, failed = match_interval_pert_mask_dna_embeddings_cached(
                    labels, self.DNA_model, cache=cache_embeddings, disk_cache=self.disk_cache,
                    dna_record=self.dna_records, FIX_WIN=524_288, jitter_window=jitter_window,
                    return_seq_info=False, flip_seq_prob=flip_prob, seed=0, max_batch_size=32,
                    DNA_emb_dim=self.dna_dim, drop_mask_prob=self.drop_mask_prob,
                    flip_mask_sign=flip_mask_sign, VERBOSE=call_verbose,
                    ccre_intervals_dict=ccre_intervals_dict, z_ctrl=z_ctrl,
                )
                self._last_seq_info = None
                return dna_ref_embeddings, failed

    def global_mean_zero(self, z: torch.Tensor) -> torch.Tensor:
        mu = z.mean(dim=0, keepdim=True)
        return (mu.pow(2).sum() / z.shape[1])
    
    def center_mean_zero(self, z: torch.Tensor, pert_ids: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        uniq, inv = torch.unique(pert_ids, return_inverse=True)
        T = uniq.numel()
        sums    = torch.zeros(T, D, device=z.device, dtype=z.dtype)
        counts = torch.zeros(T, 1, device=z.device, dtype=z.dtype)
        idx = inv.view(-1, 1).expand(B, D)
        sums.scatter_add_(0, idx, z)
        ones = torch.ones(B, 1, device=z.device, dtype=z.dtype)
        counts.scatter_add_(0, inv.view(-1, 1), ones)
        centers = sums / counts.clamp_min(1.0)
        mu = centers.mean(dim=0, keepdim=True)
        return (mu.pow(2).sum() / D)
    
    def _factorize_labels(self, labels: list, device: torch.device):
        if not labels:
            return torch.empty(0, dtype=torch.long, device=device)
        unique_to_id = {}
        ids = []
        for lab in labels:
            if lab not in unique_to_id:
                unique_to_id[lab] = len(unique_to_id)
            ids.append(unique_to_id[lab])
        return torch.tensor(ids, dtype=torch.long, device=device)
    
    def _label_mismatch_perm(self, ids: torch.Tensor, device: torch.device, max_tries: int = 50):
        # (Same shuffle logic as original, used for Contrastive Learning if enabled)
        B = ids.size(0)
        if B < 2 or torch.unique(ids).numel() < 2:
            return None
        unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
        P = unique_ids.numel()
        if P < 2: return None
        label_perm = None
        for _ in range(max_tries):
            candidate = torch.randperm(P, device=device)
            if not torch.all(candidate == torch.arange(P, device=device)):
                label_perm = candidate
                break
        if label_perm is None or torch.all(label_perm == torch.arange(P, device=device)):
            label_perm = torch.arange(P, device=device)
            for i in range(0, P - 1, 2):
                label_perm[i], label_perm[i + 1] = label_perm[i + 1], label_perm[i]
            if P % 2 == 1:
                label_perm[0], label_perm[-1] = label_perm[-1], label_perm[0]
        shuffled_labels = unique_ids[label_perm]
        perm = torch.zeros(B, dtype=torch.long, device=device)
        ar = torch.arange(B, device=device)
        for orig_idx, orig_label in enumerate(unique_ids):
            shuffled_label = shuffled_labels[orig_idx]
            mask_orig = (ids == orig_label)
            orig_indices = torch.where(mask_orig)[0]
            n_orig = orig_indices.numel()
            mask_shuffled = (ids == shuffled_label)
            shuffled_indices = torch.where(mask_shuffled)[0]
            n_shuffled = shuffled_indices.numel()
            if n_orig == n_shuffled:
                if n_orig == 1:
                    perm[orig_indices] = shuffled_indices
                else:
                    derangement_found = False
                    for _ in range(max_tries):
                        candidate_order = torch.randperm(n_shuffled, device=device)
                        candidate_mapping = shuffled_indices[candidate_order]
                        if torch.all(candidate_mapping != orig_indices):
                            perm[orig_indices] = candidate_mapping
                            derangement_found = True
                            break
                    if not derangement_found:
                        perm[orig_indices] = shuffled_indices[torch.randperm(n_shuffled, device=device)]
            elif n_orig < n_shuffled:
                available = shuffled_indices
                for orig_pos in orig_indices:
                    available = available[available != orig_pos]
                if len(available) >= n_orig:
                    selected = available[torch.randperm(len(available), device=device)[:n_orig]]
                    perm[orig_indices] = selected
                else:
                    selected = torch.randperm(n_shuffled, device=device)[:n_orig]
                    perm[orig_indices] = shuffled_indices[selected]
            else:
                for i, orig_pos in enumerate(orig_indices):
                    candidates = shuffled_indices[shuffled_indices != orig_pos]
                    if len(candidates) > 0:
                        perm[orig_pos] = candidates[torch.randint(0, len(candidates), (1,), device=device)]
                    else:
                        perm[orig_pos] = shuffled_indices[torch.randint(0, n_shuffled, (1,), device=device)]
        fixed_points = (perm == ar)
        if fixed_points.any():
            return None # Failed to find perfect shuffle
        return perm

    def save_checkpoint(self, path, optimizer=None, epoch=None, ema_model=None):
        save_dict = {
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'epoch': epoch
        }
        if ema_model is not None:
            save_dict['model_ema_state'] = ema_model.module.state_dict()
        torch.save(save_dict, path)

    def load_checkpoint(self, path, optimizer=None, ema_model=None):
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        if 'model_state' not in checkpoint:
            raise KeyError(f"'model_state' not found in checkpoint at {path}")
        self.load_state_dict(checkpoint['model_state'])
        opt_state = checkpoint.get('optimizer_state', None)
        if optimizer is not None and opt_state is not None:
            optimizer.load_state_dict(opt_state)
        ema_state = checkpoint.get('model_ema_state', None)
        if ema_model is not None and ema_state is not None:
            ema_model.module.load_state_dict(ema_state)
        return checkpoint.get('epoch', None)
    
    def _coerce_cell_line_ids(self, cell_line_ids, B: int, device: torch.device) -> torch.Tensor:
        if cell_line_ids is None:
            ids = torch.zeros(B, device=device, dtype=torch.long)
            warnings.warn("cell_line_ids is None! Setting to all zeros.")
        else:
            if isinstance(cell_line_ids, torch.Tensor):
                ids = cell_line_ids.to(device=device)
            else:
                ids = torch.as_tensor(cell_line_ids, device=device)
            ids = ids.long().view(-1)
            if ids.numel() != B:
                raise ValueError(f"cell_line_ids has {ids.numel()} elems, expected B={B}")
        return ids

