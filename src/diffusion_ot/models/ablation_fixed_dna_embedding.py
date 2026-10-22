"""
Ablation "w/o Perturb Module" (paper Fig. 4, left): ``FixedDNAEmbeddingAblation``.

Identical to ``strand_model.STRANDModel`` except that the learned DNA Perturbation Module is
bypassed: each perturbation is represented by cached DNA-encoder embeddings that are
average-pooled and kept fixed (``match_gene_pert_mask_dna_embeddings_minimal``).
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
    match_gene_pert_mask_dna_embeddings_minimal,
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



class FixedDNAEmbeddingAblation(nn.Module):
    def __init__(self,
                 DNA_model,
                 T_train=1000,
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
                # scheduler and CM options
                beta_schedule: str = 'linear_symmetric',  # 'linear_symmetric' | 'cosine'
                use_cm: bool = False,
                cm_weight: float = 0.1,
                cm_dt: float = 0.05,
                use_weighted_loss: bool = True,
                dna_cache_dir: str = None,  # Path to DiskDNACache directory,
                dna_cache_k: int = 8,
                cell_line_nums: int = 1,
                cell_line_map: dict = None,
                 ):
        super().__init__()
        
        '''
        htg_null: Hurdle Truncated Gaussian Negative Log Likelihood
        nll: Negative Log Likelihood
        mse: Mean Squared Error
        adapt_weight_type: 'uncertainty' or 'GradNorm'
        '''
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

            # self.DNA_model.unfreeze_borzoi()
        self.ctrl_dim= ctrl_dim
        
        self.dna_dim = dna_dim
        self.dna_records = dna_records  # Optional: can pre-load outside
        self.gene_annot = gene_annot  # Optional: can pre-load outside
        self.DNA_emb_len = DNA_emb_len
        
        # Initialize disk cache if path provided
        # Device will be updated after model is moved to GPU (for DDP compatibility)
        self.disk_cache = None
        if dna_cache_dir is not None:
            from ..data.cache import DiskDNACache
            # Initialize with CPU, will be updated to GPU device after model.to(device)
            self.disk_cache = DiskDNACache(dna_cache_dir, mode='r', device='cpu', k=dna_cache_k)
            if verbose:
                print(f"Initialized DiskDNACache from {dna_cache_dir} with {len(self.disk_cache.metadata)} entries")
        
        self.diffusion_model = scDiffusion1DStable(
            # dna_dim=dna_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            cond_dim=cond_dim,
            dna_dim=dna_dim,
            useFiLM=useFiLM,
            cfg_dropout_prob=cfg_dropout_prob,
        )
        
        # RNA predictor head for x_ctrl + DNA → x_pert
        self.predictor_head = DNAFullPredictor(
            # dna_dim=dna_dim,
            dna_dim=deepset_dim,  # Use the same dimension as the adaptor output
            ctrl_dim=embedding_dim,
            RNA_dim=RNA_dim,
            hidden_dim=hidden_dim,
            num_heads=1,
            num_layers=num_layers,
            loss_metric=loss_metric,
            verbose=verbose,
        )
        
        self.use_adaptive_weight = use_adaptive_weight
        # Fixed coefficients for regularizers (no adaptive weighting)
        self.w_clip = float(w_clip)
        self.contrastive_loss = use_contrastive_loss
        # scheduler and CM configuration
        self.beta_schedule = beta_schedule
        self.use_cm = bool(use_cm)
        self.cm_weight = float(cm_weight)
        self.cm_dt = float(cm_dt)
        self.use_weighted_loss = bool(use_weighted_loss)

        if use_adaptive_weight and adapt_weight_type == 'GradNorm':
            print(f'use adaptive weight')
            # learnable raw weights
            if use_contrastive_loss:
                self.raw_w = nn.Parameter(torch.ones(3))
            else:
                self.raw_w = nn.Parameter(torch.ones(2))
            # store initial losses to compute r_i(t)
            self.initial_losses = None  
        elif use_adaptive_weight and self.adapt_weight_type == 'uncertainty':
            print(f'use adaptive weight')
            # Allocate homoscedastic uncertainties for potential active losses:
            # [diff, pred] + optional [clip] + optional [cm]
            n_unc = 2 + int(self.contrastive_loss) + int(self.use_cm)
            # Initialize with zeros (sigma2=1.0) for diff, pred, clip
            # Initialize CM loss scale to log(0.1) ≈ -2.3 assuming CM loss is ~0.1 scale
            log_sigma2_init = torch.zeros(n_unc)
            if self.use_cm:
                # CM is the last parameter: [diff, pred, (clip?), cm]
                cm_idx = n_unc - 1
                log_sigma2_init[cm_idx] = math.log(0.1)  # ≈ -2.3026
            self.log_sigma2 = nn.Parameter(log_sigma2_init)
            self._log_sigma2_min = math.log(1e-3)
            self._log_sigma2_max = math.log(1e3)

        self.use_scheduler = use_scheduler
        self.schedule_scaler = None
        # self.schedule_scaler = SymmetricMixtureSchedule(cond_dim=cond_dim, T=T_train, K=4, use_s_total=False)

        self.verbose = verbose
        self.drop_mask_prob = drop_mask_prob
        # self.learned_alpha_raw = nn.Parameter(torch.tensor(-4.0), requires_grad=True)
        self.use_VIC = use_VIC
        
        self.shallow_encoder = nn.Sequential(
            nn.Linear(dna_dim, deepset_dim, bias=False),
        )
        
        self.shallow_diff_encoder = nn.Sequential(
            nn.Linear(dna_dim, deepset_dim, bias=False),
        )
        
        # self.shallow_diff_encoder = nn.Sequential(
        #     nn.Linear(dna_dim, deepset_dim),
        #     nn.GELU(),
        #     nn.Linear(deepset_dim, deepset_dim),
        #     nn.GELU(),
        #     nn.Linear(deepset_dim, deepset_dim, bias=False),
        # )
        
        self.null_pred_ln = nn.LayerNorm(deepset_dim)
        self.pert_pred_ln = nn.LayerNorm(deepset_dim)
        
        self.null_ln = nn.LayerNorm(deepset_dim)
        self.pert_ln = nn.LayerNorm(deepset_dim)
        
        self.cell_line_nums = cell_line_nums
        self.cell_line_map = cell_line_map

        self.null_dna_emb = nn.Embedding(self.cell_line_nums, cond_dim)       # for diffusion cond
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
                use_ema=True,  # Enable EMA for adaptive shuffle margins
                ema_momentum=0.99,  # High momentum for stable margins
            )
            
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
        # counters
        self.register_buffer("ema_step",  torch.zeros((), dtype=torch.long))
        self.register_buffer("mu_count",  torch.zeros((), dtype=torch.long))
        self.ema_beta = 0.5 
        
        self.weighter = PerturbationWeighter(alpha=0.5, beta=None, ema_gamma=1e-3, w_cap=5.0, prior=10.0)
        self.string_factorizer = StringLabelFactorizer()
        # self.alpha_null_raw = nn.Parameter(torch.tensor(inv_sigmoid(0.5), dtype=torch.float32))
        
    def forward(self, batch, sgRNAs, betas, alphas, alphas_cumprod, cache_embeddings=None, weight=1., 
                activate_scheduler=True, teacher_forcing=False, epoch=None, total_epoch=None, 
                jitter_window=32, flip_prob=0.1, step=0, total_step=1000,
                flip_mask_sign_prob = 0.2, scaler=None, return_pert_emb=False, gene_mu=None,
                ccre_intervals_dict=None):
        
        # Time function entry (captures DDP wrapper overhead)
        func_entry = time.time()
        func_entry_sync = time.time()
        
        # Initialize timing dictionary for fine-grained profiling
        timing = {}
        timing['forward_start'] = time.time()
        timing['func_entry'] = func_entry
        timing['func_entry_sync'] = func_entry_sync
        timing['func_entry_to_forward_start'] = timing['forward_start'] - func_entry_sync
       
        device = next(self.parameters()).device
        # self.alpha = 1.5
        # self.learned_alpha = self.sigmoid(self.learned_alpha_raw)
        self.learned_alpha = 1.0
        # Batch tensors are already on GPU (moved before model() call), just unsqueeze
        z_0 = batch["perturb"].unsqueeze(1)
        z_ctrl = batch["control"].unsqueeze(1)
        x_ctrl = batch["control_raw"]
        x_0 = batch["perturb_raw"]
        cell_line_ids = self._coerce_cell_line_ids(batch.get("cell_line_id", None), B=z_0.size(0), device=device)
        
        # warmup = step < 400
        self.ema_beta = 0.99
        
        cond_type = draw_condition(step)
        
        # Synchronize cond_type across ranks to ensure consistent branch execution
        # This prevents DDP errors from parameters being used on some ranks but not others
        cond_map = {"p_to_p": 0, "unconditional": 1, "regular": 2}
        inv_map = {v: k for k, v in cond_map.items()}
        code = torch.tensor([cond_map.get(cond_type, 2)], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            ddp_broadcast_start = time.time()
            dist.broadcast(code, src=0)
            timing['ddp_broadcast_cond'] = time.time() - ddp_broadcast_start
        cond_type = inv_map[int(code.item())]
        
        if cond_type == "p_to_p":
            weightLoss = 1.0
        else:
            weightLoss = 1.0
        # cond_type = 'unconditional'
        
        # Decide when to actually run CLIP (deterministic based on step)
        use_clip = self.contrastive_loss and (cond_type in ["p_to_p"])
        
        # Synchronize CLIP decision across ranks to avoid DDP errors
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(use_clip)], device=device, dtype=torch.int32)
            ddp_broadcast_start = time.time()
            dist.broadcast(flag, src=0)
            timing['ddp_broadcast_clip'] = time.time() - ddp_broadcast_start
            use_clip = bool(flag.item())
        
        if self.beta_schedule != 'cosine':
            if self.training:
                if np.random.rand() < 0.5:
                    # z_0 = z_0 + torch.randn_like(z_0) * 0.01
                    z_ctrl = z_ctrl + torch.randn_like(z_ctrl) * 0.05
            
        
        step = min(step, total_step)
        # alpha = torch.tensor(step / total_step).to(device)
        # if self.verbose:
        #     print(f'step {step} alpha: {alpha}')
        # if epoch is not None and total_epoch is not None:
        #     x_0 = add_gaussian_label_noise(x_0, batch["gene_sd"].to(device), epoch, total_epoch)
        labels = batch["label"]
        t0 = time.time()
        
        
        
        is_train = self.training
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            t0 = time.time()
            # if cond_type == "reverse":
            #     flip_mask_sign = True
            # else:
            flip_mask_sign = False
            
            # When return_seq_info=True, returns (embeddings, seq_info, failed) or (embeddings, seq_info_all, failed)
            # When return_seq_info=False, returns (embeddings, failed)
            # _last_seq_info is stored internally by _build_dna_embeddings when return_seq_info=True
            # Only collect seq_info every 100 steps for cCRE count logging (to reduce overhead)
            collect_ccre_info = (step % 100 == 0) if step is not None else False
            
            # Time the entire _build_dna_embeddings call
            if self.verbose:
                t_build_dna_start = time.time()
            
            build_result = self._build_dna_embeddings(
                labels,
                cache_embeddings=cache_embeddings,
                sgRNAs=sgRNAs,
                jitter_window=jitter_window,
                flip_prob=flip_prob,
                flip_mask_sign=flip_mask_sign,
                return_seq_info=collect_ccre_info,  # Only enable every 100 steps for cCRE count logging
                DNA_emb_len=self.DNA_emb_len,
                ccre_intervals_dict=ccre_intervals_dict,
                z_ctrl=z_ctrl.squeeze(1),
            )
            
            if self.verbose:
                t_build_dna_end = time.time()
                print(f"Time for _build_dna_embeddings call: {t_build_dna_end - t_build_dna_start:.4f} sec")
            
            # Mark DNA embeddings completion
            timing['dna_emb_done'] = time.time()
            
            # Time result handling and failed label processing
            if self.verbose:
                t_result_handle_start = time.time()
            
            # Handle both return formats: (embeddings, failed) or (embeddings, seq_info, failed)
            if len(build_result) == 3:
                dna_ref_embeddings, _, failed_gene_labels = build_result
            else:
                dna_ref_embeddings, failed_gene_labels = build_result
            
            if self.verbose:
                print(f"Time to get DNA embeddings using match_gene_pert_mask_dna_embeddings takes: {time.time() - t0:.4f} sec per batch")
                print(f'dna_ref_embeddings dim: {dna_ref_embeddings.shape}')
                dna_ref_embeddings_mean = dna_ref_embeddings.mean(dim=0)
                dna_ref_embeddings_std = dna_ref_embeddings.std(dim=0)
                # print(f'dna_ref_embeddings_mean: {dna_ref_embeddings_mean}')
                print(f'dna_ref_embeddings_std: {dna_ref_embeddings_std}')
                print(f'dna_ref_embeddings std all: {dna_ref_embeddings.std()}')
            # Only move to device if not already on correct device (use non_blocking for async transfer)
            if dna_ref_embeddings.device != device:
                dna_ref_embeddings = dna_ref_embeddings.to(device, non_blocking=True)
            if len(failed_gene_labels) > 0:
                mask = torch.tensor(
                    [lab not in failed_gene_labels for lab in labels],
                    dtype=torch.bool,
                    device=device
                )
                warnings.warn(f'{len(failed_gene_labels)} samples out of {len(labels)} failed to get DNA embeddings: {failed_gene_labels}')
                
                z_0 = z_0[mask]
                z_ctrl = z_ctrl[mask]
                x_ctrl = x_ctrl[mask]
                x_0 = x_0[mask]
                dna_ref_embeddings = dna_ref_embeddings[mask]
                labels = [lab for lab, keep in zip(labels, mask.tolist()) if keep]
                cell_line_ids = cell_line_ids[mask]
                
            if self.verbose:
                t_result_handle_end = time.time()
                print(f"Time for result handling and failed label processing: {t_result_handle_end - t_result_handle_start:.4f} sec")
            
            # Time setup of z_src, z_tgt, x_src, x_tgt
            if self.verbose:
                t_src_tgt_setup_start = time.time()
            
            B = z_0.size(0)
            z_src, z_tgt = z_ctrl, z_0
            x_src, x_tgt = x_ctrl, x_0
            
            if self.verbose:
                t_src_tgt_setup_end = time.time()
                print(f"Time for z_src/z_tgt/x_src/x_tgt setup: {t_src_tgt_setup_end - t_src_tgt_setup_start:.4f} sec")
            
            # Time embedding building section (weighting + encoders + CLIP)
            if self.verbose:
                t_embed_build_start = time.time()
            
            ### heuristic weighting
            # ids = self._factorize_labels(labels, device=device)
            # uniq, counts = torch.unique(ids, return_counts=True)
            # n_p = counts[ids]  # fast indexing trick: map each sample to its group count
            # weights = 1.0 / n_p.float()
            
            
            ### Adaptive weighting
            if self.use_weighted_loss:
                # Use pre-factorized IDs from batch if available (avoids expensive DDP object collectives)
                if 'label_id' in batch:
                    # Pre-factorized IDs are already integers, just convert to tensor
                    label_ids = batch['label_id']
                    if isinstance(label_ids, torch.Tensor):
                        ids = label_ids.to(device, dtype=torch.long)
                    elif isinstance(label_ids, (list, tuple, np.ndarray)):
                        ids = torch.tensor(label_ids, dtype=torch.long, device=device)
                    else:
                        # Handle scalar or other types
                        ids = torch.tensor([int(x) for x in label_ids], dtype=torch.long, device=device)
                else:
                    # Fallback for backward compatibility (slower, uses DDP collectives)
                    ids = self.string_factorizer.encode(labels).to(device)
                weights = self.weighter.update_and_get_weights(ids, loss_device=device)
            else:
                weights = None
            
            # Mark label factorization/weighting completion
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing['label_factorization_done'] = time.time()
            
           


            if cond_type == "p_to_p":
                z_b = z_0.mean(dim=0, keepdim=True).detach()
                x_b = x_0.mean(dim=0, keepdim=True).detach()
                x_ctrl_b = x_ctrl.mean(dim=0, keepdim=True).detach()
                z_ctrl_b = z_ctrl.mean(dim=0, keepdim=True).detach()
                # maintain EMA and use the debiased estimate for current forward
                z_hat     = self._ema_update(self.z_mu_ema,     z_b,     self.ema_beta, self.z_mu_step)
                x_hat     = self._ema_update(self.x_mu_ema,     x_b,     self.ema_beta, self.x_mu_step)
                zc_hat    = self._ema_update(self.z_mu_ctrl_ema, z_ctrl_b,self.ema_beta, self.z_mu_ctrl_step)
                xc_hat    = self._ema_update(self.x_mu_ctrl_ema, x_ctrl_b,self.ema_beta, self.x_mu_ctrl_step)
                
                delta_z = (z_hat - zc_hat).detach().to(z_src.device)
                delta_x = (x_hat - xc_hat).detach().to(x_src.device)
                
                z_src = z_src + delta_z.expand(B, -1, -1)
                x_src = x_src + delta_x.expand(B, -1)
                
                

                if self.verbose:
                    ### compute cosine similarity between x_tgt and x_src
                    cos_mean, cos_std = offdiag_cosine_stats((z_tgt-z_src).squeeze(1))
                    print(f'Off-diagonal cosine similarity of z_tgt (after centering): mean {cos_mean.item():.4f} std {cos_std.item():.4f}')
                    print(f'In p_to_p or warmup, use z_src mean: {z_hat.flatten()[:5]} x_src mean: {x_hat.flatten()[:5]}')
                    print(f'x_tgt after centering, mean: {x_tgt.mean().item():.4f} std: {x_tgt.std().item():.4f}')
                    print(f'dna_ref_embeddings shape is {dna_ref_embeddings.shape}')

                # 2) Build embeddings (they return (B,1,D))
                pert_diff_emb_delta = self.pert_ln(self.shallow_diff_encoder(dna_ref_embeddings))    # (B,1,D)
                pert_pred_emb_delta = self.pert_pred_ln(self.shallow_encoder(dna_ref_embeddings))
                
                
                
                L = dna_ref_embeddings.size(1)
                null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)              # (B, D)
                null_pred = self.null_pred_ln(self.null_dna_pred_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)  # (B, D)
                null = null.unsqueeze(1).expand(B, L, -1)              # (B, L, D)
                null_pred = null_pred.unsqueeze(1).expand(B, L, -1)    # (B, L, D)
                
                
                # null = null.to(dna_ref_embeddings.dtype).expand(B, dna_ref_embeddings.size(1), -1)
                # null_pred = null_pred.to(dna_ref_embeddings.dtype).expand(B, dna_ref_embeddings.size(1), -1)
                
                
                
                # gamma = torch.tanh(gamma)
                
                # pert_diff_emb_delta = pert_diff_emb_delta * gamma + beta 
                
                pert_diff_emb = pert_diff_emb_delta
                pert_pred_emb = pert_pred_emb_delta 

                # 4) Conditions
                diff_cond = pert_diff_emb + null * 0.0
                pred_cond = pert_pred_emb + null_pred * 0.0
            else: ## unconditional or regular
                pert_diff_emb_delta = self.pert_ln(self.shallow_diff_encoder(dna_ref_embeddings))
                pert_pred_emb_delta = self.pert_pred_ln(self.shallow_encoder(dna_ref_embeddings))
                
                L = dna_ref_embeddings.size(1)
                null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)              # (B, D)
                null_pred = self.null_pred_ln(self.null_dna_pred_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)  # (B, D)
                null = null.unsqueeze(1).expand(B, L, -1)              # (B, L, D)
                null_pred = null_pred.unsqueeze(1).expand(B, L, -1)    # (B, L, D)


                if cond_type == "unconditional":
                    # null = null.to(dna_ref_embeddings.dtype).expand(B, dna_ref_embeddings.size(1), -1)
                    # null_pred = null_pred.to(dna_ref_embeddings.dtype).expand(B, dna_ref_embeddings.size(1), -1)
                    
                    # Ensure encoders are in computation graph even when unused
                    # Add delta * 0.0 so gradients flow but output is unchanged
                    pert_diff_emb = null + pert_diff_emb_delta * 0.0
                    pert_pred_emb = null_pred + pert_pred_emb_delta * 0.0
                else:
                    # null = null.to(dna_ref_embeddings.dtype).expand(B, dna_ref_embeddings.size(1), -1)
                    # null_pred = null_pred.to(dna_ref_embeddings.dtype).expand(B, dna_ref_embeddings.size(1), -1)

                    pert_diff_emb = pert_diff_emb_delta + null 
                    pert_pred_emb = pert_pred_emb_delta + null_pred 
                    
                diff_cond = pert_diff_emb
                pred_cond = pert_pred_emb
            
            # Mark encoders completion
            timing['encoders_done'] = time.time()
                    
            if self.verbose:
                ## compute pred_cond cosine similarity
                print(f'dna_ref_embeddings shape is: {dna_ref_embeddings.shape}; B is {B}')
                print(f'pert_diff_emb shape: {pert_diff_emb.shape} pert_pred_emb shape: {pert_pred_emb.shape}')

                
                print(f'pred_cond shape: {pred_cond.shape} pred_cond mean: {pred_cond.mean().item():.4f} std: {pred_cond.std().item():.4f}')
                
                norm_pred_cond = F.normalize(pred_cond.mean(dim=1, keepdim=False), p=2, dim=1)
                cos_sim_matrix = torch.matmul(norm_pred_cond, norm_pred_cond.T)

                print(f'Shape of pairwise cosine similarity matrix: {cos_sim_matrix.shape}')
                cos_sim_non_diag = cos_sim_matrix.flatten()[1:].reshape(B - 1, B + 1)[:, :-1].flatten()
                print(f'Pairwise cosine (pred_cond) similarity  (mean ± std) for unique pairs: {cos_sim_non_diag.mean().item():.4f} ± {cos_sim_non_diag.std().item():.4f}')
                
                
                norm_pred_cond = F.normalize(pert_pred_emb.mean(dim=1, keepdim=False), p=2, dim=1)
                cos_sim_matrix = torch.matmul(norm_pred_cond, norm_pred_cond.T)

                print(f'Shape of pairwise cosine similarity matrix: {cos_sim_matrix.shape}')
                cos_sim_non_diag = cos_sim_matrix.flatten()[1:].reshape(B - 1, B + 1)[:, :-1].flatten()
                print(f'Pairwise cosine (pert_pred_emb) similarity (mean ± std) for unique pairs: {cos_sim_non_diag.mean().item():.4f} ± {cos_sim_non_diag.std().item():.4f}')


            clip_dict = None
            if self.contrastive_loss:
                # 1. Active Case: Run full CLIP forward pass
                if use_clip:
                    clip_dict = self.CLIP_contrast(pert_diff_emb, z_tgt.squeeze(1), labels, x1=z_src.squeeze(1), z_ctrl=z_src.squeeze(1), verbose=self.verbose)
                # 2. Inactive Case: Tax Collector (tiny L2 so grads exist but compute is cheap)
                else:
                    # Tax Collector: tiny L2 so grads exist but compute is cheap
                    tau = getattr(self, 'clip_tax', 1e-8)  # Default tax rate, can be tuned
                    l2_tax = torch.zeros((), device=pert_diff_emb.device, dtype=pert_diff_emb.dtype)
                    for p in self.CLIP_contrast.parameters():
                        if p.requires_grad:
                            l2_tax = l2_tax + p.float().pow(2).sum()
                    clip_dict = {'loss': tau * l2_tax}
            
            # Mark CLIP completion
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing['clip_done'] = time.time()
                
            if self.use_VIC:
                lambda_var, lambda_cov = 0.5, 0.01   
                L_var = variance_floor_loss(pred_cond, gamma=1.0)
                L_cov = cov_offdiag_loss(pred_cond)

            
            
            # betas_mixed, A_eff, aux = self.schedule_scaler(cond)
            if self.beta_schedule == 'cosine':
                betas_mixed, _, _ = cosine_beta_schedule(self.T_train)
            else:
                betas_mixed, _, _ = get_official_betas(self.T_train)
            betas_mixed = betas_mixed.to(device)
            
            if self.verbose and diff_cond is not None:
                cond_mean = diff_cond.mean(dim=0)
                cond_std = diff_cond.std(dim=0)
                # print(f'cond_mean: {cond_mean}')
                # print(f'cond_std: {cond_std}')
            
            if self.verbose:
                t_embed_build_end = time.time()
                print(f"Time for embedding building (weighting + encoders + CLIP): {t_embed_build_end - t_embed_build_start:.4f} sec")
            
            t2 = time.time()
            
            
            # B = z_0.size(0)
            t = torch.rand((B,), device=device)
            betas_mixed = torch.clamp(betas_mixed, 1e-10, 0.999)
            self.train_betas = betas_mixed
            loss_diff, z_0_pred = i2sb_training_step(
                self.diffusion_model,
                z_tgt.detach(), z_src.detach(), t,
                (betas_mixed, None, None),
                cond=diff_cond,
                schedule_scaler=self.schedule_scaler,
                alpha = self.learned_alpha,
                return_x0_pred=True,
                dna_tokens=None,
                scaler=scaler,
                weightLoss=weights,
            )
            # z_0_pred = z_0_pred.detach() ## detach the z_0_pred to avoid gradient flow back to the Diffusion model
            t3 = time.time()
            if self.verbose:
                print(f"Time for Diffusion Step (i2sb_training_step): {t3 - t2:.4f} sec")
            
            # Mark diffusion completion
            timing['diffusion_done'] = time.time()
            # ratio = teacher_force_ratio(epoch, start=2, end=10)
            # else:
            use_tf = False  # don't use teacher force at eval
            
            


            t_pred = torch.zeros_like(t) if use_tf else t
            # time_emb = timestep_embedding(t_pred, self.ctrl_dim)
            

            if is_train:
                z_0_pred = z_tgt if use_tf else z_0_pred  
            
            if self.verbose:
                # print(f'pert_emb_mean shape: {pert_emb.mean(dim=0).shape}; pert_emb mean: {pert_emb.mean(dim=0)}; std: {pert_emb.std(dim=0)}')
                print(f'pred_cond std all: {pred_cond.std()}')
                

            # Original supervised predictor loss
            loss_pred, predict_dict_orig = self.predictor_head(
                pred_cond, x_src, ctrl_emb=z_src, pert_emb=z_0_pred, x_pert=x_tgt, time_emb=None, weights=weights
            ) 
            t4 = time.time()
            if self.verbose:
                print(f"Time for Predictor Head: {t4 - t3:.4f} sec")

            if diff_cond is not None:
                loss_diff_all = loss_diff['loss_diff'] 
            else:
                loss_diff_all = loss_diff['loss_diff']
            
            # ---------------- Consistency-Matching (CATDM analogue, one-sided efficient) ----------------
            loss_cm = None
            loss_cm_val = torch.tensor(float('nan'), device=device, dtype=loss_diff_all.dtype)
            if self.use_cm and (cond_type != "unconditional"):
                # Two nearby timesteps along the same trajectory (anchor one at main t)
                t2 = t.detach()  # reuse main diffusion timestep per sample
                t1 = (t2 - self.cm_dt * torch.rand_like(t2)).clamp(0.0, 1.0)  # ensure t1 <= t2

                # Predict x0 at t1 with the same condition (single extra forward)
                _, z1 = i2sb_training_step(
                    self.diffusion_model,
                    z_tgt.detach(), z_src.detach(), t1,
                    (betas_mixed, None, None),
                    cond=diff_cond,
                    schedule_scaler=self.schedule_scaler,
                    alpha=self.learned_alpha,
                    return_x0_pred=True,
                    dna_tokens=None,
                    scaler=scaler,
                    weightLoss=weights,
                )

                # Consistency on pooled outputs vs the main path prediction at t
                z1p = z1.mean(dim=1).float()
                z2p = z_0_pred.mean(dim=1).float().detach()  # main path as fixed target
                loss_cm = F.mse_loss(z1p, z2p)
                loss_cm_val = loss_cm.detach()

            # Compute uncertainty metrics if using uncertainty weighting
            uncertainty_metrics = None
            if self.use_adaptive_weight and self.training:
                if self.adapt_weight_type == 'uncertainty':
                    # Adaptive weighting across active losses: [diff, pred, (clip), (cm)]
                    # NOTE: shuffle is handled OUTSIDE uncertainty weighting (fixed coefficient)
                    # Map each potential loss to its learnable log_sigma2 slot.
                    diff_idx = 0
                    pred_idx = 1
                    next_idx = 2
                    clip_idx = None
                    if self.contrastive_loss:
                        clip_idx = next_idx
                        next_idx += 1
                    cm_idx = next_idx if self.use_cm else None

                    losses = [loss_diff_all, loss_pred]
                    loss_names = ['diff', 'pred']
                    loss_indices = [diff_idx, pred_idx]
                    if self.contrastive_loss and clip_dict is not None and ('loss' in clip_dict):
                        losses.append(clip_dict['loss'].to(loss_diff_all.dtype))
                        loss_names.append('clip')
                        if clip_idx is not None:
                            loss_indices.append(clip_idx)
                    if (self.use_cm and (cond_type != "unconditional") and (loss_cm is not None)):
                        losses.append(loss_cm.to(loss_diff_all.dtype))
                        loss_names.append('cm')
                        if cm_idx is not None:
                            loss_indices.append(cm_idx)
                    loss_vec = torch.stack(losses)
                    log_sigma2_eff = self.log_sigma2.clamp(self._log_sigma2_min, self._log_sigma2_max)
                    sigma2_eff = torch.exp(log_sigma2_eff)
                    sigma2_selected = sigma2_eff[loss_indices]
                    log_sigma2_selected = log_sigma2_eff[loss_indices]
                    weighted = 0.5 * (loss_vec / sigma2_selected + log_sigma2_selected)
                    total_loss = weighted.sum()
                    
                    # Store uncertainty metrics for logging
                    uncertainty_metrics = {
                        'names': loss_names,
                        'raw_losses': loss_vec.detach().cpu().float(),  # Raw losses before weighting
                        'log_sigma2': log_sigma2_selected.detach().cpu().float(),
                        'sigma2': sigma2_selected.detach().cpu().float(),  # Scale parameters
                        'weights': (0.5 / sigma2_selected).detach().cpu().float(),
                        'weighted_terms': weighted.detach().cpu().float(),
                        'active_k': len(loss_indices),
                        'indices': loss_indices,
                    }
                    
                    if self.use_VIC:
                        total_loss = total_loss + lambda_var * L_var + lambda_cov * L_cov
                else:
                    raise ValueError(f"Invalid adapt_weight_type: {self.adapt_weight_type}")
            else:
                # Non-adaptive path keeps fixed coefficients for regularizers
                total_loss = loss_diff_all + weight*loss_pred
                if self.contrastive_loss and clip_dict is not None and ('loss' in clip_dict):
                    total_loss = total_loss + self.w_clip * clip_dict['loss'].to(total_loss.dtype)
                # Add CM as fixed regularizer
                if (self.use_cm and (cond_type != "unconditional") and (loss_cm is not None)):
                    total_loss = total_loss + self.cm_weight * loss_cm

            # Mark loss computation completion
            timing['loss_computation_done'] = time.time()
                
            # if use_clip:
            #     total_loss = total_loss + clip_dict['loss']
            
            # Always include loss_clip (NaN when inactive)
            if self.contrastive_loss and clip_dict is not None and 'loss' in clip_dict:
                loss_clip_val = clip_dict['loss'].detach()
            else:
                loss_clip_val = torch.tensor(float('nan'), device=device, dtype=loss_diff_all.dtype)
            
            # Zero-batch safety tax to keep critical heads/encoders 'used' on all ranks
            if self.training:
                tau_zero = getattr(self, 'zero_batch_tax', 1e-8)
                if tau_zero > 0:
                    # B was defined earlier as z_0.size(0) after any masking
                    # if B == 0:
                    zero_tax = torch.zeros((), device=device, dtype=total_loss.dtype)
                    critical = [
                        getattr(self, 'DNA_model', None),       # <--- ADDED
                        getattr(self, 'diffusion_model', None), # <--- ADDED
                        getattr(self, 'shallow_encoder', None),
                        getattr(self, 'shallow_diff_encoder', None),
                        getattr(self, 'null_ln', None),
                        getattr(self, 'pert_pred_ln', None),
                        getattr(self, 'null_pred_ln', None),
                        getattr(self, 'predictor_head', None),
                        getattr(self, 'CLIP_contrast', None),   # <--- ADDED (Just in case)
                    ]
                    for mod in critical:
                        if mod is not None:
                            for p in mod.parameters():
                                if p.requires_grad:
                                    zero_tax = zero_tax + p.float().pow(2).sum()
                    
                    # Ensure null_dna_pred_emb parameter is always in computation graph
                    # This prevents DDP errors when it's conditionally used in different branches
                    if hasattr(self, 'null_dna_pred_emb') and self.null_dna_pred_emb.weight.requires_grad:
                        zero_tax = zero_tax + self.null_dna_pred_emb.weight.float().pow(2).sum()
                    
                    total_loss = total_loss + tau_zero * zero_tax
            
            out = {
                'loss_diff': loss_diff_all,
                'loss_pred': loss_pred,
                'loss_clip': loss_clip_val,
                'loss_cm': loss_cm_val,
                'total_loss': total_loss,
            }
            
            # Add uncertainty metrics if available
            if uncertainty_metrics is not None:
                out['uncertainty'] = uncertainty_metrics
            
            # Extract and add cCRE counts for logging
            if hasattr(self, '_last_seq_info') and self._last_seq_info:
                ccre_counts = []
                for si in self._last_seq_info:
                    if si and isinstance(si, list) and len(si) > 0:
                        si_dict = si[0] if isinstance(si[0], dict) else si
                        ccre_intervals_rel = si_dict.get('ccre_intervals_rel', [])
                        ccre_counts.append(len(ccre_intervals_rel) if ccre_intervals_rel else 0)
                    elif isinstance(si, dict):
                        ccre_intervals_rel = si.get('ccre_intervals_rel', [])
                        ccre_counts.append(len(ccre_intervals_rel) if ccre_intervals_rel else 0)
                if len(ccre_counts) > 0:
                    out['ccre_count'] = ccre_counts
            
            # no shuffle loss stats; CM tracked via loss_cm
            
            # Mark before return
            timing['before_return'] = time.time()
            
            if return_pert_emb:
                # detach to avoid autograd references, keep on CPU to reduce GPU mem pressure
                # Use non_blocking transfer for async CPU-GPU transfer (faster)
                out['pert_emb'] = pert_pred_emb_delta.mean(dim=1).detach().to('cpu', non_blocking=True)
                out['pert_diff_emb'] = pert_diff_emb_delta.mean(dim=1).detach().to('cpu', non_blocking=True)
                out['null_emb'] = null_pred.mean(dim=1).detach().to('cpu', non_blocking=True)
                # ensure labels here reflect the post-masking list (already your case)
                out['labels'] = list(labels)
            
            # Time function exit (captures DDP wrapper overhead on return)
            func_exit_sync = time.time()
            timing['func_exit_sync'] = func_exit_sync
            timing['before_return_to_exit'] = func_exit_sync - timing['before_return']
            
            # Store timing dictionary in output for analysis
            out['_timing'] = timing

        return out
    
    @torch.no_grad()
    def predict(self, x_ctrl, z_ctrl, labels, gene_annot, 
                sgRNAs, dna_record, cell_line_ids=None, cache_embeddings=None, z_0=None, 
                flip_mask_sign=False, debug=False, return_mask=False):
        """
        Predict perturbed RNA profile from DNA + control RNA.
        
        Args:
            x_ctrl: (B, RNA_dim) torch.Tensor
            labels: List[str] of gene names
            gene_annot: pandas.DataFrame
            sgRNAs: dict or lookup object for gRNA positions
            dna_record: dict of chromosome sequences
            cache_embeddings: LRUCache (optional)
        
        Returns:
            mu_hat: predicted non-zero mean (B, RNA_dim)
            gate: probability of being expressed (B, RNA_dim)
        """
        self.eval()
        device = next(self.parameters()).device

        x_ctrl = x_ctrl.to(device)
        cell_line_ids = self._coerce_cell_line_ids(cell_line_ids, B=len(labels), device=device)

        if z_0 is not None:
            z_0 = z_0.to(device) 
        if z_ctrl is not None:
            z_ctrl = z_ctrl.to(device) 
        
        # Get DNA embedding
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            if debug:
                dna_ref_embeddings, seq_info_all, failed_gene_labels = self._build_dna_embeddings(
                    labels,
                    cache_embeddings=cache_embeddings,
                    sgRNAs=sgRNAs,
                    jitter_window=0,
                    flip_prob=0.0,
                    flip_mask_sign=flip_mask_sign,
                    return_seq_info=True,
                    DNA_emb_len=self.DNA_emb_len,
                    z_ctrl=z_ctrl.squeeze(1),
                )
                print(f"seq_info_all: {seq_info_all[0][0]['proxy_start']} - {seq_info_all[0][0]['proxy_end']}")
                print(f"seq_info_all sequence: {seq_info_all[0][0]['seq'][:10]}")
                print(f'dna_ref_embeddings: {dna_ref_embeddings}')
            else:
                dna_ref_embeddings, failed_gene_labels = self._build_dna_embeddings(
                    labels,
                    cache_embeddings=cache_embeddings,
                    sgRNAs=sgRNAs,
                    jitter_window=0,
                    flip_prob=0.0,
                    flip_mask_sign=flip_mask_sign,
                    return_seq_info=False,
                    DNA_emb_len=self.DNA_emb_len,
                    z_ctrl=z_ctrl.squeeze(1),
                )
            
            if len(failed_gene_labels) > 0:
                mask = torch.tensor(
                    [lab not in failed_gene_labels for lab in labels],
                    dtype=torch.bool,
                    device=device
                )
                warnings.warn(f'{len(failed_gene_labels)} samples out of {len(labels)} failed to get DNA embeddings: {failed_gene_labels}')
                
                z_0 = z_0[mask]
                z_ctrl = z_ctrl[mask]
                x_ctrl = x_ctrl[mask]
                # x_0 = x_0[mask]
                dna_ref_embeddings = dna_ref_embeddings[mask]
                cell_line_ids = cell_line_ids[mask]
                labels = [lab for lab, keep in zip(labels, mask.tolist()) if keep]
            else:
                mask = torch.ones(len(labels), dtype=torch.bool, device=device)
            
            # Only move to device if not already on correct device (use non_blocking for async transfer)
            if dna_ref_embeddings.device != device:
                dna_ref_embeddings = dna_ref_embeddings.to(device, non_blocking=True)
            if dna_ref_embeddings.dim() == 2:
                dna_ref_embeddings = dna_ref_embeddings.unsqueeze(1)
            # cond = self.cond_adaptor(dna_ref_embeddings, z_ctrl)
            if z_0 is None:
                raise ValueError("Currently z_0 must be provided for prediction")
            if z_0.dim() == 2:
                orig_dim = z_0.shape
                z_0 = z_0.unsqueeze(1)
                print(f'Warning: z_0 has shape {orig_dim}, unsqueezing to {z_0.shape}')
            if z_ctrl.dim() == 2:
                orig_dim = z_ctrl.shape
                z_ctrl = z_ctrl.unsqueeze(1)
                print(f'Warning: z_ctrl has shape {orig_dim}, unsqueezing to {z_ctrl.shape}')
        
            
            pert_diff_emb = self.shallow_encoder(dna_ref_embeddings)
            pert_diff_emb = self.pert_pred_ln(pert_diff_emb)
            
            # null_pred = self.null_dna_pred_emb
            # null_pred = self.null_pred_ln(null_pred) ## (B, D)
            
            # null_pred = null_pred.to(dna_ref_embeddings.dtype).expand(dna_ref_embeddings.size(0), dna_ref_embeddings.size(1), -1)
            
            B = dna_ref_embeddings.size(0)
            null_pred = self.null_pred_ln(self.null_dna_pred_emb(cell_line_ids)).to(dna_ref_embeddings.dtype)  # (B, D)
            null_pred = null_pred.unsqueeze(1).expand(B, dna_ref_embeddings.size(1), -1)  # (B, L, D)
                        
            pert_diff_emb = pert_diff_emb + null_pred
            
            t = torch.zeros((z_0.size(0),), device=device)  # Dummy time for prediction
            time_emb = timestep_embedding(t, self.ctrl_dim)
            
            pred_dict = self.predictor_head(
                None, x_ctrl, ctrl_emb=z_ctrl, pert_emb=z_0,
            )
            
            
            if self.loss_metric == 'htg_nll':
                logits, mu, sigma, mu_ref = pred_dict['logits'], pred_dict['mu'], pred_dict['sigma'], pred_dict['mu_ref']
                # x_0_pred = predict_unconditional_mean(logits, mu, sigma)
                x_0_pred = predict_unconditional_mean_residual(logits, mu, sigma, mu_ref)
                # x_0_pred = predict_unconditional_mean_delta(logits, mu, sigma, x_ctrl)
            elif self.loss_metric == 'nll':
                mu, sigma = pred_dict['mu'], pred_dict['sigma']
                x_0_pred = mu
            elif self.loss_metric == 'mse':
                mu = pred_dict['mu']
                x_0_pred = mu
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
                    Ts=(2, 6, 10, 20, 30, 50, 100),
                    scale_list=(0.1, 1.0, 2.0, 5.0),
                    scaler=None, sample_limit=5000,
                    eval_mode='macro',):
        self.eval()
        device = next(self.parameters()).device
        ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp else 0
        world_sz = dist.get_world_size() if ddp else 1
        
        per_rank_limit = None
        if sample_limit is not None and sample_limit > 0:
            per_rank_limit = math.ceil(sample_limit / world_sz)

        # ---------- Baseline ----------
        local_gen, local_gt, local_label = [], [], []
        cum = 0
        for step, batch in enumerate(val_loader):
            bsz = batch["perturb"].size(0)
            if per_rank_limit is not None and cum >= per_rank_limit:
                break
            cum += bsz
            z0  = batch["perturb"].unsqueeze(1).to(device)
            zc  = batch["control"].unsqueeze(1).to(device)
            labels = batch["label"]
            cell_line_ids = self._coerce_cell_line_ids(batch.get("cell_line_id", None), B=bsz, device=device)
            local_gen.append(zc.squeeze(1).to('cpu', dtype=torch.float32))
            local_gt.append(z0.squeeze(1).to('cpu', dtype=torch.float32))
            local_label.extend(labels)

        if ddp:
            all_gen    = gather_and_concatenate_tensors(local_gen).to(torch.float32).cpu()
            all_truth  = gather_and_concatenate_tensors(local_gt).to(torch.float32).cpu()
            all_labels = gather_and_concatenate_list(local_label)
            dist.barrier()
        else:
            all_gen    = (torch.cat(local_gen, 0) if local_gen else torch.empty(0)).to(torch.float32).cpu()
            all_truth  = (torch.cat(local_gt,  0) if local_gt  else torch.empty(0)).to(torch.float32).cpu()
            all_labels = local_label

        if rank == 0 and all_gen.numel() > 0:
            gen_np   = all_gen.numpy()
            truth_np = all_truth.numpy()
            base_err = compute_e_distance(gen_np, truth_np, all_labels, average=eval_mode)
            print(f"Baseline e-distance: {base_err:.4f}")
        else:
            base_err = float('inf')

        best_error = base_err
        best_T, best_scale, best_eta = 0, 1.0, 0

        betas_full, _, _ = get_official_betas(1000)
        betas_full = betas_full.clamp_(1e-10, 0.999)
        A_full = torch.cumsum(betas_full, dim=0)

        # ---------- Search over (T, scale, eta) ----------
        for T in Ts:
            for scale in scale_list:
                for eta in [0,0.3,0.7,1]:
                    # per-setting local buffers
                    local_generated_samples = []
                    local_ground_truths = []
                    local_labels = []

                    betas, _, _ = get_official_betas(T)
                    if isinstance(betas, np.ndarray):
                        betas = torch.from_numpy(betas)
                    betas = betas.clamp_(1e-10, 0.999).to(device, dtype=torch.float32)
                    cum = 0
                    for step, batch in enumerate(val_loader):
                        bsz = batch["perturb"].size(0)
                        if per_rank_limit is not None and cum >= per_rank_limit:
                            break
                        cum += bsz
                        z_0   = batch["perturb"].unsqueeze(1).to(device)
                        z_ctrl= batch["control"].unsqueeze(1).to(device)
                        labels= batch["label"]
                        cell_line_ids = self._coerce_cell_line_ids(batch.get("cell_line_id", None), B=bsz, device=device)

                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            dna_ref_embeddings, _ = self._build_dna_embeddings(
                                labels,
                                cache_embeddings=cache_embeddings,
                                sgRNAs=sgRNAs,
                                jitter_window=0,
                                flip_prob=0.0,
                                flip_mask_sign=flip_mask_sign,
                                return_seq_info=False,
                                DNA_emb_len=self.DNA_emb_len,
                                z_ctrl=z_ctrl.squeeze(1),
                            )
                            # Only move to device if not already on correct device (use non_blocking for async transfer)
                            if dna_ref_embeddings.device != device:
                                dna_ref_embeddings = dna_ref_embeddings.to(device, non_blocking=True)

                            pert_emb = self.shallow_diff_encoder(dna_ref_embeddings)
                            pert_emb = self.pert_ln(pert_emb)
                            # null     = self.null_ln(self.null_dna_emb)
                            # if null.dim() == 2:
                            #     null = null.unsqueeze(1)
                            # null = null.to(dna_ref_embeddings.dtype).expand(
                            #     dna_ref_embeddings.size(0), dna_ref_embeddings.size(1), -1
                            # )
                            B = dna_ref_embeddings.size(0)
                            L = dna_ref_embeddings.size(1)
                            null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=dna_ref_embeddings.dtype)              # (B, D)
                            null = null.unsqueeze(1).expand(B, L, -1)              # (B, L, D)
                            # gamma = self.p2pFiLM(torch.cat([self.ctrl_ln(z_ctrl), pert_emb.detach()], dim=-1))
                            # gamma = torch.tanh(gamma) + 1.0
                            # pert_emb = pert_emb * gamma
                            
                            pert_null_emb = null
                            # Match old behavior: bypass cond_adaptor in inference
                            cond = pert_emb + null
                            
                            # Use the same null used to build cond as CFG unconditional anchor
                            null_cfg_diff = pert_null_emb

                            generated_sample = ddpm_sample_i2sb(
                                self.diffusion_model, x_T=z_ctrl,
                                betas=betas, alphas_cumprod=None, T=T,
                                eta=eta, cond=cond, cond_null=null_cfg_diff,
                                guidance_scale=scale,
                                schedule_scaler=self.schedule_scaler
                            )

                        # accumulate locally (CPU as in your original)
                        local_generated_samples.append(
                            generated_sample.squeeze(1).to('cpu', dtype=torch.float32)
                        )
                        local_ground_truths.append(
                            z_0.squeeze(1).to('cpu', dtype=torch.float32)
                        )
                        local_labels.extend(labels)

                    # ---- single gather & single metric per (T,scale,eta) ----
                    if ddp:
                        all_generated_samples = gather_and_concatenate_tensors(local_generated_samples).to(torch.float32).cpu()
                        all_ground_truths     = gather_and_concatenate_tensors(local_ground_truths).to(torch.float32).cpu()
                        all_labels = gather_and_concatenate_list(local_labels)  # not used by metric here
                        dist.barrier()
                    else:
                        all_generated_samples = (
                            torch.cat(local_generated_samples, 0) if local_generated_samples else torch.empty(0)
                        ).to(torch.float32).cpu()
                        all_ground_truths = (
                            torch.cat(local_ground_truths, 0) if local_ground_truths else torch.empty(0)
                        ).to(torch.float32).cpu()
                        all_labels = local_labels

                    if rank == 0 and all_generated_samples.numel() > 0:
                        gen_np   = all_generated_samples.numpy()
                        truth_np = all_ground_truths.numpy()
                        loss = compute_e_distance(gen_np, truth_np, all_labels, average=eval_mode)
                        print(f"[Eval] T={T}, scale={scale:.2f}, eta={eta}: e-distance={loss:.4f}")
                        if loss < best_error or (best_T == 0 and T > 0 and loss <= base_err * (1 + 1e-4)):
                            best_error, best_T, best_scale, best_eta = loss, T, scale, eta
                            print(f"New best error: {best_error:.4f} at T={best_T}, scale={best_scale:.2f}, eta={best_eta}")

                    torch.cuda.empty_cache()

        # --- Broadcast best parameters from rank 0 to all ranks ---
        if ddp:
            best_T_tensor    = torch.tensor(best_T if rank == 0 else 0, device=device, dtype=torch.int)
            best_scale_tensor= torch.tensor(best_scale if rank == 0 else 1.0, device=device, dtype=torch.float32)
            best_eta_tensor  = torch.tensor(best_eta if rank == 0 else 0, device=device, dtype=torch.int)
            dist.broadcast(best_T_tensor, src=0)
            dist.broadcast(best_scale_tensor, src=0)
            dist.broadcast(best_eta_tensor, src=0)
            best_T    = int(best_T_tensor.item())
            best_scale= float(best_scale_tensor.item())
            best_eta  = int(best_eta_tensor.item())

        # cache best schedule
        self.T_sample   = best_T
        self.best_scale = best_scale
        self.best_eta   = best_eta
        if best_T > 0:
            betas, _, _ = get_official_betas(best_T)
            if isinstance(betas, np.ndarray):
                betas = torch.from_numpy(betas)
            self.best_beta = betas.to(device, dtype=torch.float32)
            self.best_alphas = None
            self.best_alphas_cumprod = None
        else:
            self.best_beta = self.best_alphas = self.best_alphas_cumprod = None


        
    
    @torch.no_grad()
    def sample_latent(self, z_ctrl, labels, gene_annot, sgRNAs, dna_record, cell_line_ids=None, cache_embeddings=None, flip_mask_sign=False):
        self.eval()
        device = next(self.parameters()).device
        z_ctrl = z_ctrl.to(device)
        if z_ctrl.dim() == 2:
            z_ctrl = z_ctrl.unsqueeze(1)  # Ensure shape is (B, 1, RNA_dim)
            
        cell_line_ids = self._coerce_cell_line_ids(cell_line_ids, B=z_ctrl.size(0), device=device)
        
        if not hasattr(self, 'T_sample'):
            raise ValueError("T_sample not set. Run eval_diffusion first.")
        T = self.T_sample
        
        if T == 0:
            warnings.warn("No sampling is performed, returning control RNA as generated sample")
            return z_ctrl.squeeze(1)  # (B, RNA_dim)

        # z_ctrl = self.RNA_projector(z_ctrl)
        
        betas = self.best_beta 
        alphas_cumprod = self.best_alphas_cumprod
        scale = self.best_scale
        eta = self.best_eta
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            dna_ref_embeddings, failed_gene_labels = self._build_dna_embeddings(
                labels,
                cache_embeddings=cache_embeddings,
                sgRNAs=sgRNAs,
                jitter_window=0,
                flip_prob=0.0,
                flip_mask_sign=flip_mask_sign,
                return_seq_info=False,
                DNA_emb_len=self.DNA_emb_len,
                z_ctrl=z_ctrl.squeeze(1),
            )
            if dna_ref_embeddings.dim() == 2:
                dna_ref_embeddings = dna_ref_embeddings.unsqueeze(1)
            # Only move to device if not already on correct device (use non_blocking for async transfer)
            if dna_ref_embeddings.device != device:
                dna_ref_embeddings = dna_ref_embeddings.to(device, non_blocking=True)
            # cond = self.cond_adaptor(dna_ref_embeddings,z_ctrl)
            # pert_emb = self.deepset_encoder([dna_ref_embeddings], pool="sum")
            # pert_emb = self.deepset_diff_encoder([dna_ref_embeddings], pool="sum")
            print(f'dna_ref_embeddings shape is {dna_ref_embeddings.shape}')
            pert_emb = self.shallow_diff_encoder(dna_ref_embeddings)
            print(f'pert_emb shape is {pert_emb.shape}')
            pert_emb = self.pert_ln(pert_emb)
            
            # null = self.null_dna_emb
            # null = self.null_ln(null) ## (B, D)
            
            # if null.dim() == 2:
            #     null = null.unsqueeze(1)
                
            # null = null.to(pert_emb.dtype).expand(pert_emb.size(0), pert_emb.size(1),-1)
            
            
            B = pert_emb.size(0)
            L = pert_emb.size(1)
            null = self.null_ln(self.null_dna_emb(cell_line_ids)).to(dtype=pert_emb.dtype)              # (B, D)
            null = null.unsqueeze(1).expand(B, L, -1)              # (B, L, D)
            pert_emb = pert_emb + null
            pert_null_emb = null
            cond = pert_emb
            
            # Use same null as unconditional CFG anchor
            null_cfg_diff = pert_null_emb
                
            generated_sample =  ddpm_sample_i2sb(self.diffusion_model, x_T=z_ctrl, betas=betas, alphas_cumprod=alphas_cumprod, T=T, eta=eta, 
                                                    cond=cond, cond_null=null_cfg_diff,
                                                    guidance_scale=scale, schedule_scaler=self.schedule_scaler)
            
        return generated_sample.squeeze(1).to(torch.float32)  # (B, RNA_dim)
    
    def _merge_mean(self, mu_buf, cnt_buf, batch_mean, b):
        """
        mu_buf: (1,*,D) fp32 buffer
        cnt_buf: 0-D long buffer
        batch_mean: same shape as mu_buf, any dtype/device
        """
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
        """
        Debiased EMA (Adam-style); in-place on ema_buf & step_buf.
        Returns ema_hat (debiased) tensor view.
        """
        bm = batch_mean.detach().to(device=ema_buf.device, dtype=ema_buf.dtype)
        step = step_buf.item() + 1
        if step == 1:
            ema_buf.copy_(bm)
        else:
            ema_buf.mul_(beta).add_(bm, alpha=(1.0 - beta))
        step_buf.fill_(step)
        bias_corr = 1.0 - (beta ** step)
        return ema_buf / bias_corr
    
    def _build_dna_embeddings(self,
                              labels,
                              cache_embeddings=None,
                              sgRNAs=None,
                              jitter_window=0,
                              flip_prob=0.0,
                              flip_mask_sign=False,
                              return_seq_info=False,
                              DNA_emb_len=1,
                              ccre_intervals_dict=None,
                              z_ctrl: torch.Tensor = None):
        """
        Unified entry for DNA embeddings.
        Respects self.embed_backend. Keeps the original return contract.
        """
        backend = self.embed_backend

        # auto-detect once per batch if requested
        if backend == 'auto':
            if self.verbose:
                torch.cuda.synchronize()
                t_backend_detect_start = time.time()
            
            # if any label looks like interval, use interval path; otherwise gene
            use_interval = any(_looks_like_interval_label(l) for l in labels if isinstance(l, str))
            backend = 'interval' if use_interval else 'gene'
            
            if self.verbose:
                torch.cuda.synchronize()
                t_backend_detect_end = time.time()
                print(f"Time for backend detection (auto): {t_backend_detect_end - t_backend_detect_start:.4f} sec")

        if backend == 'gene':
            # Minimal path: cache lookup + average pooling only (no CPU work, no model inference)
            call_verbose = self.verbose
            
            # Update disk_cache device to match model's device (for DDP compatibility)
            if self.disk_cache is not None:
                try:
                    # Detect device from model parameters
                    model_device = next(self.parameters()).device
                    if self.disk_cache.device != model_device:
                        self.disk_cache.update_device(model_device)
                except (StopIteration, AttributeError):
                    # Fallback if no parameters or device detection fails
                    pass
            
            if self.verbose:
                torch.cuda.synchronize()
                t_gene_path_start = time.time()
            
            dna_ref_embeddings, failed_gene_labels = match_gene_pert_mask_dna_embeddings_minimal(
                gene_labels=labels,
                cache=cache_embeddings,
                disk_cache=self.disk_cache,
                DNA_emb_dim=self.dna_dim,
                VERBOSE=call_verbose,
            )
            
            if self.verbose:
                torch.cuda.synchronize()
                t_gene_path_end = time.time()
                print(f"Time for gene path (match_gene_pert_mask_dna_embeddings_minimal): {t_gene_path_end - t_gene_path_start:.4f} sec")
            
            # Minimal function doesn't return seq_info, so set to None
            if return_seq_info:
                self._last_seq_info = None
            
            # Return format: (embeddings, failed) - always 2-tuple
            return dna_ref_embeddings, failed_gene_labels
        else:
            # INTERVAL PATH - cache-only with disk cache support
            call_verbose = self.verbose
            
            if self.verbose:
                torch.cuda.synchronize()
                t_interval_path_start = time.time()
            
            if return_seq_info:
                dna_ref_embeddings, seq_info_all, failed = match_interval_pert_mask_dna_embeddings_cached(
                    labels,
                    self.DNA_model,
                    cache=cache_embeddings,
                    disk_cache=self.disk_cache,
                    dna_record=self.dna_records,
                    FIX_WIN=524_288,
                    jitter_window=jitter_window,
                    return_seq_info=True,
                    flip_seq_prob=flip_prob,
                    seed=0,
                    max_batch_size=32,
                    DNA_emb_dim=self.dna_dim,
                    drop_mask_prob=self.drop_mask_prob,
                    flip_mask_sign=flip_mask_sign,
                    VERBOSE=call_verbose,
                    ccre_intervals_dict=ccre_intervals_dict,
                    z_ctrl=z_ctrl,
                )
                
                if self.verbose:
                    torch.cuda.synchronize()
                    t_interval_path_end = time.time()
                    print(f"Time for interval path (match_interval_pert_mask_dna_embeddings_cached with seq_info): {t_interval_path_end - t_interval_path_start:.4f} sec")
                
                # Store seq_info for cCRE count extraction
                self._last_seq_info = seq_info_all
                return dna_ref_embeddings, seq_info_all, failed
            else:
                dna_ref_embeddings, failed = match_interval_pert_mask_dna_embeddings_cached(
                    labels,
                    self.DNA_model,
                    cache=cache_embeddings,
                    disk_cache=self.disk_cache,
                    dna_record=self.dna_records,
                    FIX_WIN=524_288,
                    jitter_window=jitter_window,
                    return_seq_info=False,
                    flip_seq_prob=flip_prob,
                    seed=0,
                    max_batch_size=32,
                    DNA_emb_dim=self.dna_dim,
                    drop_mask_prob=self.drop_mask_prob,
                    flip_mask_sign=flip_mask_sign,
                    VERBOSE=call_verbose,
                    ccre_intervals_dict=ccre_intervals_dict,
                    z_ctrl=z_ctrl,
                )
                
                if self.verbose:
                    torch.cuda.synchronize()
                    t_interval_path_end = time.time()
                    print(f"Time for interval path (match_interval_pert_mask_dna_embeddings_cached without seq_info): {t_interval_path_end - t_interval_path_start:.4f} sec")
                
                self._last_seq_info = None
                return dna_ref_embeddings, failed
        
    def global_mean_zero(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D)
        mu = z.mean(dim=0, keepdim=True)           # (1, D)
        return (mu.pow(2).sum() / z.shape[1])      # scale-invariant
    
    
    def center_mean_zero(self, z: torch.Tensor, pert_ids: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) DNA embeddings for samples
        pert_ids: (B,) integer ids for perturbations
        Returns: scalar = || mean over per-pert centers ||^2 / D
                (each perturbation contributes equally, duplicates don't overweight)
        """
        B, D = z.shape
        uniq, inv = torch.unique(pert_ids, return_inverse=True)  # inv maps samples -> center row
        T = uniq.numel()
        sums   = torch.zeros(T, D, device=z.device, dtype=z.dtype)
        counts = torch.zeros(T, 1, device=z.device, dtype=z.dtype)
        idx = inv.view(-1, 1).expand(B, D)
        sums.scatter_add_(0, idx, z)
        ones = torch.ones(B, 1, device=z.device, dtype=z.dtype)
        counts.scatter_add_(0, inv.view(-1, 1), ones)
        centers = sums / counts.clamp_min(1.0)        # (T, D)
        mu = centers.mean(dim=0, keepdim=True)        # (1, D) — equal weight per perturbation
        return (mu.pow(2).sum() / D)
    
    def _factorize_labels(self, labels: list, device: torch.device):
        """
        Convert a list of labels (str/int) into contiguous integer IDs [0..T-1].
        Each unique label gets one ID, preserving first occurrence order.
        Returns a torch.LongTensor on the specified device.
        """
        # Fast path: handle empty input
        if not labels:
            return torch.empty(0, dtype=torch.long, device=device)

        # Build mapping from label -> integer ID
        unique_to_id = {}
        ids = []
        for lab in labels:
            if lab not in unique_to_id:
                unique_to_id[lab] = len(unique_to_id)
            ids.append(unique_to_id[lab])

        # Convert to tensor
        return torch.tensor(ids, dtype=torch.long, device=device)
    
    
    def _label_mismatch_perm(self, ids: torch.Tensor, device: torch.device, max_tries: int = 50):
        """
        Macro shuffling: Shuffle perturbation label assignments at the group level.
        
        For a batch with perturbation labels, groups samples by label and shuffles
        which perturbation each group gets assigned. This is more robust than
        per-sample permutation for unbalanced batches.
        
        Returns permutation array perm where diff_cond[perm] has shuffled labels,
        or None if shuffling is not feasible (e.g., all labels identical).
        """
        B = ids.size(0)
        if B < 2 or torch.unique(ids).numel() < 2:
            if self.verbose:
                print(f'No permutation possible for labels: {ids}')
            return None
        
        # Get unique labels and their counts
        unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
        P = unique_ids.numel()
        
        if P < 2:
            return None
        
        # Create shuffled mapping of unique labels (ensuring non-identity)
        # Try multiple times to ensure we get a non-identity permutation
        label_perm = None
        for _ in range(max_tries):
            candidate = torch.randperm(P, device=device)
            if not torch.all(candidate == torch.arange(P, device=device)):
                label_perm = candidate
                break
        
        # If still identity after max_tries, explicitly swap labels
        if label_perm is None or torch.all(label_perm == torch.arange(P, device=device)):
            label_perm = torch.arange(P, device=device)
            # Swap all adjacent pairs to ensure non-identity
            for i in range(0, P - 1, 2):
                label_perm[i], label_perm[i + 1] = label_perm[i + 1], label_perm[i]
            # If odd number, swap last with first
            if P % 2 == 1:
                label_perm[0], label_perm[-1] = label_perm[-1], label_perm[0]
        
        # Build mapping: original_label -> shuffled_label
        shuffled_labels = unique_ids[label_perm]
        
        # For each unique label, find all sample indices with that label
        # and map them to sample indices with the shuffled label
        # CRITICAL: Ensure no fixed points (perm[i] != i) while maintaining label mismatch
        perm = torch.zeros(B, dtype=torch.long, device=device)
        ar = torch.arange(B, device=device)
        
        for orig_idx, orig_label in enumerate(unique_ids):
            shuffled_label = shuffled_labels[orig_idx]
            
            # Find all samples with original label
            mask_orig = (ids == orig_label)
            orig_indices = torch.where(mask_orig)[0]
            n_orig = orig_indices.numel()
            
            # Find all samples with shuffled label
            mask_shuffled = (ids == shuffled_label)
            shuffled_indices = torch.where(mask_shuffled)[0]
            n_shuffled = shuffled_indices.numel()
            
            # Map original samples to shuffled positions, ensuring no fixed points
            # Strategy: create a derangement (permutation with no fixed points) within the mapping
            if n_orig == n_shuffled:
                # Same count: create a derangement to avoid fixed points
                if n_orig == 1:
                    # Single sample case: can't avoid mapping to itself if only one target
                    # But we require different label, so this should be fine
                    perm[orig_indices] = shuffled_indices
                else:
                    # Create a derangement: try permutations until we find one with no fixed points
                    derangement_found = False
                    for _ in range(max_tries):
                        candidate_order = torch.randperm(n_shuffled, device=device)
                        candidate_mapping = shuffled_indices[candidate_order]
                        # Check if any fixed points relative to orig_indices
                        if torch.all(candidate_mapping != orig_indices):
                            perm[orig_indices] = candidate_mapping
                            derangement_found = True
                            break
                    if not derangement_found:
                        # Fallback: use any permutation, we'll fix fixed points later
                        perm[orig_indices] = shuffled_indices[torch.randperm(n_shuffled, device=device)]
            elif n_orig < n_shuffled:
                # Fewer original samples: randomly select from shuffled positions
                # Ensure we don't select positions that match orig_indices
                available = shuffled_indices
                for orig_pos in orig_indices:
                    # Remove orig_pos from candidates if it appears
                    available = available[available != orig_pos]
                if len(available) >= n_orig:
                    selected = available[torch.randperm(len(available), device=device)[:n_orig]]
                    perm[orig_indices] = selected
                else:
                    # Fallback: allow some overlap, will fix fixed points later
                    selected = torch.randperm(n_shuffled, device=device)[:n_orig]
                    perm[orig_indices] = shuffled_indices[selected]
            else:
                # More original samples: randomly sample with replacement, but avoid fixed points
                for i, orig_pos in enumerate(orig_indices):
                    # Find candidates that are not the original position
                    candidates = shuffled_indices[shuffled_indices != orig_pos]
                    if len(candidates) > 0:
                        perm[orig_pos] = candidates[torch.randint(0, len(candidates), (1,), device=device)]
                    else:
                        # Fallback: use any, will fix later
                        perm[orig_pos] = shuffled_indices[torch.randint(0, n_shuffled, (1,), device=device)]
        
        # CRITICAL: Fix any remaining fixed points (perm[i] == i) while maintaining label mismatch
        fixed_points = (perm == ar)
        if fixed_points.any():
            fixed_indices = torch.where(fixed_points)[0]
            # Build a pool of all valid swap targets (different label and different position)
            for idx in fixed_indices:
                # Find all positions with different label that are not themselves
                candidates = torch.where((ar != idx) & (ids != ids[idx]))[0]
                
                if candidates.numel() == 0:
                    # No valid candidates - this should be rare but possible
                    if self.verbose:
                        print(f"Warning: Cannot fix fixed point at {idx} - no valid candidates")
                    return None
                
                # Try to find a candidate that also has a fixed point, so we can swap
                # This maintains the label mismatch property
                swap_candidates = candidates[perm[candidates] == candidates]
                if swap_candidates.numel() > 0:
                    # Swap with another fixed point
                    swap_idx = swap_candidates[0]
                    perm[idx], perm[swap_idx] = perm[swap_idx], perm[idx]
                else:
                    # Find a candidate whose current mapping is not idx
                    non_self_candidates = candidates[perm[candidates] != idx]
                    if non_self_candidates.numel() > 0:
                        # Create a cycle: idx -> candidate -> perm[candidate]
                        target = non_self_candidates[0]
                        perm[idx] = target
                        # Now we need to fix where target was pointing
                        # Find something that was pointing to target
                        pointing_to_target = torch.where(perm == target)[0]
                        if pointing_to_target.numel() > 0 and pointing_to_target[0] != idx:
                            perm[pointing_to_target[0]] = idx
                    else:
                        # Last resort: just pick any candidate
                        perm[idx] = candidates[torch.randint(0, candidates.numel(), (1,), device=device)]
        
        # Final verification: ensure ALL samples map to different position AND different label
        no_fixed_points = torch.all(perm != ar)
        different_labels = torch.all(ids[perm] != ids)
        
        if not (no_fixed_points and different_labels):
            if self.verbose:
                print(f"Macro shuffling verification failed: fixed_points={not no_fixed_points}, "
                      f"same_labels={not different_labels}")
            return None
        
        return perm
    
    
    def _coerce_cell_line_ids(self, cell_line_ids, B: int, device: torch.device) -> torch.Tensor:
        """
        Returns LongTensor shape (B,) on device, values in [0, self.cell_line_nums-1].
        Accepts list/np array/torch tensor/None.
        """
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

    
    
    def save_checkpoint(self, path, optimizer=None, epoch=None):
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'epoch': epoch
        }, path)

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['model_state'])
        if optimizer and checkpoint.get('optimizer_state'):
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint.get('epoch', None)
