# %%
'''
Train STRAND on one dataset (single GPU).

The data split (zero-shot or few-shot) is fixed upstream, by the OT index files written by
scripts/02_embed_and_match and scripts/03_make_splits; --run_group only labels the output.
Checkpoints go to <ckpt_root>/checkpoints/<run_group>/SingleGPU/<dname>/<version><ablations>/<date>/.
'''
# %load_ext autoreload
# %autoreload 2
import socket
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import random
import argparse
from datetime import datetime
import hashlib
import sys
import pickle

# Optional dependencies
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from refactored structure
from src.diffusion_ot.data import (
    perturb_control_dataloader,
    LRUCache,
)
from src.diffusion_ot.models.strand_model import STRANDModel
from PModule.DNA_modeling import BorzoiPerturbAdapter
from src.diffusion_ot.utils import (
    get_dna_record,
    get_DNA_annot,
    load_ccres_bed,
)
from src.diffusion_ot.utils.cell_line_mixture import (
    compute_cell_line_prototypes_from_arrays,
    infer_cell_line_mix_weights_from_batch,
)
from src.diffusion_ot.schedulers import symmetric_beta, EarlyStopping, get_official_betas
from src.diffusion_ot.training import (
    flatten_dict,
    GroupPlateau,
    get_saved_lrs,
    init_saved_lrs,
    refresh_saved_lrs_from_optimizer,
    toggle_grad,
    load_config,
)
from src.diffusion_ot.evaluation import compute_e_distance
from argparse import Namespace

# --- Helper Functions (Adapted for Single GPU) ---

def _ensure_single_process_group():
    try:
        if dist.is_available() and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            dist.init_process_group(backend=backend, rank=0, world_size=1)
    except Exception as e:
        print(f"[warn] single-process group init skipped: {e}")

def _macro_mse(arr_a: np.ndarray, arr_b: np.ndarray, lbls: np.ndarray):
    """Compute macro-averaged MSE across unique labels."""
    unique_lbls = np.unique(lbls)
    per_group = {}
    for lbl in unique_lbls:
        mask = (lbls == lbl)
        if not np.any(mask):
            continue
        per_group[str(lbl)] = float(np.mean((arr_a[mask] - arr_b[mask]) ** 2))
    if not per_group:
        return float('nan'), {}
    macro_val = float(np.mean(list(per_group.values())))
    return macro_val, per_group



def main_worker(config, args):
    # --- Setup ---
    print("\n===== Single GPU Training Run Configuration =====")
    for k, v in config.items(): print(f"{k}: {v}")
    if hasattr(args, '__dict__'):
        for k, v in vars(args).items(): print(f"{k}: {v}")
    print("================================================\n")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dname = args.dname
    batch_size = config['train']['batch_size']
    T_train = config['model']['T_train']
    RNA_loss = args.RNA_loss
    out_suffix = args.out_suffix
    wandb_name = args.wandb_name
    USE_SCHEDULER = not getattr(args, 'no_lr_scheduler', False)
    
    # Ablation Suffix
    ablate_distance_bias = getattr(args, 'ablate_distance_bias', False)
    ablate_ccre_bias = getattr(args, 'ablate_ccre_bias', False)
    cell_bias_mode = getattr(args, 'cell_bias_mode', 'normal')
    on_off = lambda ablated: "off" if ablated else "on"
    ablation_suffix = (
        f"_distbias-{on_off(ablate_distance_bias)}"
        f"_ccrebias-{on_off(ablate_ccre_bias)}"
        f"_cellbias-{cell_bias_mode}"
    )
    out_suffix = out_suffix + ablation_suffix
    USE_LR_SCHEDULER = True

    idx_paths = [f"{config['IO']['idx_root']}/{dname}/"]
    namelist = [dname]
    root_dir = config['IO']['root_dir']
    anndata_path = config['IO'].get('anndata_path', None)
    cell_line_mapping = config.get("cell_line_mapping", None)
    h5_dataset_name = config['IO'].get('h5_dataset_name', None)
    anndata_dataset_name = config['IO'].get('anndata_dataset_name', None)

    # --- Data Loading ---
    resume_checkpoint = getattr(args, 'resume_checkpoint', None)
    loaded_scaler = None
    start_epoch = 0
    
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        checkpoint_dir = os.path.dirname(resume_checkpoint)
        save_name = os.path.basename(resume_checkpoint).replace('.pt','')
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith('scaler_') and save_name in fname:
                with open(os.path.join(checkpoint_dir, fname), 'rb') as f:
                    loaded_scaler = pickle.load(f)
                break

    train_dataset = perturb_control_dataloader(
        namelist=namelist, idx_paths=idx_paths, root_dir=root_dir,
        h5_suffix='.X_scimilarity.h5', idx_suffix=config['IO']['idx_train_suffix'],
        scale=True, 
        cell_line_mapping=cell_line_mapping,
        anndata_path=anndata_path,
        h5_dataset_name=h5_dataset_name,
        anndata_dataset_name=anndata_dataset_name,
        # Conservative: training never allows unknown cell lines unless user explicitly enables mixture inference
        allow_unknown_cell_lines=False,
    )
    scaler = train_dataset.scaler
    
    val_dataset = perturb_control_dataloader(
        namelist=[dname], idx_paths=idx_paths, root_dir=root_dir,
        h5_suffix='.X_scimilarity.h5', idx_suffix=config['IO']['idx_val_suffix'],
        scale=train_dataset.scaler,
        cell_line_mapping=cell_line_mapping,
        anndata_path=anndata_path,
        h5_dataset_name=h5_dataset_name,
        anndata_dataset_name=anndata_dataset_name,
        allow_unknown_cell_lines=False,
    )

    test_dataset = perturb_control_dataloader(
        namelist=[dname], idx_paths=idx_paths, root_dir=root_dir,
        h5_suffix='.X_scimilarity.h5', idx_suffix=config['IO']['idx_test_suffix'],
        scale=train_dataset.scaler,
        cell_line_mapping=cell_line_mapping,
        anndata_path=anndata_path,
        h5_dataset_name=h5_dataset_name,
        anndata_dataset_name=anndata_dataset_name,
        # Only allow unknown cell lines when explicitly requested for unseen-cell-line evaluation.
        allow_unknown_cell_lines=bool(getattr(args, "infer_cell_line_mix_weights", False)),
    )

    # Checkpoint Path
    # save_path = './checkpoints/SingleGPU/'
   
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # e.g. 20251216_093000
    date_str = timestamp[:8]                             # e.g. 20251216
    run_folder_name = date_str
    
    ckpt_root = getattr(args, 'ckpt_root', os.environ.get("OUTPUT_ROOT", os.path.join(project_root, "outputs")))
    run_group = getattr(args, 'run_group', 'zero_shot')
    base_ckpt_dir = os.path.join(ckpt_root, 'checkpoints', run_group, 'SingleGPU')
    
    if '.sub.' in config['IO']['idx_val_suffix']: dname_save = dname + '_sub'
    else: dname_save = dname
    
    # Include ablation suffix in version for unique checkpoint paths
    version_with_ablation = args.version + ablation_suffix
    save_path = os.path.join(base_ckpt_dir, dname_save, version_with_ablation, run_folder_name)
    
    os.makedirs(save_path, exist_ok=True)
    save_name = f"{dname_save}_{RNA_loss}_base_{args.version}_{out_suffix}"
    
    with open(os.path.join(save_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    if USE_SCHEDULER: save_name_final = save_name + ".pt"
    else: save_name_final = save_name + "_no_scheduler.pt"

    # Loaders (Single GPU - no DistributedSampler)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # --- Resources ---
    dna_record = get_dna_record()
    gene_annot = get_DNA_annot()
    
    ccre_intervals_dict = None
    ccre_bed_path = getattr(args, 'ccre_bed_path', None)
    if ccre_bed_path and ccre_bed_path != '':
        print(f"[cCRE] Loading cCRE intervals from: {ccre_bed_path}")
        ccre_intervals_dict = load_ccres_bed(ccre_bed_path)
    
    # --- Model ---
    borzoipert = BorzoiPerturbAdapter(load_backbone=False, 
                                      out_dim=1536, cache_pool_k=1,
                                      ablate_distance_bias=args.ablate_distance_bias,
                                      ablate_ccre_bias=args.ablate_ccre_bias,
                                      cell_bias_mode=args.cell_bias_mode).to(device)
    cache_embeddings = LRUCache(max_size=3072, device=device, )
    dna_cache_dir = (args.dna_cache_dir or os.environ.get("DNA_CACHE_DIR")
                     or config['IO'].get('dna_cache_dir') or os.path.join(project_root, "outputs", "dna_cache"))

    model = STRANDModel(
        DNA_model=borzoipert,
        T_train=T_train,
        RNA_dim=train_dataset.RNA_dim,
        dna_dim=config['model']['DNA_embed_dim'],
        use_scheduler=USE_SCHEDULER,
        dna_records=dna_record,
        gene_annot=gene_annot,
        loss_metric=RNA_loss,
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        cond_dim=config['model']['cond_dim'],
        hidden_dim=config['model']['hidden_dim'],
        embedding_dim=config['model']['embedding_dim'],
        use_adaptive_weight=True,
        drop_mask_prob=0.0,
        DNA_emb_len=1,
        dna_cache_dir=dna_cache_dir,
        dna_cache_k=1,
        cell_line_nums=len(cell_line_mapping) if cell_line_mapping else 1,
        verbose=False,
    ).to(device)

    # -------- Optional: unseen cell-line mixture inference (OFF by default) --------
    # This block is a conservative extension: it is only used if the user explicitly
    # enables it via --infer_cell_line_mix_weights. Otherwise nothing changes.
    prototypes = None
    if bool(getattr(args, "infer_cell_line_mix_weights", False)):
        if cell_line_mapping is None or len(cell_line_mapping) <= 0:
            raise ValueError(
                "--infer_cell_line_mix_weights requires a non-empty `cell_line_mapping` in config "
                "containing the *seen* cell lines."
            )
        if not hasattr(train_dataset, "control") or not hasattr(train_dataset, "cell_line_ids"):
            raise ValueError(
                "Train dataset must expose `.control` and `.cell_line_ids` to compute prototypes."
            )
        # Prototypes in the same control embedding space used by the model (scaled if scaler enabled).
        proto_np = compute_cell_line_prototypes_from_arrays(
            control_emb=np.asarray(train_dataset.control),
            cell_line_ids=np.asarray(train_dataset.cell_line_ids),
            num_cell_lines=len(cell_line_mapping),
        )
        prototypes = torch.from_numpy(proto_np).to(device=device, dtype=torch.float32)
        print(f"[unseen-cell-line] Enabled mixture inference. Prototypes shape: {tuple(prototypes.shape)}")

    # --- Optimizer & Parameter Groups (Exact port from DDP) ---
    diff_params = list(model.diffusion_model.parameters())
    dna = model.DNA_model
    adapt_params = [p for p in dna.parameters() if p.requires_grad]
    
    clip_params = []
    if hasattr(model, 'CLIP_contrast') and model.CLIP_contrast is not None:
        clip_params = list(model.CLIP_contrast.parameters())
    use_clip = len(clip_params) > 0

    pred_params = [
        p for n, p in model.named_parameters()
        if (p.requires_grad and p not in set(diff_params) and p not in set(adapt_params) and p not in set(clip_params))
    ]

    base_lr = config['train']['lr'] * 20
    diff_lr = base_lr *0.5
    adapt_lr = base_lr *0.5
    pred_lr = base_lr *0.5
    weight_decay = config['train']['weight_decay'] * 0.05

    def split_hidden_nonhidden(params):
        hidden = [p for p in params if p.requires_grad and getattr(p, "ndim", 0) >= 2]
        nonhidden = [p for p in params if p.requires_grad and getattr(p, "ndim", 0) < 2]
        return hidden, nonhidden

    diff_hidden, diff_nonhidden = split_hidden_nonhidden(diff_params)
    adapt_hidden, adapt_nonhidden = split_hidden_nonhidden(adapt_params)
    pred_hidden, pred_nonhidden = split_hidden_nonhidden(pred_params)
    clip_hidden, clip_nonhidden = split_hidden_nonhidden(clip_params)

    # Matrices (hidden) and vectors (nonhidden) keep separate groups: the plateau schedulers below
    # address groups by position.
    param_groups = [
        dict(params=diff_hidden,     lr=diff_lr,  eps=1e-8),
        dict(params=diff_nonhidden,  lr=diff_lr,  eps=1e-10),
        dict(params=adapt_hidden,    lr=adapt_lr, eps=1e-8),
        dict(params=adapt_nonhidden, lr=adapt_lr, eps=1e-10),
        dict(params=pred_hidden,     lr=pred_lr,  eps=1e-8),
        dict(params=pred_nonhidden,  lr=pred_lr,  eps=1e-10),
    ]
    if use_clip:
        param_groups += [
            dict(params=clip_hidden,    lr=pred_lr, eps=1e-8),
            dict(params=clip_nonhidden, lr=pred_lr, eps=1e-10),
        ]

    _ensure_single_process_group()
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=weight_decay)

    init_saved_lrs(optimizer)
    
    if USE_LR_SCHEDULER:
        sched_diff = GroupPlateau(optimizer, [0, 1], mode='min', factor=0.5, patience=3, min_lr=1e-7, verbose=True)
        sched_pred = GroupPlateau(optimizer, [2, 3, 4, 5] + ([6, 7] if use_clip else []), mode='min', factor=0.5, patience=3, min_lr=1e-7, verbose=True)

    betas, alphas, alphas_cumprod = symmetric_beta(T_train)
    betas, alphas, alphas_cumprod = betas.to(device), alphas.to(device), alphas_cumprod.to(device)

    early_stopper = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True)

    # Resume Logic
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from {resume_checkpoint} at epoch {start_epoch}")

    # WandB
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but --use_wandb was set. Install wandb or disable the flag.")
        run_id = hashlib.md5(datetime.now().strftime("%Y%m%d_%H%M%S").encode()).hexdigest()[:8]
        flat_config = flatten_dict(config)
        flat_args = {f"arg/{k}": v for k,v in vars(args).items()}
        combined = {**flat_config, **flat_args}
        combined.update({"diff_lr": diff_lr, "adapt_lr": adapt_lr, "pred_lr": pred_lr})
        wandb.init(project=f"{args.project_name}_{args.version}", name=f"{dname}_{RNA_loss}_{wandb_name}_singleGPU_{run_id}", config=combined)

    # --- Training Loop ---
    step = 0
    warmup_step = 0
    prev_checkpoint_path = None
    total_step = 100 * 32 // batch_size 

    for epoch in range(start_epoch, config['train']['num_epochs']):
        model.train()
        total_loss = 0.0
        total_loss_diff = 0.0
        total_loss_pred = 0.0
        train_len = 0
        
        teacher_forcing = True if epoch < 0 else False
        toggle_grad(diff_params, enable=not teacher_forcing)

        data_loading_start = time.time()
        
        for batch in train_loader:
            data_loading_end = time.time()
            
            if cache_embeddings: cache_embeddings.reset_step_counters()
            if teacher_forcing: warmup_step += 1
            want_emb_log = (args.use_wandb and (step % 100 == 0))
            step += 1

            optimizer.zero_grad(set_to_none=True)
            
            # Forward
            forward_start = time.time()
            loss_dict = model(
                batch=batch, sgRNAs=None, betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod,
                cache_embeddings=cache_embeddings, weight=config['train']['loss_weight'],
                activate_scheduler=True, teacher_forcing=teacher_forcing,
                step=step-warmup_step, total_step=total_step, epoch=epoch, total_epoch=config['train']['num_epochs'],
                scaler=scaler, return_pert_emb=want_emb_log, ccre_intervals_dict=ccre_intervals_dict,
            )
            forward_end = time.time()
            forward_time = forward_end - forward_start

            loss = loss_dict['total_loss']
            loss_diff = loss_dict.get('loss_diff', torch.tensor(0.0, device=device))
            loss_pred = loss_dict.get('loss_pred', torch.tensor(0.0, device=device))
            loss_clip = loss_dict.get('loss_clip', torch.tensor(float('nan'), device=device))
            loss_cm = loss_dict.get('loss_cm', torch.tensor(float('nan'), device=device))
            
            # Logging payload construction
            if args.use_wandb:
                def _lr_family(kind):
                    idx_map = {'diff': (0, 1), 'adapt': (2, 3), 'pred': (4, 5)}
                    if use_clip: idx_map['clip'] = (6, 7)
                    i0, i1 = idx_map[kind]
                    return max(optimizer.param_groups[i0]['lr'], optimizer.param_groups[i1]['lr'])

                clip_active = bool(loss_dict.get('clip_active', False))
                cm_active = bool(loss_dict.get('cm_active', False))
                
                log_dict = {
                    "trainer/step": step, "epoch": epoch,
                    "step_train_loss": loss.item(), "step_train_loss_diff": loss_diff.item(),
                    "step_train_loss_pred": loss_pred.item(),
                    "step_train_loss_clip": loss_clip.item() if torch.is_tensor(loss_clip) else float('nan'),
                    "step_train_loss_cm": loss_cm.item() if torch.is_tensor(loss_cm) else float('nan'),
                    "learning_rate": max(_lr_family('diff'), _lr_family('adapt'), _lr_family('pred')),
                    "timing/data_loading_sec": data_loading_end - data_loading_start,
                    "timing/forward_pass_sec": forward_time,
                }
                
                # Cache stats
                if cache_embeddings:
                    st = cache_embeddings.stats(reset_step=True)
                    log_dict.update({f"cache/{k}": v for k, v in st.items()})
                
                # Uncertainty stats
                if 'uncertainty' in loss_dict and loss_dict['uncertainty'] is not None:
                    unc = loss_dict['uncertainty']
                    names = unc.get('names', [])
                    sigma2 = unc.get('sigma2', [])
                    for i, name in enumerate(names):
                        if i < len(sigma2):
                            val = sigma2[i].item() if torch.is_tensor(sigma2[i]) else sigma2[i]
                            log_dict[f'uncertainty/sigma2/{name}'] = float(val)

                # Embedding cosine similarity log
                if want_emb_log and 'pert_diff_emb' in loss_dict and loss_dict['pert_diff_emb'] is not None:
                     # (Simplifying the complex table logging from DDP for brevity, but logic is preserved)
                     emb = loss_dict['pert_diff_emb'].detach().cpu().float()
                     if emb.ndim == 2 and emb.shape[0] > 1:
                         emb = F.normalize(emb, p=2, dim=-1)
                         cos_mat = torch.mm(emb, emb.t()).numpy()
                         iu = np.triu_indices(emb.shape[0], 1)
                         log_dict["emb/cosine_diff_mean"] = float(cos_mat[iu].mean())
                         log_dict["emb/cosine_diff_max"] = float(cos_mat[iu].max())
                         log_dict["emb/cosine_diff_min"] = float(cos_mat[iu].min())

            # Backward
            backward_start = time.time()
            loss.backward()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            backward_end = time.time()
            backward_time = backward_end - backward_start

            # Clip
            clip_start = time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            clip_time = clip_end = time.time() - clip_start

            # Optimizer Step (handle CLIP groups active status)
            refresh_saved_lrs_from_optimizer(optimizer)
            if use_clip:
                saved_lrs = get_saved_lrs()
                clip_active = bool(loss_dict.get('clip_active', False))
                for group_idx in [6, 7]:
                    if clip_active and saved_lrs:
                        optimizer.param_groups[group_idx]['lr'] = saved_lrs[group_idx]
                        optimizer.param_groups[group_idx]['weight_decay'] = weight_decay
                    else:
                        optimizer.param_groups[group_idx]['lr'] = 0.0
                        optimizer.param_groups[group_idx]['weight_decay'] = 0.0

            optimizer_start = time.time()
            optimizer.step()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            optimizer_time = time.time() - optimizer_start

            if args.use_wandb:
                log_dict.update({
                    "timing/backward_pass_sec": backward_time,
                    "timing/gradient_clipping_sec": clip_time,
                    "timing/optimizer_step_sec": optimizer_time,
                    "timing/total_batch_time_sec": time.time() - data_loading_start
                })
                wandb.log(log_dict)

            bs = len(batch['perturb'])
            total_loss += loss.item() * bs
            total_loss_diff += loss_diff.item() * bs
            total_loss_pred += loss_pred.item() * bs
            train_len += bs
            
            data_loading_start = time.time()

        # Epoch Summary
        avg_loss = total_loss / train_len
        avg_loss_diff = total_loss_diff / train_len
        avg_loss_pred = total_loss_pred / train_len
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Diff: {avg_loss_diff:.4f} | Pred: {avg_loss_pred:.4f}")

        # --- Validation ---
        model.eval()
        run_val = {'loss': 0.0, 'diff': 0.0, 'pred': 0.0, 'clip': 0.0, 'cm': 0.0}
        counts = {'clip': 0, 'cm': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device (same as training loop)
                batch_gpu = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch_gpu[k] = v.to(device, non_blocking=True)
                    else:
                        batch_gpu[k] = v
                
                val_dict = model(batch=batch_gpu, sgRNAs=None, betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod,
                                 cache_embeddings=cache_embeddings, weight=config['train']['loss_weight'],
                                 activate_scheduler=True, step=step, epoch=epoch, scaler=scaler)
                bs = len(batch['perturb'])
                run_val['loss'] += val_dict['total_loss'].item() * bs
                run_val['diff'] += val_dict.get('loss_diff', torch.tensor(0.0)).item() * bs
                run_val['pred'] += val_dict.get('loss_pred', torch.tensor(0.0)).item() * bs
                
                vc = val_dict.get('loss_clip', torch.tensor(float('nan')))
                if not torch.isnan(vc):
                    run_val['clip'] += vc.item() * bs
                    counts['clip'] += bs
                
                vcm = val_dict.get('loss_cm', torch.tensor(float('nan')))
                if not torch.isnan(vcm):
                    run_val['cm'] += vcm.item() * bs
                    counts['cm'] += bs
                
                counts['total'] += bs

        avg_val_loss = run_val['loss'] / counts['total']
        avg_val_loss_diff = run_val['diff'] / counts['total']
        avg_val_loss_pred = run_val['pred'] / counts['total']
        
        if USE_LR_SCHEDULER:
            if sched_diff: sched_diff.step(avg_val_loss_diff)
            if sched_pred: sched_pred.step(avg_val_loss_pred)

        if args.use_wandb:
            wandb.log({
                "epoch": epoch, "val_loss": avg_val_loss, 
                "val_loss_diff": avg_val_loss_diff, "val_loss_pred": avg_val_loss_pred
            })

        # --- Test Evaluation ---
        # Run full metrics every 5 epochs, else just loss
        test_mode = 'full' if (epoch % 5 == 0) else 'loss_only'
        # test_res = evaluate_test_set(model, test_loader, device, betas, alphas, alphas_cumprod, 
        #                              cache_embeddings, gene_annot, dna_record, scaler, config, 
        #                              ccre_intervals_dict, mode=test_mode, epoch=epoch)
        
        # if args.use_wandb:
        #     wandb.log({"epoch": epoch, **test_res})
        
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f}")
        # if test_mode == 'full':
        #     print(f"    Test Latent E-Dist: {test_res.get('test_e_dist_macro'):.4f} | MSE: {test_res.get('test_mse_dist_macro'):.4f}")

        # --- Checkpointing & Early Stopping ---
        stop = early_stopper(avg_val_loss_pred + avg_val_loss_diff, model, epoch=epoch)
        
        # Save every epoch (with overwrite logic)
        save_name_tmp = save_name_final.replace(".pt", f"_epoch_{epoch}.pt")
        current_ckpt = os.path.join(save_path, save_name_tmp)
        latest_path = os.path.join(save_path, "latest_model.pt")
        model.save_checkpoint(latest_path, optimizer=optimizer, epoch=epoch)
        
        
        
        if early_stopper.counter == 0:
            best_path = os.path.join(save_path, "best_model.pt")
            # We can just copy 'latest' to 'best' or save again
            torch.save(torch.load(latest_path), best_path) 
            print(f"    --> New best model saved.")

        if stop:
            print(f"Early stopping at epoch {epoch}. Restoring best weights.")
            if early_stopper.best_weights:
                model.load_state_dict(early_stopper.best_weights)
            break
        

    # Save final best model
    if early_stopper.best_weights:
        model.load_state_dict(early_stopper.best_weights)
    
    final_path = os.path.join(save_path, save_name_final)
    model.save_checkpoint(final_path, optimizer=optimizer, epoch=config['train']['num_epochs'])
    
    # --- Post-Training Evaluation ---
    model.eval()
    
    # First evaluation loop - diffusion evaluation on validation set
    print("Starting post-training validation diffusion evaluation...")
    model.eval_diffusion(val_loader, cache_embeddings=cache_embeddings, sgRNAs=None,
                        dna_record=dna_record, gene_annot=gene_annot, 
                        Ts=[2, 4, 10, 14, 20, 30, 50, 100], scaler=scaler)
    
    # Second evaluation loop - latent space evaluation on test set
    print("Starting post-training test evaluation with full metrics...")
    total_samples = 0
    
    gens_list = []
    z_0_list = []
    z_ctrl_list = []
    preds_list = []
    x0_raw_list = []
    x_ctrl_raw_list = []
    labels_list = []
    
    for batch in test_loader:
        # Move batch to device
        batch_gpu = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_gpu[k] = v.to(device, non_blocking=True)
            else:
                batch_gpu[k] = v

        # Opt-in unseen cell-line logic: compute and attach mixture weights.
        # IMPORTANT: This is only active when explicitly enabled; otherwise the batch is untouched.
        cell_line_mix_weights = batch_gpu.get("cell_line_mix_weights", None)
        if (cell_line_mix_weights is None) and (prototypes is not None):
            # z_ctrl in embedding space (B,D); use batch-level convex projection by default (Occam).
            z_ctrl_cpu = batch["control"] if torch.is_tensor(batch["control"]) else torch.as_tensor(batch["control"])
            cell_line_mix_weights = infer_cell_line_mix_weights_from_batch(
                z_ctrl_batch=z_ctrl_cpu,
                prototypes=prototypes,
                level="batch",
            )
            # keep on CPU; model will move to GPU as needed
            batch_gpu["cell_line_mix_weights"] = cell_line_mix_weights
        
        z_0 = batch_gpu["perturb"]
        z_ctrl = batch_gpu["control"]
        labels = batch_gpu["label"]
        x_0 = batch_gpu["perturb_raw"]
        x_ctrl = batch_gpu["control_raw"]
        cell_line_ids = batch_gpu.get("cell_line_id", None)
        batch_size = z_0.size(0)
        
        # Sample latent space
        z_ctrl_batch = z_ctrl.unsqueeze(1)
        gens = model.sample_latent(z_ctrl_batch, 
                                   labels, 
                                   gene_annot, 
                                   None, 
                                   dna_record, 
                                   cell_line_ids=cell_line_ids,
                                   cache_embeddings=cache_embeddings,
                                   cell_line_mix_weights=cell_line_mix_weights,
                                )
        
        pred, keep_mask = model.predict(
            x_ctrl,
            z_ctrl_batch,
            labels,
            gene_annot,
            None,
            dna_record,
            cache_embeddings=cache_embeddings,
            z_0=gens,  # <-- replacing z_0 with gens
            return_mask=True,
            cell_line_ids=cell_line_ids,
            cell_line_mix_weights=cell_line_mix_weights,
        )
        
        if keep_mask is not None:
            x_0 = x_0[keep_mask]
            x_ctrl = x_ctrl[keep_mask]
            batch_size = x_0.size(0)
        
        # Convert to numpy for metric computation
        pred_cpu = pred.detach().cpu()
        x0_cpu = x_0.detach().cpu()
        x_ctrl_cpu = x_ctrl.detach().cpu()
        gens = gens.detach().cpu()
        z_0 = z_0.detach().cpu()
        z_ctrl = z_ctrl.detach().cpu()
        
        if torch.is_tensor(labels):
            labels_np = labels.detach().cpu().numpy()
        elif isinstance(labels, np.ndarray):
            labels_np = labels
        else:
            labels_np = np.asarray(labels)
        
        if keep_mask is not None:
            if isinstance(keep_mask, torch.Tensor):
                keep_mask_np = keep_mask.detach().cpu().numpy()
            else:
                keep_mask_np = np.asarray(keep_mask)
            labels_np = labels_np[keep_mask_np]
        
        labels_flat = np.asarray(labels_np).reshape(-1)
        labels_list.extend([str(lab) for lab in labels_flat])
        
        gens_list.append(gens)
        z_0_list.append(z_0)
        z_ctrl_list.append(z_ctrl)
        preds_list.append(pred_cpu)
        x0_raw_list.append(x0_cpu)
        x_ctrl_raw_list.append(x_ctrl_cpu)
        
        total_samples += batch_size
        if total_samples > 10000:
            break
    
    # Concatenate all collected data (single GPU - no gathering needed)
    if len(gens_list) > 0:
        gens_list = torch.cat(gens_list, dim=0).numpy()
        z_0_list = torch.cat(z_0_list, dim=0).numpy()
        z_ctrl_list = torch.cat(z_ctrl_list, dim=0).numpy()
        preds_list = torch.cat(preds_list, dim=0).numpy()
        x0_raw_list = torch.cat(x0_raw_list, dim=0).numpy()
        x_ctrl_raw_list = torch.cat(x_ctrl_raw_list, dim=0).numpy()
        labels_array = np.asarray(labels_list)
        
        print(f"Aggregated data shapes:")
        print(f"  gens_list: {gens_list.shape}")
        print(f"  z_0_list: {z_0_list.shape}")
        print(f"  z_ctrl_list: {z_ctrl_list.shape}")
        print(f"  preds_list: {preds_list.shape}")
        print(f"  x0_raw_list: {x0_raw_list.shape}")
        print(f"  labels: {labels_array.shape}")
        
        total_samples_macro = labels_array.shape[0]
        avg_mse, _ = _macro_mse(preds_list, x0_raw_list, labels_array)
        rng = np.random.default_rng(0)
        permuted_targets = x0_raw_list[rng.permutation(total_samples_macro)]
        avg_mse_permuted, _ = _macro_mse(preds_list, permuted_targets, labels_array)
        avg_mse_null, _ = _macro_mse(x0_raw_list, x_ctrl_raw_list, labels_array)
        avg_mse_zero, _ = _macro_mse(x0_raw_list, np.zeros_like(x0_raw_list), labels_array)
        
        # Permuted prediction MSE metrics (permute predictions, not targets)
        permuted_preds_raw = preds_list[rng.permutation(total_samples_macro)]
        avg_mse_pred_permuted_raw, _ = _macro_mse(permuted_preds_raw, x0_raw_list, labels_array)
        
        total_e_dist = compute_e_distance(gens_list, z_0_list, labels=labels_array, average='macro')
        total_null_e_dist = compute_e_distance(z_ctrl_list, z_0_list, labels=labels_array, average='macro')
        ave_mse_dist, _ = _macro_mse(gens_list, z_0_list, labels_array)
        ave_null_mse_dist, _ = _macro_mse(z_ctrl_list, z_0_list, labels_array)
        gens_inverse_list = scaler.inverse_transform(gens_list)
        z_0_inverse_list = scaler.inverse_transform(z_0_list)
        z_ctrl_inverse_list = scaler.inverse_transform(z_ctrl_list)
        total_inverse_e_dist = compute_e_distance(gens_inverse_list, z_0_inverse_list, labels=labels_array, average='macro')
        total_inverse_null_e_dist = compute_e_distance(z_ctrl_inverse_list, z_0_inverse_list, labels=labels_array, average='macro')
        ave_inverse_mse_dist, _ = _macro_mse(gens_inverse_list, z_0_inverse_list, labels_array)
        ave_inverse_null_mse_dist, _ = _macro_mse(z_ctrl_inverse_list, z_0_inverse_list, labels_array)
        
        # Permuted prediction MSE metrics for latent and inverse spaces
        permuted_gens = gens_list[rng.permutation(total_samples_macro)]
        ave_mse_pred_permuted_latent, _ = _macro_mse(permuted_gens, z_0_list, labels_array)
        permuted_gens_inverse = gens_inverse_list[rng.permutation(total_samples_macro)]
        ave_inverse_mse_pred_permuted, _ = _macro_mse(permuted_gens_inverse, z_0_inverse_list, labels_array)
        
        # Permuted prediction e-distance metrics
        total_e_dist_pred_permuted = compute_e_distance(permuted_gens, z_0_list, labels=labels_array, average='macro')
        total_inverse_e_dist_pred_permuted = compute_e_distance(permuted_gens_inverse, z_0_inverse_list, labels=labels_array, average='macro')
        
        print(f'Total samples (macro evaluation): {labels_array.shape[0]}')
        print(f'Unique perturbations: {len(np.unique(labels_array))}')
        print(f'##### Raw Space Metrics (macro) #####')
        print(f'Average MSE: {avg_mse:.6f}')
        print(f'Average Permuted MSE: {avg_mse_permuted:.6f}')
        print(f'Average Pred Permuted MSE: {avg_mse_pred_permuted_raw:.6f}')
        print(f'Average Control Perturb MSE: {avg_mse_null:.6f}')
        print(f'Average Zero Perturb MSE: {avg_mse_zero:.6f}')
        print(f'##### Latent Space Metrics (macro) #####')
        print(f'E-distance: {total_e_dist:.6f}')
        print(f'Pred Permuted E-distance: {total_e_dist_pred_permuted:.6f}')
        print(f'Null e-distance: {total_null_e_dist:.6f}')
        print(f'Average MSE distance: {ave_mse_dist:.6f}')
        print(f'Average Pred Permuted MSE distance: {ave_mse_pred_permuted_latent:.6f}')
        print(f'Average null MSE distance: {ave_null_mse_dist:.6f}')
        print(f'Inverse E-distance: {total_inverse_e_dist:.6f}')
        print(f'Inverse Pred Permuted E-distance: {total_inverse_e_dist_pred_permuted:.6f}')
        print(f'Inverse Null e-distance: {total_inverse_null_e_dist:.6f}')
        print(f'Average Inverse MSE distance: {ave_inverse_mse_dist:.6f}')
        print(f'Average Inverse Pred Permuted MSE distance: {ave_inverse_mse_pred_permuted:.6f}')
        print(f'Average Inverse null MSE distance: {ave_inverse_null_mse_dist:.6f}')
        
        if args.use_wandb:
            wandb.log({
                "avg_mse_macro": avg_mse,
                "avg_mse_permuted_macro": avg_mse_permuted,
                "avg_mse_pred_permuted_raw_macro": avg_mse_pred_permuted_raw,
                "avg_mse_null_macro": avg_mse_null,
                "avg_mse_zero_macro": avg_mse_zero,
                "e_dist_macro": total_e_dist,
                "e_dist_pred_permuted_macro": total_e_dist_pred_permuted,
                "null_e_dist_macro": total_null_e_dist,
                "mse_dist_macro": ave_mse_dist,
                "mse_pred_permuted_latent_macro": ave_mse_pred_permuted_latent,
                "null_mse_dist_macro": ave_null_mse_dist,
                "inverse_e_dist_macro": total_inverse_e_dist,
                "inverse_e_dist_pred_permuted_macro": total_inverse_e_dist_pred_permuted,
                "inverse_null_e_dist_macro": total_inverse_null_e_dist,
                "inverse_mse_dist_macro": ave_inverse_mse_dist,
                "inverse_mse_pred_permuted_macro": ave_inverse_mse_pred_permuted,
                "inverse_null_mse_dist_macro": ave_inverse_null_mse_dist,
            })
            wandb.finish()
    
    # Final cleanup
    cleanup()

def cleanup():
    # Only relevant for DDP, but placeholder kept for structure
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RNA_loss', type=str, default='htg_nll', choices=['mse', 'nll', 'htg_nll'],
                        help="Decoder likelihood. htg_nll = hurdle truncated-Gaussian residual "
                             "(the paper's decoder); mse and nll are alternatives.")
    parser.add_argument('--dname', type=str, default='k562',
                        help='Dataset folder under DATA_ROOT: k562, jurkat, rpe1 or combined_perturbqa_hvg2k.')
    parser.add_argument('--transform_count', action='store_true', default=False)
    parser.add_argument('--no_lr_scheduler', action='store_true', default=False,
                        help='Naming only: appends _no_scheduler to the final checkpoint file name. '
                             'The learning-rate scheduler (GroupPlateau) is always on.')
    parser.add_argument("--config", type=str, help="Path to config (default: configs/k562_config.yaml)")
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--version', type=str, default='v1', help='Run tag used in checkpoint paths.')
    parser.add_argument('--project_name', type=str, default='STRAND', help='Weights & Biases project prefix.')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--out_suffix', type=str, default='')
    # Optional resources (recommended to provide via env/CLI for reproducibility)
    parser.add_argument(
        '--ccre_bed_path',
        type=str,
        default=os.environ.get("CCRE_BED_PATH", ""),
        help="Optional cCRE BED (gz) path. If empty/missing, cCRE features are disabled.",
    )
    parser.add_argument(
        '--dna_cache_dir',
        type=str,
        default=None,
        help="Directory of precomputed DNA embeddings (step 02 build_dna_cache.py). Defaults to "
             "$DNA_CACHE_DIR, else IO.dna_cache_dir in the config.",
    )
    parser.add_argument('--ablate_distance_bias', action='store_true', default=False, help='Ablate distance bias in SinglePerturbEmbedding')
    parser.add_argument('--ablate_ccre_bias', action='store_true', default=False, help='Ablate cCRE bias in SinglePerturbEmbedding')
    parser.add_argument('--cell_bias_mode', type=str, default='normal', choices=['normal', 'noise'], help='Cell bias mode: normal or noise')
    parser.add_argument('--wandb_name', type=str, default='null')
    parser.add_argument(
        '--ckpt_root',
        type=str,
        default=os.environ.get("OUTPUT_ROOT", os.path.join(project_root, "outputs")),
        help='Root directory for saving checkpoints',
    )
    parser.add_argument('--run_group', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot', 'heldout_cell_line'],
                        help='Label for the checkpoint subdirectory only. It does not change the data: '
                             'the split is chosen when OT pairs are built (--split_column in '
                             'scripts/02_embed_and_match/embed_and_match.py).')
    # Opt-in unseen cell-line support (OFF by default; preserves bitwise identity otherwise)
    parser.add_argument(
        '--infer_cell_line_mix_weights',
        action='store_true',
        default=False,
        help='Enable unseen cell-line inference by computing cell-line mixture weights from control embeddings '
             'and mixing null embeddings accordingly (requires config cell_line_mapping).',
    )
    args = parser.parse_args()
    
    config_path = args.config if args.config else os.path.join(project_root, "configs", "k562_config.yaml")
    config = load_config(config_path)

    main_worker(config, args)

if __name__ == "__main__":
    main()