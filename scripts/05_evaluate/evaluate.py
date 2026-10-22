# %%
"""
STRAND checkpoint -> test-split predictions.

Loads a trained STRAND model (SCimilarity latent + HiRef-OT + DNA->RNA diffusion),
runs per-perturbation inference on the TEST split, and writes paired prediction /
ground-truth AnnData files (obs 'target_gene', control 'non-targeting').

Design: this is a *thin* driver that REUSES existing package code rather than
reimplementing inference. The heavy lifting is delegated to:
  - Model + inference:  src/diffusion_ot/models/strand_model.py
        STRANDModel.sample_latent()  (latent transport)
        STRANDModel.predict()        (latent -> RNA_dim expression)
    The inference loop below mirrors the post-training eval block of
    scripts/04_train/train.py.
  - Output contract:    create_anndata_for_celleval() below
        -> paired pred/real AnnData

CRITICAL CORRECTNESS NOTE (latent vs expression):
  Predictions are written as GENE EXPRESSION (RNA_dim, e.g. 4136 for the K562
  checkpoint whose `x_mu` buffer is (1, 4136)), NOT the 128-d SCimilarity
  latent. STRAND HAS an RNA decoder: `predictor_head` (heads.py DNAFullPredictor)
  maps (x_ctrl, z_ctrl, z_0) -> RNA_dim expression. The two-step call is:
      gens = model.sample_latent(...)              # (B, 128) perturbed LATENT
      pred = model.predict(..., z_0=gens)          # (B, RNA_dim) decoded EXPRESSION
  This driver writes `pred` (the decoded expression), NOT `gens`. (strand_model.py
  predict() returns `x_0_pred`, RNA_dim-wide, via predict_unconditional_mean_residual
  for htg_nll / `mu` for mse.)
"""

import os
import sys
import argparse
import pickle
import random

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

# --- Package imports (mirror train.py) ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.diffusion_ot.data import perturb_control_dataloader, LRUCache
from src.diffusion_ot.models.strand_model import STRANDModel
from src.diffusion_ot.models.ablation_fixed_dna_embedding import FixedDNAEmbeddingAblation
from src.diffusion_ot.models.ablation_deterministic_fusion import DeterministicFusionAblation
from PModule.DNA_modeling import BorzoiPerturbAdapter
from src.diffusion_ot.utils import get_dna_record, get_DNA_annot
from src.diffusion_ot.schedulers import get_official_betas
from src.diffusion_ot.training import load_config

# --model choices: STRAND and the two paper Fig. 4 ablations. All three share one
# constructor signature. STRAND and the fixed-DNA-embedding ablation also share
# every weight name, so the checkpoint alone cannot tell them apart: pass the
# class the checkpoint was trained with.
MODEL_CLASSES = {
    'strand': STRANDModel,
    'fixed_dna_embedding': FixedDNAEmbeddingAblation,
    'deterministic_fusion': DeterministicFusionAblation,
}


def create_anndata_for_celleval(sumstats, gene_names, pert_col="target_gene",
                                control_label="non-targeting"):
    """
    Create paired AnnData objects following the cache-eval schema.
    """
    predictions = np.asarray(sumstats["all_predictions"])
    ground_truths = np.asarray(sumstats["all_ground_truths"])
    all_control_cells = np.asarray(sumstats["all_control_cells"])

    pred_labels = list(sumstats["pred_labels"])
    real_labels = list(sumstats["real_labels"])

    n_ctrl = int(all_control_cells.shape[0])
    n_pred = int(predictions.shape[0])
    n_real = int(ground_truths.shape[0])

    pred_data = np.vstack([all_control_cells, predictions])
    pred_obs_labels = [control_label] * n_ctrl + pred_labels
    pred_cell_names = [f"Ctrl_{i}" for i in range(n_ctrl)] + [f"Pred_{i}" for i in range(n_pred)]

    adata_pred = ad.AnnData(
        X=csr_matrix(pred_data),
        obs=pd.DataFrame({pert_col: pred_obs_labels, "cell_type": ["predicted"] * len(pred_obs_labels)}, index=pred_cell_names),
        var=pd.DataFrame(index=list(map(str, gene_names))),
    )

    real_data = np.vstack([all_control_cells, ground_truths])
    real_obs_labels = [control_label] * n_ctrl + real_labels
    real_cell_names = [f"Ctrl_{i}" for i in range(n_ctrl)] + [f"Real_{i}" for i in range(n_real)]

    adata_real = ad.AnnData(
        X=csr_matrix(real_data),
        obs=pd.DataFrame({pert_col: real_obs_labels, "cell_type": ["real"] * len(real_obs_labels)}, index=real_cell_names),
        var=pd.DataFrame(index=list(map(str, gene_names))),
    )

    return adata_pred, adata_real


def build_model(config, args, RNA_dim, device, cell_line_nums=1,
                dna_record=None, gene_annot=None, dna_cache_dir=None):
    """Construct the model class named by --model with the SAME args as training
    (train.py). dna_record/gene_annot/dna_cache_dir
    are optional so a checkpoint-architecture smoke test can build the model
    without genome files or a populated DNA cache.
    """
    # BorzoiPerturbAdapter: load_backbone=False -> DNA embeddings come from the
    # disk cache at inference (matches training).
    borzoipert = BorzoiPerturbAdapter(
        load_backbone=False,
        out_dim=1536,
        cache_pool_k=1,
        ablate_distance_bias=args.ablate_distance_bias,
        ablate_ccre_bias=args.ablate_ccre_bias,
        cell_bias_mode=args.cell_bias_mode,
    ).to(device)

    model = MODEL_CLASSES[args.model](
        DNA_model=borzoipert,
        T_train=config['model']['T_train'],
        RNA_dim=RNA_dim,
        dna_dim=config['model']['DNA_embed_dim'],
        use_scheduler=True,
        dna_records=dna_record,
        gene_annot=gene_annot,
        loss_metric=args.RNA_loss,
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
        cell_line_nums=cell_line_nums,
        verbose=False,
    ).to(device)
    return model


def load_weights(model, ckpt_path, device):
    """Load STRAND weights. save_checkpoint() writes key 'model_state'
    (strand_model.py); the training-resume path uses 'model_state_dict'
    (train.py). Support both. Loading is strict: a checkpoint whose weights do not
    match the model built from --model and the ablation flags is an error.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and 'model_state' in ck:
        sd = ck['model_state']
    elif isinstance(ck, dict) and 'model_state_dict' in ck:
        sd = ck['model_state_dict']
    else:
        sd = ck  # bare state_dict
    if all(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"{ckpt_path} does not match the model built here. Check --model, "
            f"--config and the ablation flags (--ablate_distance_bias, "
            f"--ablate_ccre_bias, --cell_bias_mode).\n{e}") from None


def set_pinned_schedule(model, T, scale, eta, device):
    """Pin the sampler schedule that sample_latent() consumes.

    eval_diffusion() (strand_model.py) normally grid-searches
    (T, scale, eta) on val and caches T_sample/best_scale/best_eta/best_beta.
    We instead PIN the production inference config (scale=1 / T=100 / eta=1) --
    the standing inference invariant -- so no val grid is run. best_alphas_cumprod
    is left None, exactly as eval_diffusion leaves it.
    """
    model.T_sample = int(T)
    model.best_scale = float(scale)
    model.best_eta = float(eta)
    betas, _, _ = get_official_betas(int(T))
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas)
    model.best_beta = betas.clamp_(1e-10, 0.999).to(device, dtype=torch.float32)
    model.best_alphas = None
    model.best_alphas_cumprod = None


def get_gene_names(dataset):
    """RNA_dim gene names for AnnData.var, read from the backed source AnnData
    (dataset._adata_paths, datasets.py/664) and HVG-masked if a mask exists
    (datasets.py)."""
    import scanpy as sc
    path = next(iter(dataset._adata_paths.values()))
    a = sc.read(path, backed='r')
    names = np.asarray(a.var_names).astype(str)
    try:
        a.file.close()
    except Exception:
        pass
    mask = getattr(dataset, 'hvg_mask', None)
    if mask is not None:
        names = names[mask]
    return names


def run_inference(model, test_loader, dna_record, gene_annot, cache_embeddings,
                  device, max_samples):
    """Mirror the STRAND post-training eval loop (train.py):
    per test batch (an OT-matched control/perturb pair), sample the perturbed latent
    then decode to RNA_dim expression. Returns raw-space arrays."""
    preds_list, x0_raw_list, xctrl_raw_list, labels_list = [], [], [], []
    total = 0
    for batch in test_loader:
        batch_gpu = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

        z_ctrl = batch_gpu["control"]
        labels = batch_gpu["label"]
        x_0 = batch_gpu["perturb_raw"]
        x_ctrl = batch_gpu["control_raw"]
        cell_line_ids = batch_gpu.get("cell_line_id", None)

        z_ctrl_batch = z_ctrl.unsqueeze(1)  #
        gens = model.sample_latent(
            z_ctrl_batch, labels, gene_annot, None, dna_record,
            cell_line_ids=cell_line_ids, cache_embeddings=cache_embeddings,
            cell_line_mix_weights=None,
        )  # perturbed LATENT (B, 128)

        pred, keep_mask = model.predict(
            x_ctrl, z_ctrl_batch, labels, gene_annot, None, dna_record,
            cache_embeddings=cache_embeddings, z_0=gens, return_mask=True,
            cell_line_ids=cell_line_ids, cell_line_mix_weights=None,
        )  # decoded EXPRESSION (B, RNA_dim)

        if keep_mask is not None:
            x_0 = x_0[keep_mask]
            x_ctrl = x_ctrl[keep_mask]

        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        if keep_mask is not None:
            km = keep_mask.detach().cpu().numpy() if torch.is_tensor(keep_mask) else np.asarray(keep_mask)
            labels_np = labels_np[km]

        labels_list.extend([str(l) for l in np.asarray(labels_np).reshape(-1)])
        preds_list.append(pred.detach().cpu())
        x0_raw_list.append(x_0.detach().cpu())
        xctrl_raw_list.append(x_ctrl.detach().cpu())

        total += pred.size(0)
        if max_samples and total > max_samples:  #
            break

    preds = torch.cat(preds_list, 0).float().numpy()
    x0_raw = torch.cat(x0_raw_list, 0).float().numpy()
    xctrl_raw = torch.cat(xctrl_raw_list, 0).float().numpy()
    labels = np.asarray(labels_list)
    return preds, x0_raw, xctrl_raw, labels


def main():
    parser = argparse.ArgumentParser(description="Predict the test split with a trained STRAND checkpoint")
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, 'configs', 'k562_config.yaml'))
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='strand', choices=sorted(MODEL_CLASSES),
                        help='Model class the checkpoint was trained with: strand (the paper '
                             'model) or one of the two Fig. 4 ablations.')
    parser.add_argument('--dname', type=str, default='k562', help='Dataset folder name (idx_root/<dname>)')
    parser.add_argument('--result_path', type=str, required=True, help='Output dir for *_pred.h5ad/_real.h5ad')
    parser.add_argument('--outname', type=str, default='strand', help='Output filename suffix')
    parser.add_argument('--control_label', type=str, default='non-targeting')
    parser.add_argument('--n_control_samples', type=int, default=256,
                        help='Cap on the control-cell baseline pool written to the AnnData')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_samples', type=int, default=10000, help='Break inference after N samples')
    # Ablation flags -- must match the checkpoint's ablation tag
    parser.add_argument('--ablate_distance_bias', action='store_true', default=False)
    parser.add_argument('--ablate_ccre_bias', action='store_true', default=False)
    parser.add_argument('--cell_bias_mode', type=str, default='normal', choices=['normal', 'noise'])
    # Loss/head type -- K562 checkpoints use htg_nll (zero/mu/sigma/x_base heads)
    parser.add_argument('--RNA_loss', type=str, default='htg_nll', choices=['mse', 'nll', 'htg_nll'])
    # Pinned sampler schedule (production inference invariant; NOT the val grid)
    parser.add_argument('--sample_T', type=int, default=100,
                        help='Reverse-bridge sampling steps. The paper tunes this per dataset on '
                             'validation data over {6, 10, 14, 20, 50} for benchmarks and uses 10 '
                             'for in-silico locus scans.')
    parser.add_argument('--sample_scale', type=float, default=1.0)
    parser.add_argument('--sample_eta', type=float, default=1.0)
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='scaler.pkl for the SCimilarity latent; default: next to checkpoint')
    parser.add_argument('--dna_cache_dir', type=str, default=None,
                        help='DNA embedding cache dir (defaults to $DNA_CACHE_DIR, else config IO.dna_cache_dir)')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)
    device = torch.device(args.device)

    config = load_config(args.config)
    cell_line_mapping = config.get("cell_line_mapping", None)
    cell_line_nums = len(cell_line_mapping) if cell_line_mapping else 1
    dna_cache_dir = args.dna_cache_dir or os.environ.get("DNA_CACHE_DIR") or config['IO'].get('dna_cache_dir', None)

    # --- Scaler (SCimilarity latent standardizer). Prefer scaler.pkl next to the
    #     checkpoint (saved at train.py); else fit from the
    #     train split exactly as training does. ---
    ckpt_dir = os.path.dirname(args.checkpoint_path)
    scaler_path = args.scaler_path or os.path.join(ckpt_dir, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[scaler] loaded {scaler_path}")

    idx_paths = [f"{config['IO']['idx_root']}/{args.dname}/"]
    root_dir = config['IO']['root_dir']
    anndata_path = config['IO'].get('anndata_path', None)
    h5_dataset_name = config['IO'].get('h5_dataset_name', None)
    anndata_dataset_name = config['IO'].get('anndata_dataset_name', None)

    ds_common = dict(
        namelist=[args.dname], idx_paths=idx_paths, root_dir=root_dir,
        h5_suffix='.X_scimilarity.h5',
        cell_line_mapping=cell_line_mapping, anndata_path=anndata_path,
        h5_dataset_name=h5_dataset_name, anndata_dataset_name=anndata_dataset_name,
        allow_unknown_cell_lines=False,
    )

    if scaler is None:
        # Fit scaler from train split (mirrors training)
        train_dataset = perturb_control_dataloader(
            idx_suffix=config['IO']['idx_train_suffix'], scale=True, **ds_common)
        scaler = train_dataset.scaler
        print("[scaler] fit from train split (no scaler.pkl found)")

    test_dataset = perturb_control_dataloader(
        idx_suffix=config['IO']['idx_test_suffix'], scale=scaler, **ds_common)  #
    RNA_dim = int(test_dataset.RNA_dim)
    print(f"[data] test samples={len(test_dataset)}  RNA_dim={RNA_dim}")

    # --- Resources + model ---
    dna_record = get_dna_record()
    gene_annot = get_DNA_annot()
    cache_embeddings = LRUCache(max_size=3072, device=device)  #

    model = build_model(config, args, RNA_dim, device, cell_line_nums=cell_line_nums,
                        dna_record=dna_record, gene_annot=gene_annot, dna_cache_dir=dna_cache_dir)
    load_weights(model, args.checkpoint_path, device)
    print(f"[ckpt] loaded {args.checkpoint_path} into {type(model).__name__} (strict)")
    model.eval()
    set_pinned_schedule(model, args.sample_T, args.sample_scale, args.sample_eta, device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['train']['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    with torch.no_grad():
        preds, x0_raw, xctrl_raw, labels = run_inference(
            model, test_loader, dna_record, gene_annot, cache_embeddings, device, args.max_samples)
    print(f"[infer] pred={preds.shape}  gt={x0_raw.shape}  ctrl={xctrl_raw.shape}  "
          f"unique_perts={len(np.unique(labels))}")

    # --- Control baseline pool (cap at n_control_samples). STRAND controls are
    #     OT-paired, so we pool the collected raw controls and subsample. ---
    ctrl_pool = xctrl_raw
    if args.n_control_samples and ctrl_pool.shape[0] > args.n_control_samples:
        sel = rng.choice(ctrl_pool.shape[0], size=args.n_control_samples, replace=False)
        ctrl_pool = ctrl_pool[sel]

    gene_names = get_gene_names(test_dataset)

    # --- Paired pred/real AnnData (create_anndata_for_celleval):
    #     obs 'target_gene', control <control_label>. The test split comes from the
    #     OT index file (IO.idx_test_suffix). ---
    sumstats = {
        "all_predictions": preds,
        "all_ground_truths": x0_raw,
        "all_control_cells": ctrl_pool,
        "pred_labels": list(labels),
        "real_labels": list(labels),
    }
    adata_pred, adata_real = create_anndata_for_celleval(
        sumstats=sumstats, gene_names=gene_names,
        pert_col="target_gene", control_label=args.control_label)

    os.makedirs(args.result_path, exist_ok=True)
    cellline = str(config['model'].get('data_name', args.dname))
    fname_base = f"{args.dname}_{cellline}_{args.outname}"
    pred_path = os.path.join(args.result_path, f"{fname_base}_pred.h5ad")
    real_path = os.path.join(args.result_path, f"{fname_base}_real.h5ad")
    adata_pred.write_h5ad(pred_path, compression="gzip")
    adata_real.write_h5ad(real_path, compression="gzip")
    print(f"[write] {pred_path}\n[write] {real_path}")

    print("Done.")


if __name__ == "__main__":
    main()
