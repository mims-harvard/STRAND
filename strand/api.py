"""High-level inference interface for trained STRAND checkpoints.

``STRAND`` wraps the model classes used in the paper
(``src.diffusion_ot.models.strand_model.STRANDModel`` and
``PModule.DNA_modeling.BorzoiPerturbAdapter``) without changing their numerics:

1. the reference DNA window around a locus and its perturbation mask are encoded by the
   frozen Flashzoi/Borzoi trunk and the learned DNA Perturbation Module;
2. control cells, represented by their SCimilarity embeddings, are transported by the I2SB
   bridge sampler conditioned on that perturbation embedding;
3. the residual hurdle decoder maps the transported latents to gene expression relative to
   the control baseline.

Heavy dependencies (PyTorch, borzoi-pytorch, baskerville, Biopython) are imported lazily so
``import strand`` stays cheap.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from .locus import Locus

REPO_ROOT = Path(__file__).resolve().parents[1]
PathLike = Union[str, os.PathLike]


def _ensure_repo_on_path() -> None:
    """The research code lives in ``src/diffusion_ot`` and ``PModule`` at the repository root."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class SamplerConfig:
    """Reverse-bridge sampler settings.

    Defaults follow App. C.4 of the paper for in-silico screening: 10 diffusion steps,
    classifier-free guidance scale 1 and DDPM ``eta`` 1. For the benchmark tables the number
    of steps was selected per dataset on validation data from {6, 10, 14, 20, 50}.
    """

    steps: int = 10
    guidance_scale: float = 1.0
    eta: float = 1.0


@dataclass
class ControlCells:
    """Unperturbed cells that define the starting cellular state.

    ``expression`` is log-normalized expression in the checkpoint's gene order
    (``RNA_dim`` columns). ``latent`` holds SCimilarity embeddings of the same cells,
    already transformed with the scaler saved next to the checkpoint.
    """

    expression: np.ndarray
    latent: np.ndarray
    gene_names: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.expression = np.asarray(self.expression, dtype=np.float32)
        self.latent = np.asarray(self.latent, dtype=np.float32)
        if self.expression.ndim != 2 or self.latent.ndim != 2:
            raise ValueError("expression and latent must be 2-D (cells x features)")
        if self.expression.shape[0] != self.latent.shape[0]:
            raise ValueError("expression and latent must describe the same cells")
        if self.gene_names is not None:
            self.gene_names = np.asarray(self.gene_names).astype(str)
            if self.gene_names.shape[0] != self.expression.shape[1]:
                raise ValueError("gene_names length must match expression columns")

    def __len__(self) -> int:
        return self.expression.shape[0]

    @classmethod
    def from_anndata(
        cls,
        adata_path: PathLike,
        embeddings_h5: PathLike,
        scaler,
        obs_filter: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        embedding_key: str = "X_scimilarity",
        max_cells: Optional[int] = None,
        seed: int = 0,
    ) -> "ControlCells":
        """Load control cells from a curated AnnData and its SCimilarity embedding file.

        ``obs_filter`` selects the control population, e.g. ``{"gene": "non-targeting"}``
        (the default). Values may be a string or a list of accepted strings. Rows of
        ``embeddings_h5[embedding_key]`` must align with the AnnData rows, as produced by
        ``scripts/02_embed_and_match/embed_and_match.py``.
        """
        import anndata as ad
        import h5py

        obs_filter = {"gene": "non-targeting"} if obs_filter is None else obs_filter
        adata = ad.read_h5ad(adata_path, backed="r")
        keep = np.ones(adata.n_obs, dtype=bool)
        for key, accepted in obs_filter.items():
            accepted = [accepted] if isinstance(accepted, str) else list(accepted)
            keep &= adata.obs[key].astype(str).isin(accepted).to_numpy()
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            raise ValueError(f"no cells match {dict(obs_filter)} in {adata_path}")
        if max_cells is not None and idx.size > max_cells:
            idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_cells, replace=False))
        x = adata.X[idx]
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        with h5py.File(embeddings_h5, "r") as h5:
            emb = h5[embedding_key]
            if emb.shape[0] != adata.n_obs:
                raise ValueError("embedding rows do not match AnnData rows")
            z = np.asarray(emb[idx])
        z = scaler.transform(z) if scaler is not None else z
        return cls(x, z, np.asarray(adata.var_names))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Row indices drawn with replacement."""
        return rng.choice(len(self), size=int(n), replace=True)


@dataclass
class Prediction:
    """Predicted perturbed cells for one locus, paired with the controls they started from."""

    locus: Locus
    expression: np.ndarray  # (n_cells, RNA_dim) predicted perturbed expression
    latent: np.ndarray  # (n_cells, latent_dim) transported SCimilarity-space latents
    control_expression: np.ndarray  # (n_cells, RNA_dim) the matched control inputs
    gene_names: Optional[np.ndarray] = None

    def mean_shift(self) -> np.ndarray:
        """Control-relative response: mean predicted minus mean control expression per gene."""
        return self.expression.mean(axis=0) - self.control_expression.mean(axis=0)

    def response_table(self):
        """Per-gene summary sorted by absolute shift (pandas DataFrame)."""
        import pandas as pd

        names = (
            self.gene_names
            if self.gene_names is not None
            else np.array([f"gene_{i}" for i in range(self.expression.shape[1])])
        )
        ctrl = self.control_expression.mean(axis=0)
        pred = self.expression.mean(axis=0)
        df = pd.DataFrame(
            {"gene": names, "control_mean": ctrl, "predicted_mean": pred, "delta": pred - ctrl}
        )
        return df.reindex(df["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def to_anndata(self, include_controls: bool = True):
        """AnnData with predicted cells (and optionally the control inputs) as rows."""
        import anndata as ad
        import pandas as pd

        blocks = [self.expression]
        kind = ["predicted"] * self.expression.shape[0]
        if include_controls:
            blocks.append(self.control_expression)
            kind += ["control_input"] * self.control_expression.shape[0]
        obs = pd.DataFrame(
            {"kind": kind, "locus": self.locus.label},
            index=[f"cell_{i}" for i in range(len(kind))],
        )
        var = pd.DataFrame(
            index=self.gene_names
            if self.gene_names is not None
            else [f"gene_{i}" for i in range(self.expression.shape[1])]
        )
        out = ad.AnnData(np.vstack(blocks).astype(np.float32), obs=obs, var=var)
        out.obsm["X_latent"] = np.vstack(
            [self.latent] + ([np.full_like(self.latent, np.nan)] if include_controls else [])
        )
        out.uns["strand_locus"] = {
            "chrom": self.locus.chrom,
            "start": self.locus.start,
            "end": self.locus.end,
            "strand": self.locus.strand,
            "coordinates": "1-based inclusive",
        }
        return out


def _load_state_dict(path: PathLike, device):
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def build_model(
    model_cfg: Mapping,
    rna_dim: int,
    *,
    load_backbone: bool,
    dna_record=None,
    gene_annot=None,
    dna_cache_dir: Optional[str] = None,
    cell_bias_mode: str = "normal",
    ablate_distance_bias: bool = False,
    ablate_ccre_bias: bool = False,
    cell_line_nums: int = 1,
):
    """Construct the STRAND network with the architecture given in ``configs/*.yaml``.

    Mirrors ``scripts/04_train/train.py`` and
    ``scripts/05_evaluate/evaluate.py``.
    """
    _ensure_repo_on_path()
    from PModule.DNA_modeling import BorzoiPerturbAdapter
    from src.diffusion_ot.models.strand_model import STRANDModel

    adapter = BorzoiPerturbAdapter(
        load_backbone=load_backbone,
        out_dim=int(model_cfg.get("DNA_embed_dim", 1536)),
        cache_pool_k=1,
        ablate_distance_bias=ablate_distance_bias,
        ablate_ccre_bias=ablate_ccre_bias,
        cell_bias_mode=cell_bias_mode,
    )
    model = STRANDModel(
        DNA_model=adapter,
        T_train=int(model_cfg["T_train"]),
        RNA_dim=int(rna_dim),
        dna_dim=int(model_cfg["DNA_embed_dim"]),
        use_scheduler=True,
        dna_records=dna_record,
        gene_annot=gene_annot,
        loss_metric="htg_nll",
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        cond_dim=int(model_cfg["cond_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        use_adaptive_weight=True,
        drop_mask_prob=0.0,
        DNA_emb_len=1,
        dna_cache_dir=dna_cache_dir,
        dna_cache_k=1,
        cell_line_nums=int(cell_line_nums),
        verbose=False,
    )
    return model


class STRAND:
    """Sequence-conditioned perturbation prediction from a trained checkpoint."""

    def __init__(
        self,
        model,
        *,
        dna_record,
        gene_annot,
        device,
        sampler: SamplerConfig = SamplerConfig(),
        cache_size: int = 4096,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        _ensure_repo_on_path()
        from src.diffusion_ot.data import LRUCache

        self.model = model.to(device).eval()
        self.model.embed_backend = "interval"
        self.dna_record = dna_record
        self.gene_annot = gene_annot
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.dna_cache = LRUCache(max_size=cache_size, device=device)
        self.set_sampler(sampler)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: PathLike,
        config: PathLike,
        *,
        reference_dir: Optional[PathLike] = None,
        annotation_gtf: Optional[PathLike] = None,
        device: Optional[str] = None,
        sampler: SamplerConfig = SamplerConfig(),
        load_backbone: bool = True,
        strict: bool = True,
        cell_bias_mode: str = "normal",
    ) -> "STRAND":
        """Load a checkpoint saved by the training script.

        ``reference_dir`` holds per-chromosome FASTA files (``chr1.fasta`` ...) of the
        assembly the checkpoint was trained on (GRCh38); ``annotation_gtf`` is the transcript
        GTF used for gene lookups. Both default to the ``DNA_REFERENCE_PATH`` and
        ``GENE_ANNOTATION_PATH`` environment variables. ``load_backbone=True`` downloads the
        Flashzoi weights from Hugging Face on first use and is required for loci whose DNA
        embeddings are not in a precomputed cache.
        """
        import torch

        _ensure_repo_on_path()
        from src.diffusion_ot.training import load_config
        from src.diffusion_ot.utils import get_DNA_annot, get_dna_record

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = load_config(str(config))
        state = _load_state_dict(checkpoint, device)
        rna_dim = int(state["x_mu"].shape[1])
        # One learned null embedding per training cell line: 1 for a per-line
        # checkpoint, 3 for Combined.
        cell_line_nums = int(state["null_dna_emb.weight"].shape[0])
        dna_record = get_dna_record(str(reference_dir) if reference_dir else None)
        gene_annot = get_DNA_annot(str(annotation_gtf) if annotation_gtf else None)
        model = build_model(
            cfg["model"],
            rna_dim,
            load_backbone=load_backbone,
            dna_record=dna_record,
            gene_annot=gene_annot,
            cell_bias_mode=cell_bias_mode,
            cell_line_nums=cell_line_nums,
        )
        result = model.load_state_dict(state, strict=strict)
        if not strict and (result.missing_keys or result.unexpected_keys):
            import warnings

            warnings.warn(
                f"checkpoint/model mismatch: {len(result.missing_keys)} missing, "
                f"{len(result.unexpected_keys)} unexpected keys"
            )
        return cls(
            model,
            dna_record=dna_record,
            gene_annot=gene_annot,
            device=device,
            sampler=sampler,
            checkpoint_path=str(checkpoint),
        )

    @staticmethod
    def load_scaler(path: PathLike):
        """The scikit-learn scaler saved next to each checkpoint (``scaler.pkl``)."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------ settings
    def set_sampler(self, sampler: SamplerConfig) -> None:
        import torch

        _ensure_repo_on_path()
        from src.diffusion_ot.schedulers import get_official_betas

        betas, _, _ = get_official_betas(int(sampler.steps))
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas)
        self.model.T_sample = int(sampler.steps)
        self.model.best_scale = float(sampler.guidance_scale)
        self.model.best_eta = float(sampler.eta)
        self.model.best_beta = betas.to(self.device, dtype=torch.float32)
        self.model.best_alphas = None
        self.model.best_alphas_cumprod = None
        self.sampler = sampler

    @property
    def rna_dim(self) -> int:
        return int(self.model.x_mu.shape[1])

    # ------------------------------------------------------------------ inference
    def predict(
        self,
        controls: ControlCells,
        locus: Union[Locus, str],
        n_cells: int = 256,
        seed: int = 0,
        control_index: Optional[np.ndarray] = None,
    ) -> Prediction:
        """Predict ``n_cells`` perturbed cells for a sequence-defined perturbation at ``locus``.

        Control cells are drawn with replacement (``seed``) unless ``control_index`` is given,
        which lets several loci share the same starting cells.
        """
        import torch

        if isinstance(locus, str):
            locus = Locus.from_label(locus)
        if controls.expression.shape[1] != self.rna_dim:
            raise ValueError(
                f"controls have {controls.expression.shape[1]} genes; checkpoint expects {self.rna_dim}"
            )
        rng = np.random.default_rng(seed)
        idx = controls.sample(n_cells, rng) if control_index is None else np.asarray(control_index)
        x_ctrl = torch.as_tensor(controls.expression[idx], device=self.device)
        z_ctrl = torch.as_tensor(controls.latent[idx], device=self.device).unsqueeze(1)
        labels = [locus.label] * len(idx)
        cell_line_ids = torch.zeros(len(idx), dtype=torch.long, device=self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            z_pert = self.model.sample_latent(
                z_ctrl,
                labels,
                self.gene_annot,
                None,
                self.dna_record,
                cell_line_ids=cell_line_ids,
                cache_embeddings=self.dna_cache,
            )
            if z_pert.dim() == 2:
                z_pert = z_pert.unsqueeze(1)
            x_pert = self.model.predict(
                x_ctrl=x_ctrl,
                z_ctrl=z_ctrl,
                labels=labels,
                gene_annot=self.gene_annot,
                sgRNAs=None,
                dna_record=self.dna_record,
                cache_embeddings=self.dna_cache,
                z_0=z_pert,
                cell_line_ids=cell_line_ids,
            )
        return Prediction(
            locus=locus,
            expression=x_pert.float().cpu().numpy(),
            latent=z_pert.squeeze(1).float().cpu().numpy(),
            control_expression=controls.expression[idx],
            gene_names=controls.gene_names,
        )

    def scan(
        self,
        controls: ControlCells,
        loci: Iterable[Union[Locus, str]],
        n_cells: int = 256,
        seed: int = 0,
        shared_controls: bool = True,
    ) -> List[Prediction]:
        """Predict a series of loci. With ``shared_controls`` every locus starts from the
        same control cells, so differences between loci come from the DNA condition and the
        sampler noise only."""
        loci = list(loci)
        idx = controls.sample(n_cells, np.random.default_rng(seed)) if shared_controls else None
        return [
            self.predict(controls, locus, n_cells=n_cells, seed=seed + i, control_index=idx)
            for i, locus in enumerate(loci)
        ]


__all__ = ["SamplerConfig", "ControlCells", "Prediction", "STRAND", "build_model"]
