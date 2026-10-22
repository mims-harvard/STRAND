"""
Default configuration and paths for Diffusion-OT.

Centralizes all hard-coded paths and default settings.
"""

import os
from typing import Dict, Any
from pathlib import Path


def _repo_root() -> Path:
    """
    Resolve repository root.

    File is at: <repo>/src/diffusion_ot/config/defaults.py
    """
    return Path(__file__).resolve().parents[3]


def get_default_paths() -> Dict[str, str]:
    """
    Get default paths for DNA data, annotations, and model checkpoints.
    
    Returns:
        Dictionary mapping path names to absolute paths.
    """
    repo_root = _repo_root()

    # User-overridable roots (preferred for portability)
    data_root = os.environ.get("DATA_ROOT", str(repo_root / "data" / "processed"))
    output_root = os.environ.get("OUTPUT_ROOT", str(repo_root / "outputs"))
    
    paths = {
        # Main data root (contains per-dataset folders)
        "data_root": data_root,

        # Outputs (checkpoints, caches)
        "output_root": output_root,

        # DNA reference genome (set externally; no genome is bundled)
        "dna_reference": os.environ.get("DNA_REFERENCE_PATH", ""),

        # Gene annotation (set externally)
        "gene_annotation": os.environ.get("GENE_ANNOTATION_PATH", ""),

        # HGNC mapping (optional; used by some preprocessing helpers)
        "hgnc_annotation": os.environ.get("HGNC_COMPLETE_SET_PATH", ""),

        # cCRE BED file (optional; set externally)
        "ccre_bed": os.environ.get("CCRE_BED_PATH", ""),

        # SCimilarity model directory (optional; only needed if you run embedding generation)
        "scimilarity_model": os.environ.get("SCIMILARITY_MODEL_PATH", ""),
    }
    
    return paths


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values for model, training, and data.
    
    Returns:
        Dictionary with default configuration values.
    """
    config = {
        "model": {
            "T_train": 1000,
            "DNA_embed_dim": 1920,
            "cond_dim": 256,
            "hidden_dim": 128,
            "embedding_dim": 128,
            "num_heads": 4,
            "num_layers": 4,
            "beta_schedule": "linear_symmetric",
            "use_scheduler": True,
            "cfg_dropout_prob": 0.1,
            "DNA_emb_len": 1,
        },
        "train": {
            "batch_size": 32,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "num_epochs": 100,
            "loss_weight": 1.0,
        },
        "data": {
            "h5_suffix": "X_scimilarity.h5",
            "idx_train_suffix": "trainsub.idx.csv",
            "idx_val_suffix": "valsub.idx.csv",
            "idx_test_suffix": "testsub.idx.csv",
            "hvg_thresh": 1e-2,
            "scaler_sample_size": 10000,
        },
        "IO": {
            "root_dir": get_default_paths()["data_root"],
            "idx_root": get_default_paths()["data_root"],
        },
    }
    
    return config


# Convenience accessors
DEFAULT_PATHS = get_default_paths()
DEFAULT_CONFIG = get_default_config()

# Legacy names used across the codebase
PROCESSED_DATA_ROOT = DEFAULT_PATHS["data_root"]
SCIMILARITY_MODEL_PATH = DEFAULT_PATHS.get("scimilarity_model", "")
DNA_REFERENCE_PATH = DEFAULT_PATHS.get("dna_reference", "")
GENE_ANNOTATION_PATH = DEFAULT_PATHS.get("gene_annotation", "")
HGNC_COMPLETE_SET_PATH = DEFAULT_PATHS.get("hgnc_annotation", "")
CCRE_BED_PATH = DEFAULT_PATHS.get("ccre_bed", "")

# Common suffix defaults
H5_SUFFIX = "X_scimilarity.h5"
IDX_SUFFIX = "trainsub.idx.csv"

