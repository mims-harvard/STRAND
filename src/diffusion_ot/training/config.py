"""
Configuration utilities: Config loading and flattening.

This module contains:
- load_config: Load YAML configuration files
"""

import os
import yaml
from typing import Dict, Any


def load_config(default_path):
    """
    Load YAML configuration and apply lightweight portability transforms:
    - Expand env vars like ${DATA_ROOT}
    - Expand user home (~)
    - Resolve selected IO paths relative to the config file directory when not absolute
    """
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(default_path))

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expanduser(os.path.expandvars(obj))
        return obj

    config = _expand(config)

    # Resolve common path fields relative to config dir (keeps YAMLs portable)
    io = config.get("IO", {})
    for key in ("root_dir", "idx_root", "anndata_path", "dna_cache_dir"):
        val = io.get(key, None)
        if isinstance(val, str) and val and (not os.path.isabs(val)):
            io[key] = os.path.normpath(os.path.join(cfg_dir, val))
    config["IO"] = io
    return config


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

