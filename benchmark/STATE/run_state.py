#!/usr/bin/env python3
"""
Minimal STATE benchmark pipeline: calls the official STATE CLI for train and predict.

Paths are read from environment variables (or CLI overrides). Requires the STATE
package to be installed (e.g. `uv tool install arc-state` or `pip install arc-state`).

Environment variables (used when CLI args are not provided):
  DATA_DIR       - Base data directory (used in TOML or for adata paths).
  TOML_CONFIG    - Path to STATE TOML config (required for train).
  STATE_RUN_DIR  - Output directory for training; also used for predict (run dir).
  EMBEDDINGS_PATH -  Path to precomputed embeddings (merge into h5ad or set in TOML).
  EMBED_KEY     -  Key in adata.obsm for STATE SE embeddings (from state emb transform). Default in original pipeline: X_state.

External data (documented in README):
  - SE embeddings: Produced by the official STATE SE model (state emb transform) on full-gene raw input; pass path or merge into h5ad.
  - HVG matrix: If using HVG subset, produce filtered h5ad beforehand and set paths in TOML.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v else default


def _state_cmd() -> list[str]:
    """STATE CLI: use 'state' from PATH if available, else 'python -m state'."""
    import shutil
    if shutil.which("state"):
        return ["state"]
    return [sys.executable, "-m", "state"]


def run_train(
    toml_config: str,
    run_dir: str,
    embed_key: str | None = None,
    extra_overrides: list[str] | None = None,
) -> int:
    """Run `state tx train` with the given TOML config and output dir."""
    cmd = _state_cmd() + [
        "tx",
        "train",
        f"data.kwargs.toml_config_path={toml_config}",
        f"output_dir={run_dir}",
    ]
    if embed_key:
        cmd.append(f"data.kwargs.embed_key={embed_key}")
    if extra_overrides:
        cmd.extend(extra_overrides)

    print(f"[run_state] Running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def run_predict(run_dir: str, checkpoint: str = "last.ckpt") -> int:
    """Run `state tx predict` (or infer) on the trained run."""
    cmd = _state_cmd() + [
        "tx",
        "predict",
        f"--output_dir={run_dir}",
        f"--checkpoint={checkpoint}",
    ]
    print(f"[run_state] Running: {' '.join(cmd)}", flush=True)
    ret = subprocess.call(cmd)
    if ret != 0:
        # Try infer if predict uses different args
        cmd_infer = _state_cmd() + [
            "tx",
            "infer",
            f"--model_dir={run_dir}",
            f"--checkpoint={checkpoint}",
        ]
        print(f"[run_state] Trying: {' '.join(cmd_infer)}", flush=True)
        ret = subprocess.call(cmd_infer)
    return ret


def main() -> int:
    parser = argparse.ArgumentParser(
        description="STATE benchmark: train and/or predict via official STATE CLI. Paths from env or args."
    )
    parser.add_argument(
        "--toml_config",
        type=str,
        default=_env("TOML_CONFIG"),
        help="Path to STATE TOML config (default: TOML_CONFIG env)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=_env("STATE_RUN_DIR"),
        help="Output dir for training; run dir for predict (default: STATE_RUN_DIR env)",
    )
    parser.add_argument(
        "--embed_key",
        type=str,
        default=_env("EMBED_KEY"),
        help="Key in adata.obsm for embeddings (default: EMBED_KEY env)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint name for predict (default: last.ckpt)",
    )
    parser.add_argument(
        "step",
        type=str,
        nargs="?",
        choices=["train", "predict", "both"],
        default="both",
        help="Run train, predict, or both (default: both)",
    )
    args = parser.parse_args()

    if not args.toml_config:
        print("ERROR: --toml_config or TOML_CONFIG must be set.", file=sys.stderr)
        return 1
    if not args.run_dir:
        print("ERROR: --run_dir or STATE_RUN_DIR must be set.", file=sys.stderr)
        return 1

    os.makedirs(args.run_dir, exist_ok=True)

    if args.step in ("train", "both"):
        ret = run_train(
            toml_config=args.toml_config,
            run_dir=args.run_dir,
            embed_key=args.embed_key or None,
        )
        if ret != 0:
            return ret
    if args.step in ("predict", "both"):
        ret = run_predict(run_dir=args.run_dir, checkpoint=args.checkpoint)
        if ret != 0:
            return ret

    print("[run_state] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
