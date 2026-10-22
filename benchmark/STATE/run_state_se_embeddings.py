#!/usr/bin/env python3
"""
Prototype wrapper for STATE SE embedding generation: calls the official
`state emb transform` CLI to produce cell embeddings from full-gene raw h5ad input.

Use full-gene input (e.g. <dname>_raw_with_split.h5ad from scripts 01-04); do not
use HVG-only or log-normalized data. See benchmark/STATE/README.md for details.

Requires the STATE package (e.g. `uv tool install arc-state` or `pip install arc-state`).
CLI args match the official STATE docs: --model-folder, --checkpoint, --input, --output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _state_cmd() -> list[str]:
    """STATE CLI: use 'state' from PATH if available, else 'python -m state'."""
    import shutil
    if shutil.which("state"):
        return ["state"]
    return [sys.executable, "-m", "state"]


def run_emb_transform(
    model_folder: str,
    checkpoint: str,
    input_path: str,
    output_path: str,
    embed_key: str = "X_state",
) -> int:
    """Run `state emb transform` with the given paths and embed key."""
    cmd = _state_cmd() + [
        "emb",
        "transform",
        "--model-folder", model_folder,
        "--checkpoint", checkpoint,
        "--input", input_path,
        "--output", output_path,
        "--embed-key", embed_key,
    ]
    print(f"[run_state_se_embeddings] Running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="STATE SE embeddings: run state emb transform. Use full-gene raw h5ad as input (e.g. <dname>_raw_with_split.h5ad)."
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to STATE SE model folder (e.g. SE-600M). Passed as --model-folder to state emb transform.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SE checkpoint (e.g. se600m_epoch15.ckpt). Passed as --checkpoint to state emb transform.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input h5ad (full-gene raw, e.g. <dname>_raw_with_split.h5ad).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output h5ad or .npy (embeddings written by state emb transform).",
    )
    parser.add_argument(
        "--embed_key",
        type=str,
        default="X_state",
        help="Key in adata.obsm for embeddings (default: X_state, matching original pipeline).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    ret = run_emb_transform(
        model_folder=args.model_folder,
        checkpoint=args.checkpoint,
        input_path=os.path.abspath(args.input),
        output_path=os.path.abspath(args.output),
        embed_key=args.embed_key,
    )
    if ret != 0:
        return ret
    print("[run_state_se_embeddings] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
