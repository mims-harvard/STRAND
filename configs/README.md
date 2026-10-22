## Configs

YAML configs for training (`scripts/04_train/train.py`): one per cell line (`k562`, `jurkat`,
`rpe1`) and `combined_config.yaml` for the cross-cell-line Combined dataset, which adds
`cell_line_mapping` for cell-line-specific null embeddings.

These configs are **portable** by default via environment-variable expansion:
- `${DATA_ROOT}`: points to the data root folder created by curation/bridging
- `${OUTPUT_ROOT}`: where checkpoints/caches are stored

The loader expands env vars automatically in `src/diffusion_ot/training/config.py`.

