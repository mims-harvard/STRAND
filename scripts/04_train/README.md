## 04_train

Trains STRAND ([`src/diffusion_ot/models/strand_model.py`](../../src/diffusion_ot/models/strand_model.py)):
the frozen Flashzoi/Borzoi DNA encoder with the DNA Perturbation Module, the I²SB latent
transport module and the residual hurdle decoder (paper Fig. 2).

### Requirements
- **DNA resources** (not bundled):
  - `DNA_REFERENCE_PATH`: directory with per-chromosome FASTA files
  - `GENE_ANNOTATION_PATH`: GTF (e.g. GENCODE)
  - `CCRE_BED_PATH` (optional): BED/GZ file for cCRE intervals
- Python deps (see top-level `requirements.txt`)

Recommended environment:

```bash
export DATA_ROOT="/path/to/data_root"
export OUTPUT_ROOT="/path/to/outputs"
export DNA_REFERENCE_PATH="/path/to/hg38_fasta_dir"
export GENE_ANNOTATION_PATH="/path/to/gencode.gtf"
export CCRE_BED_PATH="/path/to/ccres.bed.gz"   # optional (can be empty)
```

### Run training (local launcher)

```bash
bash train.sh k562
```

This invokes `train.py` with the appropriate config under `configs/`.

### Run training (direct Python)

```bash
python train.py \
  --dname k562 \
  --config ../../configs/k562_config.yaml \
  --RNA_loss htg_nll \
  --out_suffix cache_line_specific \
  --version v1 \
  --no_lr_scheduler
```

Notes:
- `--run_group {zero_shot, few_shot, heldout_cell_line}` only names the checkpoint subdirectory
  (`checkpoints/<run_group>/...`). The setting itself comes from the split used when the OT
  pairs were built in `02_embed_and_match`.
- Ablation switches inside the DNA Perturbation Module: `--ablate_distance_bias`,
  `--ablate_ccre_bias`, `--cell_bias_mode`. They appear in the run name as
  `distbias-on|off`, `ccrebias-on|off`, `cellbias-<mode>`.
- The two model-level ablations of paper Fig. 4 are implemented in
  `src/diffusion_ot/models/ablation_fixed_dna_embedding.py` (w/o Perturb Module) and
  `src/diffusion_ot/models/ablation_deterministic_fusion.py` (w/o Diffusion); `train.py`
  does not select them.
- The DNA cache is read from `--dna_cache_dir`, else `$DNA_CACHE_DIR`, else the config's `IO.dna_cache_dir` (same order in `05_evaluate/evaluate.py`).
- `--use_wandb` is optional; if used, `wandb` must be installed.
- The optimizer is AdamW. The paper's runs used Muon ([KellerJordan/Muon](https://github.com/KellerJordan/Muon)), which this release does not require.

