# STRAND: Sequence-Conditioned Transport for Single-Cell Perturbations

**Authors**
* Boyang Fu
* Sameer Gabbita
* George Dasoulas
* Xiang Lin
* Shanghua Gao
* Xiaorui Su
* Soumya Ghosh
* Marinka Zitnik

[Preprint](https://arxiv.org/abs/2602.10156) | [Website](https://zitniklab.hms.harvard.edu/STRAND/)

*Advances in Neural Information Processing Systems (NeurIPS), 2026*

STRAND predicts single-cell transcriptional responses by conditioning on the regulatory DNA
sequence at the perturbed locus rather than on a gene identifier, so perturbations that target
different genomic sites within the same gene can receive different predicted responses.

## Why sequence conditioning

Most single-cell perturbation models represent a perturbation as a discrete gene identifier or a
node in a gene graph, which collapses every targeted site inside a gene into one representation and
closes the perturbation vocabulary to annotated genes. STRAND instead takes a reference DNA window
and a perturbation mask at a user-specified locus, so the perturbation is defined by sequence and
position. This supports zero-shot inference at targets unseen during training and locus-resolved
inputs across a 524 kb window at 128 bp model resolution.

![Overview of STRAND: the same gene-associated region is targeted at two distinct loci, and the two
perturbation sequences produce different predicted transcriptomic responses.](assets/figure1_overview.png)

**Figure 1.** Perturbations targeting the same gene at different genomic locations (*s*<sub>p</sub><sup>1</sup>,
*s*<sub>p</sub><sup>2</sup>) can induce distinct transcriptional responses.

## How STRAND works

Regulatory DNA and a target mask enter a frozen DNA foundation model; a learnable DNA Perturbation
Module turns the masked sequence context into a perturbation condition; a control-anchored latent
transport module carries the control cell state to the perturbed state under that condition; and a
residual hurdle decoder predicts perturbed expression relative to a reconstructed control baseline.

![STRAND architecture: DNA sequence and target mask through the frozen DNA foundation model and the
DNA Perturbation Module, control cells through the frozen RNA foundation model, joined by the
Cellular Transport Module (an I2SB bridge) and a residual decoder.](assets/figure2_architecture.png)

**Figure 2.** The frozen encoders are Borzoi (DNA, 128 bp output bins over a 524 kb window) and
SCimilarity (RNA). The learnable components are the DNA Perturbation Module *g*<sub>φ</sub>, the
Cellular Transport Module *e*<sub>θ</sub> (a control-anchored I²SB bridge), and the gene-level
decoder. An auxiliary sequence–expression alignment objective links the DNA condition to the mean
transcriptional effect.

## Quick start

```bash
git clone https://github.com/mims-harvard/STRAND.git && cd STRAND
pip install -r requirements.txt
```

Requires Python ≥ 3.10 (tested: Python 3.11, PyTorch 2.9.0). Run commands from the repository
root; tests: `python -m pytest tests`.

Point the pipeline at the reference genome, the annotation and your data roots:

```bash
export DATA_ROOT="/path/to/data_root"              # curated .h5ad + embeddings
export OUTPUT_ROOT="/path/to/outputs"              # checkpoints + DNA cache
export DNA_REFERENCE_PATH="/path/to/hg38_fasta_dir"  # per-chromosome FASTA (GRCh38)
export GENE_ANNOTATION_PATH="/path/to/gencode.v47.transcript.annotation.gtf"
export HGNC_COMPLETE_SET_PATH="/path/to/hgnc_complete_set.txt"  # gene symbol -> Ensembl ID
export CCRE_BED_PATH="/path/to/ENCFF234XEZ.bed.gz"  # ENCODE cCREs (cCRE bias)
export DNA_CACHE_DIR="$OUTPUT_ROOT/dna_cache"      # DNA embeddings, built in step 02, read by steps 04-05
```

Train a per-cell-line model with the configuration used in the paper (after data steps 01–03
below):

```bash
bash scripts/04_train/train.sh k562   # k562 | jurkat | rpe1 | combined_perturbqa_hvg2k
```

Predict the test split with a trained checkpoint (writes paired predicted / measured `.h5ad` files):

```bash
python scripts/05_evaluate/evaluate.py \
  --config configs/k562_config.yaml --dname k562 \
  --checkpoint_path /path/to/best_model.pt \
  --result_path outputs/k562_eval
```

Inspect or build a sequence-defined perturbation directly. `strand.locus` needs only NumPy, and
reproduces the coordinate and mask construction the model uses at inference:

```python
from strand import Locus, masked_tokens

locus = Locus.centered("chrX", 48_786_590, half_width=200, tag="GATA1", dataset="k562")
print(locus.label)                      # chrX:48786390-48786790(+)-GATA1-k562
print(masked_tokens(locus, 156_040_895))  # DNA context tokens selected by the mask
```

`strand.STRAND.from_checkpoint(...)` loads a trained checkpoint and exposes `predict` and `scan`
for single loci and sliding-window scans; see [docs below](#reproducibility-and-documentation) for
the model dependencies this requires.

## Reproducibility and documentation

The pipeline runs in numbered order:

| Step | What it does | Where |
| --- | --- | --- |
| 01 | Curate the PerturbQA screens into per-cell-line `.h5ad` files | [`scripts/01_curate/`](scripts/01_curate/) |
| 02 | SCimilarity embeddings and OT pairing of control and perturbed cells (the split column chosen here sets zero-shot or few-shot); Borzoi DNA embedding cache | [`scripts/02_embed_and_match/`](scripts/02_embed_and_match/) |
| 03 | Train/validation split of the OT training pairs | [`scripts/03_make_splits/`](scripts/03_make_splits/) |
| 04 | Train STRAND | [`scripts/04_train/`](scripts/04_train/) |
| 05 | Predict the test split with a trained checkpoint | [`scripts/05_evaluate/`](scripts/05_evaluate/) |

| Also | Where |
| --- | --- |
| Datasets, sources and the perturbation-coordinate protocol | [`docs/data.md`](docs/data.md) |
| Training configurations (one per cell line, plus Combined) | [`configs/`](configs/) |
| STRAND model (paper Fig. 2) and the two Fig. 4 ablations | [`src/diffusion_ot/models/`](src/diffusion_ot/models/) |
| DNA Perturbation Module over the Borzoi backbone | [`PModule/DNA_modeling.py`](PModule/DNA_modeling.py) |
| Baselines (GEARS, BioLord, STATE, STACK) | [`benchmark/`](benchmark/) |

**Perturbation coordinates.** Public CRISPRi datasets carry gene-level labels rather than exact
guide coordinates, so the default benchmark uses a TSS-centered proxy: gene symbols are resolved to
Ensembl IDs, a canonical transcript is chosen by tag priority, and a ±200 bp mask is placed on the
TSS inside a 524 kb window. Exact guide coordinates can be supplied when available; the paper
reports a sensitivity analysis comparing the two. `docs/data.md` documents the full procedure.

**External dependencies.** Not bundled here: the Borzoi/Flashzoi DNA backbone
([borzoi-pytorch](https://github.com/johahi/borzoi-pytorch)), the SCimilarity RNA encoder
([Genentech/scimilarity](https://github.com/Genentech/scimilarity)), the HiRef optimal-transport
solver ([raphael-group/HiRef](https://github.com/raphael-group/HiRef)) and the GRCh38 reference with
GENCODE annotation.

**Scope.** Training and evaluation use CRISPRi data in K562, Jurkat and RPE1. HCT116 appears only as
an external cross-study transfer evaluation and is not part of the training data. STRAND operates at
128 bp model resolution, set by the DNA encoder's output binning.

## Citation

```bibtex
@inproceedings{fu2026strand,
  title     = {{STRAND}: Sequence-Conditioned Transport for Single-Cell Perturbations},
  author    = {Fu, Boyang and Gabbita, Sameer and Dasoulas, George and Lin, Xiang and
               Gao, Shanghua and Su, Xiaorui and Ghosh, Soumya and Zitnik, Marinka},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Questions: please open an issue in this repository or contact marinka@hms.harvard.edu.
