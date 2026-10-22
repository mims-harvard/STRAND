# Data Availability and Resources

## Reference Genome and Gene Annotation

- **Reference human genome (GRCh38.p0):**  
  https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/

- **Gene annotation (GENCODE v47, used for TSS matching):**  
  https://www.gencodegenes.org/human/release_47.html

## Datasets

### Dataset 1: RPE1 (ReplogleWeissman2022_rpe1)

| Field | Detail |
| --- | --- |
| Source paper | Replogle et al., Cell 185(14):2559–2575, 2022 [link](https://pubmed.ncbi.nlm.nih.gov/35688146/) |
| Cell line | hTERT-immortalized, near-euploid retinal pigment epithelial (RPE1) cells expressing ZIM3 KRAB-dCas9-P2A-BFP |
| Perturbation type | CRISPRi targeting DepMap 20Q1 common essential genes, sampled at day 7 post-transduction |
| Library | Dual-sgRNA multiplexed CRISPRi library (two sgRNAs per gene, single lentiviral vector); sgRNA sequences in Table S1 (supplement) |
| Scale in STRAND | 2,394 unique perturbations (2,002 overlapping with PerturbQA); 209,506 cells (84.5% coverage); 7,724 genes out of 9,063 total union genes |
| Data access | Interactive browser and processed data at http://gwps.wi.mit.edu; raw sequencing on SRA under BioProject PRJNA831566 |
| File used | ReplogleWeissman2022_rpe1.h5ad (also available via pertpy.data.replogle_2022_rpe1()) |

### Dataset 2: K562 Genome-Wide (ReplogleWeissman2022_K562_gwps)

| Field | Detail |
| --- | --- |
| Source paper | Replogle et al., Cell 185(14):2559–2575, 2022 [link](https://pubmed.ncbi.nlm.nih.gov/35688146/) |
| Cell line | Chronic myeloid leukemia (CML) K562 cells expressing dCas9-BFP-KRAB |
| Perturbation type | CRISPRi targeting all expressed genes in K562 (genome-wide), sampled at day 8 post-transduction |
| Library | Dual-sgRNA multiplexed CRISPRi library targeting 9,866 expressed genes; sgRNA sequences in Table S1 (supplement) |
| Scale in STRAND | 9,867 unique perturbations (1,831 overlapping with PerturbQA); 379,676 cells (20% coverage, reflecting genome-wide scale); 7,290 genes out of 9,063 total union genes |
| Data access | Same as RPE1 — http://gwps.wi.mit.edu; SRA BioProject PRJNA831566 |
| File used | ReplogleWeissman2022_K562_gwps.h5ad (also available via pertpy.data.replogle_2022_k562_gwps()) |

### Dataset 3: Jurkat (GSE264667_jurkat_raw_singlecell_01.h5ad)

| Field | Detail |
| --- | --- |
| Source paper | Nadig, Replogle et al. (TRADE),  [link](https://www.nature.com/articles/s41588-025-02169-3) |
| Cell line | Jurkat T-cell leukemia cells expressing UCOE-EF1α-Zim3-dCas9-P2A-mCherry |
| Perturbation type | CRISPRi targeting 2,393 DepMap 20Q1 common essential genes (dual-sgRNA library dJR092), sampled at day 7 post-transduction; 56 GEM groups on 10x Genomics Chromium v3 |
| Scale in STRAND | 2,394 unique perturbations (1,540 overlapping with PerturbQA); 191,466 cells (72.81% coverage); 7,952 genes out of 9,063 total union genes |
| Data access | GEO accession GSE264667; raw sequencing on SRA under BioProject PRJNA1100571 |
| File used | GSE264667_jurkat_raw_singlecell_01.h5ad (8.7 GB, downloadable from GEO) |

## Data Processing Pipeline

The full curation and preprocessing pipeline is provided in:

- `scripts/01_curate/`
- `scripts/03_make_splits/`

## Fine-Grained Perturbation Protocol

This section is a practical guide for researchers who need to recover locus-level (genomic) information from gene-level perturbation labels. Two stages matter: (1) expression curation in `scripts/01_curate/`, and (2) model-time DNA/TSS logic in the model package `src/diffusion_ot/`. Keep them separate when reproducing results. Model-side entry points (under `src/diffusion_ot/`): `utils/dna_utils.py` (`gen_gene_TSS_sequence`), `data/grna_resolution.py` (optional guide-level refinement), `data/preprocessing.py` (HGNC / Ensembl), `data/datasets.py` (`curate_perturb_tag_obs_only`).

1) Curation: inputs, perturbation filter, union gene set, HVG

- Inputs (see `scripts/01_curate/README.md`): PerturbQA metadata CSVs (`k562-de.csv`, `rpe1-de.csv`, `jurkat-de.csv`) and the three source `.h5ad` files named in `curate_perturbqa.py`.
- Perturbation filter (`process_and_save` in `scripts/01_curate/curate_perturbqa.py`): keep cells whose `obs['gene']` is in the union of PerturbQA `pert` values for that line **or** `"non-targeting"`. Cells with any other target gene are dropped (not counted as "unmapped TSS"; they are excluded by design for the PerturbQA-aligned subset).
- Union gene set (`load_metadata_and_compute_union` in the same script): gene symbols are taken from each dataset's `adata.var`—`var['gene_name']` if present, otherwise `var.index`. The union is `union(K562, RPE1, Jurkat)` via `numpy.union1d`. The **9,063** total is the size of that union when run on the pinned source h5ads; it is **not** an expression threshold and not derived from the CSV `gene` columns (those drive perturbation lists, not the union size).
- Gene matrix alignment (`reformat_adata_genes`): each dataset is aligned to a canonical gene list; genes in the list missing from the object are zero-filled; genes present in the object but outside the canonical list are removed from the matrix.
- Combined object: the script concatenates union-aligned data and selects **top 2,000 highly variable genes** using Scanpy's Seurat v3 flavor **on control (`non-targeting`) cells only**, producing `combined_perturbqa_hvg2k.h5ad` (see module docstring in `curate_perturbqa.py`).

2) Model-time: gene identifiers and TSS localization (GENCODE)

- **Mapping symbols to Ensembl gene IDs** for DNA lookup uses HGNC (`convert_gene_name`, `convert_to_ensembl` in `data/preprocessing.py`) against `hgnc_complete_set.txt` (official symbols, aliases, previous symbols, and `ensembl_gene_id`). Dataset-specific label curation (e.g. which `obs` column holds the target) is handled in `curate_perturb_tag_obs_only` in `data/datasets.py` (e.g. Jurkat/K562/RPE1 use `obs['gene']`).
- **TSS assignment** is the default perturbation localization in STRAND, implemented in `utils/dna_utils.py` as `gen_gene_TSS_sequence(gene_ID, ...)`. It reads a **transcript-level** GENCODE GTF (**GENCODE v47**, from `GENE_ANNOTATION_PATH` unless a path is passed; align this file with the reference build cited above). The DNA cache builder (`scripts/02_embed_and_match/build_dna_cache.py`) groups transcripts by **Ensembl `gene_id`** (version removed) and looks genes up through HGNC; training and evaluation group them by GENCODE `gene_name`.
- **Canonical transcript / TSS selection policy:** among rows for that `gene_id`, the code selects one transcript row by scanning the GTF `attribute` column for tags in this **priority order**: `MANE_Select` → `Ensembl_canonical` → `appris_principal_1` → `GENCODE_Primary` → `basic`, taking the **first matching row** for the first tag tier that yields any hit. If none match, it uses the **first row** of that gene's table. This is **not** "most expressed" or "all TSSs"; it is tag-priority then fallback.
- **Strand and TSS coordinate:** strand is read from the GTF. For `+`, TSS uses the transcript **start** column; for `-`, TSS uses the **end** column (GTF coordinates are **1-based inclusive** as usual). The chosen row's start/end define TSS/TES for the window loop.

**Note on annotation version:** `gen_gene_TSS_sequence`, `get_DNA_annot` and `get_dna_record` read `GENE_ANNOTATION_PATH` and `DNA_REFERENCE_PATH` unless paths are passed. For reproducibility, point both at the GENCODE v47 transcript annotation and the GRCh38 FASTA directory.

3) Model-time: reference sequence and window

- **Reference:** sequences are read from per-chromosome FASTA files via **Biopython `SeqIO.parse`**, not `bedtools getfasta`. `DNA_REFERENCE_PATH` must be a directory holding one file per chromosome, named `chr1.fasta` … `chr22.fasta`, `chrX.fasta`, `chrY.fasta`, `chrM.fasta` (`get_dna_record` in `utils/dna_utils.py` loads all 25).
- **Window:** `gen_gene_TSS_sequence` builds a fixed-length window of **`FIX_WIN` base pairs (default 524,288)** centered on the TSS, with boundary adjustments so the window stays within the chromosome; optional `jitter_window` shifts the center within feasible bounds. The extracted substring uses **1-based** `proxy_start` / `proxy_end` converted to Python slicing as `record.seq[(proxy_start-1):proxy_end]` (0-based half-open slice).
- **Perturbation mask (default):** A **±200 bp** region centered on the TSS within the extracted window marks the perturbation site. This TSS-centered mask is the standard perturbation localization used across all experiments reported in the paper. It is computed in window-relative coordinates as a region of width `2 * tss_window` (default 200 bp on each side) around `TSS_rel` in **0-based indices**.

4) Optional: sub-TSS refinement via gRNA coordinates

When individual sgRNA genomic coordinates are available in the dataset metadata, STRAND can optionally refine the perturbation mask to the exact guide cut site rather than the TSS-centered default. This path is implemented in `data/grna_resolution.py` and is designed for future datasets that ship with per-guide positional annotations.

- Guide strings are read from `obs` columns (`gRNA_unified`, `sgID_AB`, `guide_id`, etc.—see `extract_grna_field` in `data/grna_resolution.py`). Curated catalogs (unified / trade / jiang) supply guide IDs and optional `(gene, strand, coordinate)` lookups; catalog paths can be configured via environment variables such as `DIFFUSION_OT_UNIFIED_GRNA_CSV`.
- **`lookup_curated_grnas`:** resolves by exact normalized ID, else by `(gene, strand, coordinate)`; if multiple catalog rows share the same key, the **first** entry is used.
- **`build_grna_mask_region`:** maps guide **absolute genomic coordinates** to positions inside the large window using `guide.coordinate - proxy_start`. If guide coordinates are unavailable or fall outside the window, the standard TSS-centered mask is used.

This refinement is not required for reproducing the results in the paper; all reported experiments use TSS-centered localization.

5) QC summary

| Situation | Behavior (code path) |
|-----------|-------------------------|
| Perturbation not in PerturbQA `pert` list + non-targeting | Cell dropped at curation (`process_and_save`). |
| Gene in matrix not in canonical union list | Column dropped in `reformat_adata_genes`. |
| `gene_id` missing from GTF | `gen_gene_TSS_sequence` warns and returns `None`. |
| gRNA coordinates unavailable | Standard TSS-centered ±200 bp mask is used (default behavior). |

6) End-to-end order (for STRAND-ready DNA inputs)

Download **GRCh38** reference FASTA and **GENCODE v47** transcript annotation → map perturbation gene symbols to **Ensembl `gene_id`** (HGNC) → select canonical transcript by **tag priority** and compute **TSS** with **strand** → extract a **524 kb** centered window with **Biopython** from per-chrom FASTA (1-based coords → 0-based slice) → apply **TSS ±200 bp mask** as the perturbation region within that window → run curation (`curate_perturbqa.py`) for expression matrices and HVG union as above. Optionally, if per-guide genomic coordinates are available, the mask can be refined to the exact cut site via the gRNA resolution module.