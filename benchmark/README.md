# Benchmark: SOTA baselines

We compare our method against **GEARS**, **BioLord**, **STATE**, and **STACK** (SOTA methods). Each method has a dedicated README with procedure, script inventory, and evaluation details.

| Method | Official repository | README | Brief procedure |
|--------|--------------------|--------|-----------------|
| **GEARS** | [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | [GEARS/README.md](GEARS/README.md) | h5ad → PertData → train GNN → 256 controls per pert → pred/real → cell-eval |
| **BioLord** | [nitzanlab/biolord](https://github.com/nitzanlab/biolord) | [Biolord/README.md](Biolord/README.md) | h5ad + gene/mapping/mask → format → train biolord → pred/real → cell-eval |
| **STATE** | [ArcInstitute/state](https://github.com/ArcInstitute/state) | [STATE/README.md](STATE/README.md) | raw h5ad → state emb transform (SE; X_state) → merge embeddings → TOML → state tx train → state tx predict → cell-eval |
| **STACK** | [ArcInstitute/stack](https://github.com/ArcInstitute/stack) | [STACK/README.md](STACK/README.md) | h5ad → prepare data → train STACK → generate predictions → format → cell-eval |

GEARS, BioLord, STATE, and STACK use the **cell-eval** package for evaluation. Each README describes a **single-dataset demo** (e.g. one cell line); combined/cross-cell-line variants are mentioned in one line where applicable.

**STATE** and **STACK** require external inputs (SE embeddings and pre-filtered cross-cell-line h5ads, respectively); see each method's README for paths and setup.
