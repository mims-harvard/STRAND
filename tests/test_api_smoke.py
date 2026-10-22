"""Interface smoke test with synthetic inputs.

This exercises the real model classes (DNA Perturbation Module, I2SB sampler, hurdle
decoder) on CPU with random weights, a random chromosome and random DNA-encoder features.
It checks shapes, determinism and the locus -> mask -> prediction contract. It says nothing
about biology; scientifically meaningful inference needs a trained checkpoint, the GRCh38
reference and the frozen Flashzoi encoder (see `STRAND.from_checkpoint` in strand/api.py).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from strand.api import STRAND, ControlCells, SamplerConfig, build_model  # noqa: E402
from strand.locus import N_TOKENS, Locus, bp_mask, token_mask, window_coordinates  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CHROM_LEN = 600_000
RNA_DIM = 24


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    record = {"chr1": SimpleNamespace(seq="".join(rng.choice(list("ACGT"), size=CHROM_LEN)))}
    cfg = yaml.safe_load(open(REPO / "configs" / "k562_config.yaml"))["model"]
    torch.manual_seed(0)
    model = build_model(cfg, RNA_DIM, load_backbone=False, dna_record=record)
    runner = STRAND(model, dna_record=record, gene_annot=None, device="cpu", sampler=SamplerConfig(steps=4))
    loci = [Locus.centered("chr1", 300_000, tag="siteA"), Locus.centered("chr1", 301_280, tag="siteB")]
    for i, locus in enumerate(loci):
        mask = token_mask(bp_mask(window_coordinates(locus, CHROM_LEN)))
        feats = torch.from_numpy(np.random.default_rng(10 + i).normal(size=(1, N_TOKENS, 1536)).astype(np.float32))
        key = f"{locus.chrom}:{locus.start}-{locus.end}({locus.strand})".upper()
        runner.dna_cache[key] = (feats, torch.from_numpy(mask).view(1, 1, -1), None)
    controls = ControlCells(
        np.abs(rng.normal(size=(32, RNA_DIM))).astype(np.float32),
        rng.normal(size=(32, 128)).astype(np.float32),
        np.array([f"G{i}" for i in range(RNA_DIM)]),
    )
    return runner, loci, controls


def test_checkpoint_architecture_from_config_has_expected_keys(synthetic):
    runner, _, _ = synthetic
    keys = runner.model.state_dict().keys()
    assert "x_mu" in keys and "DNA_model.perturbation.distance_bias.ell" in keys
    assert not any(k.startswith("diffusion_model.blocks.1.") for k in keys)  # num_layers: 1


def test_predict_shapes_and_contract(synthetic):
    runner, loci, controls = synthetic
    pred = runner.predict(controls, loci[0], n_cells=6, seed=3)
    assert pred.expression.shape == (6, RNA_DIM)
    assert pred.control_expression.shape == (6, RNA_DIM)
    assert pred.latent.shape == (6, 128)
    assert np.isfinite(pred.expression).all()
    assert (pred.expression >= 0).all()  # hurdle predictive mean is non-negative
    table = pred.response_table()
    assert list(table.columns) == ["gene", "control_mean", "predicted_mean", "delta"]
    adata = pred.to_anndata()
    assert adata.shape == (12, RNA_DIM) and adata.uns["strand_locus"]["start"] == loci[0].start


def test_same_seed_is_deterministic_and_loci_differ(synthetic):
    runner, loci, controls = synthetic
    idx = np.arange(6)
    a1 = runner.predict(controls, loci[0], n_cells=6, seed=5, control_index=idx).expression
    a2 = runner.predict(controls, loci[0], n_cells=6, seed=5, control_index=idx).expression
    b = runner.predict(controls, loci[1], n_cells=6, seed=5, control_index=idx).expression
    np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-6)
    assert not np.allclose(a1, b)


def test_gene_count_mismatch_is_rejected(synthetic):
    runner, loci, _ = synthetic
    bad = ControlCells(np.zeros((4, RNA_DIM + 1)), np.zeros((4, 128)))
    with pytest.raises(ValueError):
        runner.predict(bad, loci[0], n_cells=2)
