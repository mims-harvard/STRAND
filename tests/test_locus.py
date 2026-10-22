"""Coordinate and mask contract (NumPy only)."""

import numpy as np
import pytest

from strand.locus import (
    BIN_BP,
    N_TOKENS,
    WINDOW_BP,
    Locus,
    bp_mask,
    masked_tokens,
    scan,
    token_mask,
    window_coordinates,
)

CHR_LEN = 156_040_895  # chrX, GRCh38


def test_label_round_trip():
    locus = Locus.centered("chrX", 48_786_590, tag="GATA1", dataset="k562")
    assert locus.label == "chrX:48786390-48786790(+)-GATA1-k562"
    assert Locus.from_label(locus.label) == locus
    assert locus.width == 401
    assert locus.center == 48_786_590


def test_tags_cannot_break_the_label_grammar():
    assert Locus.centered("chr1", 1000, tag="e-HDAC6").tag == "e_HDAC6"
    with pytest.raises(ValueError):
        Locus("chr1", 100, 200, tag="bad-tag")
    with pytest.raises(ValueError):
        Locus("chr1", 200, 100)


def test_window_is_centered_and_mask_is_401_bp():
    locus = Locus.centered("chrX", 48_786_590)
    w = window_coordinates(locus, CHR_LEN)
    assert w.length == WINDOW_BP
    assert w.proxy_start == 48_786_590 - WINDOW_BP // 2
    m = bp_mask(w)
    assert m.sum() == 401
    assert m[w.rel_start] == 1 and m[w.rel_end] == 1


def test_mask_selects_three_tokens_at_the_window_center():
    # The window is centered on the locus, so a +/-200 bp mask always lands on the
    # same three 128 bp tokens away from chromosome ends.
    for center in (48_786_590, 48_800_590, 10_000_000):
        idx = masked_tokens(Locus.centered("chrX", center), CHR_LEN)
        assert idx.tolist() == [2047, 2048, 2049]


def test_window_shifts_inward_at_chromosome_ends():
    w = window_coordinates(Locus.centered("chr1", 1_000), 248_956_422)
    assert w.proxy_start == 1 and w.length == WINDOW_BP
    assert w.rel_start == 799 and w.rel_end == 1199  # 0-based window index of bases 800..1200
    w = window_coordinates(Locus.centered("chr1", 248_956_000), 248_956_422)
    assert w.proxy_end == 248_956_422 and w.length == WINDOW_BP


def test_token_mask_matches_nearest_neighbour_interpolation():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    mask = (rng.random(WINDOW_BP) < 0.001).astype(np.float32)
    ours = token_mask(mask)
    ref = torch.nn.functional.interpolate(
        torch.from_numpy(mask)[None, None], size=N_TOKENS, mode="nearest"
    )[0, 0].numpy()
    np.testing.assert_array_equal(ours, ref)
    assert N_TOKENS * BIN_BP == WINDOW_BP


def test_scan_grid():
    loci = scan("chrX", 48_798_590, 48_802_591, step=50, tag="eHDAC6", dataset="k562")
    assert len(loci) == 81
    assert loci[0].center == 48_798_590 and loci[-1].center == 48_802_590
    assert all(l.width == 401 for l in loci)
