"""strand.locus agrees with the functions the model calls (src/diffusion_ot)."""

import numpy as np
import pytest

from strand.locus import Locus, bp_mask, window_coordinates

dm = pytest.importorskip("src.diffusion_ot.data.dna_matching")


class _Record:
    def __init__(self, seq):
        self.seq = seq


@pytest.mark.parametrize("center", [300_000, 1_000, 999_500, 600_123])
def test_window_and_mask_match_gen_interval_centered_sequence(center):
    rng = np.random.default_rng(center)
    seq = "".join(rng.choice(list("ACGT"), size=1_000_000))
    locus = Locus.centered("chr7", center)
    info = dm.gen_interval_centered_sequence(
        locus.chrom, locus.start, locus.end, strand="+", dna_record={"chr7": _Record(seq)}
    )
    ours = window_coordinates(locus, len(seq))
    assert (info["proxy_start"], info["proxy_end"]) == (ours.proxy_start, ours.proxy_end)
    assert (info["interval_rel_start"], info["interval_rel_end"]) == (ours.rel_start, ours.rel_end)
    _, mask = dm.aggregate_mask_from_interval(info)
    np.testing.assert_array_equal(mask, bp_mask(ours))
    assert info["seq"] == seq[ours.proxy_start - 1 : ours.proxy_end]


def test_label_parser_matches():
    locus = Locus.centered("chrX", 48_800_590, strand="-", tag="eHDAC6_w32", dataset="rpe1")
    assert dm.parse_interval_label(locus.label) == (
        "chrX", locus.start, locus.end, "-", "eHDAC6_w32", "rpe1"
    )


def test_vendored_one_hot():
    from src.diffusion_ot.data import _dna_onehot

    x = _dna_onehot.dna_1hot("ACGTNacgt")
    assert x.shape == (9, 4)
    assert x[:4].argmax(1).tolist() == [0, 1, 2, 3]
    assert not x[4].any()
    np.testing.assert_array_equal(x[5:], x[:4])
