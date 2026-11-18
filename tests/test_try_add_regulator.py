"""Unit tests for the try_add_regulator heuristic."""

from __future__ import annotations

import boolmore.boolean_functions as bf
from boolmore.evaluation import constraint as cons
from boolmore.genetic.operators.constraint_enforcement import (
    impose_necessary,
    try_add_regulator,
)


def test_try_add_regulator_forward_can_fix_regulate():
    regulators = ("A", "B")
    rr = "0000"
    repaired, max_rr, max_irr, path = try_add_regulator(regulators, rr, "A")
    assert path == "forward"
    # Rightmost mask with A set among zeros is index 3 (A&B).
    assert repaired == "0001"
    assert max_rr == bf.get_uni_rr(rr, max=True)
    assert max_irr is None  # forward hit; irr not built
    min_rr = bf.get_uni_rr(repaired, max=False)
    assert cons.check_regulate(regulators, min_rr, "A", is_min=True)


def test_try_add_regulator_forward_picks_rightmost_zero():
    # k=3: A-supersets are masks 4..7. With only 6 and 7 set, zeros are 4 and 5;
    # rightmost flip target is 5 (A&C), not sparse 4 (A alone).
    regulators = ("A", "B", "C")
    rr = "00000011"
    max_rr = bf.get_uni_rr(rr, max=True)
    assert max_rr[4] == "0" and max_rr[5] == "0"
    assert max_rr[6] == "1" and max_rr[7] == "1"
    repaired, _, _, path = try_add_regulator(regulators, rr, "A", max_rr=max_rr)
    assert path == "forward"
    assert repaired == "00000111"


def test_try_add_regulator_falls_back_to_irr_when_forward_full():
    # Constant-1: forward has no A-related zeros; must use irr.
    regulators = ("A", "B")
    rr = "1111"
    assert not cons.check_regulate(
        regulators, bf.get_uni_rr(rr, max=False), "A", is_min=True
    )
    repaired, max_rr, max_irr, path = try_add_regulator(regulators, rr, "A")
    assert path == "irr"
    assert repaired != rr
    assert max_rr == "1111"
    assert max_irr is not None


def test_try_add_regulator_reuses_cached_max_and_irr():
    regulators = ("A", "B")
    rr = "0000"
    max_rr = bf.get_uni_rr(rr, max=True)
    max_irr = bf.get_max_irr(rr, max_rr=max_rr)

    calls = {"uni": 0, "irr": 0}
    orig_uni, orig_irr = bf.get_uni_rr, bf.get_max_irr

    def counting_uni(s, max=True):
        calls["uni"] += 1
        return orig_uni(s, max=max)

    def counting_irr(s, max_rr=None):
        calls["irr"] += 1
        return orig_irr(s, max_rr=max_rr)

    bf.get_uni_rr = counting_uni  # type: ignore[assignment]
    bf.get_max_irr = counting_irr  # type: ignore[assignment]
    try:
        repaired, out_max, out_irr, path = try_add_regulator(
            regulators, rr, "A", max_rr=max_rr, max_irr=max_irr
        )
        assert path == "forward"
        assert out_max == max_rr
        assert out_irr == max_irr  # unused but returned unchanged
        assert calls["uni"] == 0
        assert calls["irr"] == 0
        assert repaired != rr
    finally:
        bf.get_uni_rr = orig_uni  # type: ignore[assignment]
        bf.get_max_irr = orig_irr  # type: ignore[assignment]


def test_try_add_regulator_both_spaces_full_returns_noop():
    # Function is exactly A: already regulates; neither space has a flip.
    regulators = ("A", "B")
    rr = bf.get_uni_rr("0011", max=True)
    assert cons.check_regulate(
        regulators, bf.get_uni_rr(rr, max=False), "A", is_min=True
    )
    repaired, max_rr, max_irr, path = try_add_regulator(regulators, rr, "A")
    assert path == "noop"
    assert repaired == rr
    assert max_rr == rr


def test_try_add_regulator_necessary_only_flips_masks_with_nec():
    regulators = ("A", "B")
    rr = "0000"
    repaired, _, _, path = try_add_regulator(
        regulators, rr, "A", necessary=("B",)
    )
    assert path == "forward"
    # Only mask with both A and B is index 3.
    assert repaired == "0001"
    after = impose_necessary(regulators, repaired, "B")
    assert cons.check_regulate(
        regulators, bf.get_uni_rr(after, max=False), "A", is_min=True
    )


def test_try_add_regulator_necessary_falls_back_to_max_irr():
    regulators = ("A", "B")
    rr = bf.get_uni_rr("0101", max=True)
    assert rr[3] == "1"
    assert not cons.check_regulate(
        regulators, bf.get_uni_rr(rr, max=False), "A", is_min=True
    )
    repaired, max_rr, max_irr, path = try_add_regulator(
        regulators, rr, "A", necessary=("B",)
    )
    assert path == "irr"
    assert repaired != rr
    assert max_irr is not None
