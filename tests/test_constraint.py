import itertools
import random

import pytest

from boolmore.boolean_functions import get_uni_rr
from boolmore.evaluation.constraint import (
    check_constant,
    check_necessary,
    check_node,
    check_regulate,
    check_source,
)


EMPTY_CONSTRAINTS = {
    "fixed": [],
    "regulate": {},
    "necessary": {},
    "group": {},
    "possible_constant": [],
}


def test_base_constant_unchanged_passes():
    assert check_node((), "0", "0", EMPTY_CONSTRAINTS, "GEF1_4_10") is True


def test_base_constant_flipped_without_regulators_fails():
    assert check_node((), "1", "0", EMPTY_CONSTRAINTS, "GEF1_4_10") is False


def test_base_constant_with_edge_pool_regulator_nonconstant_passes():
    # Seed-11 case: constant 0 gains ABI1 and becomes identity-like.
    assert check_node(("ABI1",), "10", "0", EMPTY_CONSTRAINTS, "GEF1_4_10") is True


def test_base_constant_with_regulator_same_constant_passes():
    assert check_node(("ABI1",), "00", "0", EMPTY_CONSTRAINTS, "GEF1_4_10") is True


def test_base_constant_with_regulator_opposite_constant_fails():
    assert check_node(("ABI1",), "11", "0", EMPTY_CONSTRAINTS, "GEF1_4_10") is False


def test_pure_source_unchanged_passes():
    assert check_node(("A",), "10", "10", EMPTY_CONSTRAINTS, "A") is True


def test_pure_source_mutated_fails():
    assert check_node(("A",), "01", "10", EMPTY_CONSTRAINTS, "A") is False


def test_source_with_added_regulator_passes():
    # Former source that gained an edge-pool regulator is no longer frozen.
    assert check_node(("A", "B"), "1000", "10", EMPTY_CONSTRAINTS, "A") is True


@pytest.mark.parametrize("k", range(1, 5))
def test_check_helpers_precomputed_forms_match(k):
    regulators = tuple(f"R{i}" for i in range(k))
    n = 2**k
    rng = random.Random(k)

    samples = [
        "".join(bits) for bits in itertools.product("01", repeat=n)
    ] if k <= 3 else [
        "".join(rng.choice("01") for _ in range(n)) for _ in range(80)
    ]

    for rr in samples:
        min_rr = get_uni_rr(rr, max=False)
        max_rr = get_uni_rr(rr, max=True)

        assert check_constant(rr) == check_constant(rr, max_rr=max_rr)
        assert check_source(regulators, rr, regulators[0]) == check_source(
            regulators, rr, regulators[0], min_rr=min_rr
        )
        for reg in regulators:
            assert check_regulate(regulators, rr, reg) == check_regulate(
                regulators, min_rr, reg, is_min=True
            )
            assert check_necessary(regulators, rr, reg) == check_necessary(
                regulators, rr, reg, min_rr=min_rr
            )


def test_check_node_fixed_rejects_mutation():
    constraints = {
        **EMPTY_CONSTRAINTS,
        "fixed": ["N"],
    }
    assert check_node(("A", "B"), "0001", "0001", constraints, "N") is True
    assert check_node(("A", "B"), "0011", "0001", constraints, "N") is False


def test_check_node_regulate_present_and_absent():
    regulators = ("A", "B")
    rr_ok = "0100"  # A regulates; non-constant
    rr_bad = "0000"  # A does not regulate
    constraints = {
        **EMPTY_CONSTRAINTS,
        "regulate": {"N": ["A"]},
    }
    assert check_regulate(regulators, rr_ok, "A")
    assert check_node(regulators, rr_ok, "0", constraints, "N") is True

    constraints_bad = {
        **EMPTY_CONSTRAINTS,
        "regulate": {"N": ["A"]},
        "possible_constant": ["N"],
    }
    assert not check_regulate(regulators, rr_bad, "A")
    assert check_node(regulators, rr_bad, "0", constraints_bad, "N") is False


def test_check_node_necessary_present_and_absent():
    regulators = ("A", "B")
    constraints = {
        **EMPTY_CONSTRAINTS,
        "necessary": {"N": ["A"]},
        "possible_constant": ["N"],
    }
    rr_ok = "0100"
    assert check_necessary(regulators, rr_ok, "A")
    assert check_node(regulators, rr_ok, "0", constraints, "N") is True

    rr_bad = "0010"
    assert not check_necessary(regulators, rr_bad, "A")
    assert check_node(regulators, rr_bad, "0", constraints, "N") is False


def test_check_node_possible_constant_allows_constant():
    regulators = ("A", "B")
    constraints = {
        **EMPTY_CONSTRAINTS,
        "possible_constant": ["N"],
    }
    assert check_node(regulators, "0000", "0001", constraints, "N") is True
    # Without possible_constant, same constant fails for non-base-constant.
    assert check_node(regulators, "0000", "0001", EMPTY_CONSTRAINTS, "N") is False


def test_check_node_self_loop_source_rejected():
    # Node N regulated only by itself as a source-like rule should fail.
    regulators = ("N", "A")
    # Construct a source-on-N rule if possible; otherwise assert helper agreement.
    constraints = EMPTY_CONSTRAINTS
    # Identity-like on N alone would be length-2; with two regs use min form with
    # single ON at the N-source position.
    # If check_source is True, check_node must fail.
    for rr in ("".join(bits) for bits in itertools.product("01", repeat=4)):
        if check_source(regulators, rr, "N"):
            assert check_node(regulators, rr, "0001", constraints, "N") is False
            break
    else:
        pytest.skip("no length-4 source-on-N example found")


def test_check_node_opposite_constant_rejected():
    # Former constant 0 that becomes constant 1 with a regulator.
    assert check_node(("A",), "11", "0", EMPTY_CONSTRAINTS, "X") is False
    assert check_node(("A",), "00", "0", EMPTY_CONSTRAINTS, "X") is True
