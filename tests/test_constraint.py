from boolmore.evaluation.constraint import check_node


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
