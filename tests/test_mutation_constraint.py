"""Exact-outcome tests for reused min/max RR forms in mutation."""

from __future__ import annotations

import random

import pytest

import boolmore.boolean_functions as bf
from boolmore.evaluation import constraint as cons
from boolmore.genetic.operators import constraint_enforcement as ce
from boolmore.genetic.operators.mutation import (
    mutate_rr_bias,
    mutate_rr_constraint,
    regulate_repair_enabled,
)


def _empty_constraints() -> dict:
    return {
        "fixed": [],
        "regulate": {},
        "necessary": {},
        "group": {},
        "possible_constant": [],
    }


def _ref_mutate_rr_constraint(
    regulators: tuple[str, ...],
    rr: str,
    base_rr: str,
    constraints: dict,
    node: str,
    probability: float,
    bias: float = 0.5,
) -> tuple[str, bool]:
    """Pre-reuse implementation: re-canonicalizes inside each check call."""
    if len(regulators) == 0:
        if rr == base_rr:
            return rr, False
        return base_rr, True

    if len(regulators) == 1 and len(base_rr) != 1:
        if rr == base_rr:
            return rr, False
        return base_rr, True

    if node in constraints["fixed"]:
        if rr == base_rr:
            return rr, False
        return base_rr, True

    mutated_rr = ""
    redo = True
    trial = 0
    while redo:
        if trial > 10000:
            raise Exception("too many trials. check mutation parameter")

        if node in constraints["group"]:
            groups = constraints["group"][node]
            group_rr, group_regulators = ce.rr2group_rr(regulators, rr, groups)
            mutated_group_rr = mutate_rr_bias(group_rr, probability, bias)
            mutated_rr = ce.group_rr2rr(
                regulators, mutated_group_rr, group_regulators
            )
        else:
            mutated_rr = mutate_rr_bias(rr, probability, bias)

        if node in constraints["necessary"]:
            for necc in constraints["necessary"][node]:
                mutated_rr = ce.impose_necessary(regulators, mutated_rr, necc)

        redo = False
        if node in constraints["regulate"]:
            for reg in constraints["regulate"][node]:
                redo = redo or not cons.check_regulate(regulators, mutated_rr, reg)
                if redo:
                    break

        if node in regulators:
            redo = redo or cons.check_source(regulators, mutated_rr, node)

        if cons.check_constant(mutated_rr):
            max_rr = bf.get_uni_rr(mutated_rr, max=True)
            if node not in constraints["possible_constant"] and len(base_rr) != 1:
                redo = True
            elif base_rr == "1" and "0" in max_rr:
                redo = True
            elif base_rr == "0" and "1" in max_rr:
                redo = True

        if redo:
            trial += 1

    max_original = bf.get_uni_rr(rr)
    max_mutated = bf.get_uni_rr(mutated_rr)
    modified = max_original != max_mutated
    return mutated_rr, modified


def _constraint_combos(node: str, regulators: tuple[str, ...]) -> list[dict]:
    empty = _empty_constraints()
    combos = [empty]

    if len(regulators) >= 1:
        reg0 = regulators[0]
        c = _empty_constraints()
        c["regulate"] = {node: [reg0]}
        combos.append(c)

        c = _empty_constraints()
        c["necessary"] = {node: [reg0]}
        combos.append(c)

        c = _empty_constraints()
        c["regulate"] = {node: [reg0]}
        c["necessary"] = {node: [reg0]}
        combos.append(c)

    c = _empty_constraints()
    c["possible_constant"] = [node]
    combos.append(c)

    c = _empty_constraints()
    c["fixed"] = [node]
    combos.append(c)

    return combos


def _has_regulate(constraints: dict) -> bool:
    return bool(constraints.get("regulate"))


@pytest.mark.parametrize("probability", [0.05, 0.3])
@pytest.mark.parametrize("bias", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_mutate_rr_constraint_matches_reference(probability, bias, k):
    """RNG-exact vs pre-reuse reference with regulate repair disabled."""
    regulators = tuple(f"R{i}" for i in range(k))
    # Include a self-loop case for source checks.
    node = regulators[0]
    base_rr = "1" + "0" * (2**k - 1)  # single ON bit at mask 0 (constant 1)
    rr = base_rr

    with regulate_repair_enabled(False):
        for constraints in _constraint_combos(node, regulators):
            for seed in range(40):
                random.seed(seed)
                ref_out = _ref_mutate_rr_constraint(
                    regulators, rr, base_rr, constraints, node, probability, bias
                )
                ref_state = random.getstate()

                random.seed(seed)
                new_out = mutate_rr_constraint(
                    regulators, rr, base_rr, constraints, node, probability, bias
                )
                new_state = random.getstate()

                mutated_rr, modified, min_rr, max_rr = new_out
                assert (mutated_rr, modified) == ref_out
                assert new_state == ref_state
                if min_rr is None:
                    # Early-return paths (fixed / constant / single-reg) skip forms.
                    assert max_rr is None
                else:
                    assert min_rr == bf.get_uni_rr(mutated_rr, max=False)
                    assert max_rr == bf.get_uni_rr(mutated_rr, max=True)


@pytest.mark.parametrize("probability", [0.05, 0.3])
@pytest.mark.parametrize("bias", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_mutate_rr_constraint_regulate_postconditions(probability, bias, k):
    """With repair on, accepted mutants still satisfy regulate constraints."""
    regulators = tuple(f"R{i}" for i in range(k))
    node = regulators[0]
    base_rr = "1" + "0" * (2**k - 1)
    rr = base_rr

    regulate_combos = [
        c for c in _constraint_combos(node, regulators) if _has_regulate(c)
    ]
    with regulate_repair_enabled(True):
        for constraints in regulate_combos:
            for seed in range(40):
                random.seed(seed)
                mutated_rr, _modified, min_rr, max_rr = mutate_rr_constraint(
                    regulators, rr, base_rr, constraints, node, probability, bias
                )
                assert min_rr is not None and max_rr is not None
                assert min_rr == bf.get_uni_rr(mutated_rr, max=False)
                assert max_rr == bf.get_uni_rr(mutated_rr, max=True)
                for reg in constraints["regulate"][node]:
                    assert cons.check_regulate(
                        regulators, min_rr, reg, is_min=True
                    )


def test_mutate_rr_constraint_early_returns_match_reference():
    constraints = _empty_constraints()
    # Constant / no regulators
    assert mutate_rr_constraint((), "0", "0", constraints, "X", 0.5) == (
        "0",
        False,
        None,
        None,
    )
    assert mutate_rr_constraint((), "1", "0", constraints, "X", 0.5) == (
        "0",
        True,
        None,
        None,
    )
    # Single regulator (non-base-constant); identity is "01" in mask order
    assert mutate_rr_constraint(("A",), "01", "01", constraints, "B", 0.5) == (
        "01",
        False,
        None,
        None,
    )
    assert mutate_rr_constraint(("A",), "10", "01", constraints, "B", 0.5) == (
        "01",
        True,
        None,
        None,
    )
    # Fixed
    fixed = _empty_constraints()
    fixed["fixed"] = ["N"]
    assert mutate_rr_constraint(("A", "B"), "1000", "1000", fixed, "N", 0.5) == (
        "1000",
        False,
        None,
        None,
    )
    assert mutate_rr_constraint(("A", "B"), "1100", "1000", fixed, "N", 0.5) == (
        "1000",
        True,
        None,
        None,
    )
