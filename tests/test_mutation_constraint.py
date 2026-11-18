"""Exact-outcome tests for reused min/max RR forms in mutation."""

from __future__ import annotations

import random

import pytest

import boolmore.boolean_functions as bf
from boolmore.evaluation import constraint as cons
from boolmore.genetic.operators import constraint_enforcement as ce
from boolmore.genetic.operators.mutation import mutate_rr_bias, mutate_rr_constraint


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


@pytest.mark.parametrize("probability", [0.05, 0.3])
@pytest.mark.parametrize("bias", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_mutate_rr_constraint_matches_reference(probability, bias, k):
    regulators = tuple(f"R{i}" for i in range(k))
    # Include a self-loop case for source checks.
    node = regulators[0]
    base_rr = "0" * (2**k)
    base_rr = base_rr[:-1] + "1"  # single ON bit at the end
    rr = base_rr

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

            assert new_out == ref_out
            assert new_state == ref_state


def test_mutate_rr_constraint_early_returns_match_reference():
    constraints = _empty_constraints()
    # Constant / no regulators
    assert mutate_rr_constraint((), "0", "0", constraints, "X", 0.5) == (
        "0",
        False,
    )
    assert mutate_rr_constraint((), "1", "0", constraints, "X", 0.5) == (
        "0",
        True,
    )
    # Single regulator (non-base-constant)
    assert mutate_rr_constraint(("A",), "10", "10", constraints, "B", 0.5) == (
        "10",
        False,
    )
    assert mutate_rr_constraint(("A",), "01", "10", constraints, "B", 0.5) == (
        "10",
        True,
    )
    # Fixed
    fixed = _empty_constraints()
    fixed["fixed"] = ["N"]
    assert mutate_rr_constraint(("A", "B"), "0001", "0001", fixed, "N", 0.5) == (
        "0001",
        False,
    )
    assert mutate_rr_constraint(("A", "B"), "0011", "0001", fixed, "N", 0.5) == (
        "0001",
        True,
    )
