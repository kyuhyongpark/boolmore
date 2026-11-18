"""Parity tests for the optional Cython LDOI closure kernel."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest
from pyboolnet.external.bnet2primes import bnet_text2primes

import boolmore.inference.ldoi as ldoi_mod
from boolmore.inference.ldoi import (
    _LDOI_CYTHON,
    _get_compiled,
    _ldoi_closure_fast,
    _ldoi_many_cython,
    _ldoi_many_python,
    ldoi_closure_engine,
    ldoi_many,
    logical_domain_of_influence,
)

pytestmark = pytest.mark.skipif(
    not _LDOI_CYTHON or ldoi_mod._batch_closure_cy is None,
    reason="Cython LDOI extension not built",
)


def test_cython_batch_matches_python_and_scan():
    primes = bnet_text2primes("A, A\nB, A\nX, A & B\nC, 1\n")
    seeds = [{}, {"A": 1}, {"A": 1, "B": 0}, {"X": 1}]
    compiled = _get_compiled(primes)
    with ldoi_closure_engine("scan"):
        scan = [logical_domain_of_influence(primes, seed) for seed in seeds]
    py = _ldoi_many_python(compiled, seeds)
    cy = _ldoi_many_cython(compiled, seeds)
    assert cy == py == scan


def test_cython_seed_overrides_constant():
    primes = bnet_text2primes("A, 0\nB, A\n")
    compiled = _get_compiled(primes)
    seed = {"A": 1}
    with ldoi_closure_engine("scan"):
        scan = logical_domain_of_influence(primes, seed)
    assert _ldoi_many_cython(compiled, [seed])[0] == scan
    assert _ldoi_closure_fast(compiled, seed) == scan


def test_cython_random_networks_match_scan():
    rng = random.Random(1)
    formulas = [
        "0",
        "1",
        "{x}",
        "!{x}",
        "{x} & {y}",
        "{x} | {y}",
        "{x} & !{y}",
        "!{x} | {y}",
    ]
    for trial in range(20):
        n = rng.randint(3, 7)
        names = [f"N{i}" for i in range(n)]
        lines: list[str] = []
        for target in names:
            template = rng.choice(formulas)
            regs = rng.sample(names, k=min(2, n))
            fmt = {"x": regs[0], "y": regs[-1]}
            lines.append(f"{target}, {template.format(**fmt)}")
        primes = bnet_text2primes("\n".join(lines) + "\n")
        seeds: list[dict[str, int]] = []
        for _ in range(6):
            k = rng.randint(0, min(3, n))
            seed_nodes = rng.sample(names, k=k)
            seeds.append({node: rng.randint(0, 1) for node in seed_nodes})
        with ldoi_closure_engine("scan"):
            expected = [
                logical_domain_of_influence(primes, seed) for seed in seeds
            ]
        assert ldoi_many(primes, seeds) == expected


def test_dispatcher_uses_cython_when_available():
    primes = bnet_text2primes("A, A\nB, A\n")
    seeds = [{"A": 1}, {}]
    with (
        patch.object(
            ldoi_mod, "_ldoi_many_cython", wraps=ldoi_mod._ldoi_many_cython
        ) as cy_mock,
        patch.object(
            ldoi_mod, "_ldoi_many_python", wraps=ldoi_mod._ldoi_many_python
        ) as py_mock,
    ):
        out = ldoi_many(primes, seeds)
        cy_mock.assert_called_once()
        py_mock.assert_not_called()
        assert out == _ldoi_many_python(_get_compiled(primes), seeds)


def test_single_seed_public_api_uses_cython():
    primes = bnet_text2primes("A, A\nB, A\n")
    with patch.object(
        ldoi_mod, "_ldoi_one_cython", wraps=ldoi_mod._ldoi_one_cython
    ) as cy_mock:
        out = logical_domain_of_influence(primes, {"A": 1})
        cy_mock.assert_called_once()
        assert out == _ldoi_closure_fast(_get_compiled(primes), {"A": 1})
