"""Parity and dispatch tests for the optional Cython get_uni_rr kernel."""

from __future__ import annotations

import itertools
import random
from unittest.mock import patch

import pytest

import boolmore.boolean_functions as bf
from boolmore.boolean_functions import (
    _UNI_RR_CYTHON,
    _UNI_RR_CYTHON_MIN_K,
    _get_uni_rr_py,
    get_uni_rr,
)

pytestmark = pytest.mark.skipif(
    not _UNI_RR_CYTHON or bf._get_uni_rr_cy is None,
    reason="Cython get_uni_rr extension not built",
)


@pytest.mark.parametrize("max_mode", [True, False])
@pytest.mark.parametrize("k", range(0, 5))
def test_cython_matches_python_exhaustive(k, max_mode):
    n = 1 << k
    for bits in itertools.product("01", repeat=n):
        rr = "".join(bits)
        assert bf._get_uni_rr_cy(rr, max_mode) == _get_uni_rr_py(rr, max_mode)


@pytest.mark.parametrize("max_mode", [True, False])
@pytest.mark.parametrize("k", range(5, 10))
def test_cython_matches_python_random(k, max_mode):
    random.seed(k + int(max_mode))
    n = 1 << k
    for dens in (0.05, 0.3):
        for _ in range(20):
            rr = "".join("1" if random.random() < dens else "0" for _ in range(n))
            assert bf._get_uni_rr_cy(rr, max_mode) == _get_uni_rr_py(rr, max_mode)


def test_dispatcher_uses_cython_at_or_above_threshold():
    k = max(_UNI_RR_CYTHON_MIN_K, 1)
    rr = "0" * (1 << k)
    with (
        patch.object(bf, "_get_uni_rr_cy", wraps=bf._get_uni_rr_cy) as cy_mock,
        patch.object(bf, "_get_uni_rr_py", wraps=_get_uni_rr_py) as py_mock,
    ):
        out = get_uni_rr(rr, max=True)
        cy_mock.assert_called_once_with(rr, True)
        py_mock.assert_not_called()
        assert out == _get_uni_rr_py(rr, True)


def test_dispatcher_uses_python_below_threshold():
    if _UNI_RR_CYTHON_MIN_K <= 0:
        pytest.skip("Cython threshold covers all non-empty lengths")

    k = _UNI_RR_CYTHON_MIN_K - 1
    rr = "0" * (1 << k) if k >= 0 else "0"
    # Force a length strictly below the Cython cutoff.
    if len(rr) >= (1 << _UNI_RR_CYTHON_MIN_K):
        pytest.skip("unable to construct below-threshold rr")

    with (
        patch.object(bf, "_get_uni_rr_cy", wraps=bf._get_uni_rr_cy) as cy_mock,
        patch.object(bf, "_get_uni_rr_py", wraps=_get_uni_rr_py) as py_mock,
    ):
        out = get_uni_rr(rr, max=False)
        py_mock.assert_called_once_with(rr, False)
        cy_mock.assert_not_called()
        assert out == _get_uni_rr_py(rr, False)


def test_dispatcher_uses_python_for_n_leq_one_when_min_k_positive():
    """n<=1 never meets n >= 2**K_MIN when K_MIN >= 1."""
    if _UNI_RR_CYTHON_MIN_K < 1:
        pytest.skip("threshold includes n=1")

    with (
        patch.object(bf, "_get_uni_rr_cy", wraps=bf._get_uni_rr_cy) as cy_mock,
        patch.object(bf, "_get_uni_rr_py", wraps=_get_uni_rr_py) as py_mock,
    ):
        assert get_uni_rr("0", max=True) == "0"
        assert get_uni_rr("1", max=False) == "1"
        assert py_mock.call_count == 2
        cy_mock.assert_not_called()
