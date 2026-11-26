"""Equivalence: Primer-B bnet import vs pyboolnet BNetToPrime."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.io.bnet_primes import (
    bnet_file_to_primes,
    bnet_text_to_primes,
    canon_primes,
)
from boolmore.io.load import load_primes
from boolmore.io.paths import project_root

REPO = project_root()
CAD_BNET = REPO / "demo" / "CAD.bnet"
BREAST_BNET = REPO / "case_study" / "breast_cancer" / "parsimonious.bnet"


def _assert_same(text: str) -> None:
    expected = canon_primes(bnet_text2primes(text))
    got = canon_primes(bnet_text_to_primes(text))
    assert got == expected


@pytest.mark.parametrize(
    "text",
    [
        "A, 0\n",
        "A, 1\n",
        "A, A\n",
        "A, !B\nB, A\n",
        "A, B | C\nB, 1\nC, 0\n",
        "A, (B|C) & (D|E)\nB,1\nC,1\nD,1\nE,1\n",
        "A, !!B\nB, 1\n",
        "A, !(B & C)\nB, 1\nC, 1\n",
        "A, !(B | C)\nB, 1\nC, 1\n",
        "targets, factors\nErk, Erk & Mek | Mek & Raf\nMek, Erk | Mek & Raf\nRaf, !Erk | !Raf\n",
    ],
)
def test_bnet_text_matches_pyboolnet(text: str):
    _assert_same(text)


def test_auto_source_for_undefined_regulator():
    text = "A, S & B\nB, 1\n"
    _assert_same(text)
    primes = bnet_text_to_primes(text)
    assert "S" in primes
    assert primes["S"] == [[{"S": 0}], [{"S": 1}]]


def test_header_with_tab_ignored():
    text = "targets,\tfactors\nA, 1\n"
    primes = bnet_text_to_primes(text)
    assert set(primes) == {"A"}
    assert primes["A"] == [[], [{}]]


def test_cad_bnet_file_matches_pyboolnet():
    assert CAD_BNET.is_file()
    expected = canon_primes(bnet_text2primes(CAD_BNET.read_text(encoding="utf-8")))
    got = canon_primes(bnet_file_to_primes(CAD_BNET))
    assert got == expected


def test_breast_cancer_bnet_matches_pyboolnet():
    assert BREAST_BNET.is_file()
    expected = canon_primes(bnet_text2primes(BREAST_BNET.read_text(encoding="utf-8")))
    got = canon_primes(bnet_file_to_primes(BREAST_BNET))
    assert got == expected


def test_load_primes_uses_primer_path(tmp_path: Path):
    path = tmp_path / "tiny.bnet"
    path.write_text("A, !B\nB, A\n", encoding="utf-8")
    expected = canon_primes(bnet_text2primes(path.read_text(encoding="utf-8")))
    got = canon_primes(load_primes(path))
    assert got == expected
