"""Tests for NAV experiment import (TSV/CSV + column mapping)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from boolmore.experiment import NAVExperiment
from boolmore.io.nav import import_NAV_exps


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_import_tsv_default_mapping(tmp_path: Path):
    tsv = _write(
        tmp_path / "data.tsv",
        "ID\tScore\tSource\tPerturbation\tObserved node\tCategorization\n"
        "1\t1.0\tA=1\tB KO\tC\tOFF\n"
        "2\t0.5\t\tD CA\tE\tON\n",
    )

    exps = import_NAV_exps(tsv)

    assert exps == [
        NAVExperiment(
            id=1,
            fixes=(("A", 1), ("B", 0)),
            observed="C",
            outcome="OFF",
            weight=1.0,
        ),
        NAVExperiment(
            id=2,
            fixes=(("D", 1),),
            observed="E",
            outcome="ON",
            weight=0.5,
        ),
    ]


def test_tsv_writes_csv_when_missing_and_round_trips(tmp_path: Path):
    tsv = _write(
        tmp_path / "data.tsv",
        "ID\tScore\tSource\tPerturbation\tObserved node\tCategorization\n"
        "1\t1.0\tA=1\tB KO, C CA\tD\tSome\n",
    )
    csv_path = tmp_path / "data.csv"
    assert not csv_path.exists()

    from_tsv = import_NAV_exps(tsv)
    assert csv_path.exists()

    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "ID",
        "Score",
        "Source",
        "Perturbation",
        "Observed node",
        "Categorization",
    ]
    assert rows[1][2] == "A=1"
    assert rows[1][3] == "B=0,C=1"

    from_csv = import_NAV_exps(csv_path)
    assert from_csv == from_tsv


def test_tsv_does_not_overwrite_existing_csv(tmp_path: Path):
    tsv = _write(
        tmp_path / "data.tsv",
        "ID\tScore\tSource\tPerturbation\tObserved node\tCategorization\n"
        "1\t1.0\t\tX KO\tY\tOFF\n",
    )
    csv_path = _write(tmp_path / "data.csv", "sentinel\n")

    import_NAV_exps(tsv)

    assert csv_path.read_text(encoding="utf-8") == "sentinel\n"


def test_normalized_header_aliases(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "id,score,source,perturbation,Observed_Node,value\n"
        "3,2.0,S=0,T=1,U,OFF/Some\n",
    )

    exps = import_NAV_exps(csv_path)
    assert exps == [
        NAVExperiment(
            id=3,
            fixes=(("S", 0), ("T", 1)),
            observed="U",
            outcome="OFF/Some",
            weight=2.0,
        )
    ]


def test_custom_fixes_cols_merge(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Signals,Extra,Observed node,Categorization\n"
        "1,1.0,A=1,B=0,C,ON\n",
    )

    exps = import_NAV_exps(
        csv_path,
        fixes_cols=("Signals", "Extra"),
    )
    assert exps[0].fixes == (("A", 1), ("B", 0))


def test_dash_only_fixes_cell_is_empty(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,-,B=1,C,ON\n"
        "2,1.0,A=0,-,D,OFF\n",
    )

    exps = import_NAV_exps(csv_path)
    assert exps[0].fixes == (("B", 1),)
    assert exps[1].fixes == (("A", 0),)


def test_duplicate_node_in_fixes_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Signals,Observed node,Categorization\n"
        "1,1.0,A=1,A=1,B,ON\n",
    )

    with pytest.raises(ValueError, match="Duplicate node 'A'"):
        import_NAV_exps(csv_path, fixes_cols=("Source", "Signals"))


def test_duplicate_id_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,A=1,,B,ON\n"
        "1,0.5,C=0,,D,OFF\n",
    )

    with pytest.raises(ValueError, match="Duplicate id 1"):
        import_NAV_exps(csv_path)


def test_observed_in_fixes_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,A=1,B=0,A,ON\n",
    )

    with pytest.raises(
        ValueError,
        match="Node 'A' appears in both fixes and observed",
    ):
        import_NAV_exps(csv_path)


def test_inconsistent_node_in_fixes_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Signals,Observed node,Categorization\n"
        "1,1.0,A=1,A=0,B,ON\n",
    )

    with pytest.raises(ValueError, match="Inconsistent fixes for node 'A'"):
        import_NAV_exps(csv_path, fixes_cols=("Source", "Signals"))


def test_ambiguous_single_column_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Score Alias,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,2.0,,,N,OFF\n",
    )

    with pytest.raises(ValueError, match="Ambiguous columns for 'weight'"):
        import_NAV_exps(csv_path, weight_col=("Score", "Score Alias"))


def test_missing_required_column_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node\n"
        "1,1.0,,,N\n",
    )

    with pytest.raises(ValueError, match="No column found for 'outcome'"):
        import_NAV_exps(csv_path)


def test_intervention_alias_in_default_fixes(tmp_path: Path):
    tsv = _write(
        tmp_path / "data.tsv",
        "ID\tScore\tSource\tIntervention\tObserved Node\tValue\n"
        "1\t1.0\t\tFoo KO\tBar\tOFF\n",
    )

    exps = import_NAV_exps(tsv)
    assert exps[0].fixes == (("Foo", 0),)
    assert exps[0].observed == "Bar"
    assert exps[0].outcome == "OFF"
