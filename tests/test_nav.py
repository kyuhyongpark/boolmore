"""Tests for NAV experiment import (TSV/CSV + column mapping)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from boolmore.experiment import NAVExperiment
from boolmore.io.nav import import_NAV_exps
from boolmore.outcome import ComparativeOutcome


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


def test_import_writes_preprocessed_csv(tmp_path: Path):
    tsv = _write(
        tmp_path / "data.tsv",
        "ID\tScore\tSource\tPerturbation\tObserved node\tCategorization\n"
        "1\t1.0\tA=1\tB KO, C CA\tD\tSome\n",
    )
    preprocessed = tmp_path / "data_preprocessed.csv"
    assert not preprocessed.exists()

    from_tsv = import_NAV_exps(tsv)
    assert preprocessed.exists()
    assert from_tsv[0].outcome == "Some"

    with preprocessed.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "ID",
        "Score",
        "Source",
        "Perturbation",
        "Observed node",
        "Categorization",
        "Outcome range",
    ]
    assert rows[1][2] == "A=1"
    assert rows[1][3] == "B=0,C=1"
    assert rows[1][5] == "Some"
    assert rows[1][6] == "0.25,0.75"

    from_pre = import_NAV_exps(preprocessed)
    assert from_pre == from_tsv
    assert not (tmp_path / "data_preprocessed_preprocessed.csv").exists()


def test_preprocessed_csv_always_overwritten(tmp_path: Path):
    tsv = _write(
        tmp_path / "data.tsv",
        "ID\tScore\tSource\tPerturbation\tObserved node\tCategorization\n"
        "1\t1.0\t\tX KO\tY\tOFF\n",
    )
    preprocessed = _write(tmp_path / "data_preprocessed.csv", "sentinel\n")

    import_NAV_exps(tsv)

    text = preprocessed.read_text(encoding="utf-8")
    assert "sentinel" not in text
    assert "Outcome range" in text
    assert "0.0,0.0" in text


def test_canonical_input_skips_preprocessed_write(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        '1,1.0,,B=0,C,"0.0,0.0"\n',
    )

    exps = import_NAV_exps(csv_path)
    assert exps[0].outcome == (0.0, 0.0)
    assert not (tmp_path / "data_preprocessed.csv").exists()


def test_range_string_outcome_parses_to_tuple(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        '1,1.0,A=1,,B,"0.5,1.0"\n',
    )

    exps = import_NAV_exps(csv_path)
    assert exps[0].outcome == (0.5, 1.0)
    assert not (tmp_path / "data_preprocessed.csv").exists()


def test_consistent_outcome_range_column_skips_write(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization,Outcome range\n"
        '1,1.0,,B=0,C,OFF,"0.0,0.0"\n'
        '2,1.0,A=1,,D,Some,"0.25,0.75"\n',
    )

    exps = import_NAV_exps(csv_path)
    assert exps[0].outcome == "OFF"
    assert exps[1].outcome == "Some"
    assert not (tmp_path / "data_preprocessed.csv").exists()


def test_mismatched_outcome_range_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization,Outcome range\n"
        '1,1.0,,B=0,C,ON,"0.0,0.0"\n',
    )

    with pytest.raises(ValueError, match=r"Inconsistent Outcome range \(row id=1\)"):
        import_NAV_exps(csv_path)


def test_empty_outcome_range_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization,Outcome range\n"
        "1,1.0,,B=0,C,ON,\n",
    )

    with pytest.raises(ValueError, match=r"Empty Outcome range \(row id=1\)"):
        import_NAV_exps(csv_path)


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


def test_invalid_outcome_raises_at_import(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,,,N,MAYBE\n",
    )

    with pytest.raises(ValueError, match="Unexpected experiment outcome"):
        import_NAV_exps(csv_path)


def test_import_comparative_outcomes(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,A=1,,B,higher than 2\n"
        "2,1.0,,,B,lower than 1\n"
        "3,0.5,A=0,,C,same as 4\n"
        "4,0.5,,,C,same as 3\n",
    )

    exps = import_NAV_exps(csv_path)
    assert exps[0].outcome == ComparativeOutcome(relation="higher", other_id=2)
    assert exps[1].outcome == ComparativeOutcome(relation="lower", other_id=1)
    assert exps[2].outcome == ComparativeOutcome(relation="same", other_id=4)
    assert exps[3].outcome == ComparativeOutcome(relation="same", other_id=3)

    preprocessed = tmp_path / "data_preprocessed.csv"
    assert preprocessed.exists()
    with preprocessed.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1][5] == "higher than 2"
    assert rows[1][6] == "1.0,1.0"
    assert rows[2][6] == "1.0,1.0"
    assert rows[3][5] == "same as 4"
    assert rows[3][6] == "0.0,0.0"
    assert rows[4][5] == "same as 3"
    assert rows[4][6] == "0.0,0.0"


def test_comparative_consistent_outcome_range(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization,Outcome range\n"
        '1,1.0,A=1,,B,higher than 2,"1.0,1.0"\n'
        '2,1.0,,,B,lower than 1,"1.0,1.0"\n'
        '3,1.0,A=0,,C_avg,same as 4,"0.0,0.0"\n'
        '4,1.0,,,C_avg,same as 3,"0.0,0.0"\n',
    )

    exps = import_NAV_exps(csv_path)
    assert exps[0].outcome == ComparativeOutcome(relation="higher", other_id=2)
    assert exps[2].outcome == ComparativeOutcome(relation="same", other_id=4)
    assert not (tmp_path / "data_preprocessed.csv").exists()


def test_comparative_mismatched_outcome_range_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization,Outcome range\n"
        '1,1.0,A=1,,B,higher than 2,"0.0,0.0"\n'
        '2,1.0,,,B,lower than 1,"1.0,1.0"\n',
    )

    with pytest.raises(ValueError, match=r"Inconsistent Outcome range \(row id=1\)"):
        import_NAV_exps(csv_path)


def test_comparative_missing_other_id_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,,,N,higher than 99\n",
    )

    with pytest.raises(ValueError, match="missing id 99"):
        import_NAV_exps(csv_path)


def test_comparative_different_observed_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,A=1,,B,higher than 2\n"
        "2,1.0,,,C,lower than 1\n",
    )

    with pytest.raises(ValueError, match=r"different observed \('B' vs 'C'\)"):
        import_NAV_exps(csv_path)


def test_comparative_missing_reciprocal_higher_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,A=1,,B,higher than 2\n"
        "2,1.0,,,B,OFF\n",
    )

    with pytest.raises(
        ValueError,
        match=r"expects reciprocal 'lower than 1' on experiment 2",
    ):
        import_NAV_exps(csv_path)


def test_comparative_missing_reciprocal_same_raises(tmp_path: Path):
    csv_path = _write(
        tmp_path / "data.csv",
        "ID,Score,Source,Perturbation,Observed node,Categorization\n"
        "1,1.0,A=1,,B,same as 2\n"
        "2,1.0,,,B,OFF\n",
    )

    with pytest.raises(
        ValueError,
        match=r"expects reciprocal 'same as 1' on experiment 2",
    ):
        import_NAV_exps(csv_path)
