import csv
from pathlib import Path

import pytest

from boolmore.experiment import PhenotypeExperiment
from boolmore.inference.prediction import PhenotypePrediction
from boolmore.evaluation.score import EvaluationItemScore, EvaluationResult
from boolmore.io.export import export_nav_report, export_phenotype_report


def test_export_phenotype_report(tmp_path):
    exp = PhenotypeExperiment(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 1),),
        phenotype=(("P", 1),),
        expected_exists=True,
        weight=2.0,
    )

    pred = PhenotypePrediction(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 1),),
        phenotype=(("P", 1),),
        maximal_trap_spaces=[],
        predicted_exists=True,
    )

    score_item = EvaluationItemScore(
        id=1,
        weight=2.0,
        agreement=1.0,
        score=2.0,
    )

    file_path = tmp_path / "results.csv"

    result_path = export_phenotype_report(
        [exp], [pred], [score_item], file_path
    )

    assert result_path == file_path.resolve()
    assert file_path.exists()

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["expected_exists"] == "True"
    assert rows[0]["predicted_exists"] == "True"
    assert rows[0]["score"] == "2.0"


def test_export_nav_report(tmp_path):
    result = EvaluationResult(
        model_id=0,
        max_score=1.0,
        score=1.0,
        n_edges=0,
        n_self_edges=0,
        n_prime_implicants=0,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[
            EvaluationItemScore(
                id=1,
                weight=1.0,
                agreement=1.0,
                score=1.0,
                hierarchy=0,
                fixes=(("S", 1),),
                observed="A",
                outcome="ON",
                predict_value=1.0,
            )
        ],
    )

    file_path = tmp_path / "nav_score.csv"
    result_path = export_nav_report(result, file_path)

    assert result_path == file_path.resolve()
    assert isinstance(result_path, Path)

    with open(file_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["fixes"] == "S=1"
    assert rows[0]["observed_node"] == "A"
    assert rows[0]["score"] == "1.0"


def test_export_nav_report_requires_nav_fields(tmp_path):
    result = EvaluationResult(
        model_id=0,
        max_score=1.0,
        score=1.0,
        n_edges=0,
        n_self_edges=0,
        n_prime_implicants=0,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[
            EvaluationItemScore(
                id=1,
                weight=1.0,
                agreement=1.0,
                score=1.0,
            )
        ],
    )

    with pytest.raises(ValueError, match="NAV item scores"):
        export_nav_report(result, tmp_path / "bad.csv")
