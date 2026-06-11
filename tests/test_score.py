import pytest

from boolmore.eval.score import get_phenotype_score
from boolmore.core.experiment import PhenotypeExperiment
from boolmore.core.prediction import PhenotypePrediction


def test_get_phenotype_score_match():
    exp = PhenotypeExperiment(
        id=1,
        perturbation=(),
        sources=(),
        phenotype=(),
        expected_exists=True,
        weight=2.0,
    )

    pred = PhenotypePrediction(
        id=1,
        perturbation=(),
        sources=(),
        phenotype=(),
        found_phenotypes=[],
        predicted_exists=True,
        agreement=0.0,
        score=0.0,
    )

    max_score, score = get_phenotype_score([exp], [pred])

    assert pred.agreement == 1.0
    assert pred.score == 2.0
    assert max_score == 2.0
    assert score == 2.0


def test_get_phenotype_score_mismatch():
    exp = PhenotypeExperiment(
        id=1,
        perturbation=(),
        sources=(),
        phenotype=(),
        expected_exists=False,
        weight=3.0,
    )

    pred = PhenotypePrediction(
        id=1,
        perturbation=(),
        sources=(),
        phenotype=(),
        found_phenotypes=[],
        predicted_exists=True,
        agreement=0.0,
        score=0.0,
    )

    max_score, score = get_phenotype_score([exp], [pred])

    assert pred.agreement == 0.0
    assert pred.score == 0.0
    assert max_score == 3.0
    assert score == 0.0


def test_get_phenotype_score_missing_id():
    exp = PhenotypeExperiment(
        id=1,
        perturbation=(),
        sources=(),
        phenotype=(),
        expected_exists=True,
        weight=1.0,
    )

    pred = PhenotypePrediction(
        id=2,
        perturbation=(),
        sources=(),
        phenotype=(),
        found_phenotypes=[],
        predicted_exists=True,
        agreement=0.0,
        score=0.0,
    )

    with pytest.raises(ValueError):
        get_phenotype_score([exp], [pred])