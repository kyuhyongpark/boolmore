import pytest

from boolmore.evaluation.score import (
    EvaluationItemScore,
    EvaluationResult,
    get_model_score,
    get_result_kind,
)
from boolmore.evaluation.phenotype_score import get_phenotype_scores
from boolmore.experiment import PhenotypeExperiment
from boolmore.inference.prediction import PhenotypePrediction


def test_get_phenotype_score_match():
    exp = PhenotypeExperiment(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 0),),
        phenotype=(("P", 1),),
        expected_exists=True,
        weight=2.0,
        metadata={"note": "ok"},
    )

    pred = PhenotypePrediction(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 0),),
        phenotype=(("P", 1),),
        maximal_trap_spaces=[{"P": 1}],
        predicted_exists=True,
    )

    scores = get_phenotype_scores([exp], [pred])
    max_score, score = get_model_score(scores)

    assert scores[0].agreement == 1.0
    assert scores[0].score == 2.0
    assert scores[0].kind == "Phenotype"
    assert scores[0].perturbation == (("A", 1),)
    assert scores[0].sources == (("S", 0),)
    assert scores[0].phenotype == (("P", 1),)
    assert scores[0].expected_exists is True
    assert scores[0].predicted_exists is True
    assert scores[0].maximal_trap_spaces == [{"P": 1}]
    assert scores[0].metadata == {"note": "ok"}
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
        maximal_trap_spaces=[],
        predicted_exists=True,
    )

    scores = get_phenotype_scores([exp], [pred])
    max_score, score = get_model_score(scores)

    assert scores[0].agreement == 0.0
    assert scores[0].score == 0.0
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
        maximal_trap_spaces=[],
        predicted_exists=True,
    )

    with pytest.raises(ValueError):
        get_phenotype_scores([exp], [pred])


def test_get_result_kind():
    nav = EvaluationItemScore(id=1, weight=1.0, agreement=1.0, score=1.0, kind="NAV")
    pheno = EvaluationItemScore(
        id=2, weight=1.0, agreement=1.0, score=1.0, kind="Phenotype"
    )
    untyped = EvaluationItemScore(id=3, weight=1.0, agreement=1.0, score=1.0)

    assert get_result_kind([nav]) == "NAV"
    assert get_result_kind([pheno]) == "Phenotype"
    assert get_result_kind([nav, pheno]) == "Mixed"
    assert get_result_kind([untyped]) == "Mixed"
    assert get_result_kind([]) == "Mixed"

    result = EvaluationResult(
        model_id=0,
        max_score=1.0,
        score=1.0,
        n_eff_edges=0,
        n_eff_self_edges=0,
        n_prime_implicants=0,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[nav],
    )
    assert result.kind == "NAV"
