import csv
from pathlib import Path

import pytest

from boolmore.genetic.population import Candidate
from boolmore.evaluation.score import EvaluationItemScore, EvaluationResult
from boolmore.io.export import (
    export_candidate,
    export_nav_report,
    export_phenotype_report,
)
from boolmore.model import Model


def _tiny_model() -> Model:
    return Model.from_primes({
        "A": [[{"A": 0}], [{"A": 1}]],
    })


def _nav_result() -> EvaluationResult:
    return EvaluationResult(
        model_id=3,
        max_score=1.0,
        score=1.0,
        n_eff_edges=1,
        n_eff_self_edges=1,
        n_prime_implicants=1,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[
            EvaluationItemScore(
                id=1,
                weight=1.0,
                agreement=1.0,
                score=1.0,
                kind="NAV",
                hierarchy=0,
                fixes=(("S", 1),),
                observed="A",
                outcome="ON",
                predict_value=1.0,
            )
        ],
    )


def _phenotype_result() -> EvaluationResult:
    return EvaluationResult(
        model_id=5,
        max_score=2.0,
        score=2.0,
        n_eff_edges=1,
        n_eff_self_edges=1,
        n_prime_implicants=1,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[
            EvaluationItemScore(
                id=1,
                weight=2.0,
                agreement=1.0,
                score=2.0,
                kind="Phenotype",
                perturbation=(("A", 1),),
                sources=(("S", 1),),
                phenotype=(("P", 1),),
                expected_exists=True,
                predicted_exists=True,
                maximal_trap_spaces=[],
                metadata={},
            )
        ],
    )


def _generic_result() -> EvaluationResult:
    return EvaluationResult(
        model_id=7,
        max_score=2.0,
        score=2.0,
        n_eff_edges=1,
        n_eff_self_edges=1,
        n_prime_implicants=1,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[
            EvaluationItemScore(
                id=1,
                weight=2.0,
                agreement=1.0,
                score=2.0,
            )
        ],
    )


def test_export_phenotype_report(tmp_path):
    result = _phenotype_result()
    assert result.kind == "Phenotype"
    file_path = tmp_path / "results.csv"

    result_path = export_phenotype_report(result, file_path)

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
    assert "perturbation" in rows[0]
    assert "sources" in rows[0]
    assert "phenotype" in rows[0]


def test_export_phenotype_report_requires_phenotype_fields(tmp_path):
    result = _generic_result()

    with pytest.raises(ValueError, match="phenotype item scores"):
        export_phenotype_report(result, tmp_path / "bad.csv")


def test_export_nav_report(tmp_path):
    result = EvaluationResult(
        model_id=0,
        max_score=1.0,
        score=1.0,
        n_eff_edges=0,
        n_eff_self_edges=0,
        n_prime_implicants=0,
        n_extra_edges=0,
        satisfies_constraints=True,
        item_scores=[
            EvaluationItemScore(
                id=1,
                weight=1.0,
                agreement=1.0,
                score=1.0,
                kind="NAV",
                hierarchy=0,
                fixes=(("S", 1),),
                observed="A",
                outcome="ON",
                predict_value=1.0,
            )
        ],
    )

    assert result.kind == "NAV"
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
    assert rows[0]["experimental_outcome"] == "ON"
    assert rows[0]["outcome_range"] == "1.0,1.0"
    assert rows[0]["score"] == "1.0"
    assert list(rows[0].keys()) == [
        "id",
        "hierarchy",
        "fixes",
        "observed_node",
        "experimental_outcome",
        "outcome_range",
        "predict_value",
        "attractor_agreement",
        "score",
    ]


def test_export_nav_report_requires_nav_fields(tmp_path):
    result = _generic_result()

    with pytest.raises(ValueError, match="NAV item scores"):
        export_nav_report(result, tmp_path / "bad.csv")


def test_export_candidate_nav(tmp_path):
    candidate = Candidate(
        model=_tiny_model(),
        id=3,
        generation=2,
        eval_result=_nav_result(),
    )

    export_prefix = str(tmp_path / "run")
    stem = export_candidate(candidate, export_prefix)
    assert stem == (tmp_path / "run_3_gen2").resolve()

    bnet_path = tmp_path / "run_3_gen2.bnet"
    pkl_path = tmp_path / "run_3_gen2.pkl"
    score_path = tmp_path / "run_3_gen2_score.csv"

    assert bnet_path.exists()
    assert pkl_path.exists()
    assert score_path.exists()

    bnet_text = bnet_path.read_text()
    assert "# id: 3" in bnet_text
    assert "# generation: 2" in bnet_text
    assert "# score: 1.0 / 1.0 (100.0%)" in bnet_text
    assert "# n_eff_edges: 1" in bnet_text
    assert "# n_eff_self_edges: 1" in bnet_text
    assert "# model_id:" not in bnet_text
    assert "targets,\tfactors" in bnet_text

    with open(score_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["fixes"] == "S=1"
    assert rows[0]["observed_node"] == "A"
    assert rows[0]["score"] == "1.0"


def test_export_candidate_phenotype(tmp_path):
    candidate = Candidate(
        model=_tiny_model(),
        id=5,
        generation=1,
        eval_result=_phenotype_result(),
    )

    export_candidate(candidate, str(tmp_path / "pheno"))
    score_path = tmp_path / "pheno_5_gen1_score.csv"
    assert score_path.exists()

    with open(score_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["id"] == "1"
    assert rows[0]["weight"] == "2.0"
    assert rows[0]["score"] == "2.0"
    assert rows[0]["expected_exists"] == "True"
    assert rows[0]["predicted_exists"] == "True"
    assert "perturbation" in rows[0]
    assert "maximal_trap_spaces" in rows[0]


def test_export_candidate_generic_scores(tmp_path):
    result = _generic_result()
    assert result.kind == "Mixed"
    candidate = Candidate(
        model=_tiny_model(),
        id=7,
        generation=0,
        eval_result=result,
    )

    export_candidate(candidate, str(tmp_path / "generic"))
    score_path = tmp_path / "generic_7_gen0_score.csv"
    assert score_path.exists()

    with open(score_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert list(rows[0].keys()) == ["id", "weight", "agreement", "score"]
    assert rows[0]["id"] == "1"
    assert rows[0]["weight"] == "2.0"
    assert rows[0]["score"] == "2.0"


def test_export_candidate_requires_eval_result(tmp_path):
    candidate = Candidate(model=_tiny_model(), id=1, generation=0)

    with pytest.raises(ValueError, match="eval_result"):
        export_candidate(candidate, str(tmp_path / "missing"))
