"""Phenotype experiment scoring."""

from __future__ import annotations

from boolmore.experiment import PhenotypeExperiment
from boolmore.inference.prediction import PhenotypePrediction
from boolmore.evaluation.score import EvaluationItemScore


def get_phenotype_scores(
    experiments: list[PhenotypeExperiment],
    predictions: list[PhenotypePrediction],
) -> list[EvaluationItemScore]:
    """Score phenotype experiments given model predictions.

    Experiments and predictions are matched by ``id``. Agreement is ``1.0``
    when ``predicted_exists`` matches ``expected_exists``, else ``0.0``.
    Score is ``weight * agreement``.

    Parameters
    ----------
    experiments : list of PhenotypeExperiment
    predictions : list of PhenotypePrediction

    Returns
    -------
    list of EvaluationItemScore

    Raises
    ------
    ValueError
        If a prediction has no matching experiment.
    """
    exp_by_id = {exp.id: exp for exp in experiments}
    scores: list[EvaluationItemScore] = []
    missing: list[int] = []

    for pred in predictions:
        exp = exp_by_id.get(pred.id)
        if exp is None:
            missing.append(pred.id)
            continue

        agreement = 1.0 if pred.predicted_exists == exp.expected_exists else 0.0
        scores.append(
            EvaluationItemScore(
                id=pred.id,
                weight=exp.weight,
                agreement=agreement,
                score=exp.weight * agreement,
                kind="Phenotype",
                perturbation=exp.perturbation,
                sources=exp.sources,
                phenotype=exp.phenotype,
                expected_exists=exp.expected_exists,
                predicted_exists=pred.predicted_exists,
                maximal_trap_spaces=pred.maximal_trap_spaces,
                metadata=dict(exp.metadata),
            )
        )

    if missing:
        raise ValueError(
            f"No Experiment found for Prediction id(s): {missing}"
        )

    return scores
