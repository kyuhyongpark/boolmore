"""Model evaluation: per-experiment scores, aggregation, and reporting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from boolmore.model import Model
from boolmore.experiment import PhenotypeExperiment, NAVExperiment
from boolmore.inference.prediction import PhenotypePrediction
from boolmore.evaluation.agreement import get_agreements
from boolmore.evaluation.metrics import get_model_complexity
from boolmore.evaluation.constraint import check_model_constraints
from boolmore.types import AgreeType, Assignment


@dataclass(frozen=True)
class EvaluationItemScore:
    """Score contribution of a single experiment.

    Attributes
    ----------
    id : int
        Experiment identifier.
    weight : float
        Experiment weight (maximum score if fully agreed).
    agreement : float
        Agreement between prediction and observation in ``[0, 1]``.
    score : float
        Contribution to the model score (may be hierarchy-adjusted for NAV).
    hierarchy : int or None
        Number of non-default interventions (NAV only).
    fixes : Assignment or None
        Intervention assignment (NAV only).
    observed : str or None
        Observed node name (NAV only).
    outcome : str or None
        Experimental outcome label (NAV only).
    predict_value : float or None
        Predicted average value of the observed node (NAV only).
    """

    id: int
    weight: float
    agreement: float
    score: float
    hierarchy: int | None = None
    fixes: Assignment | None = None
    observed: str | None = None
    outcome: str | None = None
    predict_value: float | None = None


@dataclass
class EvaluationResult:
    """Aggregated evaluation of one model against a set of experiments.

    Attributes
    ----------
    model_id : int
        Identifier of the evaluated model (e.g. candidate id).
    max_score : float
        Sum of experiment weights.
    score : float
        Sum of per-experiment scores.
    n_edges : int
        Number of effective edges.
    n_self_edges : int
        Number of effective self-edges.
    n_prime_implicants : int
        Total number of prime-implicant literals.
    n_extra_edges : int
        Number of edges beyond the base model.
    satisfies_constraints : bool
        Whether the model satisfies all configured constraints.
    item_scores : list of EvaluationItemScore
        Per-experiment score details.
    """

    model_id: int
    max_score: float
    score: float
    n_edges: int
    n_self_edges: int
    n_prime_implicants: int
    n_extra_edges: int
    satisfies_constraints: bool
    item_scores: list[EvaluationItemScore]


def get_model_score(
    item_scores: list[EvaluationItemScore],
) -> tuple[float, float]:
    """Aggregate per-experiment scores into ``(max_score, score)``.

    Parameters
    ----------
    item_scores : list of EvaluationItemScore

    Returns
    -------
    max_score : float
        Sum of experiment weights.
    score : float
        Sum of per-experiment scores.
    """
    max_score = sum(item.weight for item in item_scores)
    total_score = sum(item.score for item in item_scores)
    return max_score, total_score


def intervention_hierarchy(
    fixes: Assignment,
    default_sources: dict[str, int],
) -> int:
    """Count interventions that differ from default source settings.

    Parameters
    ----------
    fixes : Assignment
        Intervention assignment ``((node, value), ...)``.
    default_sources : dict of str to int
        Default values for source nodes.

    Returns
    -------
    int
        Number of nodes intervened away from their default (or any
        intervention on a non-default source).
    """
    hierarchy = 0
    for node, value in fixes:
        if node in default_sources and value == default_sources[node]:
            continue
        hierarchy += 1
    return hierarchy


def _is_intervention_subset(
    candidate: dict[str, int],
    reference: dict[str, int],
    default_sources: dict[str, int],
) -> bool:
    """Return True if ``candidate`` is a hierarchical subset of ``reference``.

    A candidate intervention is a subset when every non-default (or
    non-source) setting also appears with the same value in ``reference``.
    Default source settings are ignored.
    """
    for node, value in candidate.items():
        if node in default_sources:
            if value == default_sources[node]:
                continue
            if reference.get(node) != value:
                return False
        elif reference.get(node) != value:
            return False
    return True


def get_flat_scores(
    agreements: AgreeType,
    default_sources: dict[str, int] | None = None,
) -> list[EvaluationItemScore]:
    """Compute per-experiment scores without hierarchy adjustment.

    Each score is ``weight * agreement``. Hierarchy depth is still recorded
    on each item for reporting.

    Parameters
    ----------
    agreements : AgreeType
        Nested map ``observed_node -> fixes -> (id, weight, outcome,
        predict_value, agreement)``.
    default_sources : dict of str to int, optional
        Default source settings used only to compute hierarchy depth.

    Returns
    -------
    list of EvaluationItemScore
    """
    if default_sources is None:
        default_sources = {}

    results: list[EvaluationItemScore] = []

    for observed_node, by_fixes in agreements.items():
        for fixes, data in by_fixes.items():
            exp_id, weight, outcome, predict_value, agreement = data
            results.append(
                EvaluationItemScore(
                    id=exp_id,
                    weight=weight,
                    agreement=agreement,
                    score=weight * agreement,
                    hierarchy=intervention_hierarchy(fixes, default_sources),
                    fixes=fixes,
                    observed=observed_node,
                    outcome=outcome,
                    predict_value=predict_value,
                )
            )

    return results


def get_hierarchy_scores(
    agreements: AgreeType,
    default_sources: dict[str, int],
) -> list[EvaluationItemScore]:
    """Compute per-experiment scores with intervention hierarchy.

    For each experiment, the score is the experiment weight multiplied by
    the agreement of every hierarchical subset intervention (including
    itself). Models must agree with more general (fewer) interventions
    before receiving credit for more specific ones.

    Parameters
    ----------
    agreements : AgreeType
        Nested map ``observed_node -> fixes -> (id, weight, outcome,
        predict_value, agreement)``.
    default_sources : dict of str to int
        Default source settings defining the top of the hierarchy. These
        sources should have a defined value in every experiment.

    Returns
    -------
    list of EvaluationItemScore
    """
    results: list[EvaluationItemScore] = []

    for observed_node, by_fixes in agreements.items():
        for fixes, data in by_fixes.items():
            fixes_dict = dict(fixes)
            subset_fixes = [
                other
                for other in by_fixes
                if _is_intervention_subset(dict(other), fixes_dict, default_sources)
            ]

            exp_id, weight, outcome, predict_value, agreement = data
            score = weight
            for other in subset_fixes:
                *_, other_agreement = by_fixes[other]
                score *= other_agreement

            results.append(
                EvaluationItemScore(
                    id=exp_id,
                    weight=weight,
                    agreement=agreement,
                    score=score,
                    hierarchy=intervention_hierarchy(fixes, default_sources),
                    fixes=fixes,
                    observed=observed_node,
                    outcome=outcome,
                    predict_value=predict_value,
                )
            )

    return results


def get_NAV_scores(
    experiments: list[NAVExperiment],
    predictions,
    default_sources: dict[str, int] | None = None,
    hierarchy: bool = True,
) -> list[EvaluationItemScore]:
    """Score NAV experiments given model predictions.

    Parameters
    ----------
    experiments : list of NAVExperiment
    predictions
        Average attractor values keyed by intervention assignment.
    default_sources : dict of str to int, optional
        Default source settings for hierarchy scoring.
    hierarchy : bool, default True
        If True, apply hierarchy-adjusted scoring; otherwise use flat scores.

    Returns
    -------
    list of EvaluationItemScore
    """
    if default_sources is None:
        default_sources = {}

    agreements = get_agreements(experiments, predictions)

    if hierarchy:
        return get_hierarchy_scores(agreements, default_sources)
    return get_flat_scores(agreements, default_sources)


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
            )
        )

    if missing:
        raise ValueError(
            f"No Experiment found for Prediction id(s): {missing}"
        )

    return scores


class Evaluator:
    """Evaluate models against experiments using prediction and scoring functions.

    Parameters
    ----------
    base : Model
        Base / reference model (also exposed for GA setup and comparison).
    experiments : list
        Experiments to evaluate against (NAV or phenotype).
    prediction_fn : callable
        ``prediction_fn(primes, experiments) -> predictions``.
    score_fn : callable
        ``score_fn(experiments, predictions) -> list[EvaluationItemScore]``.
    constraints : dict
        Constraint specification passed to :func:`check_model_constraints`.
    run_type : str
        Experiment modality, e.g. ``"NAV"`` or ``"Phenotype"``.
    hierarchy : bool, default True
        Whether NAV scoring applies intervention hierarchy.
    default_sources : dict of str to int, optional
        Default source-node settings defining the top of the hierarchy.
    generate_defaults : bool, default False
        Whether default sources were (or should be) inferred from the base model.
    """

    def __init__(
        self,
        *,
        base: Model,
        experiments,
        prediction_fn: Callable[..., Any],
        score_fn: Callable[..., list[EvaluationItemScore]],
        constraints: dict,
        run_type: str,
        hierarchy: bool = True,
        default_sources: dict[str, int] | None = None,
        generate_defaults: bool = False,
    ):
        self.base = base
        self.experiments = experiments
        self.prediction_fn = prediction_fn
        self.score_fn = score_fn
        self.constraints = constraints
        self.run_type = run_type
        self.hierarchy = hierarchy
        self.default_sources = dict(default_sources) if default_sources is not None else {}
        self.generate_defaults = generate_defaults

    def evaluate(self, model: Model, model_id: int = -1) -> EvaluationResult:
        """Score ``model`` and return an aggregated :class:`EvaluationResult`.

        Parameters
        ----------
        model : Model
            Model to evaluate.
        model_id : int, default -1
            Identifier stored on the result (e.g. GA candidate id).

        Returns
        -------
        EvaluationResult
        """
        predictions = self.prediction_fn(model.primes, self.experiments)
        item_scores = self.score_fn(self.experiments, predictions)
        max_score, score = get_model_score(item_scores)
        complexity = get_model_complexity(model)
        satisfies_constraints = check_model_constraints(model, self.constraints)

        if not satisfies_constraints:
            print("ERROR: model does not follow constraints")

        return EvaluationResult(
            model_id=model_id,
            max_score=max_score,
            score=score,
            n_edges=complexity["n_eff_edges"],
            n_self_edges=complexity["n_eff_self_edges"],
            n_prime_implicants=complexity["n_prime_implicants"],
            n_extra_edges=complexity["n_extra_edges"],
            satisfies_constraints=satisfies_constraints,
            item_scores=item_scores,
        )


def format_score(result: EvaluationResult) -> str:
    """Format a result score like GA console output.

    Parameters
    ----------
    result : EvaluationResult

    Returns
    -------
    str
        e.g. ``"score: 12.5 / 20 (62.5%)"``.
    """
    pct = result.score / result.max_score * 100 if result.max_score else 0.0
    return (
        f"score: {round(result.score, 2)} / {result.max_score} "
        f"({round(pct, 1)}%)"
    )


def write_nav_report(result: EvaluationResult, path: str | Path) -> Path:
    """Write a per-experiment NAV score CSV from an :class:`EvaluationResult`.

    Parameters
    ----------
    result : EvaluationResult
        Must contain NAV ``item_scores`` (with ``fixes`` set).
    path : str or Path
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If any item score is missing NAV fields.
    """
    rows = []
    for item in result.item_scores:
        if item.fixes is None:
            raise ValueError("write_nav_report requires NAV item scores")
        rows.append({
            "id": item.id,
            "hierarchy": item.hierarchy,
            "fixes": ",".join(f"{n}={v}" for n, v in item.fixes),
            "observed_node": item.observed,
            "experimental_outcome": item.outcome,
            "predict_value": item.predict_value,
            "attractor_agreement": item.agreement,
            "score": item.score,
        })
    output = Path(path)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.resolve()
