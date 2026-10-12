"""Model evaluation: result types, aggregation, Evaluator, and reporting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from boolmore.model import Model
from boolmore.evaluation.metrics import get_model_complexity
from boolmore.evaluation.constraint import check_model_constraints
from boolmore.types import Assignment


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
