from dataclasses import dataclass
from pathlib import Path

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
    id: int
    weight: float
    agreement: float
    score: float
    # NAV-only (None for phenotype)
    hierarchy: int | None = None
    fixes: Assignment | None = None
    observed: str | None = None
    outcome: str | None = None
    predict_value: float | None = None


@dataclass
class EvalResult:
    model_id: int
    max_score: float
    score: float
    n_edges: int
    n_self_edges: int
    n_prime_implicants: int
    n_extra_edges: int
    following_constraints: bool
    outcomes: list[EvaluationItemScore]


class Evaluator:
    """Full model evaluation: experiments, base model, and scoring context."""

    def __init__(
        self,
        *,
        base: Model,
        exps,
        prediction_fn,
        score_fn,
        constraints: dict,
        run_type: str,
        hierarchy: bool = True,
        default_sources: dict[str, int] | None = None,
        generate_defaults: bool = False,
    ):
        self.base = base
        self.exps = exps
        self.prediction_fn = prediction_fn
        self.score_fn = score_fn
        self.constraints = constraints
        self.run_type = run_type
        self.hierarchy = hierarchy
        self.default_sources = dict(default_sources) if default_sources is not None else {}
        self.generate_defaults = generate_defaults

    def evaluate(self, model: Model, id: int = -1) -> EvalResult:
        predictions = self.prediction_fn(model.primes, self.exps)
        score_items: list[EvaluationItemScore] = self.score_fn(self.exps, predictions)
        max_score, score = get_model_score(score_items)
        complexity = get_model_complexity(model)
        following_constraints = check_model_constraints(model, self.constraints)
        result = EvalResult(
            model_id=id,
            max_score=max_score,
            score=score,
            n_edges=complexity["n_eff_edges"],
            n_self_edges=complexity["n_eff_self_edges"],
            n_prime_implicants=complexity["n_prime_implicants"],
            n_extra_edges=complexity["n_extra_edges"],
            following_constraints=following_constraints,
            outcomes=score_items,
        )

        if not following_constraints:
            print("ERROR: model does not follow constraints")

        return result


def format_score(result: EvalResult) -> str:
    """Format score the same way as GA console output."""
    pct = result.score / result.max_score * 100 if result.max_score else 0.0
    return (
        f"score: {round(result.score, 2)} / {result.max_score} "
        f"({round(pct, 1)}%)"
    )


def write_nav_report(result: EvalResult, file: str | Path) -> Path:
    """Write a per-experiment NAV score CSV from an existing ``EvalResult``."""
    rows = []
    for item in result.outcomes:
        if item.fixes is None:
            raise ValueError("write_nav_report requires NAV outcomes")
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
    path = Path(file)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path.resolve()


def intervention_hierarchy(
    fixes: Assignment,
    default_sources: dict[str, int],
) -> int:
    """Count interventions that differ from default source settings."""
    hierarchy = 0
    for node, value in fixes:
        if node in default_sources:
            if value != default_sources[node]:
                hierarchy += 1
        else:
            hierarchy += 1
    return hierarchy


def get_non_hierarchy_scores(
    agreements: AgreeType,
    default_sources: dict[str, int] | None = None,
) -> list[EvaluationItemScore]:
    if default_sources is None:
        default_sources = {}

    ID = 0
    MAX_SCORE = 1
    OUTCOME_VALUE = 2
    PREDICT_VALUE = 3
    AGREEMENT = 4

    results: list[EvaluationItemScore] = []

    for observed_node in agreements:
        for fixes in agreements[observed_node]:
            data = agreements[observed_node][fixes]
            base_weight = data[MAX_SCORE]
            agreement = data[AGREEMENT]

            # no hierarchy / subset effects
            score = base_weight * agreement

            results.append(
                EvaluationItemScore(
                    id=data[ID],
                    weight=base_weight,
                    agreement=agreement,
                    score=score,
                    hierarchy=intervention_hierarchy(fixes, default_sources),
                    fixes=fixes,
                    observed=observed_node,
                    outcome=data[OUTCOME_VALUE],
                    predict_value=data[PREDICT_VALUE],
                )
            )

    return results


def get_hierarchy_scores(
    agreements: AgreeType,
    default_sources: dict[str, int],
) -> list[EvaluationItemScore]:
    """
    Returns per-experiment scores when given attractor agreements.

    Parameters
    ----------
    agreements : AgreeType
        collection of all data
        key : str
            observed_node
        value : dict[FixesType, tuple]
            data for the node
            key : FixesType
            value : tuple[int, float, str, float, float]
                data for given fixes - (id, max_score, outcome_value, predict_value, agreement)

    default_sources : dict[str, int]
        Shows the default settings for the source nodes, which is considered the top of the hierarchy
        These source nodes must have a defined value in every experiments.

    Returns
    -------
    list[EvaluationItemScore]
        per-experiment weights, agreements, and hierarchy-adjusted scores
    """

    ID = 0
    MAX_SCORE = 1
    OUTCOME_VALUE = 2
    PREDICT_VALUE = 3
    AGREEMENT = 4

    results: list[EvaluationItemScore] = []

    for observed_node in agreements:
        for fixes in agreements[observed_node]:
            fixes_dict = {key: value for (key, value) in fixes}

            subset_fixes_set = set()
            for other_fixes in agreements[observed_node]:
                other_fixes_dict = {key: value for (key, value) in other_fixes}

                is_subset = True
                for node in other_fixes_dict:
                    if node in default_sources:
                        if other_fixes_dict[node] == default_sources[node]:
                            continue
                        elif other_fixes_dict[node] == fixes_dict[node]:
                            continue
                        else:
                            is_subset = False
                            break
                    else:
                        if node not in fixes_dict:
                            is_subset = False
                            break
                        elif other_fixes_dict[node] == fixes_dict[node]:
                            continue
                        else:
                            is_subset = False
                            break

                if is_subset:
                    subset_fixes_set.add(other_fixes)

            data = agreements[observed_node][fixes]
            base_weight = data[MAX_SCORE]

            current_score = base_weight
            for subset_fixes in subset_fixes_set:
                current_score *= agreements[observed_node][subset_fixes][AGREEMENT]

            results.append(
                EvaluationItemScore(
                    id=data[ID],
                    weight=base_weight,
                    agreement=data[AGREEMENT],
                    score=current_score,
                    hierarchy=intervention_hierarchy(fixes, default_sources),
                    fixes=fixes,
                    observed=observed_node,
                    outcome=data[OUTCOME_VALUE],
                    predict_value=data[PREDICT_VALUE],
                )
            )

    return results


def get_NAV_scores(
    exps: list[NAVExperiment],
    predictions,
    default_sources: dict | None = None,
    hierarchy: bool = True,
) -> list[EvaluationItemScore]:
    """
    Returns per-experiment scores when given experiments and predictions.

    Can be modified to meet the desired criteria.
    """
    if default_sources is None:
        default_sources = {}

    agreements = get_agreements(exps, predictions)

    if hierarchy:
        return get_hierarchy_scores(agreements, default_sources)
    return get_non_hierarchy_scores(agreements, default_sources)


def get_phenotype_scores(
    experiments: list[PhenotypeExperiment],
    predictions: list[PhenotypePrediction],
) -> list[EvaluationItemScore]:
    """
    Update each PhenotypePrediction with its agreement and score.

    agreement is 1.0 if predicted_exists matches the corresponding
    Experiment's expected_exists, and 0.0 otherwise.

    score is weight * agreement.

    Experiments and Predictions are matched by their id.

    Returns
    -------
    max_score : float
        Sum of the experiment weights.

    score : float
        Sum of the individual prediction scores.
    TODO: return or modify input, not do both at the same time.
    """
    exp_by_id = {exp.id: exp for exp in experiments}

    scores: list[EvaluationItemScore] = []

    missing = []

    for pred in predictions:
        exp = exp_by_id.get(pred.id)
        if exp is None:
            missing.append(pred.id)
            continue

        agreement = (
            1.0 if pred.predicted_exists == exp.expected_exists else 0.0
        )

        score = exp.weight * agreement

        scores.append(
            EvaluationItemScore(
                id=pred.id,
                weight=exp.weight,
                agreement=agreement,
                score=score,
            )
        )

    if missing:
        raise ValueError(
            f"No Experiment found for Prediction id(s): {missing}"
        )

    return scores


def get_model_score(
    scores: list[EvaluationItemScore],
) -> tuple[float, float]:

    max_score = sum(s.weight for s in scores)
    total_score = sum(s.score for s in scores)

    return max_score, total_score
