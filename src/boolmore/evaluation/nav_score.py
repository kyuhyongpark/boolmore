"""NAV experiment scoring, including intervention hierarchy."""

from __future__ import annotations

from boolmore.experiment import NAVExperiment
from boolmore.evaluation.agreement import get_agreements
from boolmore.evaluation.score import EvaluationItemScore
from boolmore.inference.prediction import NAVPrediction
from boolmore.types import AgreeType, Assignment


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
    predictions: dict[Assignment, NAVPrediction],
    default_sources: dict[str, int] | None = None,
    hierarchy: bool = True,
) -> list[EvaluationItemScore]:
    """Score NAV experiments given model predictions.

    Parameters
    ----------
    experiments : list of NAVExperiment
    predictions : dict of Assignment to NAVPrediction
        Predictions keyed by intervention assignment.
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
