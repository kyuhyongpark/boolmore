"""NAV experiment scoring, including intervention hierarchy."""

from __future__ import annotations

from boolmore.experiment import NAVExperiment, merge_assignments
from boolmore.evaluation.agreement import get_agreements
from boolmore.evaluation.score import EvaluationItemScore
from boolmore.inference.prediction import NAVPrediction
from boolmore.outcome import ComparativeOutcome
from boolmore.types import AgreeType, Assignment


def intervention_hierarchy(
    intervention: Assignment,
    default_sources: dict[str, int] | None = None,
) -> int:
    """Count hierarchy depth for an intervention.

    Parameters
    ----------
    intervention : Assignment
        Intervention assignment ``((node, value), ...)``.
    default_sources : dict of str to int, optional
        When provided (legacy path with empty experiment context), settings
        that match ``default_sources`` are ignored. When omitted or empty and
        the intervention is already context-relative, depth is
        ``len(intervention)``.

    Returns
    -------
    int
        Number of nodes in the intervention that count toward hierarchy.
    """
    if not default_sources:
        return len(intervention)

    hierarchy = 0
    for node, value in intervention:
        if node in default_sources and value == default_sources[node]:
            continue
        hierarchy += 1
    return hierarchy


def _strip_default_sources(
    assignment: Assignment,
    default_sources: dict[str, int],
) -> Assignment:
    """Drop assignments that match ``default_sources``."""
    return tuple(
        (node, value)
        for node, value in assignment
        if node not in default_sources or default_sources[node] != value
    )


def _is_intervention_subset(
    candidate: dict[str, int],
    reference: dict[str, int],
    default_sources: dict[str, int] | None = None,
) -> bool:
    """Return True if ``candidate`` is a hierarchical subset of ``reference``.

    When ``default_sources`` is provided (legacy path), default source
    settings in ``candidate`` are ignored. Otherwise every candidate
    setting must appear with the same value in ``reference``.
    """
    if not default_sources:
        return all(reference.get(node) == value for node, value in candidate.items())

    for node, value in candidate.items():
        if node in default_sources:
            if value == default_sources[node]:
                continue
            if reference.get(node) != value:
                return False
        elif reference.get(node) != value:
            return False
    return True


def _effective_hierarchy_parts(
    context: Assignment,
    intervention: Assignment,
    default_sources: dict[str, int],
) -> tuple[Assignment, Assignment, bool]:
    """Return ``(context, intervention, use_legacy_defaults)`` for scoring.

    Explicit non-empty context uses the intervention as-is. Empty context
    (legacy) treats ``default_sources`` as the shared context and strips
    matching defaults from the intervention for hierarchy math.
    """
    if context:
        return context, intervention, False
    legacy_context = tuple(sorted(default_sources.items(), key=lambda item: item[0]))
    return legacy_context, _strip_default_sources(intervention, default_sources), True


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
        Nested map ``observed -> context -> intervention -> entries``.
    default_sources : dict of str to int, optional
        Used only for hierarchy depth on the legacy empty-context path.

    Returns
    -------
    list of EvaluationItemScore
    """
    if default_sources is None:
        default_sources = {}

    results: list[EvaluationItemScore] = []
    fixes_cache: dict[tuple[Assignment, Assignment], Assignment] = {}

    for observed_node, by_context in agreements.items():
        for context, by_intervention in by_context.items():
            for intervention, entries in by_intervention.items():
                _, eff_intervention, _use_legacy = _effective_hierarchy_parts(
                    context, intervention, default_sources
                )
                hierarchy = len(eff_intervention)
                fixes_key = (context, intervention)
                fixes = fixes_cache.get(fixes_key)
                if fixes is None:
                    fixes = merge_assignments(context, intervention)
                    fixes_cache[fixes_key] = fixes
                for exp_id, weight, outcome, predict_value, agreement in entries:
                    results.append(
                        EvaluationItemScore(
                            id=exp_id,
                            weight=weight,
                            agreement=agreement,
                            score=weight * agreement,
                            kind="NAV",
                            hierarchy=hierarchy,
                            fixes=fixes,
                            context=context,
                            intervention=intervention,
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

    Multiplication is scoped to experiments that share the same observed
    node and the same (effective) context. Within that group, the score of
    a definitive experiment is its weight multiplied by every definitive
    agreement whose intervention is a hierarchical subset (including
    itself). Comparative outcomes are scored flat and never multiply into
    another experiment's hierarchy product.

    For legacy experiments with empty ``context``, ``default_sources``
    defines the unique shared context (settings matching defaults are
    ignored when measuring depth and subsets).

    Parameters
    ----------
    agreements : AgreeType
        Nested map ``observed -> context -> intervention -> entries``.
    default_sources : dict of str to int
        Default source settings for the legacy empty-context path.

    Returns
    -------
    list of EvaluationItemScore
    """
    results: list[EvaluationItemScore] = []
    fixes_cache: dict[tuple[Assignment, Assignment], Assignment] = {}

    # Re-group by (observed, effective_context) so legacy empty contexts that
    # share default_sources land in one hierarchy pool.
    grouped: dict[
        tuple[str, Assignment],
        dict[
            Assignment,
            list[tuple[Assignment, Assignment, list, bool]],
        ],
    ] = {}

    for observed_node, by_context in agreements.items():
        for context, by_intervention in by_context.items():
            for intervention, entries in by_intervention.items():
                eff_context, eff_intervention, use_legacy = _effective_hierarchy_parts(
                    context, intervention, default_sources
                )
                pool = grouped.setdefault((observed_node, eff_context), {})
                pool.setdefault(eff_intervention, []).append(
                    (context, intervention, entries, use_legacy)
                )

    for (observed_node, _eff_context), by_eff_intervention in grouped.items():
        as_dicts = {eff: dict(eff) for eff in by_eff_intervention}
        abs_agreements: dict[Assignment, list[float]] = {
            eff: [
                other_agreement
                for _oc, _oi, other_entries, _oul in bundles
                for _id, _w, other_outcome, _pv, other_agreement in other_entries
                if not isinstance(other_outcome, ComparativeOutcome)
            ]
            for eff, bundles in by_eff_intervention.items()
        }
        products: dict[Assignment, float] = {}
        for eff, ref_dict in as_dicts.items():
            product = 1.0
            for other, other_dict in as_dicts.items():
                if _is_intervention_subset(other_dict, ref_dict):
                    for other_agreement in abs_agreements[other]:
                        product *= other_agreement
            products[eff] = product

        for eff_intervention, bundles in by_eff_intervention.items():
            hierarchy = len(eff_intervention)
            product = products[eff_intervention]
            for context, intervention, entries, _use_legacy in bundles:
                fixes_key = (context, intervention)
                fixes = fixes_cache.get(fixes_key)
                if fixes is None:
                    fixes = merge_assignments(context, intervention)
                    fixes_cache[fixes_key] = fixes
                for exp_id, weight, outcome, predict_value, agreement in entries:
                    if isinstance(outcome, ComparativeOutcome):
                        score = weight * agreement
                    else:
                        score = weight * product

                    results.append(
                        EvaluationItemScore(
                            id=exp_id,
                            weight=weight,
                            agreement=agreement,
                            score=score,
                            kind="NAV",
                            hierarchy=hierarchy,
                            fixes=fixes,
                            context=context,
                            intervention=intervention,
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
    related_nodes: dict[str, list[str]] | None = None,
    *,
    validate_co_located: bool = True,
) -> list[EvaluationItemScore]:
    """Score NAV experiments given model predictions.

    Parameters
    ----------
    experiments : list of NAVExperiment
    predictions : dict of Assignment to NAVPrediction
        Predictions keyed by effective ``fixes`` assignment.
    default_sources : dict of str to int, optional
        Default source settings for legacy hierarchy scoring when experiment
        ``context`` is empty.
    hierarchy : bool, default True
        If True, apply hierarchy-adjusted scoring; otherwise use flat scores.
    related_nodes : dict of str to list of str, optional
        Precomputed observed-base → sibling node names shared across
        candidates (see :func:`~boolmore.evaluation.agreement.build_related_nodes`).
    validate_co_located : bool, default True
        Forwarded to :func:`~boolmore.evaluation.agreement.get_agreements`.
        GA evaluators set this False after import-time validation.

    Returns
    -------
    list of EvaluationItemScore
    """
    if default_sources is None:
        default_sources = {}

    agreements = get_agreements(
        experiments,
        predictions,
        related_nodes=related_nodes,
        validate_co_located=validate_co_located,
    )

    if hierarchy:
        return get_hierarchy_scores(agreements, default_sources)
    return get_flat_scores(agreements, default_sources)
