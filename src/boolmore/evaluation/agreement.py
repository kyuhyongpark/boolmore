"""Agreement between NAV experiment outcomes and model predictions."""

from __future__ import annotations

import re
from collections.abc import Iterable

from boolmore.experiment import NAVExperiment
from boolmore.inference.prediction import NAVPrediction
from boolmore.outcome import ComparativeOutcome, resolve_outcome_range
from boolmore.types import AgreeType, Assignment, OutcomeType

_AVG_SUFFIX = "_avg"


def line(x: float, start: tuple[float, float], end: tuple[float, float]) -> float:
    """Linearly interpolate ``y`` for ``x`` on the segment from ``start`` to ``end``.

    Returns ``0.0`` when ``x`` lies outside ``[start[0], end[0]]``.

    Parameters
    ----------
    x : float
        Horizontal coordinate.
    start : tuple[float, float]
        Segment start ``(x1, y1)``.
    end : tuple[float, float]
        Segment end ``(x2, y2)``.

    Returns
    -------
    float
        Interpolated ``y``, or ``0.0`` outside the segment.
    """
    if not start[0] <= x <= end[0]:
        return 0.0

    slope = (end[1] - start[1]) / (end[0] - start[0])
    return start[1] + slope * (x - start[0])


def _score_range(x: float, lo: float, hi: float) -> float:
    """Score ``x`` against plateau ``[lo, hi]`` with linear shoulders to 0."""
    parts: list[float] = []
    if lo > 0.0:
        parts.append(line(x, (0.0, 0.0), (lo, 1.0)))
    if lo < hi:
        parts.append(line(x, (lo, 1.0), (hi, 1.0)))
    if hi < 1.0:
        parts.append(line(x, (hi, 1.0), (1.0, 0.0)))
    if not parts:
        return 1.0
    return max(parts)


def _agreement_score(outcome: OutcomeType, predict_value: float | None) -> float:
    """Score agreement of a predicted average value against an observed outcome.

    Outcomes are float ranges ``(lo, hi)``, scalar floats (expanded to a
    ±0.1 band, or point ranges at ``0.0`` / ``1.0``), string labels that
    map to those ranges, or comparative outcomes (resolved to ``(1, 1)`` or
    ``(0, 0)``). Agreement is ``1.0`` inside the range and drops linearly
    to ``0`` outside (shoulders from ``0`` and/or to ``1``).

    Unknown predictions (``None`` / ``?``) score ``1.0`` for optimistic
    upper-bound scoring. Forced floats are clamped to ``[0, 1]`` (so tiny
    FP overshoots do not fall off the curves).
    """
    if predict_value is None:
        return 1.0

    x = min(1.0, max(0.0, predict_value))
    lo, hi = resolve_outcome_range(outcome)
    return _score_range(x, lo, hi)


def _comparative_predict_value(
    outcome: ComparativeOutcome,
    this_value: float | None,
    other_value: float | None,
) -> float | None:
    """Oriented or absolute difference for a comparative outcome.

    Returns ``None`` when either side is unknown (optimistic scoring).
    """
    if this_value is None or other_value is None:
        return None
    if outcome.relation == "higher":
        return this_value - other_value
    if outcome.relation == "lower":
        return other_value - this_value
    return abs(this_value - other_value)


def _related_node_names(base: str, node_names: Iterable[str]) -> list[str]:
    """Return names matching ``base``, ``base_<int>``, or ``base_<A-Z>``."""
    pattern = re.compile(rf"^(?:{re.escape(base)}|{re.escape(base)}_(?:\d+|[A-Z]))$")
    return [name for name in node_names if pattern.fullmatch(name)]


def _average_related_values(
    average_values: dict[str, float | None],
    observed_avg: str,
) -> float | None:
    """Mean of sibling node values for an ``*_avg`` observed name.

    Raises
    ------
    ValueError
        If ``observed_avg`` does not end with ``_avg``, the base is empty,
        or no related sibling nodes are present in ``average_values``.
    """
    if not observed_avg.endswith(_AVG_SUFFIX):
        raise ValueError(f"Expected observed name ending with {_AVG_SUFFIX!r}: {observed_avg!r}")
    base = observed_avg[: -len(_AVG_SUFFIX)]
    if not base:
        raise ValueError(f"Empty base for average observed node {observed_avg!r}")

    related = _related_node_names(base, average_values)
    if not related:
        raise ValueError(
            f"No related nodes found for {observed_avg!r} "
            f"(looked for {base!r}, {base}_<int>, or {base}_<A-Z>)"
        )

    floats: list[float] = []
    for name in related:
        value = average_values[name]
        if value is None:
            return None
        floats.append(value)
    return sum(floats) / len(floats)


def _predict_value_for_observed(
    average_values: dict[str, float | None],
    observed: str,
) -> float | None:
    """Look up a predicted value, resolving ``*_avg`` as the sibling mean."""
    if observed.endswith(_AVG_SUFFIX):
        return _average_related_values(average_values, observed)
    return average_values[observed]


def check_co_located_outcomes(
    existing: Iterable[tuple[int, OutcomeType]],
    exp_id: int,
    outcome_value: OutcomeType,
    observed_node: str,
    fixes: Assignment,
) -> None:
    """Reject redundant absolute or same-``other_id`` comparative entries.

    Multiple experiments may share ``(observed_node, fixes)`` when at most
    one outcome is non-comparative and every comparative outcome targets a
    distinct ``other_id``.

    Raises
    ------
    ValueError
        If ``outcome_value`` conflicts with an existing co-located outcome.
    """
    new_is_comparative = isinstance(outcome_value, ComparativeOutcome)
    for other_id, other_outcome in existing:
        other_is_comparative = isinstance(other_outcome, ComparativeOutcome)
        if not new_is_comparative and not other_is_comparative:
            raise ValueError(
                f"Duplicate experimental entry: {other_id} and {exp_id} "
                f"share observed={observed_node!r} fixes={fixes!r} "
                f"with non-comparative outcomes"
            )
        if (
            new_is_comparative
            and other_is_comparative
            and outcome_value.other_id == other_outcome.other_id
        ):
            raise ValueError(
                f"Duplicate experimental entry: {other_id} and {exp_id} "
                f"share observed={observed_node!r} fixes={fixes!r} "
                f"and comparative other_id={outcome_value.other_id}"
            )


def get_agreements(
    experiments: list[NAVExperiment],
    predictions: dict[Assignment, NAVPrediction],
) -> AgreeType:
    """Return attractor agreements for experimental outcomes vs model predictions.

    Agreements are keyed by observed node so interventions for that node
    are easy to find.

    Parameters
    ----------
    experiments : list of NAVExperiment
    predictions : dict of Assignment to NAVPrediction
        Predictions keyed by intervention assignment. Average node values
        are read from ``NAVPrediction.average_values``. Observed names
        ending in ``*_avg`` are scored as the mean of related sibling
        nodes (``base``, ``base_<int>``, or ``base_<A-Z>``), or ``None``
        if any sibling is unknown.

    Returns
    -------
    AgreeType
        Nested mapping ``observed_node -> fixes -> list of (id, max_score,
        outcome_value, predict_value, agreement)``. Multiple experiments may
        share the same ``(observed, fixes)`` when at most one outcome is
        non-comparative and comparative outcomes use distinct ``other_id``
        values. ``predict_value`` may be ``None`` (``?``) for LDOI
        upper-bound scoring; then ``agreement`` is ``1.0``. For comparative
        outcomes, ``predict_value`` is the oriented or absolute difference
        of the two experiments' absolute averages.

    Raises
    ------
    ValueError
        If an outcome label or range is invalid, a comparative reference id
        is missing, an ``*_avg`` observed name has no related siblings, or
        if two experiments share the same observed node and intervention
        assignment with conflicting outcomes (two non-comparative outcomes,
        or two comparatives with the same ``other_id``).
    """
    experiments_by_id = {exp.id: exp for exp in experiments}
    agreements: AgreeType = {}
    for exp in experiments:
        exp_id = exp.id
        max_score = exp.weight
        fixes = exp.fixes
        observed_node = exp.observed
        outcome_value = exp.outcome

        if isinstance(outcome_value, ComparativeOutcome):
            try:
                other = experiments_by_id[outcome_value.other_id]
            except KeyError as exc:
                raise ValueError(
                    f"Comparative outcome for experiment {exp_id} references "
                    f"missing id {outcome_value.other_id}"
                ) from exc
            this_value = _predict_value_for_observed(
                predictions[fixes].average_values, observed_node
            )
            other_value = _predict_value_for_observed(
                predictions[other.fixes].average_values, other.observed
            )
            predict_value = _comparative_predict_value(
                outcome_value, this_value, other_value
            )
        else:
            predict_value = _predict_value_for_observed(
                predictions[fixes].average_values, observed_node
            )

        agreement = _agreement_score(outcome_value, predict_value)

        by_fixes = agreements.setdefault(observed_node, {})
        entries = by_fixes.setdefault(fixes, [])
        check_co_located_outcomes(
            ((other_id, other_outcome) for other_id, _, other_outcome, _, _ in entries),
            exp_id,
            outcome_value,
            observed_node,
            fixes,
        )
        entries.append(
            (
                exp_id,
                max_score,
                outcome_value,
                predict_value,
                agreement,
            )
        )

    return agreements
