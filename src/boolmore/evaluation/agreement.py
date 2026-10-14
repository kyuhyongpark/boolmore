"""Agreement between NAV experiment outcomes and model predictions."""

from __future__ import annotations

from boolmore.experiment import NAVExperiment
from boolmore.inference.prediction import NAVPrediction
from boolmore.types import AgreeType, Assignment, OutcomeType

_OUTCOME_RANGES: dict[str, tuple[float, float]] = {
    "ON": (1.0, 1.0),
    "Some/ON": (0.5, 1.0),
    "Some": (0.25, 0.75),
    "OFF/Some": (0.0, 0.5),
    "OFF": (0.0, 0.0),
}


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


def _resolve_range(outcome: OutcomeType) -> tuple[float, float]:
    """Resolve an outcome label or float range to ``(lo, hi)`` in ``[0, 1]``."""
    if isinstance(outcome, tuple):
        if len(outcome) != 2:
            raise ValueError(f"Unexpected experiment outcome: {outcome!r}")
        lo, hi = float(outcome[0]), float(outcome[1])
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"Outcome range must satisfy 0 <= lo <= hi <= 1; got {outcome!r}"
            )
        return lo, hi

    if isinstance(outcome, str):
        try:
            return _OUTCOME_RANGES[outcome]
        except KeyError:
            raise ValueError(f"Unexpected experiment outcome: {outcome!r}") from None

    raise ValueError(f"Unexpected experiment outcome: {outcome!r}")


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

    Outcomes are float ranges ``(lo, hi)``, or string labels that map to
    those ranges. Agreement is ``1.0`` inside the range and drops linearly
    to ``0`` outside (shoulders from ``0`` and/or to ``1``).

    Unknown predictions (``None`` / ``?``) score ``1.0`` for optimistic
    upper-bound scoring. Forced floats are clamped to ``[0, 1]`` (so tiny
    FP overshoots do not fall off the curves).
    """
    if predict_value is None:
        return 1.0

    x = min(1.0, max(0.0, predict_value))
    lo, hi = _resolve_range(outcome)
    return _score_range(x, lo, hi)


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
        are read from ``NAVPrediction.average_values``.

    Returns
    -------
    AgreeType
        Nested mapping ``observed_node -> fixes -> (id, max_score,
        outcome_value, predict_value, agreement)``. ``predict_value`` may
        be ``None`` (``?``) for LDOI upper-bound scoring; then
        ``agreement`` is ``1.0``.

    Raises
    ------
    ValueError
        If an outcome label or range is invalid, or if two experiments
        share the same observed node and intervention assignment.
    """
    agreements: AgreeType = {}
    for exp in experiments:
        exp_id = exp.id
        max_score = exp.weight
        fixes = exp.fixes
        observed_node = exp.observed
        outcome_value = exp.outcome

        predict_value = predictions[fixes].average_values[observed_node]
        agreement = _agreement_score(outcome_value, predict_value)

        by_fixes = agreements.setdefault(observed_node, {})
        if fixes in by_fixes:
            other_id = by_fixes[fixes][0]
            raise ValueError(
                f"Duplicate experimental entry: {other_id} and {exp_id} "
                f"share observed={observed_node!r} fixes={fixes!r}"
            )

        by_fixes[fixes] = (
            exp_id,
            max_score,
            outcome_value,
            predict_value,
            agreement,
        )

    return agreements
