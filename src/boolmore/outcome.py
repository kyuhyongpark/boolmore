"""NAV outcome labels, floats, and ranges used for agreement scoring."""

from __future__ import annotations

from boolmore.types import OutcomeType

OUTCOME_RANGES: dict[str, tuple[float, float]] = {
    "ON": (1.0, 1.0),
    "Some/ON": (0.5, 1.0),
    "Some": (0.25, 0.75),
    "OFF/Some": (0.0, 0.5),
    "OFF": (0.0, 0.0),
}
_DEFAULT_FLOAT_HALF_WIDTH = 0.1


def format_outcome_range(lo: float, hi: float) -> str:
    """Serialize a resolved outcome range as ``lo,hi``."""
    return f"{lo},{hi}"


def resolve_outcome_range(outcome: OutcomeType) -> tuple[float, float]:
    """Resolve an outcome label, scalar float, or float range to ``(lo, hi)`` in ``[0, 1]``.

    Scalar floats in ``(0, 1)`` expand to a band of half-width
    ``_DEFAULT_FLOAT_HALF_WIDTH`` clamped to ``[0, 1]``. Exact ``0.0`` and
    ``1.0`` stay point ranges.
    """
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
            return OUTCOME_RANGES[outcome]
        except KeyError:
            raise ValueError(f"Unexpected experiment outcome: {outcome!r}") from None

    if isinstance(outcome, float):
        if not (0.0 <= outcome <= 1.0):
            raise ValueError(
                f"Outcome float must satisfy 0 <= x <= 1; got {outcome!r}"
            )
        if outcome == 0.0 or outcome == 1.0:
            return outcome, outcome
        half = _DEFAULT_FLOAT_HALF_WIDTH
        return max(0.0, outcome - half), min(1.0, outcome + half)

    raise ValueError(f"Unexpected experiment outcome: {outcome!r}")


def parse_outcome_cell(text: str) -> OutcomeType:
    """Parse a CSV/TSV outcome cell into an ``OutcomeType``.

    Accepts known labels, ``lo,hi`` range strings, or a scalar float string.
    """
    value = text.strip()
    if not value:
        raise ValueError("Empty outcome cell")

    if value in OUTCOME_RANGES:
        return value

    if "," in value:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Unexpected experiment outcome: {value!r}")
        try:
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"Unexpected experiment outcome: {value!r}") from exc
        return resolve_outcome_range((lo, hi))

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Unexpected experiment outcome: {value!r}") from exc
