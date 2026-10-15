"""NAV outcome labels, floats, and ranges used for agreement scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_OUTCOME_LABEL_RANGES: dict[str, tuple[float, float]] = {
    "ON": (1.0, 1.0),
    "Some/ON": (0.5, 1.0),
    "Some": (0.25, 0.75),
    "OFF/Some": (0.0, 0.5),
    "OFF": (0.0, 0.0),
}
_DEFAULT_FLOAT_HALF_WIDTH = 0.1
_COMPARATIVE_THAN_RE = re.compile(r"^(higher|lower) than (\d+)$")
_COMPARATIVE_SAME_RE = re.compile(r"^same as (\d+)$")

ComparativeRelation = Literal["higher", "lower", "same"]


@dataclass(frozen=True)
class ComparativeOutcome:
    """Outcome that compares this experiment's prediction to another by id."""

    relation: ComparativeRelation
    other_id: int

    def __str__(self) -> str:
        if self.relation == "same":
            return f"same as {self.other_id}"
        return f"{self.relation} than {self.other_id}"


OutcomeType = str | float | tuple[float, float] | ComparativeOutcome


def format_outcome(outcome: OutcomeType) -> str:
    """Serialize an outcome for display or CSV export."""
    if isinstance(outcome, ComparativeOutcome):
        return str(outcome)
    if isinstance(outcome, tuple):
        return format_outcome_range(float(outcome[0]), float(outcome[1]))
    return str(outcome)


def format_outcome_range(lo: float, hi: float) -> str:
    """Serialize a resolved outcome range as ``lo,hi``."""
    return f"{lo},{hi}"


def resolve_outcome_range(outcome: OutcomeType) -> tuple[float, float]:
    """Resolve an outcome label, scalar float, float range, or comparative
    outcome to ``(lo, hi)`` in ``[0, 1]``.

    Scalar floats in ``(0, 1)`` expand to a band of half-width
    ``_DEFAULT_FLOAT_HALF_WIDTH`` clamped to ``[0, 1]``. Exact ``0.0`` and
    ``1.0`` stay point ranges. Comparative ``higher`` / ``lower`` map to
    ``(0.5, 1.0)`` so a difference of ``0.5`` still scores full agreement;
    ``same`` maps to ``(0.0, 0.0)``.
    """
    if isinstance(outcome, ComparativeOutcome):
        if outcome.relation == "same":
            return 0.0, 0.0
        return 0.5, 1.0

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
            return _OUTCOME_LABEL_RANGES[outcome]
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


def parse_outcome(text: str) -> OutcomeType:
    """Parse an outcome string into an ``OutcomeType``.

    Accepts known labels, ``lo,hi`` range strings, a scalar float string,
    or comparative phrases ``higher than ID``, ``lower than ID``,
    ``same as ID``.
    """
    value = text.strip()
    if not value:
        raise ValueError("Empty outcome cell")

    if value in _OUTCOME_LABEL_RANGES:
        return value

    than_match = _COMPARATIVE_THAN_RE.fullmatch(value)
    if than_match is not None:
        relation = than_match.group(1)
        assert relation in ("higher", "lower")
        return ComparativeOutcome(relation=relation, other_id=int(than_match.group(2)))

    same_match = _COMPARATIVE_SAME_RE.fullmatch(value)
    if same_match is not None:
        return ComparativeOutcome(relation="same", other_id=int(same_match.group(1)))

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
