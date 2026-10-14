"""Shared type aliases used across boolmore modules."""

from boolmore.outcome import ComparativeOutcome, OutcomeType

PrimeType = list[list[dict[str, int]]]
Assignment = tuple[tuple[str, int], ...]
FixesType = Assignment  # historical alias for Assignment
ExpType = tuple[int, float, Assignment, str, OutcomeType]
PredictType = dict[Assignment, dict[str, float]]
AgreeType = dict[
    str, dict[Assignment, tuple[int, float, OutcomeType, float | None, float]]
]

__all__ = [
    "AgreeType",
    "Assignment",
    "ComparativeOutcome",
    "ExpType",
    "FixesType",
    "OutcomeType",
    "PredictType",
    "PrimeType",
]
