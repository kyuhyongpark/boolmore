"""Shared type aliases used across boolmore modules."""

PrimeType = list[list[dict[str, int]]]
Assignment = tuple[tuple[str, int], ...]
FixesType = Assignment  # historical alias for Assignment
OutcomeType = str | tuple[float, float]
ExpType = tuple[int, float, Assignment, str, OutcomeType]
PredictType = dict[Assignment, dict[str, float]]
AgreeType = dict[
    str, dict[Assignment, tuple[int, float, OutcomeType, float | None, float]]
]
