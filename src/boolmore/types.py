"""Shared type aliases used across boolmore modules."""

PrimeType = list[list[dict[str, int]]]
Assignment = tuple[tuple[str, int], ...]
FixesType = Assignment  # historical alias for Assignment
ExpType = tuple[int, float, Assignment, str, str]
PredictType = dict[Assignment, dict[str, float]]
AgreeType = dict[
    str, dict[Assignment, tuple[int, float, str, float | None, float]]
]
