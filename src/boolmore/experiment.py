from __future__ import annotations

from dataclasses import dataclass, field

from boolmore.types import Assignment, OutcomeType


def merge_assignments(
    context: Assignment,
    intervention: Assignment,
) -> Assignment:
    """Return ``context`` overridden by ``intervention`` as a sorted Assignment."""
    merged = dict(context)
    merged.update(dict(intervention))
    return tuple(sorted(merged.items(), key=lambda item: item[0]))


@dataclass(frozen=True)
class PhenotypeExperiment:
    id: int

    perturbation: Assignment
    sources: Assignment
    phenotype: Assignment
    expected_exists: bool

    weight: float = 1.0

    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NAVExperiment:
    """NAV experiment with background context and an intervention deviation.

    Parameters
    ----------
    id, observed, outcome, weight :
        Experiment identity and observation.
    context :
        Background assignment (culture, cell line, …).
    intervention :
        Deviation from ``context`` (drug treatment / overrides).
    fixes :
        Legacy constructor alias. When provided, sets ``context=()`` and
        ``intervention=fixes``. Do not combine with ``context`` /
        ``intervention``.

    Notes
    -----
    Effective simulation constants are exposed as the ``fixes`` property:
    ``merge_assignments(context, intervention)``.
    """

    id: int
    observed: str
    outcome: OutcomeType
    context: Assignment = ()
    intervention: Assignment = ()
    weight: float = 1.0
    _fixes: Assignment = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        id: int,
        observed: str,
        outcome: OutcomeType,
        context: Assignment = (),
        intervention: Assignment = (),
        weight: float = 1.0,
        *,
        fixes: Assignment | None = None,
    ) -> None:
        if fixes is not None:
            if context or intervention:
                raise ValueError(
                    "Provide either fixes= or context=/intervention=, not both"
                )
            context = ()
            intervention = tuple(fixes)
        context_t = tuple(context)
        intervention_t = tuple(intervention)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "context", context_t)
        object.__setattr__(self, "intervention", intervention_t)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self, "_fixes", merge_assignments(context_t, intervention_t)
        )

    @property
    def fixes(self) -> Assignment:
        """Effective fixes: context overridden by intervention."""
        return self._fixes
