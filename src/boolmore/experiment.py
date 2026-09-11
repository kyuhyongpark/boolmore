from dataclasses import dataclass, field


Assignment = tuple[tuple[str, int], ...]

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
    id: int

    fixes: Assignment
    observed: str
    outcome: str

    weight: float = 1.0