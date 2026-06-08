from dataclasses import dataclass

Assignment = tuple[tuple[str, int], ...]

@dataclass(frozen=True)
class Experiment:
    id: int

    perturbation: Assignment
    sources: Assignment
    phenotype: Assignment
    expected_exists: bool

    weight: float = 1.0