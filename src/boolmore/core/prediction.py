from dataclasses import dataclass

Assignment = tuple[tuple[str, int], ...]

@dataclass(frozen=True)
class PhenotypePrediction:
    id: int

    perturbation: Assignment
    sources: Assignment
    phenotype: Assignment
    
    found_phenotypes: list[dict[str, int]]
    predicted_exists: bool