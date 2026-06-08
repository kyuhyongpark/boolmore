from dataclasses import dataclass

Assignment = tuple[tuple[str, int], ...]

@dataclass
class Prediction:
    id: int

    perturbation: Assignment
    sources: Assignment
    phenotype: Assignment
    
    found_phenotypes: list[dict[str, int]]
    predicted_exists: bool
    
    agreement: float
    score: float