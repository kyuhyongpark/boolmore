from dataclasses import dataclass

from boolmore.model import Model
from boolmore.evaluation.score import EvalResult

@dataclass
class Candidate:
    model:Model
    eval_result:EvalResult

def sort_population(
    population: list[Candidate],
    order_by: list[str] | None = None,
) -> list[Candidate]:
    """
    Sort by:
        1. score (highest first)
        2. tie-breakers in order_by (lowest first)

    Example:
        sort_population(pop, ["n_edges"])
        sort_population(pop, ["n_self_edges", "n_prime_implicants"])
    """
    if order_by is None:
        order_by = []

    for attr in reversed(order_by):
        population = sorted(
            population,
            key=lambda x: getattr(x.model, attr),
        )

    population = sorted(
        population,
        key=lambda x: x.eval_result.score,
        reverse=True,
    )

    return population