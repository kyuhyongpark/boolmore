from dataclasses import dataclass

from boolmore.model import Model
from boolmore.evaluation.score import EvalResult
from boolmore.evaluation.constraint import check_model_constraints


@dataclass
class Candidate:
    model:Model
    id:int = -1
    generation:int = 0
    eval_result:EvalResult | None = None


def describe_candidate(candidate: Candidate) -> str:
    result = candidate.eval_result
    model = candidate.model

    description = (
        f"id {candidate.id}, "
        f"generation {candidate.generation}, "
        f"score {round(result.score, 1)}/{result.max_score} "
        f"({round(result.score / result.max_score * 100, 1)}%), "
        f"extra edges {model.extra_edges}, "
        f"n edges {result.n_edges}, "
        f"n self edges {result.n_self_edges}, "
        f"n prime implicants {result.n_prime_implicants}"
    )

    if not check_model_constraints(model):
        description += "\nERROR: model does not follow constraints"

    return description


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
            key=lambda x: getattr(x.eval_result, attr),
        )

    population = sorted(
        population,
        key=lambda x: x.eval_result.score,
        reverse=True,
    )

    return population