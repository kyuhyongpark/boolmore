from dataclasses import dataclass, fields

from boolmore.model import Model
from boolmore.evaluation.score import EvaluationResult


@dataclass
class Candidate:
    model:Model
    id:int = -1
    generation:int = 0
    eval_result:EvaluationResult | None = None

    def info(self) -> str:

        result = self.eval_result

        lines = [
            f"# id: {self.id}",
            f"# generation: {self.generation}",
        ]

        if result is not None:
            for field in fields(result):
                if field.name == "item_scores":
                    continue
                lines.append(f"# {field.name}: {getattr(result, field.name)}")

        info = "\n".join(lines)

        info += "\n" + self.model.info()

        return info

    def summary(self, order_by: list[str]) -> str:
        result = self.eval_result

        if result is None:
            raise Exception("Candidate has no evaluation result")

        items = [
            f"id {self.id}",
            f"generation {self.generation}",
            f"score {round(result.score, 1)}/{result.max_score} "
            f"({round(result.score / result.max_score * 100, 1)}%)"
        ]

        for attr in order_by:
            items.append(f"{attr} {getattr(result, attr)}")

        return ", ".join(items)


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