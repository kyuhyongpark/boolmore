"""Model complexity and function-diff helpers."""

from boolmore.model import Model
from boolmore.types import PrimeType


def get_model_complexity(model: Model) -> dict[str, int]:

    if model.edges == []:
        raise ValueError("Model has no edges")

    n_self_edges = 0
    n_prime_implicants = 0
    
    for node in model.primes:
        regulators_set = set()
        for prime_implicant in model.primes[node][1]:
            n_prime_implicants += len(prime_implicant)

            for reg in prime_implicant:
                regulators_set.add(reg)
        if node in regulators_set:
            n_self_edges += 1

    return {
        "n_eff_edges": len(model.get_edges(effective=True)),
        "n_eff_self_edges": n_self_edges,
        "n_prime_implicants": n_prime_implicants,
        "n_extra_edges": len(model.get_edges(source="edge_pool"))
    }


def compare_model_functions(model1: Model, model2: Model
    ) -> dict[str, tuple[dict[str, PrimeType], dict[str, PrimeType]]]:
    differences = {}

    for node in model1.primes:
        for value in (0, 1):
            primes1 = sorted(sorted(d.items()) for d in model1.primes[node][value])
            primes2 = sorted(sorted(d.items()) for d in model2.primes[node][value])

            if primes1 != primes2:
                differences[node] = (
                    model1.primes[node],
                    model2.primes[node],
                )
                break

    return differences
