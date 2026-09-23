from boolmore.model import Model

PrimeType = list[list[dict[str, int]]]

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