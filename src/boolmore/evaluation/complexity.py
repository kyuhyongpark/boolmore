from boolmore.model import Model

def get_model_complexity(model:Model)->dict[str, int]:
    n_edges = 0
    n_self_edges = 0
    n_prime_implicants = 0
    
    for node in model.primes:
        regulators_set = set()
        for prime_implicant in model.primes[node][1]:
            n_prime_implicants += len(prime_implicant)

            for reg in prime_implicant:
                regulators_set.add(reg)
        n_edges += len(regulators_set)
        if node in regulators_set:
            n_self_edges += 1

    return {
        "n_edges": n_edges,
        "n_self_edges": n_self_edges,
        "n_prime_implicants": n_prime_implicants
    }