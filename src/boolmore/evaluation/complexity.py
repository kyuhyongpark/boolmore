from boolmore.model import Model

def get_model_complexity(model:Model)->dict[str, int]:

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
        "n_extra_edges": len(model.get_edges(source="extra_edges"))
    }