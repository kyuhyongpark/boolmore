import random

from boolmore.model import Model

def mix_models(model_id:int, model1:Model, model2:Model) -> Model:
    """
    For each node, take the rule from one of the parent model randomly.

    Parameters
    ----------
    model1, model2 - parent models                              :Model class
    
    Returns
    -------
    mixed_model - model that takes functions from the parents   :Model class
    """
    mixed_model = Model()
    mixed_model.id = model_id
    mixed_model.generation = max(model1.generation,model2.generation) + 1

    mixed_model.base = model1.base
    mixed_model.constraints = model1.constraints
    mixed_model.edge_pool = model1.edge_pool
    mixed_model.name = model1.name

    for node in model1.rr_dict:
        # get mutated_rr from rr
        rnd = random.random()
        if rnd < 0.5:
            get = model1
        else:
            get = model2
        mixed_model.primes[node] = get.primes[node]
        mixed_model.regulators_dict[node] = get.regulators_dict[node]
        mixed_model.signs_dict[node] = get.signs_dict[node]
        mixed_model.rr_dict[node] = get.rr_dict[node]
        for edge in get.extra_edges:
            if edge[1] == node:
                mixed_model.extra_edges.append(edge)

    return mixed_model