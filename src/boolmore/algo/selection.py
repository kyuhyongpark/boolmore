import numpy as np

from boolmore.core.model import Model

def sort_population(population: list[Model], hierarchy):
    population = sorted(population, key=lambda x: (len(x.extra_edges), x.complexity))
    if hierarchy:
        population = sorted(population, key=lambda x: x.score, reverse=True)
    else:
        population = sorted(population, key=lambda x: x.non_hierarchy_score, reverse=True)
    return population

def reproduction_bias(population: list[Model]):
    weights = list(range(1, len(population)+1))
    weights.reverse()
    p = np.array(weights)/np.sum(np.array(weights))
    return p