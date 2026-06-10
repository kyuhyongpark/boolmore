import numpy as np

from boolmore.core.model import Model

def sort_population(population: list[Model]):
    population = sorted(population, key=lambda x: (len(x.extra_edges), x.complexity))
    population = sorted(population, key=lambda x: x.score, reverse=True)
    return population

def reproduction_bias(population: list[Model]):
    weights = list(range(1, len(population)+1))
    weights.reverse()
    p = np.array(weights)/np.sum(np.array(weights))
    return p