import random

import numpy as np

from boolmore.genetic.population import Candidate, sort_population


class Selector:
    def __init__(self, keep: int, order_by: list[str]):
        self.keep = keep
        self.order_by = order_by

    def select_survivors(self, population:list[Candidate]):
        population = sort_population(population, self.order_by)
        return population[:self.keep]

    def select_single_parent(self, population):
        population = sort_population(population, self.order_by)
        p = parent_selection_probability(population)
        return random.choices(population, weights=p, k=1)[0]

    def select_parents(self, population):
        population = sort_population(population, self.order_by)
        p = parent_selection_probability(population)
        return np.random.choice(population, size=2, replace=False, p=p)


def parent_selection_probability(population: list[Candidate]):
    weights = list(range(1, len(population)+1))
    weights.reverse()
    p = np.array(weights)/np.sum(np.array(weights))
    return p