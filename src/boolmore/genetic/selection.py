import random
from itertools import count

import numpy as np

from boolmore.model import Model
from boolmore.genetic.population import Candidate, sort_population
from boolmore.genetic.generation.crossover import mix_models


class Selector:
    def __init__(self, keep: int, order_by: list[str]):
        self.keep = keep
        self.order_by = order_by
    def select_survivors(self, population:list[Candidate]):
        population = sort_population(population, self.order_by)
        return population[:self.keep]

def reproduction_bias(population: list[Candidate]):
    weights = list(range(1, len(population)+1))
    weights.reverse()
    p = np.array(weights)/np.sum(np.array(weights))
    return p

class Reproducer:
    def __init__(self, per_iter: int, keep: int, mix: int, order_by: list[str]):
        self.per_iter = per_iter
        self.keep = keep
        self.mix = mix
        self.order_by = order_by
        self._id_gen = count(start=1)

    def get_next_id(self):
        return next(self._id_gen)

    def asexual(self, population: list[Candidate], prob, edge_prob)->list[Model]:
        population = sort_population(population, self.order_by)
        p = reproduction_bias(population)
        # number of offsprings to generate
        # total population should be keep + per_iter
        n = self.keep + self.per_iter - len(population)
        # generate (per_iter) new models
        offsprings = []
        targets = random.choices(population, weights=p, k=n)
        for target in targets:
            new_model = target.model.mutate(self.get_next_id(), prob, edge_prob)
            offsprings.append(new_model)    
        return offsprings

    def sexual(self, population: list[Candidate])->list[Model]:
        population = sort_population(population, self.order_by)
        p = reproduction_bias(population)
        parents_lst = []
        for j in range(self.mix):
            model_choice = np.random.choice(population, size = 2, replace = False, p=p)
            parents_lst.append(model_choice)
        mixed_offsprings = []
        for parents in parents_lst:
            mixed_model = mix_models(self.get_next_id(), parents[0].model, parents[1].model)
            mixed_offsprings.append(mixed_model)
        return mixed_offsprings


