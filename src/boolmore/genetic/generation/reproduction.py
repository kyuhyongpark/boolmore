from itertools import count

from boolmore.model import Model
from boolmore.genetic.population import Candidate
from boolmore.genetic.selection import Selector
from boolmore.genetic.generation.mutation import mutate_model
from boolmore.genetic.generation.crossover import mix_models


class Reproducer:
    def __init__(self, selector:Selector, constraints:dict):
        self.selector = selector
        self.constraints = constraints
        self._id_gen = count(start=1)

    def _next_id(self):
        return next(self._id_gen)

    def asexual(self, population: list[Candidate], prob, edge_prob, n)->list[Candidate]:
        offsprings = []
        for _ in range(n):
            target = self.selector.select_single_parent(population)
            new_model = mutate_model(target.model, prob, edge_prob, self.constraints)
            offsprings.append(
                Candidate(new_model, self._next_id(), target.generation + 1,)
                )    
        return offsprings

    def sexual(self, population: list[Candidate], n)->list[Candidate]:
        mixed_offsprings = []
        for _ in range(n):
            parents = self.selector.select_parents(population)
            mixed_model = mix_models(parents[0].model, parents[1].model)
            mixed_offsprings.append(
                Candidate(mixed_model,
                          self._next_id(),
                          max(parents[0].generation, parents[1].generation) + 1)
                )
        return mixed_offsprings
