import random
from collections.abc import Callable
from dataclasses import dataclass, field

from joblib import Parallel, delayed
import numpy as np

from boolmore.evaluation.score import Evaluator
from boolmore.genetic.population import Candidate, sort_population
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.selection import Selector


@dataclass
class GAConfig:
    """
    stop_if_max : bool
        if True, stop if the maximum score is reached

    starting_gen : int
        generation number assigned to the starting model

    total_iter : int
        total number of iterations
    per_iter : int
        new models generated per iteration
    keep : int
        models to carry over tot he next iteration
    mix : int
        number of models to mix from the keep

    prob : float | dict[int, float]
        probability for each digit in the rule representation to mutate.
        if given a dict, each value is used as probability starting from each key iteration.
    edge_prob : float
        probability to add/delete extra edge, default 0.5
    
    order_by : list[str]
        list of attributes to sort by

    seed : int | None
        seed for the random number generator
    core : int
        number of cores to use
    """
    stop_if_max: bool

    starting_gen: int

    total_iter: int
    per_iter: int
    keep: int
    mix: int

    prob: float | dict[int, float]
    edge_prob: float

    order_by: list[str]

    seed: int | None
    core: int

    def __post_init__(self):
        if type(self.prob) == float:
            prob_list = [self.prob] * self.total_iter
        
        # if prob is a dictionary, make a list of probabilities
        # with the same length as total_iter
        elif type(self.prob) == dict:
            # ensure that 1 is in the key of the dictionary
            # change every key to int
            self.prob = {int(k): v for k, v in self.prob.items()}
            assert 1 in self.prob, "1 must be in the keys of the dictionary"

            prob_list = []
            for i in range(1, self.total_iter+1):
                if i in self.prob:
                    prob_list.append(self.prob[i])
                else:
                    prob_list.append(prob_list[-1])
        
        self.prob_list = prob_list

@dataclass
class GAState:
    iteration: int
    population: list[Candidate]
    generated: int = 0
    score_threshold: float | None = None
    pruned_ids: list[int] = field(default_factory=list)
    n_evaluated: int = 0

class GeneticAlgorithm:
    def __init__(
            self,
            config:GAConfig,
            evaluator:Evaluator,
            selector:Selector,
            reproducer:Reproducer,
            ):
        self.config = config
        self.evaluator = evaluator
        self.selector = selector
        self.reproducer = reproducer
    
    def initialize_population(self, population: list[Candidate], start:Candidate):
        # this ensures that models worse than the start are not carried on.
        # also ensures that same number of models are generated in the first iteration as in the other iterations.
        for i in range(self.config.keep):
            population.append(start)
        return population

    def new_candidates(
        self,
        offsprings: list[Candidate],
        evaluator: Evaluator,
        score_threshold: float | None = None,
    ) -> list[Candidate]:
        if self.config.core > 1:
            results = Parallel(n_jobs=self.config.core)(
                delayed(evaluator.evaluate)(
                    candidate.model, candidate.id, score_threshold
                )
                for candidate in offsprings
            )
            new_candidates = []
            for result in results:
                for candidate in offsprings:
                    if candidate.id == result.model_id:
                        candidate.eval_result = result
                        new_candidates.append(candidate)
            return new_candidates
        else:
            # single core
            new_candidates = []
            for candidate in offsprings:
                result = evaluator.evaluate(
                    candidate.model, candidate.id, score_threshold
                )
                candidate.eval_result = result
                new_candidates.append(candidate)
            return new_candidates

    def next_state(self, state: GAState) -> GAState:
        iteration = state.iteration + 1
        population = self.selector.select_survivors(state.population)
        generated = state.generated
        pruned_ids: list[int] = []
        n_evaluated = 0

        score_threshold = None
        if (
            self.evaluator.ldoi_prune
            and self.evaluator.nav_method == "min_traps"
            and population
        ):
            survivor_scores = [
                c.eval_result.score
                for c in population
                if c.eval_result is not None
            ]
            if survivor_scores:
                score_threshold = min(survivor_scores)

        if iteration > 1:
            # mix the good ones
            mixed_offsprings = self.reproducer.sexual(population, self.config.mix)
            candidates = self.new_candidates(
                mixed_offsprings, self.evaluator, score_threshold
            )
            population.extend(candidates)
            generated += len(candidates)
            n_evaluated += len(candidates)
            pruned_ids.extend(
                c.id
                for c in candidates
                if c.eval_result is not None and c.eval_result.is_upper_bound
            )

        # total population should be keep + per_iter
        offsprings = self.reproducer.asexual(
            population=population,
            prob=self.config.prob_list[iteration-1],
            edge_prob=self.config.edge_prob,
            n = self.config.keep + self.config.per_iter - len(population))
        candidates = self.new_candidates(
            offsprings, self.evaluator, score_threshold
        )
        population.extend(candidates)
        generated += len(candidates)
        n_evaluated += len(candidates)
        pruned_ids.extend(
            c.id
            for c in candidates
            if c.eval_result is not None and c.eval_result.is_upper_bound
        )

        # Rank the population
        population = sort_population(population, self.config.order_by)

        return GAState(
            iteration=iteration,
            population=population,
            generated=generated,
            score_threshold=score_threshold,
            pruned_ids=pruned_ids,
            n_evaluated=n_evaluated,
            )


def ga_main(
        start:Candidate,
        evaluator:Evaluator,
        selector:Selector,
        reproducer:Reproducer,
        config:GAConfig,
        on_generation: Callable[[GAState], None] | None = None,
        ) -> list[GAState]:
    """
    Main part of the genetic algorithm.

    Parameters
    ----------
    start : Candidate
        the starting model and its evaluation
    evaluator : Evaluator
        evaluator to get predictions and scores
    selector : Selector
        selector to select survivors
    reproducer : Reproducer
        reproducer to generate offsprings
    config : GAConfig
        configuration
    on_generation : callable, optional
        called after each iteration with the new ``GAState`` (e.g. for logging)

    Returns
    -------
    list[GAState]
    """
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    ga = GeneticAlgorithm(
        config,
        evaluator,
        selector,
        reproducer,
        )

    initial_state = GAState(
        iteration=0,
        population=ga.initialize_population([], start)
        )
    states = [initial_state]
    
    for _ in range(config.total_iter):
        state = ga.next_state(states[-1])
        states.append(state)

        print(
            format_iteration_summary(
                state,
                config.order_by,
                total_iter=config.total_iter,
                per_iter=config.per_iter,
                starting_gen=config.starting_gen,
            )
        )

        if on_generation is not None:
            on_generation(state)

        if (
            config.stop_if_max
            and state.population[0].eval_result.score == state.population[0].eval_result.max_score
            ):
            print("max score reached")
            break

    return states


def format_iteration_summary(
    state: GAState,
    order_by: list[str],
    *,
    total_iter: int,
    per_iter: int,
    starting_gen: int = 0,
) -> str:
    """One-line console summary for a finished GA iteration."""
    best = state.population[0]
    result = best.eval_result
    if result is None:
        raise ValueError("best candidate has no evaluation result")

    # ~per_iter new models each iteration; ids are 1..generated.
    max_generated = total_iter * per_iter
    # Up to +1 (asexual) on iter 1, then up to +2/iter (sexual then asexual).
    max_gen = starting_gen + max(1, 2 * total_iter - 1)

    iter_w = len(str(total_iter))
    gen_w = len(str(max_generated))
    id_w = len(str(max_generated))
    generation_w = len(str(max_gen))
    prune_w = len(str(per_iter))

    parts = [
        f"iter {state.iteration:>{iter_w}}/{total_iter}",
        f"generated {state.generated:>{gen_w}}",
    ]

    if state.score_threshold is not None or state.pruned_ids:
        if state.score_threshold is None:
            thresh = "n/a"
        else:
            thresh = f"{state.score_threshold:.2f}"
        parts.append(
            f"pruned {len(state.pruned_ids):>{prune_w}}/{state.n_evaluated:<{prune_w}} "
            f"(thresh {thresh})"
        )

    pct = round(result.score / result.max_score * 100, 1) if result.max_score else 0.0
    parts.extend([
        f"best id {best.id:>{id_w}}",
        f"gen {best.generation:>{generation_w}}",
        f"score {round(result.score, 1)}/{result.max_score:g} ({pct}%)",
    ])
    for attr in order_by:
        parts.append(f"{attr} {getattr(result, attr)}")

    return " | ".join(parts)


__all__ = [
    "GAConfig",
    "GAState",
    "GeneticAlgorithm",
    "ga_main",
]
