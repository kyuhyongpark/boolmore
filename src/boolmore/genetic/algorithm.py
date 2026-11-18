import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from joblib import Parallel, delayed
import numpy as np

from boolmore.evaluation.score import Evaluator
from boolmore.genetic.population import Candidate, sort_population
from boolmore.genetic.operators.crossover import mix_models
from boolmore.genetic.operators.mutation import mutate_model
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.selection import Selector
from boolmore.model import Model


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


@dataclass(frozen=True)
class GenerationSummary:
    """Cheap per-iteration snapshot (scalars only; no models or item_scores)."""

    iteration: int
    generated: int
    n_evaluated: int
    n_pruned: int
    score_threshold: float | None
    best_id: int
    best_generation: int
    best_score: float
    best_max_score: float
    order_by_values: dict[str, float | int]


@dataclass
class GAOutcome:
    """Final GA population plus compact iteration history."""

    final: GAState
    history: list[GenerationSummary]


@dataclass(frozen=True)
class _ReproduceJob:
    """One reproduce+evaluate unit of work (built on the main process)."""

    kind: Literal["sexual", "asexual"]
    model_id: int
    generation: int
    parent1: Model
    parent2: Model | None = None
    prob: float | None = None
    edge_prob: float | None = None
    seed: int | None = None


def _generation_summary(state: GAState, order_by: list[str]) -> GenerationSummary:
    best = state.population[0]
    result = best.eval_result
    if result is None:
        raise ValueError("best candidate has no evaluation result")
    return GenerationSummary(
        iteration=state.iteration,
        generated=state.generated,
        n_evaluated=state.n_evaluated,
        n_pruned=len(state.pruned_ids),
        score_threshold=state.score_threshold,
        best_id=best.id,
        best_generation=best.generation,
        best_score=result.score,
        best_max_score=result.max_score,
        order_by_values={attr: getattr(result, attr) for attr in order_by},
    )


def _job_seed(core: int) -> int | None:
    """Per-job seed for parallel workers; ``None`` keeps serial global RNG."""
    if core > 1:
        return random.randint(0, 2**31 - 1)
    return None


def _reproduce_and_evaluate(
    job: _ReproduceJob,
    constraints: dict,
    evaluator: Evaluator,
    score_threshold: float | None,
) -> Candidate:
    """Reproduce one offspring then evaluate it (pickle-friendly worker)."""
    if job.kind == "sexual":
        if job.parent2 is None:
            raise ValueError("sexual job requires parent2")
        model = mix_models(job.parent1, job.parent2, seed=job.seed)
    elif job.kind == "asexual":
        if job.prob is None or job.edge_prob is None:
            raise ValueError("asexual job requires prob and edge_prob")
        model = mutate_model(
            job.parent1,
            job.prob,
            job.edge_prob,
            constraints,
            seed=job.seed,
        )
    else:
        raise ValueError(f"unknown reproduce kind: {job.kind!r}")

    result = evaluator.evaluate(
        model,
        job.model_id,
        score_threshold,
        detail=False,
    )
    return Candidate(model, job.model_id, job.generation, result)


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
        """Evaluate already-built offsprings (legacy two-phase path / benches)."""
        if self.config.core > 1:
            results = Parallel(n_jobs=self.config.core)(
                delayed(evaluator.evaluate)(
                    candidate.model,
                    candidate.id,
                    score_threshold,
                    detail=False,
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
                    candidate.model,
                    candidate.id,
                    score_threshold,
                    detail=False,
                )
                candidate.eval_result = result
                new_candidates.append(candidate)
            return new_candidates

    def _run_jobs(
        self,
        jobs: list[_ReproduceJob],
        score_threshold: float | None,
    ) -> list[Candidate]:
        """Run reproduce+evaluate jobs serially or with ``batch_size=1`` Parallel."""
        if not jobs:
            return []
        constraints = self.reproducer.constraints
        evaluator = self.evaluator
        if self.config.core > 1:
            return Parallel(n_jobs=self.config.core, batch_size=1)(
                delayed(_reproduce_and_evaluate)(
                    job, constraints, evaluator, score_threshold
                )
                for job in jobs
            )
        return [
            _reproduce_and_evaluate(job, constraints, evaluator, score_threshold)
            for job in jobs
        ]

    def _sexual_jobs(self, population: list[Candidate], n: int) -> list[_ReproduceJob]:
        jobs: list[_ReproduceJob] = []
        for _ in range(n):
            parents = self.selector.select_parents(population)
            jobs.append(
                _ReproduceJob(
                    kind="sexual",
                    model_id=self.reproducer._next_id(),
                    generation=max(parents[0].generation, parents[1].generation) + 1,
                    parent1=parents[0].model,
                    parent2=parents[1].model,
                    seed=_job_seed(self.config.core),
                )
            )
        return jobs

    def _asexual_jobs(
        self,
        population: list[Candidate],
        n: int,
        prob: float,
        edge_prob: float,
    ) -> list[_ReproduceJob]:
        jobs: list[_ReproduceJob] = []
        for _ in range(n):
            parent = self.selector.select_single_parent(population)
            jobs.append(
                _ReproduceJob(
                    kind="asexual",
                    model_id=self.reproducer._next_id(),
                    generation=parent.generation + 1,
                    parent1=parent.model,
                    prob=prob,
                    edge_prob=edge_prob,
                    seed=_job_seed(self.config.core),
                )
            )
        return jobs

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
            # mix the good ones (select on main; reproduce+evaluate in parallel)
            candidates = self._run_jobs(
                self._sexual_jobs(population, self.config.mix),
                score_threshold,
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
        n_asexual = self.config.keep + self.config.per_iter - len(population)
        candidates = self._run_jobs(
            self._asexual_jobs(
                population,
                n_asexual,
                self.config.prob_list[iteration - 1],
                self.config.edge_prob,
            ),
            score_threshold,
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
        ) -> GAOutcome:
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
    GAOutcome
        Final population plus compact per-iteration history. Past
        populations are not retained in memory.
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

    state = GAState(
        iteration=0,
        population=ga.initialize_population([], start)
        )
    history: list[GenerationSummary] = []

    for _ in range(config.total_iter):
        state = ga.next_state(state)
        history.append(_generation_summary(state, config.order_by))

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

    return GAOutcome(final=state, history=history)


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
    "GAOutcome",
    "GenerationSummary",
    "GeneticAlgorithm",
    "ga_main",
]
