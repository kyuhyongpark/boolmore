import datetime
import json
import os
import random
import pickle
from dataclasses import dataclass
from functools import partial
from joblib import Parallel, delayed

import numpy as np
from pyboolnet.external.bnet2primes import bnet_file2primes
from pystablemotifs.format import primes2bnet

from boolmore.boolean_functions import prime2bnet
from boolmore.model import Model

from boolmore.evaluation.score import (
    Evaluator,
    get_NAV_scores, get_phenotype_scores
)
from boolmore.evaluation.compare import compare_model_functions
from boolmore.inference.prediction import get_NAV_prediction, get_phenotype_prediction
from boolmore.io.load import import_NAV_exps, import_phenotypes
from boolmore.io.export import export_model

from boolmore.genetic.population import Candidate, sort_population
from boolmore.genetic.generation.reproduction import Reproducer
from boolmore.genetic.selection import Selector

FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict]


@dataclass
class GAConfig:
    """
    stop_if_max : bool
        if True, stop if the maximum score is reached

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

    def new_candidates(self, offsprings:list[Candidate], evaluator:Evaluator)-> list[Candidate]:
        if self.config.core > 1:
            results = Parallel(n_jobs=self.config.core)(delayed(evaluator.evaluate)(candidate.model, candidate.id) for candidate in offsprings)
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
                result = evaluator.evaluate(candidate.model, candidate.id)
                candidate.eval_result = result
                new_candidates.append(candidate)
            return new_candidates

    def next_state(self, state: GAState) -> GAState:
        iteration = state.iteration + 1
        population = self.selector.select_survivors(state.population)
        generated = state.generated

        if iteration > 1:
            # mix the good ones
            mixed_offsprings = self.reproducer.sexual(population, self.config.mix)
            candidates = self.new_candidates(mixed_offsprings, self.evaluator)
            population.extend(candidates)
            generated += len(candidates)

        # total population should be keep + per_iter
        offsprings = self.reproducer.asexual(
            population=population,
            prob=self.config.prob_list[iteration-1],
            edge_prob=self.config.edge_prob,
            n = self.config.keep + self.config.per_iter - len(population))
        candidates = self.new_candidates(offsprings, self.evaluator)
        population.extend(candidates)
        generated += len(candidates)

        # Rank the population
        population = sort_population(population, self.config.order_by)
        best = population[0]

        return GAState(
            iteration=iteration,
            population=population,
            generated=generated
            )


def ga_main(
        start:Candidate,
        evaluator:Evaluator,
        selector:Selector,
        reproducer:Reproducer,
        config:GAConfig,
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
    core : int, optional
        number of cores to use, by default 1

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
            f"iteration {state.iteration}, "
            f"generated {state.generated}, "
            f"{state.population[0].summary(config.order_by)}"
            )
        
        if (
            config.stop_if_max
            and state.population[0].eval_result.score == state.population[0].eval_result.max_score
            ):
            print("max score reached")
            break

    return states


def log_candidate(fp, candidate: Candidate, label, path):
    fp.write(f"\n# {label}: {os.path.abspath(path)}\n")
    fp.write(candidate.info() + "\n")
    fp.write("# targets,\tfactors\n")

    bnet = candidate.model.bnet
    for line in bnet.splitlines():
        fp.write("# " + line + "\n")


def export_models(
        states: list[GAState],
        export_name: str,
        export_top: int = 0,
        export_thresh: float = 0.0,
        export_same: bool = False,
        ):
    final = states[-1].population[0]

    # Always export the final best model
    export_model(final.model, file_name=f"{export_name}_{final.id}_gen{final.generation}")

    # Export top models from each generation
    if export_top:
        for state in states[1:]:
            for candidate in state.population[:export_top]:
                if candidate.eval_result.score > export_thresh:
                    export_model(candidate.model, file_name=f"{export_name}_{candidate.id}_gen{candidate.generation}")

    # Export all models tied with the best model in the final generation
    if export_same:
        for candidate in states[-1].population:
            if candidate.eval_result.score == final.eval_result.score:
                export_model(candidate.model, filename=f"{export_name}_{candidate.id}_gen{candidate.generation}")


def run_ga(
        json_file:str|None=None,
        run_type:str|None=None,
        data_file:str|None=None,
        base_file:str|None=None,
        start_file:str|None=None,
        stop_if_max:bool|None=None,
        hierarchy:bool|None=None,
        seed:int|None=None,
        run_name:str|None=None,
        export_top:int=0,
        export_thresh:float=0.0,
        export_name:str|None=None,
        export_same:bool=False,
        core:int=1,
        )-> tuple[Candidate, Candidate, list[GAState]]:
    """
    Imports parameters, experiments, base model in the json file.
    Runs genetic algorithm and exports refined models.
    
    Instead of json file, locations of the data and base model can be used.
    In this case, constraints or extra edges are inapplicable.

    Parameters
    ----------
    json_file : str | None
        location of the json file containing parameters
    
    Optional
    --------
    run_type : str | None
        "Phenotype" or "NAV"
    data_file : str | None
        location of the data file that overrides the json file.
        Must be given if json_file is not given.
    base_file : str | None
        location of the base file that overrides the json file.
        Must be given if json_file is not given.
    start_file : str | None
        location of the bnet file of the starting model
        if None, base_model is the starting model
    stop_if_max : bool
        if True, stop when the max score is reached
    hierarchy : bool
        if True, hierarchy scoring is used

    seed : int | None
        random seed for reproducibility
    run_name : str | None
        models are exported as (run_name)_id_gen.bnet
        log is exported as (run_name)_log.txt
        if None, takes the start_model name
    export_top : int
        export top models from each generation
    export_thresh : float
        export models with scores higher than this
    export_name : str | None
        name of the exported models
    export_same : bool
        if True, export models with the same score for the final generation
    core : int
        if larger than 1, model evaluation is done in parallel

    Returns
    -------
    base : Candidate
        the base model and its evaluation
    start : Candidate
        the starting model and its evaluation
    states : list[GAState]
        the states of the genetic algorithm

    """
    # ---------- Load run configuration ----------
    if json_file is not None:
        if any(x is not None for x in [run_type, data_file, base_file, start_file, stop_if_max, hierarchy]):
            raise ValueError("If json file is given, other parameters should not be given.")

        with open(json_file) as f:
            config_dict = json.load(f)

    else:
        if any(x is None for x in [run_type, data_file, base_file]):
            raise ValueError("If json file is not given, run_type, data_file and base_file must be given.")

        config_dict = {
            "run_type": run_type,
            "data": data_file,
            "base": base_file,
            "start": start_file if start_file is not None else base_file,
            "hierarchy": hierarchy if hierarchy is not None else True,
            "default_sources": {},
            "generate_defaults": True,
            "constraints": {
                "fixed": [],
                "regulate": {},
                "necessary" : {},
                "group": {},
                "possible_constant": []
                },
            "edge_pool": [],
            "parameters" : {
                "stop_if_max": stop_if_max if stop_if_max is not None else True,
                "starting_gen": 0,
                "total_iter": 10,
                "per_iter" : 10,
                "keep" : 2,
                "mix" : 0,
                "prob" : 0.1,
                "edge_prob" : 0.5,
                "order_by" : ["n_extra_edges", "n_prime_implicants"]
                }
            }

    RUN_TYPE = config_dict["run_type"]
    DATA = config_dict["data"]
    BASE = config_dict["base"]
    START = config_dict["start"]
    HIERARCHY = config_dict["hierarchy"]
    DEFAULT_SOURCES = config_dict["default_sources"]
    GENERATE_DEFAULTS = config_dict["generate_defaults"]
    CONSTRAINTS = config_dict["constraints"]
    EDGE_POOL = config_dict["edge_pool"]
    parameters = config_dict["parameters"].copy()

    STARTING_GEN = parameters.pop("starting_gen")
    config = GAConfig(**parameters, seed=seed, core=core)

    if run_name is None:
        run_name = START.split("/")[-1][:-5]
    
    LOG = run_name + "_log.txt"

    if export_name is None:
        export_name = run_name

    # ---------- Load base and starting model ----------
    print(f"Loading base model from {os.path.abspath(BASE)}")
    if BASE.endswith(".bnet"):
        base_primes = bnet_file2primes(BASE)
    elif BASE.endswith(".pkl"):
        with open(BASE, "rb") as f:
            base_primes = pickle.load(f)
    else:
        raise ValueError(f"Unsupported base file format: {BASE}")
    base_model = Model.from_primes(base_primes, edge_pool=EDGE_POOL)
    print("Base model loaded.")

    if os.path.abspath(START) == os.path.abspath(BASE):
        start_primes = base_primes
    else:
        print(f"Loading starting model from {os.path.abspath(START)}")
        start_primes = bnet_file2primes(START)
    start_model = Model.from_primes(start_primes, base=base_model)
    print("Starting model loaded.")

    # ---------- Load experimental data ----------
    print(f"Loading experimental data from {os.path.abspath(DATA)}")
    if RUN_TYPE == "NAV":
        if GENERATE_DEFAULTS:
            for node in base_primes:
                if base_primes[node] == [[{node:0}], [{node:1}]]:
                    DEFAULT_SOURCES[node] = 0

        exps = import_NAV_exps(DATA)
        prediction_fn = get_NAV_prediction
        score_fn = partial(get_NAV_scores, default_sources=DEFAULT_SOURCES, hierarchy=HIERARCHY)
    elif RUN_TYPE == "Phenotype":
        exps = import_phenotypes(DATA)
        prediction_fn = get_phenotype_prediction
        score_fn = get_phenotype_scores
    else:
        raise ValueError(f"Unsupported run type: {RUN_TYPE}")
    evaluator = Evaluator(exps=exps, prediction_fn=prediction_fn, score_fn=score_fn, constraints=CONSTRAINTS)
    print("Experimental data loaded.\n")

    # ---------- Evaluate base and start models ----------
    start_single = datetime.datetime.now()
    base = Candidate(model=base_model, eval_result=evaluator.evaluate(base_model))    
    end_single = datetime.datetime.now()
    print(base.model.info())
    print(
        f"score: {round(base.eval_result.score,2)}"
        f" / {base.eval_result.max_score}"
        f" ({round(base.eval_result.score/base.eval_result.max_score*100,1)}%)\n")
    print(
        f"\tElapsed time for single evaluation: {end_single-start_single}\n"
        f"\tEstimated GA evaluation time: {(end_single-start_single)*config.total_iter*config.per_iter/core}\n"
        )

    start = Candidate(start_model, 0, STARTING_GEN, evaluator.evaluate(start_model, 0))
    print(start.model.info())
    print(f"score: {round(start.eval_result.score,2)} / {start.eval_result.max_score} ({round(start.eval_result.score/start.eval_result.max_score*100,1)}%)")
    print()

    # ---------- Run genetic algorithm ----------
    start_time = datetime.datetime.now()

    selector = Selector(keep=config.keep, order_by=config.order_by)
    reproducer = Reproducer(selector=selector, constraints=CONSTRAINTS)
    states = ga_main(start, evaluator, selector, reproducer, config)

    end_time = datetime.datetime.now()

    print()
    export_models(states, export_name, export_top, export_thresh, export_same)

    final = states[-1].population[0]

    # ---------- Write log ----------
    with open(LOG, "w") as fp:
        fp.write(f"# {RUN_TYPE=}\n")
        fp.write(f"# DATA: {os.path.abspath(DATA)}\n")
        fp.write(f"# {HIERARCHY=}\n")
        fp.write(f"# {DEFAULT_SOURCES=}\n")
        fp.write(f"# {GENERATE_DEFAULTS=}\n")
        fp.write(f"# {CONSTRAINTS=}\n")
        fp.write(f"# {EDGE_POOL=}\n\n")

        fp.write(f"# seed: {config.seed}\n")
        fp.write(f"# stop if max: {config.stop_if_max}\n")
        fp.write(f"# starting gen: {STARTING_GEN}\n")
        fp.write(f"# total iterations: {config.total_iter}\n")
        fp.write(f"# per iteration: {config.per_iter}\n")
        fp.write(f"# keep: {config.keep}\n")
        fp.write(f"# mix: {config.mix}\n")
        fp.write(f"# prob: {config.prob}\n")
        fp.write(f"# edge prob: {config.edge_prob}\n")
        fp.write(f"# order by: {config.order_by}\n\n")

        fp.write(f"# {core=}\n\n")

        log_candidate(fp, base, "BASE", BASE)
        log_candidate(fp, start, "START", START)

        fp.write(f"\n# {start_time=}\n")
        fp.write(f"# {end_time=}\n")
        fp.write(f"# elapsed time: {end_time-start_time}\n\n")

        for state in states[1:]:
            candidate = state.population[0]
            fp.write(candidate.summary(config.order_by) + "\n")
        log_candidate(fp, final, "FINAL", f"{export_name}_{final.id}_gen{final.generation}.bnet")

    # ---------- Analyze and report results ----------
    differences = compare_model_functions(start.model, final.model)

    print(f"""
        The algorithm ran for {len(states)-1} iterations,
        generating {states[-1].generated} models.
        Mutated {len(differences)} functions, 
        and increased score from {round(start.eval_result.score,2)} / {start.eval_result.max_score} ({round(start.eval_result.score/start.eval_result.max_score*100,1)}%)
        to {round(final.eval_result.score,2)} / {final.eval_result.max_score} ({round(final.eval_result.score/final.eval_result.max_score*100,1)}%).\n
        Total elapsed time: {end_time-start_time}""")
    print()

    print("-----modified functions-----")
    for node, (prime1, prime2) in differences.items():
        print("start:" + prime2bnet(node, prime1))
        print("final:" + prime2bnet(node, prime2))

    return base, start, states
