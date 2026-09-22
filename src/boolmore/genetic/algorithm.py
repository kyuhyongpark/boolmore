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
    Evaluator, EvalResult,
    get_NAV_scores, get_phenotype_scores
)
from boolmore.inference.prediction import get_NAV_prediction, get_phenotype_prediction
from boolmore.io.load import import_NAV_exps, import_phenotypes

from boolmore.genetic.population import Candidate, describe_candidate, sort_population
from boolmore.genetic.generation.reproduction import Reproducer
from boolmore.genetic.selection import Selector

FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict]


@dataclass
class GAConfig:
    """
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

    stop_if_max : bool
        if True, stop when the max score is reached. default True
    core : int
        if larger than 1, model evaluation is done in parallel      :int
    seed : int | None
        random seed for reproducibility
    """
    total_iter: int
    per_iter: int
    keep: int
    mix: int

    prob: float | dict[int, float]
    edge_prob: float

    order_by: list[str]
    
    stop_if_max: bool
    core: int
    seed: int | None

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
            reproducer:Reproducer
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

    def new_candidates(self, offsprings:list[Model], evaluator:Evaluator)-> list[EvalResult]:
        if self.config.core > 1:
            results = Parallel(n_jobs=self.config.core)(delayed(evaluator.evaluate)(new_model) for new_model in offsprings)
            new_candidates = []
            for result in results:
                for model in offsprings:
                    if model.id == result.model_id:
                        new_candidate = Candidate(model, result)
                        new_candidates.append(new_candidate)
            return new_candidates
        else:
            # single core
            new_candidates = []
            for new_model in offsprings:
                result = evaluator.evaluate(new_model)
                new_candidate = Candidate(new_model, result)
                new_candidates.append(new_candidate)
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
        reproducer
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
            f"{describe_candidate(state.population[0])}"
            )
        
        if (
            config.stop_if_max
            and state.population[0].eval_result.score == state.population[0].eval_result.max_score
            ):
            print("max score reached")
            break

    return states


def log_candidate(fp, candidate, label, path):
    fp.write(f"\n# {label}: {os.path.abspath(path)}\n")
    fp.write(
        f"# score: {candidate.eval_result.score} / "
        f"{candidate.eval_result.max_score} "
        f"({candidate.eval_result.score / candidate.eval_result.max_score * 100}%)\n"
    )
    fp.write(f"# extra edges: {candidate.model.extra_edges}\n")
    fp.write("# targets,\tfactors\n")

    bnet = primes2bnet(candidate.model.primes)
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
    final.model.name = export_name
    final.model.export()

    # Export top models from each generation
    if export_top:
        for state in states[1:]:
            for candidate in state.population[:export_top]:
                if candidate.eval_result.score > export_thresh:
                    candidate.model.name = export_name
                    candidate.model.export()

    # Export all models tied with the best model in the final generation
    if export_same:
        for candidate in states[-1].population:
            if candidate.eval_result.score == final.eval_result.score:
                candidate.model.name = export_name
                candidate.model.export()


def run_ga(run_type:str,
        json_file:str|None=None, start_model:str|None=None, run_name:str|None=None,
           data_file:str|None=None, base_file:str|None=None, parameter_dict:dict|None=None,
           export_top:int=0, export_thresh:float=0.0, export_name:str|None=None, export_same:bool=False,
           stop_if_max:bool=True, core:int=2, seed:int|None = None,
           hierarchy:bool=True,
           )-> tuple[Model, Model, Model, list]:
    """
    Imports parameters, experiments, base model in the json file.
    Runs genetic algorithm and exports refined models.
    
    Instead of json file, locations of the data and base model can be used.
    In this case, constraints or extra edges are inapplicable.

    Parameters
    ----------
    run_type : str
        "Phenotype" or "NAV"
    json_file : str | None
        location of the json file containing parameters
    
    Optional
    --------
    start_model : str | None
        location of the bnet file of the starting model
        if None, base_model is the starting model
    run_name : str | None
        models are exported as (run_name)_id_gen.bnet
        log is exported as (run_name)_log.txt
        if None, takes the start_model name
    data_file : str | None
        location of the data file that overrides the json file.
        Must be given if json_file is not given.
    base_file : str | None
        location of the base file that overrides the json file.
        Must be given if json_file is not given.
    parameter_dict : dict | None
        If given, overwrites any parameters.

    stop_if_max : bool
        if True, stop when the max score is reached
    core : int
        if larger than 1, model evaluation is done in parallel
    seed : int | None
        random seed for reproducibility

    Returns
    -------
    base : Candidate
        the base model and its evaluation
    start : Candidate
        the starting model and its evaluation
    final : Candidate
        the final model and its evaluation
    log : list
        the log

    """
    # ---------- Load run configuration ----------
    # load json file if given
    if json_file != None:
        f = open(json_file)
        json_dict = json.load(f)

        parameters = json_dict["parameters"]

        # take model specific data from the json file
        DATA:str = json_dict["data"]
        BASE:str = json_dict["base"]
        DEFAULT_SOURCES = json_dict["default_sources"]
        generate_default_sources = False
        CONSTRAINTS = json_dict["constraints"]
        EDGE_POOL = json_dict["edge_pool"]

    else:
        assert data_file != None and base_file != None, "either json or the data and base files should be provided"
        
        parameters = {"starting_gen" : 0,
                      "total_iterations" : 10,
                      "per_iteration" : 10,
                      "keep" : 2,
                      "mix" : 0,
                      "prob" : 0.1,
                      "edge_prob" : 0.5,
                      "order_by" : ["n_extra_edges", "n_prime_implicants"]}

        # all source nodes being 0 is considered the default
        DEFAULT_SOURCES = {}
        generate_default_sources = True
        # no constraint is assumed
        CONSTRAINTS = {"fixed": [], "regulate": {}, "necessary" : {},
                            "group": {}, "possible_constant": []}
        # no extra edge is assumed
        EDGE_POOL = []

    # ---------- Resolve input files and parameters ----------
    # if data file is given, overwrite DATA
    if data_file != None:
        DATA = data_file

    # if base file is given, overwrite BASE
    if base_file != None:
        BASE = base_file

    # if starting model is not given, take the base as the start
    if start_model != None:
        START_MODEL = start_model
    else:
        START_MODEL = BASE

    if run_name is None:
        run_name = START_MODEL.split("/")[-1][:-5]
    LOG = run_name + "_log.txt"

    if export_name is None:
        export_name = run_name

    # if parameter_dict is given, overwrite parameters
    if parameter_dict != None:
        parameters.update(parameter_dict)

    # take parameters
    STARTING_GEN = parameters["starting_gen"]
    TOTAL_ITERATIONS = parameters["total_iterations"]
    PER_ITERATION = parameters["per_iteration"]
    KEEP = parameters["keep"]
    MIX = parameters["mix"]
    PROB = parameters["prob"]
    EDGE_PROB = parameters["edge_prob"]
    ORDER_BY = parameters["order_by"]

    # ---------- Load base model ----------
    print(f"Loading base model from {os.path.abspath(BASE)}")
    if BASE.endswith(".bnet"):
        base_primes = bnet_file2primes(BASE)
    elif BASE.endswith(".pkl"):
        with open(BASE, "rb") as f:
            base_primes = pickle.load(f)
    base_model = Model.import_model(base_primes, constraints=CONSTRAINTS,
                              edge_pool=EDGE_POOL)
    print("Base model loaded.")

    # ---------- Load experimental data ----------
    print(f"Loading experimental data from {os.path.abspath(DATA)}")
    if run_type == "NAV":
        if generate_default_sources:
            for node in base_primes:
                if base_primes[node] == [[{node:0}], [{node:1}]]:
                    DEFAULT_SOURCES[node] = 0

        exps = import_NAV_exps(DATA)
        prediction_fn = get_NAV_prediction
        score_fn = partial(get_NAV_scores, default_sources=DEFAULT_SOURCES, hierarchy=hierarchy)
    elif run_type == "Phenotype":
        exps = import_phenotypes(DATA)
        prediction_fn = get_phenotype_prediction
        score_fn = get_phenotype_scores
    evaluator = Evaluator(exps=exps, prediction_fn=prediction_fn, score_fn=score_fn)
    print("Experimental data loaded.\n")

    # ---------- Evaluate base and start models ----------
    start_single = datetime.datetime.now()
    base = Candidate(base_model, evaluator.evaluate(base_model))    
    end_single = datetime.datetime.now()
    base.model.info()
    print(f"score: {round(base.eval_result.score,2)} / {base.eval_result.max_score} ({round(base.eval_result.score/base.eval_result.max_score*100,1)}%)")
    print(f"""
          Elapsed time for single evaluation: {end_single-start_single}
          Estimated total run time: {(end_single-start_single)*TOTAL_ITERATIONS*PER_ITERATION}""")
    print()

    if START_MODEL == BASE:
        start_primes = base_primes
    else:
        print(f"Loading starting model from {os.path.abspath(START_MODEL)}")
        start_primes = bnet_file2primes(START_MODEL)
    start_model = Model.import_model(start_primes, id=0, generation=STARTING_GEN, base=base.model)
    print("Starting model loaded.")
    start_model.name = run_name
    start = Candidate(start_model, evaluator.evaluate(start_model))
    start.model.info()
    print(f"score: {round(start.eval_result.score,2)} / {start.eval_result.max_score} ({round(start.eval_result.score/start.eval_result.max_score*100,1)}%)")
    print()

    # ---------- Run genetic algorithm ----------
    start_time = datetime.datetime.now()

    config = GAConfig(total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                      prob=PROB, edge_prob=EDGE_PROB, order_by=ORDER_BY,
                      stop_if_max=stop_if_max, core=core, seed=seed)
    selector = Selector(keep=config.keep, order_by=config.order_by)
    reproducer = Reproducer(selector=selector)
    states = ga_main(start, evaluator, selector, reproducer, config)

    end_time = datetime.datetime.now()

    export_models(states, export_name, export_top, export_thresh, export_same)

    final = states[-1].population[0]

    # ---------- Write log ----------
    print("Writing log...")
    with open(LOG, "w") as fp:
        fp.write(f"# {run_type=}\n")
        fp.write(f"# DATA: {os.path.abspath(DATA)}\n")
        fp.write(f"# {DEFAULT_SOURCES=}\n")
        fp.write(f"# {CONSTRAINTS=}\n")
        fp.write(f"# {EDGE_POOL=}\n\n")

        fp.write(f"# total_iterations: {parameters['total_iterations']}\n")
        fp.write(f"# per_iteration: {parameters['per_iteration']}\n")
        fp.write(f"# keep: {parameters['keep']}\n")
        fp.write(f"# mix: {parameters['mix']}\n")
        fp.write(f"# prob: {parameters['prob']}\n")
        fp.write(f"# edge_prob: {parameters['edge_prob']}\n")
        fp.write(f"# order_by: {parameters['order_by']}\n\n")

        fp.write(f"# {stop_if_max=}\n")
        fp.write(f"# {core=}\n")
        fp.write(f"# {seed=}\n\n")

        log_candidate(fp, base, "BASE", BASE)
        log_candidate(fp, start, "START", START_MODEL)

        fp.write(f"\n# {start_time=}\n")
        fp.write(f"# {end_time=}\n")
        fp.write(f"# elapsed time: {end_time-start_time}\n\n")

        fp.write("iteration,top score,extra edges,n_edges,n_self_edges,n_prime_implicants,best_model\n")
        for state in states[1:]:
            candidate = state.population[0]
            fp.write(
                f"{state.iteration},"
                f"{candidate.eval_result.score},"
                f"\"{candidate.model.extra_edges}\","
                f"{candidate.eval_result.n_edges},"
                f"{candidate.eval_result.n_self_edges},"
                f"{candidate.eval_result.n_prime_implicants},"
                f"\"{candidate.model.id}_gen{candidate.model.generation}\"\n"
                )

        log_candidate(fp, final, "FINAL", f"{export_name}_{final.model.id}_gen{final.model.generation}.bnet")

    # ---------- Analyze and report results ----------
    mutated = set()
    for node in start.model.primes:
        for value in [0, 1]:
            sorted_primes1 = sorted([sorted(d.items()) for d in start.model.primes[node][value]])
            sorted_primes2 = sorted([sorted(d.items()) for d in final.model.primes[node][value]])
            if sorted_primes1 != sorted_primes2:
                mutated.add(node)
    mutated = sorted(list(mutated))

    print(f"""
        The algorithm ran for {len(states)-1} iterations,
        generating {states[-1].generated} models.
        Mutated {len(mutated)} functions, 
        and increased score from {round(start.eval_result.score,2)} / {start.eval_result.max_score} ({round(start.eval_result.score/start.eval_result.max_score*100,1)}%)
        to {round(final.eval_result.score,2)} / {final.eval_result.max_score} ({round(final.eval_result.score/final.eval_result.max_score*100,1)}%).\n
        Total elapsed time: {end_time-start_time}""")
    print()

    print("-----modified functions-----")
    for node in mutated:
        print("start:" + prime2bnet(node, start.model.primes[node]))
        print("final:" + prime2bnet(node, final.model.primes[node]))

    return base, start, states
