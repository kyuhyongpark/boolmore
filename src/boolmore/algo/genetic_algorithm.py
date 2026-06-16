import datetime
import json
import os
import random
import pickle
from dataclasses import dataclass
from functools import partial
from itertools import count
from joblib import Parallel, delayed

import numpy as np
from pyboolnet.external.bnet2primes import bnet_file2primes
from pystablemotifs.format import primes2bnet

from boolmore.core.conversions import prime2bnet
from boolmore.io.load import import_NAV_exps, import_phenotypes
from boolmore.core.model import Model, mix_models
from boolmore.algo.inference import get_NAV_prediction, get_phenotype_prediction
from boolmore.eval.score import get_NAV_scores, get_phenotype_scores, get_model_score

FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict]


@dataclass
class EvalResult:
    model_id: int
    max_score: float
    score: float
    details: any

class Evaluator:
    def __init__(self, exps, prediction_fn, score_fn):
        """
        exps : list of experiment dataclasses
        """
        self.exps = exps
        self.prediction_fn = prediction_fn
        self.score_fn = score_fn

    def evaluate(self, model:Model):
        predictions = self.prediction_fn(model.primes, self.exps)
        score_items = self.score_fn(self.exps, predictions)
        max_score, score = get_model_score(score_items)
        result = EvalResult(model_id=model.id,
                            max_score=max_score,
                            score=score,
                            details=[predictions, score_items])
        return result

@dataclass
class Candidate:
    model:Model
    eval_result:EvalResult

def sort_population(
    population: list[Candidate],
    order_by: list[str] | None = None,
) -> list[Candidate]:
    """
    Sort by:
        1. score (highest first)
        2. tie-breakers in order_by (lowest first)

    Example:
        sort_population(pop, ["n_edges"])
        sort_population(pop, ["n_self_edges", "n_prime_implicants"])
    """
    if order_by is None:
        order_by = []

    for attr in reversed(order_by):
        population = sorted(
            population,
            key=lambda x: getattr(x.model, attr),
        )

    population = sorted(
        population,
        key=lambda x: x.eval_result.score,
        reverse=True,
    )

    return population

class Selector:
    def __init__(self, config:GAConfig):
        self.keep = config.keep
        self.order_by = config.order_by
    def select_survivors(self, population:list[Candidate]):
        population = sort_population(population, self.order_by)
        return population[:self.keep]

def reproduction_bias(population: list[Candidate]):
    weights = list(range(1, len(population)+1))
    weights.reverse()
    p = np.array(weights)/np.sum(np.array(weights))
    return p

class Reproducer:
    def __init__(self, config:GAConfig):
        self.per_iter = config.per_iter
        self.keep = config.keep
        self.mix = config.mix
        self.order_by = config.order_by
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
    log: list
    best: Candidate | None = None
    best_score: float = 0
    generated: int = 0

class GeneticAlgorithm:
    def __init__(self, config:GAConfig):
        self.per_iter = config.per_iter
        self.keep = config.keep
        self.mix = config.mix
        self.core = config.core
    
    def initialize_population(self, population: list[Candidate], start:Candidate):
        # this ensures that models worse than the start are not carried on.
        # also ensures that same number of models are generated in the first iteration as in the other iterations.
        for i in range(self.keep):
            population.append(start)
        return population

    def new_candidates(self, offsprings:list[Model], evaluator:Evaluator)-> list[EvalResult]:
        if self.core > 1:
            results = Parallel(n_jobs=self.core)(delayed(evaluator.evaluate)(new_model) for new_model in offsprings)
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
        models are exported as (run_name)_id_gen.bent
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

    if run_name == None:
        run_name = START_MODEL.split("/")[-1][:-5]
    LOG = run_name + "_log.txt"

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

    print(f"Loading base model from {os.path.abspath(BASE)}")
    if BASE.endswith(".bnet"):
        base_primes = bnet_file2primes(BASE)
    elif BASE.endswith(".pkl"):
        with open(BASE, "rb") as f:
            base_primes = pickle.load(f)
    base_model = Model.import_model(base_primes, constraints=CONSTRAINTS,
                              edge_pool=EDGE_POOL)
    print("Base model loaded.")

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
    print("Experimental data loaded.\n")

    start_single = datetime.datetime.now()
    predictions = prediction_fn(base_model.primes, exps)
    score_items = score_fn(exps, predictions)
    max_score, score = get_model_score(score_items)
    base_eval = EvalResult(model_id=base_model.id, max_score=max_score, score=score, details=[predictions, score_items])
    base = Candidate(base_model, base_eval)    
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
    predictions = prediction_fn(start_model.primes, exps)
    score_items = score_fn(exps, predictions)
    max_score, score = get_model_score(score_items)
    start_eval = EvalResult(model_id=start_model.id, max_score=max_score, score=score, details=[predictions, score_items])
    start = Candidate(start_model, start_eval)
    start.model.info()
    print(f"score: {round(start.eval_result.score,2)} / {start.eval_result.max_score} ({round(start.eval_result.score/start.eval_result.max_score*100,1)}%)")
    print()

    fp = open(LOG, "w")

    fp.write(f"# DATA: {os.path.abspath(DATA)}\n")
    fp.write(f"# {DEFAULT_SOURCES=}\n")
    fp.write(f"# {CONSTRAINTS=}\n")
    fp.write(f"# {EDGE_POOL=}\n\n")

    fp.write(f"# {TOTAL_ITERATIONS=}\n")
    fp.write(f"# {PER_ITERATION=}\n")
    fp.write(f"# {KEEP=}\n")
    fp.write(f"# {MIX=}\n")
    fp.write(f"# {PROB=}\n")
    fp.write(f"# {EDGE_PROB=}\n")
    fp.write(f"# {ORDER_BY=}\n\n")

    fp.write(f"# {stop_if_max=}\n")
    fp.write(f"# {core=}\n")
    fp.write(f"# {seed=}\n\n")

    fp.write(f"# BASE: {os.path.abspath(BASE)}\n")
    fp.write(f"# extra edges: {base.model.extra_edges}\n")
    fp.write(f"# score: {base.eval_result.score} / {base.eval_result.max_score} ({base.eval_result.score/base.eval_result.max_score*100}%)\n")
    fp.write("# targets,\tfactors\n")
    base_bnet = primes2bnet(base.model.primes)
    for line in base_bnet.split("\n"):
        fp.write("# " + line + "\n")
    fp.write(f"\n\n# START MODEL: {os.path.abspath(START_MODEL)}\n")
    if BASE != START_MODEL:
        fp.write(f"# score: {start.eval_result.score} / {start.eval_result.max_score} ({start.eval_result.score/start.eval_result.max_score*100}%)\n")
        fp.write(f"# extra edges: {start.model.extra_edges}\n")
        fp.write("# targets,\tfactors\n")
        start_bnet = primes2bnet(start.model.primes)
        for line in start_bnet.split("\n"):
            fp.write("# " + line + "\n")
    fp.close()

    start_time = datetime.datetime.now()
    evaluator = Evaluator(exps=exps, prediction_fn=prediction_fn, score_fn=score_fn)
    config = GAConfig(total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                      prob=PROB, edge_prob=EDGE_PROB, order_by=ORDER_BY,
                      stop_if_max=stop_if_max, core=core, seed=seed)
    selector = Selector(config)
    reproducer = Reproducer(config)
    final, log = ga_main(start, evaluator, selector, reproducer, config,
                         export_top=export_top, export_thresh=export_thresh, export_name=export_name, export_same=export_same)
    end_time = datetime.datetime.now()

    fp = open(LOG, "a")

    fp.write(f"\n# {start_time=}\n")
    fp.write(f"# {end_time=}\n")
    fp.write(f"# elapsed time: {end_time-start_time}\n\n")

    fp.write("iteration,top score,extra edges,n_edges,n_self_edges,n_prime_implicants,best_model\n")
    for iter in log:
        fp.write(f"{iter[0]},{iter[1]},\"{iter[2]}\",{iter[3]},{iter[4]},{iter[5]},\"{iter[6]}\"\n")

    mutated = set()
    for node in start.model.primes:
        for value in [0, 1]:
            sorted_primes1 = sorted([sorted(d.items()) for d in start.model.primes[node][value]])
            sorted_primes2 = sorted([sorted(d.items()) for d in final.model.primes[node][value]])
            if sorted_primes1 != sorted_primes2:
                mutated.add(node)
    mutated = sorted(list(mutated))

    print(f"""
        The algorithm ran for {log[-1][0]} iterations,
        generating {log[-1][0]*PER_ITERATION} models.
        Mutated {len(mutated)} functions, 
        and increased score from {round(start_eval.score,2)} / {start_eval.max_score} ({round(start_eval.score/start_eval.max_score*100,1)}%)
        to {round(final.eval_result.score,2)} / {final.eval_result.max_score} ({round(final.eval_result.score/final.eval_result.max_score*100,1)}%).\n
        Total elapsed time: {end_time-start_time}""")
    print()

    final.model.export()
    final.model.info()
    print()

    print("-----modified functions-----")
    for node in mutated:
        print("start:" + prime2bnet(node, start.model.primes[node]))
        print("final:" + prime2bnet(node, final.model.primes[node]))

    return base, start, final, log

def ga_main(start:Candidate,
            evaluator:Evaluator,
            selector:Selector,
            reproducer:Reproducer,
            config:GAConfig,
            export_top:int=0, export_thresh:float=0.0, export_name:str|None=None, export_same:bool=False
            ) -> tuple[Model, list]:
    """
    Main part of the genetic algorithm.

    Parameters
    ----------
    base: Candidate
        the base model and its evaluation
    start : Candidate
        the starting model and its evaluation
    evaluator : Evaluator
        evaluator to get predictions and scores
    config : GAConfig

    export_top : int
        number of models to export at each iteration, default 0
    export_thresh : float
        only models with scores above this threshold are exported, default 0.0
    export_name : str
        models are exported as (export_name)_id_gen.txt
        if None, use the start model name


    Returns
    -------
    final : Candidate
        the final model and its evaluation
    log : list[list[]]
        [[iteration #, top score, extra_edges, n_edges, n_self_edges, n_prime_implicants, best_model], ...]

    """
    total_iter = config.total_iter
    edge_prob = config.edge_prob
    stop_if_max = config.stop_if_max
    seed = config.seed
    prob_list = config.prob_list
    order_by = config.order_by

    if export_name == None:
        export_name = start.model.name

    if seed != None:
        random.seed(seed)
        np.random.seed(seed)

    state = GAState(population=[], iteration=0, log=[])
    ga = GeneticAlgorithm(config)

    ### First iteration ###
    state.iteration = 1
    state.population = ga.initialize_population(state.population, start)

    # generate (per_iter) new models
    offsprings = reproducer.asexual(state.population, prob=prob_list[0], edge_prob=edge_prob)
    new_candidates = ga.new_candidates(offsprings, evaluator)
    state.population.extend(new_candidates)
    state.generated += len(new_candidates)

    state.population = sort_population(state.population, order_by)
    final = state.population[0]
    print(f"iteration {state.iteration}, ",
          f"generated {state.generated}, ",
          f"top score {round(final.eval_result.score,1)}/{final.eval_result.max_score} ",
          f"({round(final.eval_result.score/final.eval_result.max_score*100,1)}%)",
          f"extra edges {final.model.extra_edges}, ",
          f"n edges {final.model.n_edges}, ",
          f"n self edges {final.model.n_self_edges}, ",
          f"n prime implicants {final.model.n_prime_implicants}")
    if not final.model.check_constraint():
        print("ERROR: model does not follow constraints")
    
    final_info = str(final.model.id) + "_gen" + str(final.model.generation)
    state.log.append([1, final.eval_result.score, final.model.extra_edges, final.model.n_edges, final.model.n_self_edges, final.model.n_prime_implicants, final_info])

    # Export models that exceed the threshold score
    for i in range(export_top):
        if state.population[i].eval_result.score > export_thresh:
            state.population[i].model.name = export_name
            state.population[i].model.export()
    
    ### Second to last iterations ###
    for i in range(2,total_iter+1):
        state.iteration = i

        # select the survivors
        state.population = selector.select_survivors(state.population)
    
        # mix the good ones
        mixed_offsprings = reproducer.sexual(state.population)
        new_candidates = ga.new_candidates(mixed_offsprings, evaluator)
        state.population.extend(new_candidates)
        state.generated += len(new_candidates)
    
        offsprings = reproducer.asexual(state.population, prob=prob_list[i-1], edge_prob=edge_prob)
        new_candidates = ga.new_candidates(offsprings, evaluator)
        state.population.extend(new_candidates)
        state.generated += len(new_candidates)

        state.population = sort_population(state.population, order_by)
        final = state.population[0]
        print(f"iteration {i}, ",
              f"generated {state.generated}, ",
              f"top score {round(final.eval_result.score,1)}/{final.eval_result.max_score} ",
              f"({round(final.eval_result.score/final.eval_result.max_score*100,1)}%)",
              f"extra edges {final.model.extra_edges}, ",
              f"n edges {final.model.n_edges}, ",
              f"n self edges {final.model.n_self_edges}, ",
              f"n prime implicants {final.model.n_prime_implicants}")
        if not final.model.check_constraint():
            print("ERROR: model does not follow constraints")
        
        final_info = str(final.model.id) + "_gen" + str(final.model.generation)
        state.log.append([i, final.eval_result.score, final.model.extra_edges, final.model.n_edges, final.model.n_self_edges, final.model.n_prime_implicants, final_info])

        # Export models that exceed the threshold score
        for j in range(export_top):
            if state.population[j].eval_result.score > export_thresh:
                state.population[j].model.name = export_name
                state.population[j].model.export()

        # Stop iteration if max score is reached
        if stop_if_max and state.population[0].eval_result.score == state.population[0].eval_result.max_score:
            print("max score reached")
            break

    if export_same:
        for candidate in state.population:
            if candidate.eval_result.score == final.eval_result.score:
                candidate.model.name = export_name
                candidate.model.export()

    return final, state.log
