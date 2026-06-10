import datetime
import json
import os
import random
from dataclasses import dataclass

from joblib import Parallel, delayed
import numpy as np
from pyboolnet.external.bnet2primes import bnet_file2primes

import boolmore.config
from boolmore.core.conversions import prime2bnet
from boolmore.io.load import import_exps
from boolmore.core.model import Model, mix_models

from boolmore.algo.selection import reproduction_bias, sort_population
from boolmore.algo.inference import get_NAV_prediction

FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict]


@dataclass
class EvalResult:
    model_id: int
    predictions: PredictType
    max_score: float
    score: float

class Evaluator:
    def __init__(self, fixes_list, exps, hierarchy):
        """
        exps : list[ExpType]
            exp : ExpType
                info of a single experiment                
                exp[0] : int
                    id of the experiment
                exp[1] : float
                    max_score for the experiment
                exp[2] : FixesType
                    fixes - ((node A, value1), (node B, value2), ...)
                exp[3] : str
                    observed_node
                exp[4] : str
                    outcome_value - one of OFF, OFF/Some, Some, Some/ON, ON
        fixes_list : list[FixesType]
            summarized list of fixes for convenience
            fixes : FixesType
                ((node A, value1), (node B, value2), ...)
        """
        self.fixes_list = fixes_list
        self.exps = exps
        self.hierarchy = hierarchy

    def evaluate(self, model:Model):
        predictions = get_NAV_prediction(model.primes, self.fixes_list)
        model.predictions = predictions
        max_score, score = model.get_model_score(self.exps, hierarchy=self.hierarchy)
        result = EvalResult(model_id=model.id,
                            predictions=predictions,
                            max_score=max_score,
                            score=score)
        return result


class Selector:
    def __init__(self, config:GAConfig):
        self.keep = config.keep
    def select_survivors(self, population):
        return population[:self.keep]


class Reproducer:
    def __init__(self, config:GAConfig):
        self.per_iter = config.per_iter
        self.keep = config.keep
        self.mix = config.mix

    def asexual(self, population, prob, edge_prob):
        population = sort_population(population)
        p = reproduction_bias(population)
        # number of offsprings to generate
        # total population should be keep + per_iter
        n = self.keep + self.per_iter - len(population)
        # generate (per_iter) new models
        offsprings = []
        targets = random.choices(population, weights=p, k=n)
        for target in targets:
            new_model = target.mutate(prob, edge_prob)
            offsprings.append(new_model)    
        return offsprings

    def sexual(self, population):
        population = sort_population(population)
        p = reproduction_bias(population)
        parents_lst = []
        for j in range(self.mix):
            model_choice = np.random.choice(population, size = 2, replace = False, p=p)
            parents_lst.append(model_choice)
        mixed_offsprings = []
        for parents in parents_lst:
            mixed_model = mix_models(parents[0], parents[1])
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
    population: list[Model]
    log: list
    best: Model | None = None
    best_score: float = 0

class GeneticAlgorithm:
    def __init__(self, config:GAConfig):
        self.per_iter = config.per_iter
        self.keep = config.keep
        self.mix = config.mix
        self.core = config.core
    
    def initialize_population(self, population, start):
        # this ensures that models worse than the start are not carried on.
        # also ensures that same number of models are generated in the first iteration as in the other iterations.
        for i in range(self.keep):
            population.append(start)
        return population

    def evaluate_offsprings(self, offsprings, evaluator):
        if self.core > 1:
            results = Parallel(n_jobs=self.core)(delayed(evaluator.evaluate)(new_model) for new_model in offsprings)
            for result in results:
                for new_model in offsprings:
                    if new_model.id == result.model_id:
                        new_model.predictions = result.predictions
                        new_model.max_score = result.max_score
                        new_model.score = result.score
        else:
            for new_model in offsprings:
                result = evaluator.evaluate(new_model)
                new_model.predictions = result.predictions
                new_model.max_score = result.max_score
                new_model.score = result.score



def run_ga(json_file:str|None=None, start_model:str|None=None, run_name:str|None=None,
           data_file:str|None=None, base_file:str|None=None, parameter_dict:dict|None=None,
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
    json_file : str | None
        location of the json file containing parameters
    
    Optional
    --------
    start_model : str | None
        location of the txt file of the starting model
        if None, base_model is the starting model
    run_name : str | None
        models are exported as (run_name)_id_gen.txt
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
    base : Model
        the base model
    start : Model
        the starting model
    final : Model
        the final model
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
        CONSTRAINTS = json_dict["constraints"]
        EDGE_POOL = json_dict["edge_pool"]

    else:
        assert data_file != None and base_file != None, "either json or the data and base files should be provided"
        
        parameters = {"starting_id" : 1,
                      "starting_gen" : 1,
                      "total_iterations" : 10,
                      "per_iteration" : 10,
                      "keep" : 2,
                      "mix" : 0,
                      "prob" : 0.1,
                      "edge_prob" : 0.5,
                      "export_top" : 0,
                      "export_threshold" : 0.0}

        # all source nodes being 0 is considered the default
        DEFAULT_SOURCES = {}
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
    STARTING_ID = parameters["starting_id"]
    STARTING_GEN = parameters["starting_gen"]
    TOTAL_ITERATIONS = parameters["total_iterations"]
    PER_ITERATION = parameters["per_iteration"]
    KEEP = parameters["keep"]
    MIX = parameters["mix"]
    PROB = parameters["prob"]
    EDGE_PROB = parameters["edge_prob"]
    EXPORT_TOP = parameters["export_top"]
    EXPORT_THRESHOLD = parameters["export_threshold"]

    boolmore.config.id = STARTING_ID

    print(f"Loading experimental data from {os.path.abspath(DATA)}")
    exps, fixes_list = import_exps(DATA)
    print("Experimental data loaded.\n")

    print(f"Loading base model from {os.path.abspath(BASE)}")
    base_primes = bnet_file2primes(BASE)
    base = Model.import_model(base_primes, constraints=CONSTRAINTS,
                              edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)
    print("Base model loaded.")
    start_single = datetime.datetime.now()
    predictions = get_NAV_prediction(base.primes, fixes_list)
    base.predictions = predictions
    base.get_model_score(exps, hierarchy=hierarchy)
    end_single = datetime.datetime.now()
    base.info()
    print(f"""
          Elapsed time for single evaluation: {end_single-start_single}
          Estimated total run time: {(end_single-start_single)*TOTAL_ITERATIONS*PER_ITERATION}""")
    print()

    print(f"Loading starting model from {os.path.abspath(START_MODEL)}")
    primes = bnet_file2primes(START_MODEL)
    start = Model.import_model(primes, boolmore.config.id, STARTING_GEN, base)
    print("Starting model loaded.")
    start.name = run_name
    predictions = get_NAV_prediction(start.primes, fixes_list)
    start.predictions = predictions
    start.get_model_score(exps, hierarchy=hierarchy)
    start.info()
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
    fp.write(f"# {EDGE_PROB=}\n\n")

    fp.write(f"# {stop_if_max=}\n")
    fp.write(f"# {core=}\n")
    fp.write(f"# {seed=}\n\n")

    fp.write(f"# BASE: {os.path.abspath(BASE)}\n")
    fp.write(f"# extra edges: {base.extra_edges}\n")
    fp.write(f"# score: {base.score} / {base.max_score} ({base.score/base.max_score*100}%)\n")
    with open(BASE, "r") as base_text:
        for line in base_text:
            if not line.startswith("#") and not line.isspace():
                fp.write("# " + line)
    fp.write(f"\n\n# START MODEL: {os.path.abspath(START_MODEL)}\n")
    if BASE != START_MODEL:
        fp.write(f"# score: {start.score} / {start.max_score} ({start.score/start.max_score*100}%)\n")
        fp.write(f"# extra edges: {start.extra_edges}\n")
        with open(START_MODEL, "r") as model_text:
            for line in model_text:
                if not line.startswith("#") and not line.isspace():
                    fp.write("# " + line)
    fp.close()

    start_time = datetime.datetime.now()
    evaluator = Evaluator(fixes_list=fixes_list, exps=exps, hierarchy=hierarchy)
    config = GAConfig(total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                      prob=PROB, edge_prob=EDGE_PROB,
                      stop_if_max=stop_if_max, core=core, seed=seed)
    selector = Selector(config)
    reproducer = Reproducer(config)
    final, log = ga_main(start, evaluator, selector, reproducer, config,
                         export_top=EXPORT_TOP, export_thresh=EXPORT_THRESHOLD)
    end_time = datetime.datetime.now()

    fp = open(LOG, "a")

    fp.write(f"\n# {start_time=}\n")
    fp.write(f"# {end_time=}\n")
    fp.write(f"# elapsed time: {end_time-start_time}\n\n")

    fp.write("iteration,top score,extra edges,complexity\n")
    for iter in log:
        fp.write(f"{iter[0]},{iter[1]},\"{iter[2]}\",{iter[3]}\n")

    mutated = set()
    for node in start.primes:
        for value in [0, 1]:
            sorted_primes1 = sorted([sorted(d.items()) for d in start.primes[node][value]])
            sorted_primes2 = sorted([sorted(d.items()) for d in final.primes[node][value]])
            if sorted_primes1 != sorted_primes2:
                mutated.add(node)
    mutated = sorted(list(mutated))

    print(f"""
        The algorithm ran for {log[-1][0]} iterations,
        generating {boolmore.config.id-STARTING_ID} models.
        Mutated {len(mutated)} functions, 
        and increased score from {round(start.score,2)} / {start.max_score} ({round(start.score/start.max_score*100,1)}%)
        to {round(final.score,2)} / {final.max_score} ({round(final.score/final.max_score*100,1)}%).\n
        Total elapsed time: {end_time-start_time}""")
    print()

    final.export()
    final.info()
    print()

    print("-----modified functions-----")
    for node in mutated:
        print("start:" + prime2bnet(node, start.primes[node]))
        print("final:" + prime2bnet(node, final.primes[node]))

    return base, start, final, log

def ga_main(start:Model,
            evaluator:Evaluator,
            selector:Selector,
            reproducer:Reproducer,
            config:GAConfig,
            export_top:int=0, export_thresh:float=0.0, export_name:str|None=None,
            ) -> tuple[Model, list]:
    """
    Main part of the genetic algorithm.

    Parameters
    ----------
    start : Model
        the starting model
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
    final : Model
        the final model
    log : list[list[]]
        [[iteration #, top score, extra_edges, complexity], ...]

    """
    total_iter = config.total_iter
    edge_prob = config.edge_prob
    stop_if_max = config.stop_if_max
    seed = config.seed
    prob_list = config.prob_list

    if export_name == None:
        export_name = start.name

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
    ga.evaluate_offsprings(offsprings, evaluator)
    state.population.extend(offsprings)

    state.population = sort_population(state.population)
    final = state.population[0]
    print(f"iteration {state.iteration}, generated {boolmore.config.id-start.id}, top score {round(final.score,1)}/{final.max_score} ({round(final.score/final.max_score*100,1)}%)")
    if not final.check_constraint():
        print("ERROR: model does not follow constraints")
    
    state.log.append([1, final.score, final.extra_edges, final.complexity])

    # Export models that exceed the threshold score
    for i in range(export_top):
        state.population[i].name = export_name
        state.population[i].export(threshold=export_thresh)
    
    ### Second to last iterations ###
    for i in range(2,total_iter+1):
        state.iteration = i

        # select the survivors
        state.population = selector.select_survivors(state.population)
    
        # mix the good ones
        mixed_offsprings = reproducer.sexual(state.population)
        ga.evaluate_offsprings(mixed_offsprings, evaluator)
        state.population.extend(mixed_offsprings)
    
        offsprings = reproducer.asexual(state.population, prob=prob_list[i-1], edge_prob=edge_prob)
        ga.evaluate_offsprings(offsprings, evaluator)
        state.population.extend(offsprings)

        state.population = sort_population(state.population)
        final = state.population[0]
        print(f"iteration {i}, generated {boolmore.config.id-start.id}, top score {round(final.score,1)}/{final.max_score} ({round(final.score/final.max_score*100,1)}%)")
        if not final.check_constraint():
            print("ERROR: model does not follow constraints")
        
        state.log.append([i, final.score, final.extra_edges, final.complexity])

        # Export models that exceed the threshold score
        for j in range(export_top):
            state.population[j].name = export_name
            state.population[j].export(threshold=export_thresh)

        # Stop iteration if max score is reached
        if stop_if_max and state.population[0].score == state.population[0].max_score:
            print("max score reached")
            break

    return final, state.log
