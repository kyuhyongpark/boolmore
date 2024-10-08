import json
import random
import pystablemotifs as sm
import boolmore.config
import time
import numpy as np

from boolmore.model import Model, mix_models
from boolmore.experiment import import_exps
from boolmore.conversions import prime2bnet
from joblib import Parallel, delayed


FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]

JSON = "boolmore/case_study/data/ABA_2017.json"
START_MODEL = "boolmore/case_study/baseline_models/ABA_GA_base_A.txt"
RUN_NAME = "test"
DATA = None
BASE = None

def run_ga(json_file:str|None=None, start_model:str|None=None, run_name:str|None=None,
           data_file:str|None=None, base_file:str|None=None, stop_if_max:bool=True, core:int=2,
           seed:int|None = None):
    """
    Imports parameters, experiments, base model in the json file.
    Runs genetic algorithm and exports refined models.
    
    Instead of json file, locations of the data and base model can be used.
    In this case, all parameters are set to default,
    and constraints or extra edges are inapplicable.

    Parameters
    ----------
    json_file   - location of the json file containing parameters   :str
    start_model - location of the txt file of the starting model    :str|None
                  if None, base_model is the starting model
    run_name    - models are exported as (run_name)_id_gen.txt      :str|None
                  log is exported as (run_name)_log.txt
                  if None, takes the start_model name
    
    Optional
    --------
    data_file   - location of the data file if json file is not given   :str|None
    base_file   - location of the base file if json file is not given   :str|None

    stop_if_max - if True, stop when the max score is reached                 :bool
    core        - if larger than 1, model evaluation is done in parallel      :int
    seed        - random seed                                                 :int|None


    """
    if json_file != None:
        f = open(json_file)
        json_dict = json.load(f)

        # take parameters from the json file
        STARTING_ID = json_dict["parameters"]["starting_id"]
        STARTING_GEN = json_dict["parameters"]["starting_gen"]
        TOTAL_ITERATIONS = json_dict["parameters"]["total_iterations"]
        PER_ITERATION = json_dict["parameters"]["per_iteration"]
        KEEP = json_dict["parameters"]["keep"]
        MIX = json_dict["parameters"]["mix"]
        PROB = json_dict["parameters"]["prob"]
        EDGE_PROB = json_dict["parameters"]["edge_prob"]
        EXPORT_TOP = json_dict["parameters"]["export_top"]
        EXPORT_THRESHOLD = json_dict["parameters"]["export_threshold"]

        # take model specific data from the json file
        DATA:str = json_dict["data"]
        BASE:str = json_dict["base"]
        DEFAULT_SOURCES = json_dict["default_sources"]
        CONSTRAINTS = json_dict["constraints"]
        EDGE_POOL = json_dict["edge_pool"]

    else:
        assert data_file != None and base_file != None, "either json or the data and base files should be provided"
        STARTING_ID = 1
        STARTING_GEN = 1
        TOTAL_ITERATIONS = 100
        PER_ITERATION = 100
        KEEP = 20
        MIX = 0
        PROB = 0.01
        EDGE_PROB = 0.5
        EXPORT_TOP = 0
        EXPORT_THRESHOLD = 0.0
        DEFAULT_SOURCES = {}
        CONSTRAINTS = {"fixed": [], "regulate": {}, "necessary" : {},
                            "group": {}, "possible_constant": []}
        EDGE_POOL = []
        DATA = data_file
        BASE = base_file

    # if starting model is not given, take the base as the start
    if start_model != None:
        START_MODEL = start_model
    else:
        START_MODEL = BASE

    if run_name == None:
        run_name = START_MODEL.split("/")[-1][:-4]
    LOG = run_name + "_log.txt"

    if seed != None:
        random.seed(seed)

    boolmore.config.id = STARTING_ID

    print("Loading experimental data . . .")
    exps, fixes_list = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = sm.format.import_primes(BASE)
    base = Model.import_model(base_primes, constraints=CONSTRAINTS,
                              edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)
    print("Base model loaded.")
    base.get_predictions(fixes_list)
    base.get_model_score(exps)
    base.info()
    print()

    print("Loading starting model . . .")
    primes = sm.format.import_primes(START_MODEL)
    start = Model.import_model(primes, boolmore.config.id, STARTING_GEN, base)
    print("Starting model loaded.")
    start.name = run_name
    start.get_predictions(fixes_list)
    start.get_model_score(exps)
    start.info()
    print()

    fp = open(LOG, "w")

    fp.write(f"# {DATA=}\n")
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
    fp.write(f"# {core=}\n\n")

    fp.write(f"# {BASE=}\n")
    fp.write(f"# extra edges: {base.extra_edges}\n")
    fp.write(f"# score: {base.score}\n")
    with open(BASE, "r") as base_text:
        for line in base_text:
            if not line.startswith("#") and not line.isspace():
                fp.write("# " + line)
    fp.write(f"\n# {START_MODEL=}\n")
    if BASE != START_MODEL:
        fp.write(f"# score: {start.score}\n")
        fp.write(f"# extra edges: {start.extra_edges}\n")
        with open(START_MODEL, "r") as model_text:
            for line in model_text:
                if not line.startswith("#") and not line.isspace():
                    fp.write("# " + line)

    start_time = time.time()
    final, log = ga_main(start, exps, fixes_list,
                         total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                         prob=PROB, edge_prob=EDGE_PROB,
                         export_top=EXPORT_TOP, export_thresh=EXPORT_THRESHOLD,
                         stop_if_max=stop_if_max, core=core)
    end_time = time.time()

    fp.write("\niteration\ttop score\textra edges\tcomplexity\n")
    for sth in log:
        fp.write("\t".join([str(x) for x in sth])+"\n")

    mutated = set()
    for node in start.primes:
        for value in [0, 1]:
            sorted_primes1 = sorted([sorted(d.items()) for d in start.primes[node][value]])
            sorted_primes2 = sorted([sorted(d.items()) for d in final.primes[node][value]])
            if sorted_primes1 != sorted_primes2:
                mutated.add(node)
    mutated = sorted(list(mutated))

    print()
    print(f"""\tThe algorithm ran for {log[-1][0]} iterations,
        generating {boolmore.config.id-STARTING_ID} models.
        Mutated {len(mutated)} functions, 
        and increased score from {round(start.score,2)} to {round(final.score,2)}.\n
        Total elapsed time: {time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))}
        Time per model: {(end_time-start_time)/(boolmore.config.id-STARTING_ID)} s""")
    print()

    final.export()
    final.info()
    print()

    for node in mutated:
        print("start:" + prime2bnet(node, start.primes[node]))
        print("final:" + prime2bnet(node, final.primes[node]))

    return base, start, final

def ga_main(start:Model, exps:list[ExpType], fixes_list:list[FixesType],
            total_iter:int=100, per_iter:int=100, keep:int=20, mix:int=0,
            prob:float|list[float]=0.01, edge_prob:float=0.5,
            export_top:int=0, export_thresh:float=0.0, export_name:str|None=None,
            stop_if_max:bool=True, core:int=2, seed:int|None=None,
            ) -> tuple[Model, list]:
    """
    Main part of the genetic algorithm

    Parameters
    ----------
    start      - the start model                            :Model
    exps       - list of exp                                :list[ExpType]
        exp      - info of a single experiment                :ExpType = tuple[...]
            exp[0] - id of the experiment                       :int
            exp[1] - max_score for the experiment               :float
            exp[2] - fixes                                      :FixesType = tuple[tuple[str, int]]
                     ((node A, value1), (node B, value2), ...)
            exp[3] - observed_node                              :str
            exp[4] - outcome_value                              :str
                     one of OFF, OFF/Some, Some, Some/ON, ON
    fixes_list - summarized list of fixes for convenience   :list[FixesType]
        fixes    - ((node A, value1), (node B, value2), ...)  :FixesType = tuple[tuple[str, int]]


    total_iter - total number of iterations, default 100                :int
    per_iter   - models per iterations, default 100                     :int
    keep       - models to carry over tot he next iteration, default 20 :int
    mix        - number of models to mix from the keep, default 0       :int

    prob      - probability for each digit in the rule representation to mutate, default 0.01   :float | list[float]
                if given a list, each probability is applied for each iteration
                last item is used for the remaining iterations if the length is shorter.
    edge_prob - probability to add/delete extra edge, default 0.5                               :float
 
    export_top    - number of models to export at each iteration, default 0     :int
    export_thresh - only models above this threshold are exported, default 0.0  :float
    export_name   - models are exported as (export_name)_id_gen.txt             :str
                    if None, use the start model name
                    
    stop_if_max - if True, stop when the max score is reached. default True   :bool
    core        - if larger than 1, model evaluation is done in parallel      :int
    seed        - random seed                                                 :int|None


    Returns
    -------
    final   - the final model       :Model
    log     - log of iterations     :list[list[]]
              [[iteration, top score, extra_edges, complexity], ...]
    """
    assert per_iter > keep, "We need more models per iteration than the number of models to carry over to the next iteration"

    if export_name == None:
        export_name = start.name

    if type(prob) == float:
        prob_list = [prob] * total_iter
    elif len(prob) < total_iter:
        prob_list = prob
        prob_list.extend([prob[-1]] * (total_iter-len(prob)))
    else:
        prob_list = prob

    if seed != None:
        random.seed(seed)
        np.random.seed(seed)

    log = []
    iteration = []

    ### First iteration ###

    # this ensures that models worse than the start are not carried on.
    # also ensures that same number of models are generated in the first iteration as in the other iterations.
    for i in range(keep):
        iteration.append(start)

    # generate (per_iter - keep) new models
    new_model_lst = []
    for i in range(per_iter-keep):
        new_model = start.mutate(prob_list[0], edge_prob)
        new_model_lst.append(new_model)    
    if core > 1:
        results = Parallel(n_jobs=core)(delayed(parallel_pred_and_score)(new_model, fixes_list, exps) for new_model in new_model_lst)
        for result in results:
            for new_model in new_model_lst:
                if new_model.id == result[0]:
                    new_model.predictions = result[1]
                    new_model.score = result[2]
    else:
        for new_model in new_model_lst:
            new_model.get_predictions(fixes_list)
            new_model.get_model_score(exps)
    iteration.extend(new_model_lst)
    
    iteration = sorted(iteration, key=lambda x: (len(x.extra_edges), x.complexity))
    iteration = sorted(iteration, key=lambda x: x.score, reverse=True)

    # Export models that exceed the threshold score
    for i in range(export_top):
        iteration[i].name = export_name
        iteration[i].export(threshold=export_thresh)

    final = iteration[0]

    print(f"iteration 1, genearted {boolmore.config.id-start.id}, top score {round(final.score,2)}/{final.max_score} ({round(final.score/final.max_score*100,1)}%)")

    if not final.check_constraint():
        print("ERROR: model does not follow constraints")

    log.append([1, final.score, final.extra_edges, final.complexity])
    
    ### Second to last iterations ###
    for i in range(2,total_iter+1):
        new_iteration = []
        # keep the good ones
        for j in range(keep):
            new_iteration.append(iteration[j])
    
        # mix the good ones
        to_be_mixed = sorted(new_iteration, key=lambda x: (len(x.extra_edges), x.complexity))
        to_be_mixed = sorted(to_be_mixed, key=lambda x: x.score, reverse=True)
        weights = list(range(1, keep+1))
        weights.reverse()
        p = np.array(weights)/np.sum(np.array(weights))

        mixed_model_lst = []
        for j in range(mix):
            model_choice = np.random.choice(to_be_mixed, size = 2, replace = False, p=p)
            mixed_model = mix_models(model_choice[0], model_choice[1])
            mixed_model_lst.append(mixed_model)    
        if core > 1:
            results = Parallel(n_jobs=core)(delayed(parallel_pred_and_score)(mixed_model, fixes_list, exps) for mixed_model in mixed_model_lst)
            for result in results:
                for mixed_model in mixed_model_lst:
                    if mixed_model.id == result[0]:
                        mixed_model.predictions = result[1]
                        mixed_model.score = result[2]
        else:
            for mixed_model in mixed_model_lst:
                mixed_model.get_predictions(fixes_list)
                mixed_model.get_model_score(exps)
        new_iteration.extend(mixed_model_lst)
    
        # mutate the good ones
        weights = list(range(1, keep+mix+1))
        weights.reverse()
        new_iteration = sorted(new_iteration, key=lambda x: (len(x.extra_edges), x.complexity))
        new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)
        targets = random.choices(new_iteration, weights=weights, k=per_iter-keep-mix)
        new_model_lst = []
        for target in targets:
            new_model = target.mutate(prob_list[i-1], edge_prob)
            new_model_lst.append(new_model)    
        if core > 1:
            results = Parallel(n_jobs=core)(delayed(parallel_pred_and_score)(new_model, fixes_list, exps) for new_model in new_model_lst)
            for result in results:
                for new_model in new_model_lst:
                    if new_model.id == result[0]:
                        new_model.predictions = result[1]
                        new_model.score = result[2]
        else:
            for new_model in new_model_lst:
                new_model.get_predictions(fixes_list)
                new_model.get_model_score(exps)
        new_iteration.extend(new_model_lst)
      
        new_iteration = sorted(new_iteration, key=lambda x: (len(x.extra_edges), x.complexity))
        new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)
    
        final = new_iteration[0]

        print(f"iteration {i}, generated {boolmore.config.id-start.id}, top score {round(final.score,2)}/{final.max_score} ({round(final.score/final.max_score*100,1)}%)")

        if not final.check_constraint():
            print("ERROR: model does not follow constraints")

        log.append([i, final.score, final.extra_edges, final.complexity])

        # Export models that exceed the threshold score
        for j in range(export_top):
            new_iteration[j].name = export_name
            new_iteration[j].export(threshold=export_thresh)

        # Stop iteration if max score is reached
        if stop_if_max and new_iteration[0].score == new_iteration[0].max_score:
            print("max score reached")
            break

        iteration = new_iteration

    return final, log

def parallel_pred_and_score(model, fixes_list, exps):
    model.get_predictions(fixes_list)
    model.get_model_score(exps)

    return model.id, model.predictions, model.score

if __name__=="__main__":
    run_ga(JSON, START_MODEL, RUN_NAME, DATA, BASE)