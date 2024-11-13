import pystablemotifs as sm
import boolmore.config
import json

from boolmore.experiment import import_exps
from boolmore.model import Model
from boolmore.genetic_algorithm import ga_main


REPEAT = 50
PRAM_LIST = [
    [20,7,2,0],
    [20,7,2,1],
    [20,7,2,2],
    [20,7,2,3],
    [20,7,2,4],
    # [20,7,2,5],
    ]

total = 100


json_file = "boolmore/case_study/data/ABA_2017.json"
start_path = "boolmore/case_study/baseline_models/ABA_GA_base_A.txt"

f = open(json_file)
json_dict = json.load(f)

# take parameters from the json file
PROB = json_dict["parameters"]["prob"]
EDGE_PROB = json_dict["parameters"]["edge_prob"]

# take model specific data from the json file
DATA:str = json_dict["data"]
BASE:str = json_dict["base"]
DEFAULT_SOURCES = json_dict["default_sources"]
CONSTRAINTS = json_dict["constraints"]
EDGE_POOL = json_dict["edge_pool"]


fp = open("boolmore/optimization/mix_log.csv", "a")
fp.write("sample_model,iter,pop,keep,mix,prob")
for i in range(total + 1):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

    
exps, fixes_list = import_exps(DATA)

base_primes = sm.format.import_primes(BASE)
base = Model.import_model(base_primes, constraints=CONSTRAINTS, edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)

primes = sm.format.import_primes(start_path)
start = Model.import_model(primes, boolmore.config.id, 1, base)
start.get_predictions(fixes_list)
start.get_model_score(exps)

for parameters in PRAM_LIST:

    for run_number in range(REPEAT):
        final, log = ga_main(start, exps, fixes_list,
                            total_iter=parameters[0], per_iter=parameters[1], keep=parameters[2], mix=parameters[3],
                            prob=PROB, edge_prob=EDGE_PROB,
                            stop_if_max=True, core=6, seed=run_number)

        fp = open("boolmore/optimization/mix_log.csv", "a")
        fp.write(f"ABA_GA_base_A,{parameters[0]},{parameters[1]},{parameters[2]},{parameters[3]},\"{PROB}\",{start.score}")

        for iteration in log:
            # extra commas to make the output csv file neat
            for i in range(int(parameters[1]-parameters[2])):
                fp.write(",")
            fp.write(f"{iteration[1]}")

        # if the genetic algorithm ended early, have the final score appended.
        if len(log) != parameters[0]:
            for i in range(parameters[0] - len(log)):
                # extra commas to make the output csv file neat
                for i in range(int(parameters[1]-parameters[2])):
                    fp.write(",")
                # fp.write(",")
                fp.write(f"{log[-1][1]}")

        fp.write("\n")
        fp.close()

