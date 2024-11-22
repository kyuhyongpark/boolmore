# run with the location of this script as the working directory

import json

from pyboolnet.external.bnet2primes import bnet_file2primes

import boolmore.config
from boolmore.experiment import import_exps
from boolmore.model import Model
from boolmore.genetic_algorithm import ga_main


REPEAT = 50
PROB = 0.01
TOTAL_ITERATIONS = 20
PER_ITERATION = 5
KEEP = 2
MIX_LIST = [0,
            1,
            2,
            3,
            4]


total = TOTAL_ITERATIONS * PER_ITERATION


json_file = "../../../case_study/ABA/data/ABA_2017.json"
start_path = "../../../case_study/ABA/baseline_models/ABA_GA_base_A.bnet"

data_loc = "../../../case_study/ABA/data/data_20230925.tsv"
base_loc = "../../../case_study/ABA/baseline_models/ABA_GA_base_A.bnet"

results_directory = "../results"

f = open(json_file)
json_dict = json.load(f)

# take parameters from the json file
EDGE_PROB = json_dict["parameters"]["edge_prob"]

# take model specific data from the json file
DEFAULT_SOURCES = json_dict["default_sources"]
CONSTRAINTS = json_dict["constraints"]
EDGE_POOL = json_dict["edge_pool"]


exps, fixes_list = import_exps(data_loc)

base_primes = bnet_file2primes(base_loc)
base = Model.import_model(base_primes, constraints=CONSTRAINTS, edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)

primes = bnet_file2primes(start_path)
start = Model.import_model(primes, boolmore.config.id, 1, base)
start.get_predictions(fixes_list)
start.get_model_score(exps)

fp = open(f"{results_directory}/mix_log.csv", "a")
fp.write("sample_model,seed,iter,pop,keep,mix,prob")
for i in range(total + 1):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

for mix in MIX_LIST:

    for run_number in range(REPEAT):

        boolmore.config.id = 0

        MIX = mix

        final, log = ga_main(start, exps, fixes_list,
                            total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                            prob=PROB, edge_prob=EDGE_PROB,
                            stop_if_max=True, core=6, seed=run_number+1)

        fp = open(f"{results_directory}/mix_log.csv", "a")

        fp.write(f"ABA_GA_base_A,{run_number+1},{TOTAL_ITERATIONS},{PER_ITERATION},{KEEP},{MIX},\"{PROB}\",{start.score}")

        for iteration in log:
            # extra commas to make the output csv file neat
            for i in range(int(PER_ITERATION)):
                fp.write(",")
            fp.write(f"{iteration[1]}")

        # if the genetic algorithm ended early, have the final score appended.
        if len(log) != TOTAL_ITERATIONS:
            for i in range(TOTAL_ITERATIONS - len(log)):
                # extra commas to make the output csv file neat
                for j in range(int(PER_ITERATION)):
                    fp.write(",")
                fp.write(f"{log[-1][1]}")        

        fp.write("\n")
        fp.close()

