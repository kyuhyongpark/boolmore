import pystablemotifs as sm
import boolmore.config
import json

from boolmore.experiment import import_exps
from boolmore.model import Model
from boolmore.genetic_algorithm import ga_main


TOTAL_ITERATIONS = 10
PER_ITERATION = 10
KEEP = 2
MIX = 0
PROB_LIST = [0.01,
             0.05,
             0.1,
             0.2,
             0.5,
             [0.5,0.5,0.5,0.1]]

total = TOTAL_ITERATIONS * (PER_ITERATION-KEEP)


json_file = "boolmore/case_study/data/ABA_2017.json"

f = open(json_file)
json_dict = json.load(f)

# take parameters from the json file
EDGE_PROB = json_dict["parameters"]["edge_prob"]

# take model specific data from the json file
DATA:str = json_dict["data"]
BASE:str = json_dict["base"]
DEFAULT_SOURCES = json_dict["default_sources"]
CONSTRAINTS = json_dict["constraints"]
EDGE_POOL = json_dict["edge_pool"]


exps, fixes_list = import_exps(DATA)

base_primes = sm.format.import_primes(BASE)
base = Model.import_model(base_primes, constraints=CONSTRAINTS, edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)

fp = open("boolmore/optimization/probability_log.csv", "a")
fp.write("sample_model,iter,pop,keep,mix,prob")
for i in range(total + 1):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

for i in range(25):    
    start_path = "boolmore/optimization/data/ABA_scramble_"+str(i+1)+".txt"

    primes = sm.format.import_primes(start_path)
    start = Model.import_model(primes, boolmore.config.id, 1, base)
    start.get_predictions(fixes_list)
    start.get_model_score(exps)
    start.info()
    print()


    for prob in PROB_LIST:
        final, log = ga_main(start, exps, fixes_list,
                            total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP,
                            prob=prob, edge_prob=EDGE_PROB,
                            stop_if_max=True, core=6)

        fp = open("boolmore/optimization/probability_log.csv", "a")

        fp.write(f"ABA_scrambled_{i+1},{TOTAL_ITERATIONS},{PER_ITERATION},{KEEP},{MIX},\"{prob}\",{start.score}")

        for iteration in log:
            # extra commas to make the output csv file neat
            for i in range(int(total/TOTAL_ITERATIONS)):
                fp.write(",")
            fp.write(f"{iteration[1]}")

        # if the genetic algorithm ended early, have the final score appended.
        if len(log) != TOTAL_ITERATIONS:
            for i in range(len(log) - TOTAL_ITERATIONS):
                # extra commas to make the output csv file neat
                for i in range(int(total/TOTAL_ITERATIONS)):
                    fp.write(",")
                fp.write(f"{log[-1][1]}")        

        fp.write("\n")

        fp.close()
