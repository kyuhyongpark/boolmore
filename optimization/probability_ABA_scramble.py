# import boolmore.genetic_algorithm
import os
import re
import itertools as it
import pystablemotifs as sm
import boolmore.config
import time
import json

from boolmore.experiment import import_exps
from boolmore.model import Model
from boolmore.genetic_algorithm import ga_main
from boolmore.benchmarks import generate_experiments


TOTAL_ITERATIONS = 10
PER_ITERATION = 10
KEEP = 2
PROB_LIST = [[0.5,0.5,0.5,0.1]]



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

fp = open("boolmore/optimization/probability log.csv", "a")
fp.write(f"# {TOTAL_ITERATIONS=},{PER_ITERATION=},{KEEP=}\n")
fp.close()

for i in range(25):
    start = base.mutate(0.5, 0)

    start_time = time.time()
    start.get_predictions(fixes_list)
    start.get_model_score(exps)
    print("time for model evaluation:", time.time()-start_time)
    
    start.export("boolmore/optimization/data/ABA_scramble_"+str(i+1)+".txt")


    for prob in PROB_LIST:
        final, log = ga_main(start, exps, fixes_list,
                            total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP,
                            prob=prob, edge_prob=EDGE_PROB,
                            stop_if_max=True, core=6)

        fp = open("boolmore/optimization/probability log.csv", "a")

        fp.write(f"ABA_scrambled,{prob},{start.score}")
        for iteration in log:
            fp.write(f",{iteration[1]}")
        fp.write("\n")
        fp.close()
