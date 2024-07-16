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

base_directory = "boolmore/benchmarks/benchmark_models"


# sample_models = ["Cell Cycle Transcription by Coupled CDK and Network Oscillators",
#                  "Metabolic Interactions in the Gut Microbiome",
#                  "Mammalian Cell Cycle 2006",
#                  "T-LGL Survival Network 2011 Reduced Network"]
# data_directory = "boolmore/benchmarks/results"

sample_models = ["IL-1 Signaling",
                 "Glucose Repression Signaling 2009",
                 "Signaling in Macrophage Activation",
                 "Influenza A Virus Replication Cycle"]
data_directory = "boolmore/optimization/data"

# ### Generate data
# for model_name in sample_models:
#     base_primes = sm.format.import_primes(base_directory + "/" + model_name + ".txt")
#     for i in range(5):
#         exps, fixes_list = generate_experiments(base_primes, export=True, file_name="boolmore/optimization/data/"+model_name+"_data_"+str(i+1)+".tsv")
        
#         base = Model.import_model(base_primes)
        
#         start = base.mutate(0.5, 0)

#         start_time = time.time()
#         start.get_predictions(fixes_list)
#         start.get_model_score(exps)
#         print("time for model evaluation:", time.time()-start_time)
        
#         start.export("boolmore/optimization/data/"+model_name+"_start_"+str(i+1)+".txt")


TOTAL_ITERATIONS = 10
PER_ITERATION = 10
KEEP = 2
# PROB_LIST = [0.01, 0.05, 0.1, 0.2, 0.5]
PROB_LIST = [[0.5,0.5,0.5,0.1]]



### benchmark runs
base_paths = dict()
data_paths = dict()
start_paths = dict()

for sample_model in sample_models:
    for filename in os.listdir(base_directory):
        if filename == sample_model+".txt":
            base_paths[sample_model] = base_directory + "/" + filename

    data_paths[sample_model] = []
    start_paths[sample_model] = []
    for filename in os.listdir(data_directory):
        if filename.startswith(sample_model):
            if filename.endswith(".tsv"):
                data_paths[sample_model].append(data_directory + "/" + filename)
            elif re.match(r".+_start_", filename):
                start_paths[sample_model].append(data_directory + "/" + filename)

fp = open("boolmore/optimization/probability log.csv", "a")
fp.write(f"# {TOTAL_ITERATIONS=},{PER_ITERATION=},{KEEP=}\n")
fp.close()

for sample_model in sample_models:
    print(sample_model)
    for prob, base, data, start in it.product(PROB_LIST,
                                              [base_paths[sample_model]],
                                              data_paths[sample_model],
                                              start_paths[sample_model]):
        
        exps, fixes_list = import_exps(data)
    
        base_primes = sm.format.import_primes(base)
        base = Model.import_model(base_primes)
    
        primes = sm.format.import_primes(start)
        start = Model.import_model(primes, boolmore.config.id, 1, base)
        start.get_predictions(fixes_list)
        start.get_model_score(exps)
        start.info()
        print()

        final, log = ga_main(start, exps, fixes_list,
                            total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP,
                            prob=prob, edge_prob=0,
                            stop_if_max=True, core=4)

        fp = open("boolmore/optimization/probability log.csv", "a")

        fp.write(f"{sample_model},{prob},{start.score}")
        for iteration in log:
            fp.write(f",{iteration[1]}")
        fp.write("\n")
        fp.close()

