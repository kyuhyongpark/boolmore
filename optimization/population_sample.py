# import boolmore.genetic_algorithm
import os
import re
import itertools as it
import pystablemotifs as sm
import boolmore.config
import math
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

# sample_models = ["IL-1 Signaling",
#                  "Glucose Repression Signaling 2009",
#                  "Signaling in Macrophage Activation",
#                  "Influenza A Virus Replication Cycle"]
sample_models = ["Influenza A Virus Replication Cycle"]
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

PRAM_LIST = [[1,101,1,0],
             [2,51,1,0],
             [5,21,1,0],
             [10,11,1,0],
             [20,6,1,0],
             [50,3,1,0],
             [100,2,1,0]]


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

fp = open("boolmore/optimization/population log.csv", "a")
fp.write("sample_model,iter,pop,keep,mix,prob")
for i in range(101):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

for sample_model in sample_models:
    print(sample_model)
    for parameters, base_path, data, start_path in it.product(PRAM_LIST,
                                                    [base_paths[sample_model]],
                                                    data_paths[sample_model],
                                                    start_paths[sample_model]):
        
        # prob = [0.5] * math.ceil(parameters[0] * 0.3)
        # prob.append(0.1)
        prob = 0.2

        exps, fixes_list = import_exps(data)
    
        base_primes = sm.format.import_primes(base_path)
        base = Model.import_model(base_primes)
    
        primes = sm.format.import_primes(start_path)
        start = Model.import_model(primes, boolmore.config.id, 1, base)
        start.get_predictions(fixes_list)
        start.get_model_score(exps)
        start.info()
        print()

        final, log = ga_main(start, exps, fixes_list,
                            total_iter=parameters[0], per_iter=parameters[1], keep=parameters[2], mix=parameters[3],
                            prob=prob, edge_prob=0,
                            stop_if_max=True, core=6)

        fp = open("boolmore/optimization/population log.csv", "a")

        fp.write(f"{sample_model},{parameters[0]},{parameters[1]},{parameters[2]},{parameters[3]},\"{prob}\",{start.score}")
        for iteration in log:
            
            # extra commas to make the output csv file neat
            for i in range(int(100/parameters[0])):
                fp.write(",")

            # fp.write(",")
            fp.write(f"{iteration[1]}")
            last_score = iteration[1]

        # if the genetic algorithm ended early, have the final score appended.
        if len(log) != parameters[0]:
            for i in range(len(log) - parameters[0]):
                # extra commas to make the output csv file neat
                for i in range(int(100/parameters[0])):
                    fp.write(",")
                # fp.write(",")
                fp.write(f"{last_score}")

        fp.write("\n")
        fp.close()

