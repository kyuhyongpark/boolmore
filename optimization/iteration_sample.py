import os
import re
import itertools as it
import pystablemotifs as sm
import boolmore.config

from boolmore.experiment import import_exps
from boolmore.model import Model
from boolmore.genetic_algorithm import ga_main

PROB = 0.2
PRAM_LIST = [[1,101,1,0],
             [2,51,1,0],
             [5,21,1,0],
             [10,11,1,0],
             [20,6,1,0],
             [50,3,1,0],
             [100,2,1,0]]

total = 100

sample_models = ["Cell Cycle Transcription by Coupled CDK and Network Oscillators",
                 "Metabolic Interactions in the Gut Microbiome",
                 "Mammalian Cell Cycle 2006",
                 "T-LGL Survival Network 2011 Reduced Network",
                 "IL-1 Signaling",
                 "Glucose Repression Signaling 2009",
                 "Signaling in Macrophage Activation",
                 "Influenza A Virus Replication Cycle"]

base_directory = "boolmore/benchmarks/benchmark_models"

data_directory = "boolmore/optimization/data"


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

fp = open("boolmore/optimization/iteration_log.csv", "a")
fp.write("sample_model,iter,pop,keep,mix,prob")
for i in range(total + 1):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

for sample_model in sample_models:
    print(sample_model)
    for parameters, base_path, data, start_path in it.product(PRAM_LIST,
                                                    [base_paths[sample_model]],
                                                    data_paths[sample_model],
                                                    start_paths[sample_model]):
        
        prob = PROB

        exps, fixes_list = import_exps(data)
    
        base_primes = sm.format.import_primes(base_path)
        base = Model.import_model(base_primes)
    
        primes = sm.format.import_primes(start_path)
        start = Model.import_model(primes, boolmore.config.id, 1, base)
        start.get_predictions(fixes_list)
        start.get_model_score(exps)

        final, log = ga_main(start, exps, fixes_list,
                            total_iter=parameters[0], per_iter=parameters[1], keep=parameters[2], mix=parameters[3],
                            prob=prob, edge_prob=0,
                            stop_if_max=True, core=6)

        fp = open("boolmore/optimization/iteration_log.csv", "a")

        fp.write(f"{sample_model},{parameters[0]},{parameters[1]},{parameters[2]},{parameters[3]},\"{prob}\",{start.score}")
        for iteration in log:
            # extra commas to make the output csv file neat
            for i in range(int(parameters[1]-parameters[2])):
                fp.write(",")
            fp.write(f"{iteration[1]}")

        # if the genetic algorithm ended early, have the final score appended.
        if len(log) != parameters[0]:
            for i in range(len(log) - parameters[0]):
                # extra commas to make the output csv file neat
                for i in range(int(parameters[1]-parameters[2])):
                    fp.write(",")
                fp.write(f"{log[-1][1]}")

        fp.write("\n")
        fp.close()

