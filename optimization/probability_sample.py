import os
import re
import itertools as it
import pystablemotifs as sm
import boolmore.config

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

fp = open("boolmore/optimization/probability_log.csv", "a")
fp.write("sample_model,iter,pop,keep,mix,prob")
for i in range(total + 1):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

for sample_model in sample_models:
    print(sample_model)
    for prob, base, data, start_path in it.product(PROB_LIST,
                                              [base_paths[sample_model]],
                                              data_paths[sample_model],
                                              start_paths[sample_model]):

        exps, fixes_list = import_exps(data)
    
        base_primes = sm.format.import_primes(base)
        base = Model.import_model(base_primes)
    
        primes = sm.format.import_primes(start_path)
        start = Model.import_model(primes, boolmore.config.id, 1, base)
        start.get_predictions(fixes_list)
        start.get_model_score(exps)

        final, log = ga_main(start, exps, fixes_list,
                            total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                            prob=prob, edge_prob=0,
                            stop_if_max=True, core=4)


        fp = open("boolmore/optimization/probability_log.csv", "a")

        fp.write(f"{sample_model},{TOTAL_ITERATIONS},{PER_ITERATION},{KEEP},{MIX},\"{prob}\",{start.score}")

        for iteration in log:
            # extra commas to make the output csv file neat
            for i in range(int(total/TOTAL_ITERATIONS)):
                fp.write(",")
            fp.write(f"{iteration[1]}")

        # if the genetic algorithm ended early, have the final score appended.
        if len(log) != TOTAL_ITERATIONS:
            for i in range(TOTAL_ITERATIONS - len(log)):
                # extra commas to make the output csv file neat
                for j in range(int(total/TOTAL_ITERATIONS)):
                    fp.write(",")
                fp.write(f"{log[-1][1]}")
                
        fp.write("\n")
        fp.close()

