# run with the location of this script as the working directory

import os
import re
import itertools as it

from pyboolnet.external.bnet2primes import bnet_file2primes

import boolmore.config
from boolmore.experiment import import_exps
from boolmore.model import Model
from boolmore.genetic_algorithm import ga_main

PROB = 0.2
TOTAL_ITERATIONS = 20
PER_ITERATION = 5
KEEP = 2
MIX_LIST = [0,
            1,
            2,
            3,
            4,
            5]


total = TOTAL_ITERATIONS * PER_ITERATION


sample_models = ["Cell Cycle Transcription by Coupled CDK and Network Oscillators",
                 "Metabolic Interactions in the Gut Microbiome",
                 "Mammalian Cell Cycle 2006",
                 "T-LGL Survival Network 2011 Reduced Network",
                 "IL-1 Signaling",
                 "Glucose Repression Signaling 2009",
                 "Signaling in Macrophage Activation",
                 "Influenza A Virus Replication Cycle"]

base_directory = "../../models"

data_directory = "../data"

results_directory = "../results"


### benchmark runs
base_paths = dict()
data_paths = dict()
start_paths = dict()

for sample_model in sample_models:
    for filename in os.listdir(base_directory):
        if filename == sample_model+".bnet":
            base_paths[sample_model] = f"{base_directory}/{filename}"

    data_paths[sample_model] = []
    start_paths[sample_model] = []
    for filename in os.listdir(data_directory):
        if filename.startswith(sample_model):
            if filename.endswith(".tsv"):
                data_paths[sample_model].append(f"{data_directory}/{filename}")
            elif re.match(r".+_start_", filename):
                start_paths[sample_model].append(f"{data_directory}/{filename}")

fp = open(f"{results_directory}/mix_log.csv", "a")
fp.write("sample_model,seed,iter,pop,keep,mix,prob")
for i in range(total + 1):
    fp.write(f",{i}")
fp.write("\n")
fp.close()

for sample_model in sample_models:
    print(sample_model)
    seed = 0
    for mix, data, start_path in it.product(MIX_LIST, data_paths[sample_model], start_paths[sample_model]):
        
        MIX = mix

        boolmore.config.id = 0
        seed += 1

        exps, fixes_list = import_exps(data)
    
        base_primes = bnet_file2primes(base_paths[sample_model])
        base = Model.import_model(base_primes)
    
        primes = bnet_file2primes(start_path)
        start = Model.import_model(primes, boolmore.config.id, 1, base)
        start.get_predictions(fixes_list)
        start.get_model_score(exps)

        final, log = ga_main(start, exps, fixes_list,
                            total_iter=TOTAL_ITERATIONS, per_iter=PER_ITERATION, keep=KEEP, mix=MIX,
                            prob=PROB, edge_prob=0,
                            stop_if_max=True, core=6, seed=seed)

        fp = open(f"{results_directory}/mix_log.csv", "a")

        fp.write(f"{sample_model},{seed},{TOTAL_ITERATIONS},{PER_ITERATION},{KEEP},{MIX},\"{PROB}\",{start.score}")
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
