import os
import itertools as it

from matplotlib.pylab import f
from pyboolnet.external.bnet2primes import bnet_file2primes

import boolmore.config
from boolmore.experiment import import_exps
from boolmore.genetic_algorithm import ga_main
from boolmore.model import Model

# we compare hierarchy and non-hierarchy
COMPARE = [True, False]

parameters = {
    "total_iter" : 5,
    "per_iter" : 5,
    "keep" : 1,
    "mix" : 0,
    "prob" : 0.2,
    }

sample_models = [
    "Cell Cycle Transcription by Coupled CDK and Network Oscillators",
    "Metabolic Interactions in the Gut Microbiome",
    "Mammalian Cell Cycle 2006",
    "T-LGL Survival Network 2011 Reduced Network",
    "IL-1 Signaling",
    "Glucose Repression Signaling 2009",
    "Signaling in Macrophage Activation",
    "Influenza A Virus Replication Cycle"
    ]

base_directory = "../models"

data_directory = "./data"

result_directory = "./results"


### benchmark runs
base_paths = dict()
start_paths = dict()
train_paths = dict()
valid_paths = dict()

for sample_model in sample_models:

    for filename in os.listdir(base_directory):
        if filename == f"{sample_model}.bnet":
            base_paths[sample_model] = f"{base_directory}/{filename}"

    start_paths[sample_model] = []
    train_paths[sample_model] = []
    valid_paths[sample_model] = []
    for filename in os.listdir(data_directory):
        if filename.startswith(f"{sample_model}_start"):
            start_paths[sample_model].append(f"{data_directory}/{filename}")
        elif filename.startswith(f"{sample_model}_train"):
            train_paths[sample_model].append(f"{data_directory}/{filename}")
        elif filename.startswith(f"{sample_model}_valid"):
            valid_paths[sample_model].append(f"{data_directory}/{filename}")

fp = open("./results/log.csv", "a")
fp.write("sample_model,hierarchy,seed,total_iter,per_iter,keep,mix,prob,")
fp.write("start_t_h_score,start_t_nh_score,final_t_h_score,final_t_nh_score,")
fp.write("start_v_h_score,start_v_nh_score,final_v_h_score,final_v_nh_score\n")
fp.close()

for sample_model in sample_models:
    print(sample_model)
    seed=0
    for start_path, train_data, hierarchy in it.product(start_paths[sample_model], train_paths[sample_model], COMPARE):
        
        if hierarchy:
            seed+=1

        train, fixes_list = import_exps(train_data)
    
        base_primes = bnet_file2primes(base_paths[sample_model])
        base = Model.import_model(base_primes)
    
        primes = bnet_file2primes(start_path)
        start = Model.import_model(primes, boolmore.config.id, 1, base)
        start.get_predictions(fixes_list)
        start.get_model_score(train)

        final, log = ga_main(start, train, fixes_list,
                            total_iter=parameters["total_iter"],
                            per_iter=parameters["per_iter"],
                            keep=parameters["keep"],
                            mix=parameters["mix"],
                            prob=parameters["prob"], edge_prob=0,
                            stop_if_max=True, core=6, seed=seed,
                            hierarchy=hierarchy)

        fp = open("./results/log.csv", "a")

        fp.write(f"""{sample_model},{hierarchy},{seed},""")
        fp.write(f"""{parameters["total_iter"]},{parameters["per_iter"]},""")
        fp.write(f"""{parameters["keep"]},{parameters["mix"]},\"{parameters["prob"]}\",""")

        fp.write(f"{start.score},{start.non_hierarchy_score},{final.score},{final.non_hierarchy_score},")

        # get validation scores
        valid, fixes_list = import_exps(train_data.replace("train", "valid"))
        start.get_predictions(fixes_list)
        start.get_model_score(valid)

        final.get_predictions(fixes_list)
        final.get_model_score(valid)

        fp.write(f"{start.score},{start.non_hierarchy_score},{final.score},{final.non_hierarchy_score}\n")
        fp.close()


    break
