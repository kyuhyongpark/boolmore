# Generate scrambled ABA models for optimization
# run with the location of this script as the working directory

import json

from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.experiment import import_exps
from boolmore.model import Model

# take ABA model specific data from the json file
json_file = "../../../case_study/ABA/data/ABA_2017.json"
data_loc = "../../../case_study/ABA/data/data_20230925.tsv"
base_loc = "../../../case_study/ABA/baseline_models/ABA_GA_base_A.bnet"

f = open(json_file)
json_dict = json.load(f)

DEFAULT_SOURCES = json_dict["default_sources"]
CONSTRAINTS = json_dict["constraints"]
EDGE_POOL = json_dict["edge_pool"]

exps, fixes_list = import_exps(data_loc)

base_primes = bnet_file2primes(base_loc)
base = Model.import_model(base_primes, constraints=CONSTRAINTS, edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)

# generate scrambled models
for i in range(25):
    start = base.mutate(0.5, 0, seed=i)

    start.get_predictions(fixes_list)
    start.get_model_score(exps)
    
    start.export(f"../data/ABA_scramble_{i+1}.bnet")