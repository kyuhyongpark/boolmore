# Generate scrambled ABA models for optimization

import pystablemotifs as sm
import json

from boolmore.experiment import import_exps
from boolmore.model import Model

# take ABA model specific data from the json file
json_file = "boolmore/case_study/data/ABA_2017.json"

f = open(json_file)
json_dict = json.load(f)

DATA:str = json_dict["data"]
BASE:str = json_dict["base"]
DEFAULT_SOURCES = json_dict["default_sources"]
CONSTRAINTS = json_dict["constraints"]
EDGE_POOL = json_dict["edge_pool"]

exps, fixes_list = import_exps(DATA)

base_primes = sm.format.import_primes(BASE)
base = Model.import_model(base_primes, constraints=CONSTRAINTS, edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)

# generate scrambled models
for i in range(25):
    start = base.mutate(0.5, 0)

    start.get_predictions(fixes_list)
    start.get_model_score(exps)
    
    start.export("boolmore/optimization/data/ABA_scramble_"+str(i+1)+".txt")