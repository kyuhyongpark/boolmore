import datetime

from boolmore.genetic_algorithm import run_ga

JSON = "./data/ABA_A.json"
START_MODEL = "./baseline_models/ABA_GA_base_A.bnet"

overwrite_parameters = {
    "total_iterations" : 100,
    "per_iteration" : 100,
    "keep" : 20,
    "mix" : 0,
    }

for i in range(3, 30):
    RUN_NAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    run_ga(JSON, START_MODEL, RUN_NAME, parameter_dict=overwrite_parameters, core=6, seed=i)