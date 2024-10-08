import datetime

from boolmore.genetic_algorithm import run_ga

JSON = "./data/ABA_2017.json"
START_MODEL = "./baseline_models/ABA_GA_base_A.txt"
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

run_ga(JSON, START_MODEL, RUN_NAME, core=6, seed=0)