import datetime

from boolmore.genetic_algorithm import run_ga

JSON = "./data/config.json"
START_MODEL = "./data/start.bnet"
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

run_ga(JSON, START_MODEL, RUN_NAME, core=6, seed=0)