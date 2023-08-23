import sys
sys.path.insert(0, './BoolMoRe/boolmore')

import pystablemotifs as sm
from benchmarks import *
from model import Model

BASE = './BoolMoRe/baseline_models/ABA_full_20230407.txt'
N_EXPS = 505

def test_generate_experiments():

    print("Loading primes . . .")
    base_primes = sm.format.import_primes(BASE)
    print("Primes loaded.\n")

    print("Generating artificial experimental data . . .")
    exps, pert = generate_experiments(base_primes, N_EXPS)
    print("Experimental data generated.\n")

    print("Loading base model . . .")
    base = Model.import_model(base_primes)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    starting_model = base.mutate(0.5, 0)
    starting_model.get_predictions(pert)
    starting_model.get_model_score(exps)
    starting_model.info()
    print()

    assert base.score == N_EXPS, 'base model did not get max score'

test_generate_experiments()