import sys
sys.path.insert(0, "../boolmore")

import pystablemotifs as sm
from benchmarks import *
from model import Model
from experiment import import_exps
from genetic_algorithm import ga_main

import io
from contextlib import redirect_stdout

BASE = "../demo/CAD.txt"
DATA = "test_bench_data.tsv"
START = "test_bench_start.txt"

def test_generate_experiments():

    base_primes = sm.format.import_primes(BASE)

    exps1, pert1 = generate_experiments(base_primes, export=True, file_name=DATA)

    base = Model.import_model(base_primes)
    base.get_predictions(pert1)
    base.get_model_score(exps1)

    assert base.score == base.max_score, "base model did not get max score"

    start_model = base.mutate(0.5, 0)
    start_model.get_predictions(pert1)
    start_model.get_model_score(exps1)
    start_model.export(START)

    base_score_1 = base.score
    start_score_1 = start_model.score

    exps2, pert2 = import_exps(DATA)
    assert exps1 == exps2
    assert pert1 == pert2

    base.get_predictions(pert2)
    base.get_model_score(exps2)

    start_model.get_predictions(pert2)
    start_model.get_model_score(exps2)

    base_score_2 = base.score
    start_score_2 = start_model.score
    assert base_score_1 == base_score_2, "base model did not get same score"
    assert start_score_1 == start_score_2, "starting model did not get same score"

def test_train_and_valid():
    exps, pert = import_exps(DATA)
    train_1, valid_1, valid_ids_1 = train_and_valid(exps, ratio=0.2)

    train_2, valid_2, valid_ids_2 = train_and_valid(exps, valid_ids=valid_ids_1)
    assert train_1 == train_2
    assert valid_1 == valid_2
    assert valid_ids_1 == valid_ids_2

    base_primes = sm.format.import_primes(BASE)
    base = Model.import_model(base_primes)
    base.get_predictions(pert)

    start_primes = sm.format.import_primes(START)
    start_model = Model.import_model(start_primes)
    start_model.get_predictions(pert)

    base.get_model_score(train_1)
    base.info()
    start_model.get_model_score(train_1)
    start_model.info()

    base.get_model_score(valid_1)
    base.info()
    start_model.get_model_score(valid_1)
    start_model.info()

def test_benchmark_run():

    model_name = BASE.split("/")[-1][:-4]

    base_primes = sm.format.import_primes(BASE)

    exps, fixes_list = generate_experiments(base_primes, export=True, file_name=model_name+"_data.tsv")
    train, valid, valid_ids = train_and_valid(exps, ratio=0.2)

    base = Model.import_model(base_primes)
    base.name = model_name

    start = base.mutate(0.5, 0)
    start.get_predictions(fixes_list)
    start.get_model_score(train)
    start.export(model_name+"_start.txt")

    print("model name:",model_name)
    print("start training score:",start.score)

    start.get_model_score(valid)
    print("valid ids:", valid_ids)
    print("start validation score:",start.score)

    # set a trap and redirect stdout
    trap = io.StringIO()

    with redirect_stdout(trap):
        final, log = ga_main(start, train, fixes_list, export_name="test_bench", per_iter=10, keep=2, prob=0.2)

    # # getting redirected output
    # captured_stdout = trap.getvalue()
    # print(captured_stdout)

    print("final training score:",final.score)
    final.get_model_score(valid)
    print("final validation score:",final.score)

    print("training score curve: "+",".join([str(x[1]) for x in log]))

test_generate_experiments()
test_train_and_valid()
test_benchmark_run()