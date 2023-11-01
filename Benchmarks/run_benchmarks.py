import sys
sys.path.insert(0, '../boolmore')

import os
import io
import pystablemotifs as sm

from benchmarks import *
from genetic_algorithm import ga_main
from contextlib import redirect_stdout


REPEAT = 5
DIRECTORY = "benchmark_models"
FILE = "benchmark_log.csv"
N_LIMIT = 30

fp = open(FILE, "a")
fp.write("model_name,N,repeat,start_train,final_train,valid_ids,start_valid,final_valid,")
fp.write(",".join([str(x) for x in range(1,101)]) + "\n")
fp.close()

for model_file in os.listdir(DIRECTORY):
    BASE = model_file

    model_name = BASE.split("/")[-1][:-4]

    if model_name == "Cortical Area Development":
        continue

    if model_name == "Cell Cycle Transcription by Coupled CDK and Network Oscillators":
        continue

    base_primes = sm.format.import_primes(DIRECTORY+"/"+BASE)

    # skip the model if importing the primes fails
    if base_primes == None:
        print(model_name, "- failed to import primes\n")
        continue

    N = len(base_primes)

    if N > N_LIMIT:
        print(model_name, f"- has more than {N_LIMIT} nodes\n")
        continue

    for i in range(REPEAT):
        exps, fixes_list = generate_experiments(base_primes, export=True, file_name=model_name+"_data_"+str(i+1)+".tsv")
        train, valid, valid_ids = train_and_valid(exps, ratio=0.2)

        base = Model.import_model(base_primes)
        base.name = model_name

        start = base.mutate(0.5, 0)
        start.get_predictions(fixes_list)
        start.get_model_score(train)
        start.export(model_name+"_start_"+str(i+1)+".txt")

        start_train = start.score
        start.get_model_score(valid)
        start_valid = start.score

        print("model name:", model_name)
        print("model size", N)
        print("start training score:", start_train)
        print("valid ids:", valid_ids)
        print("start validation score:", start_valid)

        # set a trap and redirect stdout
        trap = io.StringIO()

        with redirect_stdout(trap):
            final, log = ga_main(start, train, fixes_list, export_name="bench_"+model_name+"_"+str(i))

        # # getting redirected output
        # captured_stdout = trap.getvalue()
        # print(captured_stdout)

        final.export(model_name+"_final_"+str(i+1)+".txt")

        final_train = final.score
        final.get_model_score(valid)
        final_valid = final.score

        print("final training score:", final_train)
        print("final validation score:", final_valid)
        print("training score curve: "+",".join([str(x[1]) for x in log]))

        valid_ids_str = "\"" + ",".join([str(x) for x in valid_ids]) + "\""
        lst = [model_name, N, i, start_train, final_train, valid_ids_str, start_valid, final_valid]
        lst.append([x[1] for x in log])
        with open(FILE, mode="a") as fp:
            fp.write(",".join([str(x) for x in lst]) + "\n")
    