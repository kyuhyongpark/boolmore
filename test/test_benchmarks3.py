import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from benchmarks import *
from mutation import *
from Model import *
import config
from cana.datasets.bio import load_all_cell_collective_models
import cubewalkers as cw
import pystablemotifs as sm
from pyboolnet.external.bnet2primes import bnet_text2primes

N_threshold = 20
repeats = 3
ITERATIONS = 100
PER_ITERATION = 100
KEEP = 20
EXPORT_TOP = 1
PROB = 0.01
EDGE_PROB = 0
FILE = 'benchmark_20221123_cell_coll.txt'

nets = []
nets = load_all_cell_collective_models()
fp = open(FILE, "w")

for net in nets:
    N = net.Nnodes

    # skip models with more than N_threshold
    if N > N_threshold:
        continue

    print("- - - - - - - - - -")
    fp.write("- - - - - - - - - -")
    bnet = sm.format.booleannet2bnet(cw.conversions.network_rules_from_cana(net))
    primes = bnet_text2primes(bnet)
    if primes == None:
        print(net.name, 'failed\n')
        fp.write(net.name + 'failed\n')
        continue

    print(net.name)
    fp.write(net.name + '\n')
    print('N =', N, '\n')
    fp.write('N = ' + str(N) + '\n')

    base = Model.import_model(primes)

    N_EXPS = 5 * N
    STARTING_ID = 1
    STARTING_GEN = 1
    for repeat in range(repeats):
        config.id = STARTING_ID

        print(f'### {repeat+1} iteration out of {repeats}')
        exps, pert = generate_experiments(primes, N_EXPS)

        print("original model:")
        base.get_predictions(pert)
        base.get_model_score(exps)
        base.info()
        fp.write('base score: ' + str(base.score) + '\n')
        print()

        print("starting model:")
        starting_model = base.mutate(0.5, 0)
        starting_model.get_predictions(pert)
        starting_model.get_model_score(exps)
        starting_model.info()
        fp.write('starting score: ' + str(starting_model.score) + '\n')
        print()

        ### Genetic Algorithm
        # print("- - - - - iteration ", 0, " - - - - -")
        iteration = []
        iteration.append(starting_model)
        for i in range(PER_ITERATION-1):
            new_model = starting_model.mutate(PROB, EDGE_PROB)
            new_model.get_predictions(pert)
            new_model.get_model_score(exps)
            iteration.append(new_model)

        iteration = sorted(iteration, key=lambda x: x.score, reverse=True)

        print('0,', iteration[0].score)
        fp.write('0, ' + str(iteration[0].score) + '\n')

        log = []

        for j in range(ITERATIONS):
            new_iteration = []
            # keep the best ones
            for l in range(KEEP):
                new_iteration.append(iteration[l])

            # mix the best ones
            to_be_mixed = sorted(new_iteration, key=lambda x: x.score, reverse=True)
            weights = list(range(1, KEEP+1))
            weights.reverse()
            for m in range(KEEP):
                mix = random.choices(to_be_mixed, weights=weights, k=2)
                mixed_model = mix_models(mix[0], mix[1])
                mixed_model.get_predictions(pert)
                mixed_model.get_model_score(exps)
                new_iteration.append(mixed_model)

            # mutate the best ones
            weights = list(range(1, 2*KEEP+1))
            weights.reverse()
            new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)
            targets = random.choices(new_iteration, weights=weights, k=PER_ITERATION-2*KEEP)
            for target in targets:
                new_model = target.mutate(PROB, EDGE_PROB)
                new_model.get_predictions(pert)
                new_model.get_model_score(exps)
                new_iteration.append(new_model)

            new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)

            print(str(j+1) + ', ' + str(new_iteration[0].score))
            fp.write(str(j+1) + ', ' + str(new_iteration[0].score) + '\n')
            log.append(new_iteration[0].score)

            if new_iteration[0].score == N_EXPS:
                print('Top score achieved at '+ str(j))
                fp.write('Top score achieved at '+ str(j) + '\n')
                break
            elif len(log) > 10 and log[-1] == log[-10]:
                print('stagnated')
                fp.write('stagnated\n')
                break

            iteration = new_iteration
