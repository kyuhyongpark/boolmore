import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from benchmarks import *
from mutation import *
from Model import *
import config
import pystablemotifs as sm

# BASE = '../PyStableMotifs/models/ABA_full.txt'
# N_EXPS = 425

# BASE = '../cubewalkers/models/Cortical Area Development.txt'
# BASE = '../PyStableMotifs/models/phase_switch.txt'
# BASE = '../PyStableMotifs/models/ABA_reduced.txt'
BASE = '../PyStableMotifs/models/EMT_reduced.txt'

N_EXPS = 40

NAME = 'benchmark'

STARTING_ID = 1
STARTING_GEN = 1

ITERATIONS = 100
PER_ITERATION = 100
KEEP = 20
EXPORT_TOP = 1
EXPORT_THRESHOLD = 10000
PROB = 0.01
EDGE_PROB = 0

config.id = STARTING_ID

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

print(BASE)
print('number of experiments:', N_EXPS)

### Genetic Algorithm
print("- - - - - iteration ", 0, " - - - - -")

iteration = []
iteration.append(starting_model)
for i in range(PER_ITERATION-1):
    new_model = starting_model.mutate(PROB, EDGE_PROB)
    new_model.get_predictions(pert)
    new_model.get_model_score(exps)
    iteration.append(new_model)

iteration = sorted(iteration, key=lambda x: x.score, reverse=True)

average = 0
for i in range(EXPORT_TOP):
    iteration[i].export(NAME, EXPORT_THRESHOLD)
    average += iteration[i].score
average /= EXPORT_TOP
print("average score for top ", EXPORT_TOP, " : ", average)

for i in range(ITERATIONS):
    print("- - - - - iteration ", i+1, " - - - - -")
    new_iteration = []
    # keep the best ones
    for j in range(KEEP):
        new_iteration.append(iteration[j])

    # mix the best ones
    to_be_mixed = sorted(new_iteration, key=lambda x: x.score, reverse=True)
    weights = list(range(1, KEEP+1))
    weights.reverse()
    for j in range(KEEP):
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

    average = 0
    for i in range(EXPORT_TOP):
        new_iteration[i].export(NAME, EXPORT_THRESHOLD)
        average += new_iteration[i].score
    average /= EXPORT_TOP
    print("average score for top ", EXPORT_TOP, " : ", average)

    iteration = new_iteration
