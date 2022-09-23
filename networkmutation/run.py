import pystablemotifs as sm
from Model import *
from experiment import *
from mutation import *
import config

# import cProfile, pstats
# # Create the code profiler.
# pr = cProfile.Profile()
#
# # Start the code profiler.
# pr.enable()

# DATA = 'data.txt'
# DATA = 'data_osc.txt'
# DATA = 'data_pair_signal.txt'
DATA = 'data_pair_significance.txt'

CONSTRAINTS = {
'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'ROS', 'cADPR', 'cGMP'},
'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'K_efflux':('KEV','KOUT'), 'PP2CA':('RCARs',), 'Ca2osc':('CaIM','CIS')},
'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1','K_efflux'), 'Malate':('PEPC', 'AnionEM')},
'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
'possible_source': {'AquaporinPIP2_1',}
}

### Same edge with different sign is not yet allowed
EDGE_POOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'))
# EDGE_POOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'))

BASE = '../PyStableMotifs/models/ABA_full.txt'

# MODEL = '../PyStableMotifs/models/ABA_full.txt'
# model with added inhibitory edge from PA to ABI2
# MODEL = '../PyStableMotifs/models/ABA_full_with_edge.txt'
# model with the calcium oscillation node
# MODEL = '../PyStableMotifs/models/ABA_calosc.txt'
# model with the rule for KEV fixed and self loop in vacuolar acidification deleted
# MODEL = '../PyStableMotifs/models/ABA_full_fix.txt'
# model with the rule for Ca2osc and CIS changed
# MODEL = '../PyStableMotifs/models/ABA_calosc_cis.txt'

# GA model with an extra edge
MODEL = '../../Work/Network Inference/ABA/20220531/edge_7961_gen49.txt'
# GA model with no extra edges
# MODEL = '../../Work/Network Inference/ABA/20220518/7885_gen48.txt'
# GA model with calcium oscillation node
# MODEL = '../networkmutation/20220601/best/osc_7546_gen49.txt'
# GA model with calcium oscillation node and CIS modified
# MODEL = '../networkmutation/20220603/best/osc_cis_8072_gen47.txt'
# GA model with the pair-significance scoring method
# MODEL = '../networkmutation/_7621_gen48_mod.txt'

# MODEL = '../networkmutation/random1.txt'
# MODEL = '../networkmutation/random4.txt'
# MODEL = '../networkmutation/osc_test1.txt'

NAME = 'from0531'

STARTING_ID = 1
STARTING_GEN = 49

ITERATIONS = 50
PER_ITERATION = 100
KEEP = 20
EXPORT_TOP = 1
EXPORT_THRESHOLD = 123
PROB = 0.01
EDGE_PROB = 0.1

config.id = STARTING_ID

print("Loading experimental data . . .")
exps, pert = import_exps(DATA)
print("Experimental data loaded.\n")

print("Loading base model . . .")
base_primes = sm.format.import_primes(BASE)
base = Model.Model.import_model(base_primes, CONSTRAINTS, EDGE_POOL)
print("Base model loaded.")
base.get_predictions(pert)
base.get_model_score(exps)
base.info()
print()

print("Loading starting model . . .")
primes = sm.format.import_primes(MODEL)
n1 = Model.Model.import_model(primes, CONSTRAINTS, EDGE_POOL, id=config.id, generation=STARTING_GEN, base=base)
print("Starting model loaded.")
n1.get_predictions(pert)
n1.get_model_score(exps)
n1.info()
print()

# ### make one mutated model
# new_model = n1.mutate(PROB, EDGE_PROB)
# new_model.get_predictions(pert)
# new_model.get_model_score(exps)
# new_model.info()
# new_model.export(NAME)
#
# from pyboolnet.prime_implicants import primes_are_equal
# print(primes_are_equal(n1.primes,new_model.primes))

## Genetic Algorithm
print("- - - - - iteration ", 0, " - - - - -")

iteration = []
iteration.append(n1)
for i in range(PER_ITERATION-1):
    new_model = n1.mutate(PROB, EDGE_PROB)
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
        print("mixing", mix[0].score, "and", mix[1].score)
        mixed_model = mix_models(mix[0], mix[1])
        mixed_model.get_predictions(pert)
        mixed_model.get_model_score(exps)
        print("which became", mixed_model.score)
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

# # Stop the profiling.
# pr.disable()
#
# # Print the code profiler.
# ps = pstats.Stats(pr).sort_stats("cumulative")
# ps.print_stats()
