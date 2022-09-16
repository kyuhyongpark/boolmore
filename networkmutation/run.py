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

MODEL = '../PyStableMotifs/models/ABA_full.txt'
# model with added inhibitory edge from PA to ABI2
# MODEL = '../PyStableMotifs/models/ABA_full_with_edge.txt'
# model with the calcium oscillation node
# MODEL = '../PyStableMotifs/models/ABA_calosc.txt'
# model with the rule for KEV fixed and self loop in vacuolar acidification deleted
# MODEL = '../PyStableMotifs/models/ABA_full_fix.txt'
# model with the rule for Ca2osc and CIS changed
# MODEL = '../PyStableMotifs/models/ABA_calosc_cis.txt'

# model with an extra edge
# MODEL = '../../Work/Network Inference/ABA/20220531/edge_7961_gen49.txt'
# model with no extra edges
# MODEL = '../../Work/Network Inference/ABA/20220518/7885_gen48.txt'
# model from GA with calcium oscillation node
# MODEL = '../networkmutation/20220601/best/osc_7546_gen49.txt'
# model from GA with calcium oscillation node and CIS modified
# MODEL = '../networkmutation/20220603/best/osc_cis_8072_gen47.txt'
# model from GA with the pair-significance scoring method
# MODEL = '../networkmutation/_7621_gen48_mod.txt'

# MODEL = '../networkmutation/random1.txt'
# MODEL = '../networkmutation/osc_test1.txt'

# BASE = '../PyStableMotifs/models/ABA_full.txt'

# DATA = 'data.txt'
# DATA = 'data_osc.txt'
# DATA = 'data_pair_signal.txt'
DATA = 'data_pair_significance.txt'

NAME = 'test_crossover'

CONSTRAINT = {
'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'ROS', 'cADPR', 'cGMP'},
'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'K_efflux':('KEV','KOUT'), 'PP2CA':('RCARs',), 'Ca2osc':('CaIM','CIS')},
'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1','K_efflux'), 'Malate':('PEPC', 'AnionEM')},
'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
'possible_source': {'AquaporinPIP2_1',}
}

EDGEPOOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'))
# EDGEPOOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'))

STARTING_ID = 0
STARTING_GEN = 0
ITERATIONS = 5
PER_ITERATION = 10
KEEP = 2
EXPORT_TOP = 1
EXPORT_THRESHOLD = 120
PROB = 0.01
EDGE_PROB = 0.5

offspring = int(PER_ITERATION/KEEP - 1)
config.id = STARTING_ID

# base_primes = sm.format.import_primes(BASE)
# base = Model.import_model(base_primes, given_regulators = None, given_signs = None)
# regulators = base.regulators
# signs = base.signs

print("Loading model . . .")
primes = sm.format.import_primes(MODEL)

print("Model loaded.")
print()

regulators = None
signs = None
n0 = Model.Model.import_model(primes, extra_edges = [], id = config.id, generation = STARTING_GEN, given_regulators = regulators, given_signs = signs)
exps, pert = import_exps(DATA)

n0.get_predictions(pert)
n0.get_model_score(exps)
n0.info()
# print('following contraints', n0.check_constraint(CONSTRAINT))

# ### make one mutated model
# new_model = n0.mutate(CONSTRAINT, EDGEPOOL, PROB, EDGE_PROB)
# new_model.get_predictions(pert)
# new_model.get_model_score(exps)
# new_model.info()
# # new_model.export(NAME)
# # print('following contraints', new_model.check_constraint(CONSTRAINT))

print("- - - - - iteration ", 0, " - - - - -")

iteration = []
iteration.append(n0)
for i in range(PER_ITERATION-1):
    new_model = n0.mutate(CONSTRAINT, EDGEPOOL, PROB, EDGE_PROB)
    new_model.get_predictions(pert)
    new_model.get_model_score(exps)
    iteration.append(new_model)

iteration = sorted(iteration, key=lambda x: x.score, reverse=True)

for model in iteration:
    model.info()
    assert model.check_constraint(CONSTRAINT) == True

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
        new_model = target.mutate(CONSTRAINT, EDGEPOOL, PROB, EDGE_PROB)
        new_model.get_predictions(pert)
        new_model.get_model_score(exps)
        new_iteration.append(new_model)

    new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)

    for model in new_iteration:
        model.info()
        assert model.check_constraint(CONSTRAINT) == True

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
