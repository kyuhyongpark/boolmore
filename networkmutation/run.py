from mutation import *
import pystablemotifs as sm
from pprint import pprint
import itertools
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

# MODEL = '../networkmutation/random3.txt'
# MODEL = '../networkmutation/osc_test1.txt'

# BASE = '../PyStableMotifs/models/ABA_full.txt'

# DATA = 'data.txt'
# DATA = 'data_osc.txt'
# DATA = 'data_pair_signal.txt'
DATA = 'data_pair_significance.txt'


NAME = 'test_crossover'
# NAME = 'osc_cis'

CONSTRAINT = {
'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'ROS', 'cADPR', 'cGMP'},
'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'K_efflux':('KEV','KOUT'), 'PP2CA':('RCARs',), 'Ca2osc':('CaIM','CIS')},
'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1','K_efflux'), 'Malate':('PEPC', 'AnionEM')},
'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
'possible_source': {'AquaporinPIP2_1',}
}
# CONSTRAINT = {
# 'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'ROS', 'cADPR', 'cGMP'},
# 'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'K_efflux':('KEV','KOUT'), 'PP2CA':('RCARs',), 'CIS':('CaIM',)},
# 'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1','K_efflux'), 'Malate':('PEPC', 'AnionEM')},
# 'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
# 'possible_source': {'AquaporinPIP2_1',}
# }

EDGEPOOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'))
# EDGEPOOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'))

STARTING_ID = 0
STARTING_GEN = 0
GENERATIONS = 100
PER_GENERATION = 100
KEEP = 20
EXPORT_TOP = 1
EXPORT_THRESHOLD = 120
PROB = 0.01
EDGE_PROB = 0.0

offspring = int(PER_GENERATION/KEEP - 1)

config.id = STARTING_ID

# base_primes = sm.format.import_primes(BASE)
# base = Network.import_network(base_primes, given_regulators = None, given_signs = None)
# regulators = base.regulators
# signs = base.signs

print("Loading network . . .")
primes = sm.format.import_primes(MODEL)

print("Network loaded.")
print()

regulators = None
signs = None
n0 = Network.import_network(primes, extra_edges = [], id = config.id, generation = STARTING_GEN, given_regulators = regulators, given_signs = signs)
exps, pert = import_exps_new(DATA)
# exps = import_exps(DATA)


n0.get_predictions(pert)
# n0.get_predictions(exps)
n0.get_network_score(exps)
n0.info()
# print('following contraints', n0.check_constraint(CONSTRAINT))

# make one mutated network
# new_network = n0.mutate(CONSTRAINT, EDGEPOOL, PROB, EDGE_PROB)
# new_network.get_predictions(pert)
# new_network.get_network_score(exps)
# new_network.info()
# print('following contraints', new_network.check_constraint(CONSTRAINT))
# new_network.export(NAME)


print("- - - - - generation ", 0, " - - - - -")

generation = []
generation.append(n0)
for i in range(PER_GENERATION-1):
    new_network = n0.mutate(CONSTRAINT, EDGEPOOL, PROB, EDGE_PROB)
    new_network.get_predictions(pert)
    new_network.get_network_score(exps)
    generation.append(new_network)

generation = sorted(generation, key=lambda x: x.score, reverse=True)


for network in generation:
    network.info()
    assert network.check_constraint(CONSTRAINT) == True

average = 0
for i in range(EXPORT_TOP):
    generation[i].export(NAME, EXPORT_THRESHOLD)
    average += generation[i].score
average /= EXPORT_TOP
print("average score for top ", EXPORT_TOP, " : ", average)

for i in range(GENERATIONS):
    print("- - - - - generation ", i+1, " - - - - -")
    new_generation = []
    to_be_mixed = []
    for j in range(KEEP):
        new_generation.append(generation[j])
        to_be_mixed.append(generation[j])
        # # New offspring
        # for k in range(offspring):
        #     new_network = generation[j].mutate(CONSTRAINT, EDGEPOOL, 0.01, 0.1)
        #     new_network.get_predictions(pert)
        #     new_network.get_network_score(exps)
        #     new_generation.append(new_network)

    for j in range(PER_GENERATION-KEEP):
        mix = random.choices(to_be_mixed, k=2)
        # print("mixing", mix[0].id, "and", mix[1].id)
        mixed_model = mix_models(mix[0], mix[1], CONSTRAINT, PROB)
        mixed_model.get_predictions(pert)
        mixed_model.get_network_score(exps)
        new_generation.append(mixed_model)

    new_generation = sorted(new_generation, key=lambda x: x.score, reverse=True)

    for network in new_generation:
        network.info()
        assert network.check_constraint(CONSTRAINT) == True

    average = 0
    for i in range(EXPORT_TOP):
        new_generation[i].export(NAME, EXPORT_THRESHOLD)
        average += new_generation[i].score
    average /= EXPORT_TOP
    print("average score for top ", EXPORT_TOP, " : ", average)

    generation = new_generation

# # Stop the profiling.
# pr.disable()
#
# # Print the code profiler.
# ps = pstats.Stats(pr).sort_stats("cumulative")
# ps.print_stats()
