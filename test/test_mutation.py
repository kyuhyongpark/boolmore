import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from mutation import *
import pystablemotifs as sm
from pyboolnet.external.bnet2primes import bnet_text2primes

# import cProfile, pstats
#
# # Create the code profiler.
# pr = cProfile.Profile()
#
# # Start the code profiler.
# pr.enable()

# ### check manipulation of rr
# rr = '01100100011001000110010001100100011001000110010001100100011001000110010001100100011001000110010001100100011001000110010001100100'
# print('ori_rr : ', rr)
# rr = mutate_rr(rr, 0.05)
# print('mut_rr : ', rr)
# max_rr = get_uni_rr(rr, max = True)
# min_rr = get_uni_rr(rr, max = False)
# print('max_rr : ', max_rr)
# print('min_rr : ', min_rr)

# # check rr2prime
# rr = '1111'
# irr = get_max_irr(rr)
# print(rr)
# print(irr)
# regulators = ('A', 'B')
# signs = '11'
# prime1 = rr2prime(regulators, rr, signs, inverted = False)
# prime2 = rr2prime(regulators, irr, signs, inverted = True)
# print(prime1)
# print(prime2)

# ### basic functions check
# import time
#
# prime = [[{'pHc': 0}, {'ROS': 1}, {'RCARs': 1}, {'PA': 1}], [{'PA': 0, 'RCARs': 0, 'ROS': 0, 'pHc': 1}]]
# print(prime)
# regulators, rr, signs = prime2rr(prime)
# print(regulators, rr, signs)
# rr = mutate_rr(rr, 0.05)
# print(regulators, rr, signs)
#
# print('- - - - old method - - - -')
# s = time.time()
#
# rule = rr2bnet(regulators, rr, signs)
# prime0 = list(bnet_text2primes(rule).values())[0]
#
# e = time.time()
# print('it took', e - s)
# print(prime0)
#
# print('- - - - new method - - - -')
# s = time.time()
#
# irr = get_max_irr(rr)
# prime1 = rr2prime(regulators, rr, signs, inverted = False)
# prime2 = rr2prime(regulators, irr, signs, inverted = True)
#
# e = time.time()
#
# print('it took', e - s)
# print(prime1)
# print(prime2)
#
# print(prime1 == prime2)

### test mutate_rr_constraint

CONSTRAINTS = {
'fixed': {'Ca2osc', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'K_efflux':('KEV','KOUT'), 'PP2CA':('RCARs',)},
'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1','K_efflux'), 'Malate':('PEPC', 'AnionEM')},
'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
'possible_source': {'AquaporinPIP2_1','GEF1_4_10'}
}

regulators = ('ROS', 'RCARs')
rr = '0000'
constraints = CONSTRAINTS
node = 'HAB1'
probability = 0.01

new_rr, modified = mutate_rr_constraint(regulators, rr, constraints, node, probability, bias = 0.5)
print(new_rr)

# ### test add_regulator
# regulators = ('A','B','C')
# rr = '11001100'
# signs = '111'
# new_regulator = 'D'
# new_sign = '1'
#
# added_regulators, added_rr, added_signs = add_regulator(regulators, rr, signs, new_regulator, new_sign)
#
# print(added_regulators, added_rr, added_signs)

# # test delete_regulator
# regulators = ('A','C')
# rr = '1000'
# signs = '11'
# target_regulator = 'C'
#
# deleted_regulators, deleted_rr, deleted_signs = delete_regulator(regulators, rr, signs, target_regulator)
#
# print(deleted_regulators, deleted_rr, deleted_signs)

# ### test mix_models
# rule1 = '''
# A, B | C
# B, B & A
# C, C
# '''
# rule2 = '''
# A, B & C
# B, B
# C, C | !A
# '''
# primes1 = bnet_text2primes(rule1)
# primes2 = bnet_text2primes(rule2)
#
# print(sm.format.primes2bnet(primes1))
# print(sm.format.primes2bnet(primes2))
#
# EDGEPOOL = (('A', 'B', '1'), ('A', 'C', '0'))
# model1 = Network.import_network(primes1, extra_edges = [('A', 'B', '1')], id = 1, generation = 1)
# model2 = Network.import_network(primes2, extra_edges = [('A', 'C', '0')], id = 2, generation = 1)
#
# # model1.info()
# # model2.info()
#
# mix_model = mix_models(model1, model2)
#
# mix_model.info()
# print(sm.format.primes2bnet(mix_model.primes))


# # Stop the profiling.
# pr.disable()
#
# # Print the code profiler.
# ps = pstats.Stats(pr).sort_stats("cumulative")
# ps.print_stats()
