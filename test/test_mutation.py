import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from mutation import *
import pystablemotifs as sm

# import cProfile, pstats
#
# # Create the code profiler.
# pr = cProfile.Profile()
#
# # Start the code profiler.
# pr.enable()

# # check manipulation of rr
# rr = '01100100011001000110010001100100011001000110010001100100011001000110010001100100011001000110010001100100011001000110010001100100'
# print('ori_rr : ', rr)
# rr = mutate_rr(rr, 0.05)
# print('mut_rr : ', rr)
# max_rr = get_uni_rr(rr, max = True)
# min_rr = get_uni_rr(rr, max = False)
# print('max_rr : ', max_rr)
# print('min_rr : ', min_rr)
# max_rr2 = get_uni_rr2(rr, max = True)
# min_rr2 = get_uni_rr2(rr, max = False)
# print('max_rr2: ', max_rr2)
# print('min_rr2: ', min_rr2)
#
# print(max_rr == max_rr2)
# print(min_rr == min_rr2)

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

# # basic functions check
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


# # constraint check
MODEL = '../PyStableMotifs/models/ABA_full.txt'
CONSTRAINT = {
'fixed': {'Ca2_ATPase', 'Ca2c', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'ROS', 'cADPR', 'cGMP'},
'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'K_efflux':('KEV', 'KOUT'), 'PP2CA':('RCARs',)},
'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1','K_efflux'), 'Malate':('PEPC', 'AnionEM')},
'group': {'PA':(('PC', 'PLDalpha'),('PC', 'PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2', 'Sph'),)},
'possible_source': {'AquaporinPIP2_1',}
}
#
print("Loading network . . .")
primes = sm.format.import_primes(MODEL)
# #
# print("Network loaded.")
# print()
#
# nodes = []
# for condition in CONSTRAINT:
#     for node in CONSTRAINT[condition]:
#         assert node in primes, "The node does not exist"
#         nodes.append(node)
#
# for node in nodes:
#     print(node)
#
#     prime = primes[node]
#     regulators, rr, signs = prime2rr(prime)
#     constraints = CONSTRAINT
#     mutated_rr, modified = mutate_rr_constraint(regulators, rr, constraints, node, 0.5)
#
#     # print(mutated_rr)
#     # print(modified)
#
#     if not modified:
#         mutated_prime = prime
#     else:
#         mutated_prime = rr2prime(regulators, mutated_rr, signs, inverted = False)
#         print(prime)
#         print(mutated_prime)

for node in primes:
    if node in CONSTRAINT['fixed']:
        continue
    prime = primes[node]
    regulators, rr, signs = prime2rr(prime)
    print(node, len(regulators))

# # test add_regulator
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

# # Stop the profiling.
# pr.disable()
#
# # Print the code profiler.
# ps = pstats.Stats(pr).sort_stats("cumulative")
# ps.print_stats()
