import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from constraint import *
from mutation import *
import pystablemotifs as sm

# import cProfile, pstats
#
# # Create the code profiler.
# pr = cProfile.Profile()
#
# # Start the code profiler.
# pr.enable()

# # test constraint on real model
# MODEL = '../PyStableMotifs/models/ABA_full.txt'
#
# CONSTRAINT = {
# 'fixed': {},
# 'regulate': {'ABI1':('RCARs')},
# 'necessary' : {},
# 'group': {'PA':(('PC', 'PLDalpha'),('PC', 'PLDdelta'),('DAG','DAGK'))}
# }
#
# print("Loading network . . .")
# primes = sm.format.import_primes(MODEL)
# #
# print("Network loaded.")
# print()
#
# node = 'PA'
# print(node)
# # nodes = ('pHc', 'PA')
# groups = CONSTRAINT['group'][node]
# print(groups)
# # prime = [[{'pHc': 0}, {'ROS': 1}, {'RCARs': 1}, {'PA': 1}], [{'PA': 0, 'RCARs': 0, 'ROS': 0, 'pHc': 1}]]
# prime = primes[node]
# regulators, rr, signs = m.prime2rr(prime)
# constraints = CONSTRAINT
#
# print(prime)
# print(regulators, rr, signs)

# # test constraint on toy example
# node = 'X'
# signs = '1111'
# rrs = {'X':'1100001111000011'}
# regulatorss = {'X': ('A','B','C','D')}
# constraints = {'group': {'X':(('A','B'),)}, 'necessary':{'X':('C',)}}
# # constraints = {'group': {'X':(('A','B'),('A','C'),('B','C'))}}
#
# rr = rrs[node]
# regulators = regulatorss[node]
# groups = constraints['group'][node]
# print('rr:', rr)
# print('regulators:',regulators)
# print('groups:', groups)
#
# group_rr, group_regulators = rr2group_rr(regulators, rr, groups)
# print('group_rr:',group_rr)
#
# mutated_group_rr = mutate_rr(group_rr, 0.5)
# print('mutated_group_rr:',mutated_group_rr)
#
# mutated_rr = group_rr2rr(regulators, mutated_group_rr, groups, group_regulators)
# print('mutated_rr:',mutated_rr)
# print(rr2prime(regulators, mutated_rr, signs))
#
# if node in constraints['necessary']:
#     for necc in constraints['necessary'][node]:
#         mutated_rr = impose_necessary(regulators, mutated_rr, necc)
# print('mutated_rr:',mutated_rr)
# print(rr2prime(regulators, mutated_rr, signs))


# # Stop the profiling.
# pr.disable()
#
# # Print the code profiler.
# ps = pstats.Stats(pr).sort_stats("cumulative")
# ps.print_stats()
