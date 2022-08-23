import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from constraint import *
import pystablemotifs as sm

# import cProfile, pstats
#
# # Create the code profiler.
# pr = cProfile.Profile()
#
# # Start the code profiler.
# pr.enable()

# MODEL = '../PyStableMotifs/models/ABA_full.txt'
# model with an extra edge and rule for closure is not modified
MODEL2 = '../networkmutation/20220519/best/7993_gen50.txt'

primes = sm.format.import_primes(MODEL)

n0 = m.Network.import_network(primes, id = 1, generation = 0)

for node in n1.regulators.keys():
    if n1.regulators[node] != n2.regulators[node]:
        print(node)
        print(n1.regulators[node])
        print(n2.regulators[node])




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
# regulators, rr, signs = prime2rr(prime)
# constraints = CONSTRAINT
#
# print(prime)
# print(regulators, rr, signs)

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
