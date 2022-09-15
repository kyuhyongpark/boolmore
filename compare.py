# import two models, and reduce the source nodes
# And then find the nodes whose rules are different

import pystablemotifs as sm
import pyboolnet.trap_spaces
from pprint import pprint


MODEL1 = '../PyStableMotifs/models/ABA_full.txt'
# model with added inhibitory edge from PA to ABI2
# MODEL1 = '../PyStableMotifs/models/ABA_full_with_edge.txt'
# model with the calcium oscillation node
# MODEL1 = '../PyStableMotifs/models/ABA_calosc.txt'
# model with the rule for KEV fixed and self loop in vacuolar acidification deleted
# MODEL1 = '../PyStableMotifs/models/ABA_full_fix.txt'
# model with the rule for Ca2osc and CIS changed
# MODEL1 = '../PyStableMotifs/models/ABA_calosc_cis.txt'

# model with an extra edge
# MODEL2 = '../networkmutation/20220531/edge_7961_gen49.txt'
# model with no extra edges
# MODEL2 = '../networkmutation/20220518/7885_gen48.txt'
# model from GA with calcium oscillation node
# MODEL1 = '../networkmutation/20220601/best/osc_7546_gen49.txt'
# model from GA with calcium oscillation node and CIS modified
# MODEL2 = '../networkmutation/20220603/best/osc_cis_8072_gen47.txt'
# model from GA using pair-significance
MODEL2 = '../networkmutation/_7621_gen48.txt'


# MODEL = '../networkmutation/random1.txt'
# MODEL = '../networkmutation/osc_test1.txt'

# DATA1 = 'data.txt'
# DATA2 = 'data_osc.txt'

primes1 = sm.format.import_primes(MODEL1)
primes2 = sm.format.import_primes(MODEL2)

pprimes1 = pyboolnet.prime_implicants.percolate(primes1, remove_constants = False, copy=True)
# sm.format.pretty_print_prime_rules({k:pprimes1[k] for k in sorted(pprimes1)})

pprimes2 = pyboolnet.prime_implicants.percolate(primes2, remove_constants = False, copy=True)
# sm.format.pretty_print_prime_rules({k:pprimes2[k] for k in sorted(pprimes2)})

# print the different functions
different_nodes = []
only_in_pprimes1 = []
only_in_pprimes2 = []

for node in pprimes1:
    if node not in pprimes2:
        only_in_pprimes1.append(node)
        continue
    if pprimes1[node] == pprimes2[node]:
        pass
    else:
        different_nodes.append(node)

for node in pprimes2:
    if node not in pprimes1:
        only_in_pprimes2.append(node)

for node in different_nodes:
    k = node
    v = pprimes1[node]
    s = k + "* = "
    sl = []
    for c in v[1]:
        sll = []
        for kk,vv in c.items():
            if vv: sli = kk
            else: sli = '!'+kk
            sll.append(sli)
        if len(sll) > 0:
            sl.append(' & '.join(sll))
    if len(sl) > 0:
        s += ' | '.join(sl)
    if v[1]==[]:
        s = k + "* = 0"
    if v[1]==[{}]:
        s = k + "* = 1"
    print('old  :',s)
    k = node
    v = pprimes2[node]
    s = k + "* = "
    sl = []
    for c in v[1]:
        sll = []
        for kk,vv in c.items():
            if vv: sli = kk
            else: sli = '!'+kk
            sll.append(sli)
        if len(sll) > 0:
            sl.append(' & '.join(sll))
    if len(sl) > 0:
        s += ' | '.join(sl)
    if v[1]==[]:
        s = k + "* = 0"
    if v[1]==[{}]:
        s = k + "* = 1"
    print('new:',s)

print('only in pprimes 1', only_in_pprimes1)
print('only in pprimes 2', only_in_pprimes2)

# n1 = m.Network.import_network(primes1, id = 1, generation = 0)
# n2 = m.Network.import_network(primes2, id = 1, generation = 0)
#
# exps1 = m.import_exps(DATA1)
# exps2 = m.import_exps(DATA2)
#
# # print the different predictions
# def get_predictions_str(primes, exps):
#     """
#     Get predictions of perturbations in a string form
#     """
#     predictions = {}
#     for exp in exps:
#         perturbation = {}
#         for fix in exp:
#             perturbation[fix[0]] = fix[1]
#         # print("- - - - - - - - - -")
#         # print("fixed: ", perturbation)
#
#         new_primes = primes.copy()
#         for node in perturbation.keys():
#             assert node in new_primes.keys(), "perturbed node should be in the network"
#             if int(perturbation[node]) == 0:
#                 new_primes[node] = [[{}],[]]
#             else:
#                 new_primes[node] = [[],[{}]]
#
#         tr = pyboolnet.trap_spaces.compute_trap_spaces(new_primes, "min")
#
#         for i in tr:
#             for node in primes.keys():
#                 if node not in i.keys():
#                     i[node] = '?'
#                 else:
#                     i[node] = str(i[node])
#
#         result = {}
#         for node in primes.keys():
#             if node not in result.keys():
#                 result[node] = 0.0
#             for i in tr:
#                 if i[node] == '1':
#                     result[node] += (1.0)
#                 elif i[node] == '?':
#                     result[node] += (0.5)
#             result[node] = str(result[node]) + '/' + str(len(tr))
#
#         predictions[exp] = result
#
#     return predictions
#
# predictions1 = get_predictions_str(primes1, exps1)
# predictions2 = get_predictions_str(primes2, exps2)
#
# only_in_exps1 = {}
# only_in_exps2 = {}
#
# for fix in exps1:
#     if fix not in exps2:
#         nodes = []
#         for result in exps1[fix]:
#             nodes.append(result[0])
#         only_in_exps1[fix] = nodes
#         continue
#     difference = False
#     for result in exps1[fix]:
#         node = result[0]
#         if node not in predictions2[fix]:
#             if fix not in only_in_exps1:
#                 only_in_exps1[fix] = []
#             only_in_exps1[fix].append(node)
#             continue
#         predict_value1 = predictions1[fix][node]
#         predict_value2 = predictions2[fix][node]
#         if predict_value1 != predict_value2:
#             if difference == False:
#                 print("- - - - - - - - - -")
#                 print('fix:', fix)
#                 difference = True
#             print('exps1:', result)
#             print("original model", predictions1[fix][node])
#             print("model with osc", predictions2[fix][node])
#
# for fix in exps2:
#     if fix not in exps1:
#         nodes = []
#         for result in exps2[fix]:
#             nodes.append(result[0])
#         only_in_exps2[fix] = nodes
#         continue
#     for result in exps2[fix]:
#         node = result[0]
#         if node not in predictions1[fix]:
#             if fix not in only_in_exps2:
#                 only_in_exps2[fix] = []
#             only_in_exps2[fix].append(node)
#
# print("- - - - - - - - - - ONLY IN ONE EXPS - - - - - - - - - -")
# for fix in only_in_exps1:
#     print('fix:', fix)
#     for node in only_in_exps1[fix]:
#         print('node', node)
#         print("original model", predictions1[fix][node])
#
# for fix in only_in_exps2:
#     print('fix:', fix)
#     for node in only_in_exps2[fix]:
#         print('node', node)
#         print("model with osc", predictions2[fix][node])
