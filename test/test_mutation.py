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

# test mix_rr
rr1 = '0101'
rr2 = '1111'
print(mix_rr(rr1, rr2))

# # Stop the profiling.
# pr.disable()
#
# # Print the code profiler.
# ps = pstats.Stats(pr).sort_stats("cumulative")
# ps.print_stats()
