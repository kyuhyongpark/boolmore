import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from experiment import *

location = '../networkmutation/data_0822.txt'

data, pert = import_exps_new(location)
for sth in data:
    print(sth)

print(pert)

predictions = {(('ABA', '0'),):{'Closure':0, 'ROS':0},
(('ABA', '1'),):{'Closure':1, 'ROS':1},
(('ABA', '0'),):{'Closure':0, 'ROS':0},
(('ABA', '0'), ('CPK3_21', '0')): {'Closure':0},
(('ABA', '1'), ('CPK3_21', '0')): {'Closure':1}}

print(get_score_new(data, predictions, 0))
