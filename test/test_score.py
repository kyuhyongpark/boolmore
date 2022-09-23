import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from experiment import *
from score import *

location = '../networkmutation/data_0923.txt'

data, pert = import_exps(location)
# for sth in data:
#     print(sth)
#
# print(pert)

predictions = {(('ABA', '0'),):{'Closure':0},
(('ABA', '1'),):{'Closure':1},
(('ABA', '0'), ('SCAB1', '0')): {'Closure':0, 'Actin_Reorganization':0},
(('ABA', '1'), ('SCAB1', '0')): {'Closure':1, 'Actin_Reorganization':1},
(('ABA', '0'), ('SCAB1', '1')): {'Closure':0, 'Actin_Reorganization':0},
(('ABA', '1'), ('SCAB1', '1')): {'Closure':1, 'Actin_Reorganization':0.2},
(('ABA', '1'), ('GEF1_4_10','1')): {'Closure':1},
(('ABA', '0'), ('PA','1')): {'Closure':0.2}}

print(get_score(data, predictions, []))
