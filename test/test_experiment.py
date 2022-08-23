import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from experiment import *

location = '../networkmutation/data_0822.txt'

data, pert = import_exps_new(location)
for sth in data:
    print(sth)

print(pert)
