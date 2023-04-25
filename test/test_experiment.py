import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from experiment import *

location = 'networkmutation/test/data_test_20230424.txt'

data, pert = import_exps(location)
for sth in data:
    print(sth)

print(pert)
