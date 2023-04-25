import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from experiment import *

location = 'networkmutation/test/test_data_20230425.txt'

data, pert = import_exps(location)
for sth in data:
    print(sth)

print(pert)
