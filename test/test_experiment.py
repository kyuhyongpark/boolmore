import sys
sys.path.insert(0, "./boolmore/boolmore")

from experiment import *

# test data for tsv input, hierarchy scoring
# experimental data input should be a tsv file with 6 columns

# ID: id of the experiment
# e.g. 1

# SCORE: The model will get this score if it agrees with all the experiments in the
#        higher hierarchy.
# e.g. 1.0

# SOURCE: The source nodes, and the values (0 or 1)
# e.g. A=1,B=0,C=0

# PERTURBATION: The nodes, and the intervention (KO or CA)
# e.g. A CA, B KO,C KO

# NODE - the observed node
# e.g. Closure

# VALUE - one of OFF, OFF/Some, Some, Some/ON, ON
# e.g. Some/ON

location = "test_data.txt"

fp = open(location, "w")
fp.write(
"""
ID	Score	Source	Intervention	Observed	Node	Value
1	1.0	ABA=0		Closure	OFF
2	1.0	ABA=1		Closure	ON
3	1.0	ABA=0	SCAB1 KO	Closure	OFF
4	1.0	ABA=1	SCAB1 KO	Closure	ON
5	1.0	ABA=0	SCAB1 CA	Closure	OFF
6	1.0	ABA=1	SCAB1 CA	Closure	Some/ON
# duplicate with different data
#7	1.0	ABA=1	SCAB1 CA	Closure	ON
8	1.0	ABA=0	SCAB1 KO	Actin_Reorganization	OFF
9	1.0	ABA=1	SCAB1 KO	Actin_Reorganization	ON
10	1.0	ABA=0	SCAB1 CA	Actin_Reorganization	OFF
11	1.0	ABA=1	SCAB1 CA	Actin_Reorganization	Some/ON
12	1.0	ABA=1	GEF1_4_10 CA	Closure	Some/ON
13	1.0	ABA=0	PA CA	Closure	Some/ON 
"""
)
fp.close()

data, pert, max_score = import_exps(location)
for sth in data:
    print(sth)

print(pert)

print(max_score)
