from joblib import Parallel, delayed
import time
import pyboolnet.trap_spaces
import pystablemotifs as sm
from boolmore.model import Model


bnet = "boolmore/benchmarks/benchmark_models/MAPK Cancer Cell Fate Network.txt"
# bnet = "boolmore/case_study/baseline_models/ABA_GA_base_A.txt"

repeat = 100


def get_min_trap_pyboolnet(model):
    tr = pyboolnet.trap_spaces.compute_trap_spaces(model.primes, "min")
    return model.id

def mutate_and_min(model):
    mutated = model.mutate(0.5, 0.0)
    tr = pyboolnet.trap_spaces.compute_trap_spaces(mutated.primes, "min")
    return mutated.id

primes = sm.format.import_primes(bnet)

base = Model.import_model(primes)

### mutate first and get minimal trap spaces - linear
start = time.time()
lst = []
for i in range(repeat):
    lst.append(base.mutate(0.5, 0.0))

results = []
for model in lst:
    results.append(get_min_trap_pyboolnet(model))
print(results)
print("mutate first and min_trap, linear")
print(time.time() - start)

### mutate first and get minimal trap spaces - parallel
start = time.time()
lst = []
for i in range(repeat):
    lst.append(base.mutate(0.5, 0.0))

results = Parallel(n_jobs=6)(delayed(get_min_trap_pyboolnet)(model) for model in lst)
print(results)
print("mutate first and min_trap, parallel")
print(time.time() - start)

### mutate and get minimal trap spaces
start = time.time()
results = []
for i in range(repeat):
    results.append(mutate_and_min(base))
print(results)
print("mutate and min_trap, linear")
print(time.time() - start)

start = time.time()
results = Parallel(n_jobs=6)(delayed(mutate_and_min)(base) for i in range(repeat))
print(results)
print("mutate and min_trap, parallel")
print(time.time() - start)