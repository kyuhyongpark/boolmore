# generate data for optimization
# only the data for the large models are generated.
# for the small models, use data pre-generated for the benchmark.

import pystablemotifs as sm

from boolmore.model import Model
from boolmore.benchmarks import generate_experiments

base_directory = "boolmore/benchmarks/benchmark_models"

sample_models = ["IL-1 Signaling",
                 "Glucose Repression Signaling 2009",
                 "Signaling in Macrophage Activation",
                 "Influenza A Virus Replication Cycle"]

data_directory = "boolmore/optimization/data"

### Generate data
for model_name in sample_models:
    base_primes = sm.format.import_primes(base_directory + "/" + model_name + ".txt")
    for i in range(5):
        exps, fixes_list = generate_experiments(base_primes, export=True, file_name="boolmore/optimization/data/"+model_name+"_data_"+str(i+1)+".tsv")
        
        base = Model.import_model(base_primes)
        
        start = base.mutate(0.5, 0)

        start.get_predictions(fixes_list)
        start.get_model_score(exps)
        
        start.export("boolmore/optimization/data/"+model_name+"_start_"+str(i+1)+".txt")