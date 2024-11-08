# generate data for hierarchy scoring comparison

from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.model import Model
from boolmore.benchmark import generate_experiments, train_and_valid, export_exps

base_directory = "../models"

sample_models = ["Cell Cycle Transcription by Coupled CDK and Network Oscillators",
                 "Metabolic Interactions in the Gut Microbiome",
                 "Mammalian Cell Cycle 2006",
                 "T-LGL Survival Network 2011 Reduced Network",
                 "IL-1 Signaling",
                 "Glucose Repression Signaling 2009",
                 "Signaling in Macrophage Activation",
                 "Influenza A Virus Replication Cycle"]

data_directory = "./data"

### Generate data
for model_name in sample_models:

    base_primes = bnet_file2primes(f"{base_directory}/{model_name}.bnet")
    
    for i in range(5):
        exps, fixes_list = generate_experiments(base_primes, seed=i)
        
        train, valid, valid_ids = train_and_valid(exps, ratio=0.2, seed=i)

        export_exps(base_primes, train, f"{data_directory}/{model_name}_train_{i+1}.tsv")
        export_exps(base_primes, valid, f"{data_directory}/{model_name}_valid_{i+1}.tsv")

        base = Model.import_model(base_primes)
        
        start = base.mutate(0.5, 0, seed=i)
        
        start.export(f"{data_directory}/{model_name}_start_{i+1}.bnet", details=False)