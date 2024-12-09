# generate data for optimization
# run with the location of this script as the working directory

from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.model import Model
from boolmore.benchmark import generate_experiments

base_directory = "../../models"

sample_models = ["Cell Cycle Transcription by Coupled CDK and Network Oscillators",
                 "Metabolic Interactions in the Gut Microbiome",
                 "Mammalian Cell Cycle 2006",
                 "T-LGL Survival Network 2011 Reduced Network",
                 "IL-1 Signaling",
                 "Glucose Repression Signaling 2009",
                 "Signaling in Macrophage Activation",
                 "Influenza A Virus Replication Cycle"]

data_directory = "../data"

### Generate data
for model_name in sample_models:
    base_primes = bnet_file2primes(f"{base_directory}/{model_name}.bnet")
    for i in range(5):
        exps, fixes_list = generate_experiments(base_primes, seed = i,
                                                export=True, file_name=f"../data/{model_name}_data_{i+1}.tsv")
        
        base = Model.import_model(base_primes)
        
        start = base.mutate(0.5, 0, seed=i)

        start.get_predictions(fixes_list)
        start.get_model_score(exps)
        
        start.export(f"../data/{model_name}_start_{i+1}.bnet")