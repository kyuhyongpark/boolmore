# Case study: ABA-induced stomatal closure

Here we demonstrate using `boolmore` to refine the ABA-induced stomatal closure model.
See the manuscript for details ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.14.567002v2)).

## File Descriptions

- `run.py` runs the model refinement process and saves the log and the final model.

- `detailed_report.py` saves tsv files containing the detailed reports of model agreement with the experiments. It also compares the starting model with the final model and analyzes the dynamics of the final model.


### 1. baseline_models
This directory contains the ABA models from different sources, and the baseline models that are refined using the genetic algorithm. See the comments of the bnet files for descriptions of their modifications.

- `ABA_2006_Li.bnet`: Song Li et al. (2006) Predicting essential components of signal transduction networks: a dynamic model of guard cell abscisic acid signaling

- `ABA_2017_Albert.bnet`: Reka Albert et al. (2017) A new discrete dynamic model of ABA-induced stomatal closure predicts key feedback loops

- `ABA_2017_VA-KEV_del.bnet`: The same model as the above but the edge from Vacuolar_Acidification to KEV is deleted.

- `ABA_2018_Waidyarathne.bnet`: Pramuditha Waidyarathne et al. (2018) Boolean Calcium Signalling Model Predicts Calcium Role in Acceleration & Stability of Abscisic Acid-Mediated Stomatal Closure

- `ABA_2019_Mahewshwari.bnet`: Parul Maheshwari et al. (2019) Model-driven discovery of calcium-related protein-phosphatase inhibition in plant guard cell signaling

- `ABA_GA_base_A.bnet`: The baseline A model. See the Supplementary Information of the manuscript for the details.

- `ABA_GA_base_B.bnet`: The baseline B model, which has a node calcium_oscillation to represent the oscillations of cytosolic calcium. See the Supplementary Information of the manuscript for the details.


### 2. data
This directory contains the necessary data to run the genetic algorithm or score the model.

- `data_A.tsv` and `data_B.tsv` contain the perturbation-observation pairs for running the genetic algorithm from the baseline A and baseline B models respectively. `data_A.tsv` is also used for running the genetic algorithm from the VA-KEV_del model.

- `ABA_A.json` and `ABA_B.json` contain the parameters of the run and the constraints of the regulatory functions for the baseline A and baseline B models respectively. `ABA_A_more_edges.json` and `ABA_B_more_edges.json` allow more edges to be added, and have the constraints modified accordingly. `ABA_2017_VA-KEV_del.json` is used for the VA-KEV_del model.

- Other data and json files are only used for scoring the respective old models. 2017_Albert model is scored using `data_A.tsv` and `ABA_A.json`.


### 3. generated_models
This directory contains the GA-refined models and the log of the run.

- `GA1_A.bnet` and `GA1_B.bnet` are the models generated using the standard run, using `ABA_GA_base_A.bnet` and `ABA_GA_base_A.bnet` as baseline and starting models respectively. The model refinement used biological constraints and a pool of extra edges with experimental support specified in `ABA_A.json` and `ABA_B.json`, with experimental data in `data_A.tsv` and `data_B.tsv`.

- For GA0_A and GA0_B, no edges were added during the run by setting the edge addition probability to 0. Models stay consistent to the interaction graph of the baseline models, but may have some edges deleted. All other settings were the same as the GA1. 

- GA2_A and GA2_B are generated using the GA1 models as the starting model, and allowing more hypothetical edges in the pool of extra edges specified in `ABA_A_more_edges.json` and `ABA_B_more_edges.json`. The baseline models and the data were the same as GA1.

- `GA_VA-KEV_del.bnet` is generated using `ABA_2017_VA-KEV_del.bnet` as the baseline and the starting model. Otherwise, it used the same data, constraints, and extra edges as GA1.


### 4. analysis_result
This directory contains the detailed scores of the 14 models, including the old models, the baseline models, and the GA-refined models.
