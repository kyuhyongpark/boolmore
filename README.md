# boolmore
Boolean Model Refiner

The algorithm takes an existing Boolean model and refines it to fit better with the given experimental results. Hundreds or thousands of Boolean models are explored in the process, each being consistent with the interaction graph of the starting model and any mechanistic constraints given (such as a documented enzyme-substrate pair). The score (or fitness) of each model is determined by comparing its attractors with the experimental data for every perturbation (fixed node state, such as gene KO).  
We showcase the strength of our algorithm by a case study on a plant signaling model. After several hours of automatic refinement, the fittest models recapture and surpass the accuracy gain achieved over 10 years of manual revision and provide new, testable predictions.

## Install
```
pip install git+https://github.com/kyuhyongpark/boolmore
```
<details>
   <summary>Trouble shooting for Mac users</summary>
   <br>
   Installation of boolmore automatically installs PyBoolNet.<br>
   For Mac Users, PyBoolNet may not properly install gringo, one of PyBoolNet's requisites.<br>
   <br>
   If the below example does not run properly,<br>
   ones on a mac should use homebrew to install 

   ```
   brew install clingo
   ```

   then one may need to overwrite the libraries for pyboolnet to work. For example,

   ```
   ln -s /usr/local/bin/gringo /Users/.../python3.11/site-packages/pyboolnet/binaries/gringo-4.4.0/gringo_mac64
   ```
   Here /usr/local/bin/gringo is where the new gringo is installed.<br>
   .../pyboolnet/binaries/gringo-4.4.0/gringo_mac64 is where the PyBoolNet looks for gringo.<br>
</details>

## Quickstart with an example run

1. install `boolmore`
2. download 4 files from demo

   `CAD.txt` : This is the baseline model  
   `CAD_start.txt` : This is the scrambled starting model  
   `CAD_data.tsv` : This is the generated artificial experiments as in benchmarks  
   `CAD_config.json` : This contains the path to above files and other constraints
       
4. in `CAD_config.json`, modify paths for data and base

   ```json
   "data": "your_path/CAD.txt"
   
   "base": "your_path/CAD_data.tsv"
   ```

5. run python script such as

   ```python
   import boolmore.genetic_algorithm

   boolmore.genetic_algorithm.run_ga("your_path/CAD_config.json", "your_path/CAD_start.txt")
   ```

## Details for configuring input files

### 1. Experimental data input
Write the experimental results in a spreadsheet in a following format. Export in a tsv file.

Example

| ID | Score | Source | Perturbation | Observed node | Categorization |
| -- | ----- | ------ | ------------ | ------------- | -------------- |
|  6 | 	1.0 |  ABA=0 |       ROS KO |       Closure |            OFF |
|  9 |   1.0 |  ABA=1 | PLDalpha KO, RBOH KO | Closure | OFF |

- ID : id of the experiment for easy reference  
- Score : A model that agrees with this experiment will get this score.  
- Source : Values of the source nodes. For example, source1=0, source2=1. *Note that the values of all source nodes must be specified for every experiment.*  
- Perturbation : Any known interventions. For example, node1 KO, node2 CA, node3 KO  
- Observed node : the observed node  
- Categorization : one of OFF, OFF/Some, Some, Some/ON, ON  

### 2. Base model
This model is the base of all generated models. Every edge in the base model can be deleted and then added back at any time. The Boolean functions should be specified in the following ways.

Example 1
```
A* = A and (not B or C) 
```

Example 2
```
A* = A & (!B | C) 
```
### 3. json file
For example, see case_study/data/ABA_2017.json

```json
{
"parameters" : {
    "starting_id": 1,           
    "starting_gen": 1,
    "total_iterations": 100,
    "per_iteration": 100,
    "keep": 20,
    "prob": 0.01,
    "edge_prob": 0.5
    "export_top": 1,
    "export_threshold": 430,
    },

"data": "boolmore/ABA_case_study/data/data_20230925.tsv",

"base": "boolmore/ABA_case_study/baseline_models/ABA_GA_base_A.txt",

"default_sources": {"ABA":0},

"constraints": {"fixed": ["Ca2_ATPase", "Ca2c", "Closure"],
                "regulate": {"ABI1":["RCARs"],
                             "OST1":["ABI1","ABI2"]},
                "necessary" : {"8-nitro-cGMP":["cGMP"],
                               "Malate":["PEPC", "AnionEM"]},
                "group": {"PA":[["PC","PLDalpha"],["PC","PLDdelta"],["DAG","DAGK"]],
                          "S1P_PhytoS1P":[["SPHK1_2","Sph"]]},
                "possible_constant": []},

"edge_pool": [["Ca2c", "ABI2", "0"],
              ["PA", "ABI2", "0"],
              ["PA", "Microtubule_Depolymerization", "1"]]
}
```
#### parameters
- starting_id, starting_gen ( int ) : modify in case the user wants to keep track of the generations and the number of models generated in the repeated use of the algorithm  
- total_iterations, per_iteration, keep ( int ) : For example, one can have 100 total iterations, 100 models per iteration and keep top 20 to the next iteration  
- prob ( float ) : probability for each digit in the rule representation to mutate  
- edge_prob ( float ) : probability to add/delete extra edge from the edge_pool  
- export_top ( int ), export_threshold ( float ) : In case the user wants to export intermediate outcomes. e.g. top 2 of every iterations are exported if their score is higher than 430.  

#### default_sources 
{ node_name: value ( int ) } : Decides which values of the sources are taken as the default value. If given an empty dictionary, all sources being 0 will be taken as the default value.

The default value is important when determining the hierarchy between experiments. For example, say source=0 is taken as the default value. For a model to get score from an observation with source=1, it should also agree with the observation with source=0 (but not the other way around).

#### constraints
Constraints limit the searchspace, and also ensures that only reasonable models are generated.

- fixed ( list[ str ] ) : a list of nodes whose functions will not be mutated  
- regulate ( dict{ str: list[ str ] } ) : the regulated node (key) is guaranteed to be regulated by the regulators (nodes in the list). i.e., the edge is preserved.  
- necessary ( dict{ str: list[ str ] } ) : certain values of the regulators (nodes in the list) are necessary for the activation of the regulated node (key). In case of activation, the regulators being ON is necessary. In case of inhibition, the regulators being OFF is necessary.  
- group ( dict{ str: list[ list[ str ] ] } ): the regulators must appear in a group in the regulatory function. In case of {A: [[B,C], [D,E]]}, the regulators B and C must appear together with an AND rule, and same goes with regulators D and E. For example, A* = B & C | D & E, A* = B & C, A* = B & C & D & E are allowed, and A*= B & D, A* = B | E are not allowed.  
- possible_constant ( list[ str ] ): only the nodes in this list are allowed to become constant nodes. The nodes that were constant nodes in the base model can become constant nodes even if they are not on this list.  

#### edge_pool
[ [ regulator, target, sign ], ... ] : Only edges in the edge_pool can be added to the model. "0" represents a negative edge, and "1" represents a positive edge. Added edge can be deleted througout the iterations. *Note that having two edges with same regulator and target but with different signs is not allowed.*


### 4. Starting model
The genetic algorithm starts from this model. If the start model is not given, the base model is taken as the start model. *Note that any edge that is present in the starting model but not in the base model should be included in the pool of extra edges.*
