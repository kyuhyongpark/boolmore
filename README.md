# BoolMoRe
Boolean Model Refiner

Takes an existing Boolean model and refines it to fit better with the given experimental results. Hundreds or thousands of Boolean models are explored in the process, each being consistent to the interaction graph of the starting model and any extra constraints given. The score (or fitness) of the models are determined by comparing the attractors with the experimental data for every perturbation (or fixes, such as gene KO).

We showcase the strength of our algorithm by a case study on plant stomatal model. After several hours of automatic refinement, the fittest models recapture and surpass the accuracy gain achieved over 10 years of manual revision and provide new, testable predictions.

## Install
```
pip install git+https://github.com/kyuhyongpark/BoolMoRe
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
2. download 4 files from Examples and tutorials

   `Cortical_Area_Development.txt` : This is the baseline model  
   `CAD_starting.txt` : This is the scrambled starting model  
   `CAD_data.tsv` : This is the generated artificial experiments as in benchmarks  
   `CAD_config.json` : This contains the path to above files and other constraints
       
4. in `CAD_config.json`, modify paths for data and base

   ```
   "data": "your_path/Cortical_Area_Development.txt"
   
   "base": "your_path/CAD_data.tsv"
   ```

5. run python script such as

   ```
   import boolmore.genetic_algorithm

   boolmore.genetic_algorithm.run_ga('your_path/CAD_config.json', 'your_path/CAD_starting.txt')
   ```
