# Demo

Demonstration of `boolmore` on the Cortical Area Development (CAD) model.

## Quickstart

1. Install `boolmore`
    ```
    pip install git+https://github.com/kyuhyongpark/boolmore
    ```
2. Download these 5 files from the `demo` directory  
   `CAD.bnet` : This is the baseline model  
   `CAD_start.bnet` : This is the scrambled starting model  
   `CAD_data.tsv` : This file contains the artificial experiments generated as in the benchmarks  
   `CAD_config.json` : This file contains the path to the above files and other constraints  
   `run.py` : This is the run file
       
3. Run the python script such as
    ```
    cd boolmore/demo
    python run.py
    ```

## Description

The parameters are adjusted for the small model size (5 nodes)
1. mutation probability: 0.2 (compared to 0.01 for the validation benchmarks)  
2. number of iterations: 10 (compared to 100 for the validation benchmarks)  
3. models per iteration: 10 (compared to 100 for the validation benchmarks)  
4. models to keep per iteration: 2 (compared to 20 for the validation benchmarks)  
5. models to mix per iteration: 2 (compared to 20 for the validation benchmarks)

The log will be stored as "demo_log.txt"
The final model will be stored as "demo_id_gen.bnet"

## Generate your own benchmark
Use `create_your_own_benchmark.ipynb` to generate different artificial experiment datasets and starting models, using different seeds.
