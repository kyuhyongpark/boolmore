# Demo

Demonstration of `boolmore` on the Cortical Area Development (CAD) model.

## Quickstart

1. install `boolmore`
    ```
    pip install git+https://github.com/kyuhyongpark/boolmore
    ```
2. download 5 files from the `demo` directory  
   `CAD.txt` : This is the baseline model  
   `CAD_start.txt` : This is the scrambled starting model  
   `CAD_data.tsv` : This is the generated artificial experiments as in benchmarks  
   `CAD_config.json` : This contains the path to above files and other constraints  
   `run.py` : This is the run file
       
3. run python script such as
    ```
    cd boolmore/demo
    python run.py
    ```

## Description

Parameters are adjusted for the small model of size 5  
1. mutation probability: 0.2 (compared to 0.01 for the validation benchmarks)  
2. number of iterations: 10 (compared to 100 for the validation benchmarks)  
3. models per iteration: 10 (compared to 100 for the validation benchmarks)  
4. models to keep per iteration: 2 (compared to 20 for the validation benchmarks)  
5. models to mix per iteration: 2 (compared to 20 for the validation benchmarks)

the log will be stored as "demo_log.txt"
the final model will be stored as "demo_id_gen.bnet"

## Generate your own benchmark
Use `create_your_own_benchmark.ipynb` to generate different artificial experiment dataset and the starting model using different seeds.