# BoolMoRe
Boolean Model Refiner


## Install
```
pip install git+https://github.com/troonmel/BoolMoRe
```

## Quickstart with an example run

1. install `boolmore`
2. download 4 files from Examples and tutorials

   `Cortical_Area_Development.txt` : This is the baseline model  
   `CAD_scrambled.txt` : This is the scrambled starting model  
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

   boolmore.genetic_algorithm.run_ga('your_path/CAD_config.json', 'your_path/CAD_scrambled.txt')
   ```
