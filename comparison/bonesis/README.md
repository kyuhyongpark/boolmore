## Description

1. constraints and extra edges are implemented in aeon file
- "fixed" constraint is implemented by specifying the function in aeon file.
- "regulatory" constraint is implemented by specifying the regulation as "->" or "-|" in the aeon file.
Other regulations are written as "->?" or "-|?", which allows them to be non-functional.
- "necessary" constraint cannot be implemented fully. Instead, they are implemented as "regulatory".
- "group" constraint is implemented by creating new nodes with fixed inputs, and making this node regulate the original target.
For example, PC and PLDalpha regulating PC in a group is implemented by creating a node PLDalpha_complex that has a rule `PC & PLDalpha` which regulates PA.
- Extra edges are added into the aeon file from the beginning.

2. Experimental results categorized as "OFF/Some", "Some", and "Some/ON" are ignored.
This is because there is no way in bonesis to specify that a certain node must oscillate or be bistable.

3. Experimental results categorized as "OFF" or "ON" are implemented in the best possible way.
It is implemented by a constraint that a trap space that satisfy the experimental result exists, and that all fixed points, if exist, must satisfy the experimental result.
For example, say we have a result Closure "ON" when ABA=1.
We create a data dictionary {"Closure_ON" : {"Closure: 1}}, and run following script:
```
with bo.mutant({"ABA":1}):
    bo.fixed(bo.obs("Closure_ON"))
    bo.all_fixpoints(bo.obs("Closure_ON"))
```
This ensures that when ABA=1, the models have a trap space with Closure=1, and all of their fixed points satisfy Closure=1.
However, note that some models may have trap space with Closure=0 when ABA=1, as this is not forbidden.


## Scripts

Install following packages to run the scripts.

```
pip install bonesis
pip install biodivine_aeon
pip install ipykernel

Package           Version
----------------- -----------
asttokens         2.4.1
beautifulsoup4    4.12.3
biodivine_aeon    1.1.1
bonesis           0.6.7
boolean.py        4.0
cffi              1.17.1
clingo            5.7.1
colomoto_jupyter  0.8.9
colorama          0.4.6
comm              0.2.2
debugpy           1.8.7
decorator         5.1.1
executing         2.1.0
ipykernel         6.29.5
ipython           8.28.0
jedi              0.19.1
jupyter_client    8.6.3
jupyter_core      5.7.2
matplotlib-inline 0.1.7
mpbn              3.8
nest-asyncio      1.6.0
networkx          3.4.2
numpy             2.1.2
packaging         24.1
pandas            2.2.3
parso             0.8.4
pip               24.2
platformdirs      4.3.6
prompt_toolkit    3.0.48
psutil            6.1.0
pure_eval         0.2.3
pycparser         2.22
pydot             3.0.2
pyeda             0.29.0
Pygments          2.18.0
pyparsing         3.2.0
python-dateutil   2.9.0.post0
pytz              2024.2
pywin32           308
pyzmq             26.2.0
scipy             1.14.1
setuptools        65.5.0
six               1.16.0
soupsieve         2.6
stack-data        0.6.3
tornado           6.4.1
tqdm              4.66.5
traitlets         5.14.3
typing_extensions 4.12.2
tzdata            2024.2
wcwidth           0.2.13
```

### toy model

`toy.ipynb` illustrates how the constraints are implemented using aeon as described above.
Furthermore, it illustrates how the experimental input used in boolmore is converted to constraints for bonesis.

### ABA case study

Run `ABA_A.ipynb` or `ABA_B.ipynb` to reproduce results of bonesis on ABA case study.

Extra experimental results are skipped, as specfied in skip_list by their IDs.
Even with just the experimental results categorized as "OFF" or "ON", there is no Boolean model that can satisfy them all.
This is shown by running `bo.is_satisfiable()`.
Therefore, we need to ignore more experimental results. We took the baseline model or the GA generated model, and skipped all experimental results that each cannot satisfy.
This ensures that the experiments are satisfiable, as we know that there exist at least one model that satisfy them.

bonesis could not generate any model within 24 hours in 4 settings.
- ABA_A with 291 experiments satisfied in ABA_GA1_A
- ABA_A with 172 experiments satisfied in ABA_base_A
- ABA_B with 311 experiments satisfied in ABA_GA1_B
- ABA_B with 177 experiments satisfied in ABA_base_B
