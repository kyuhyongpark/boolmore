import json

from pyboolnet.trap_spaces import compute_trap_spaces
from pyboolnet.prime_implicants import percolate
from pyboolnet.prime_implicants import find_predecessors
from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.model import Model
from boolmore.experiment import import_exps
from boolmore.conversions import prime2bnet


SETTINGS = "./data/config.json"
START = "./data/start.bnet"
FINAL = "./results/20231121_6870_gen68.bnet"

def get_detailed_report(json_file:str, final:str, start:str|None=None):

    f = open(json_file)
    json_dict = json.load(f)

    # take model specific data from the json file
    DEFAULT_SOURCES = json_dict["default_sources"]
    CONSTRAINTS = json_dict["constraints"]
    EDGE_POOL = json_dict["edge_pool"]

    DATA = json_dict["data"]
    BASE = json_dict["base"]

    # if starting model is not given, take the base as the start
    if start == None:
        start = BASE

    FILE0 = start.split("/")[-1][:-5]+"_score.tsv"
    FILE1 = final.split("/")[-1][:-5]+"_score.tsv"

    print("Loading experimental data . . .")
    exps, pert = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = bnet_file2primes(BASE)
    base = Model.import_model(base_primes,
                              constraints=CONSTRAINTS, edge_pool=EDGE_POOL,
                              default_sources=DEFAULT_SOURCES)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    print("Loading starting model . . .")
    start_primes = bnet_file2primes(start)
    n0 = Model.import_model(start_primes, base=base)
    print("Starting model loaded.")
    n0.get_predictions(pert)
    n0.get_model_score(exps, report=True, file=FILE0)
    n0.info()
    print()

    print("Loading final model . . .")
    final_primes = bnet_file2primes(final)
    n1 = Model.import_model(final_primes, base=base)
    print("Final model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps, report=True, file=FILE1)
    n1.info()
    print()


    print("Comparing the starting model and the final model . . .\n")

    ### find nodes that have different functions
    different_nodes = []
    for node in start_primes: # type: ignore
        if node not in final_primes:
            different_nodes.append(node)
            continue
        if start_primes[node] == final_primes[node]: # type: ignore
            pass
        else:
            different_nodes.append(node)
    for node in final_primes: # type: ignore
        if node not in start_primes:
            different_nodes.append(node)
    print("Nodes with changed functions: ", sorted(different_nodes))
    print()

    ### find the edges that are deleted or added
    deleted_regulators = {}
    added_regulators = {}
    for node in base_primes:
        start_regulators = find_predecessors(start_primes, [node])
        final_regulators = find_predecessors(final_primes, [node])
        deleted_regulator_list = [x for x in start_regulators if x not in final_regulators]
        added_regulator_list = [x for x in final_regulators if x not in start_regulators]
        if len(deleted_regulator_list) != 0:
            deleted_regulators[node] = deleted_regulator_list
        if len(added_regulator_list) != 0:
            added_regulators[node] = added_regulator_list
    print("deleted regulators: ",deleted_regulators)
    print()
    print("added regulators: ",added_regulators)
    print()

    ### percolate the constants and reduce the model
    pprimes0 = percolate(start_primes, remove_constants = True, copy=True)
    print("percolated rules of the starting model: ")
    sorted_primes = {k:pprimes0[k] for k in sorted(pprimes0)}
    for k in sorted_primes:
        s = prime2bnet(k, sorted_primes[k])
        print(s)
    print()

    pprimes1 = percolate(final_primes, remove_constants = True, copy=True)
    print("percolated rules of the final model: ")
    sorted_primes = {k:pprimes1[k] for k in sorted(pprimes1)}
    for k in sorted_primes:
        s = prime2bnet(k, sorted_primes[k])
        print(s)
    print()

    ### find nodes that have different functions
    different_nodes = []
    for node in pprimes0: # type: ignore
        if node not in pprimes1:
            different_nodes.append(node)
            continue
        if pprimes0[node] == pprimes1[node]: # type: ignore
            pass
        else:
            different_nodes.append(node)
    for node in pprimes1: # type: ignore
        if node not in pprimes0:
            different_nodes.append(node)
    print("Nodes with changed percolated functions: ", sorted(different_nodes))
    print()


    print("Analyzing the final model . . .")

    ### find stable motifs
    stable_motifs = compute_trap_spaces(pprimes1, "max") # type: ignore
    print("Stable Motifs:")
    for stable_motif in stable_motifs:
        print(stable_motif)
    print()

    ### fix extra nodes to find conditionally stable motifs
    ### WARNING: fixing constant nodes do not work, as they are already percolated
    print("Conditionally Stable Motifs:")
    fixes = [{"Food":0, "oxidative_stress":0},
             {"Food":0, "oxidative_stress":1},
             {"Food":1, "oxidative_stress":0},
             {"Food":1, "oxidative_stress":1}]
    for fix in fixes:
        print("fix - ", fix)
        pprimes2 = percolate(pprimes1, add_constants = fix, remove_constants = True, copy=True) # type: ignore
        conditinally_stable_motifs = compute_trap_spaces(pprimes2, "max") # type: ignore
        for conditinally_stable_motif in conditinally_stable_motifs:
            if conditinally_stable_motif not in stable_motifs:
                print(conditinally_stable_motif)
    print()

    ### fix nodes to find minimal trapspaces
    ### fixing constant nodes are not allowed
    print("minimal trapspaces:")
    fixes = [{"Food":0, "oxidative_stress":0},
             {"Food":0, "oxidative_stress":1},
             {"Food":1, "oxidative_stress":0},
             {"Food":1, "oxidative_stress":1}]
    for fix in fixes:
        print("- - - - - - - - - -")
        print("fix -", fix)
        pprimes3 = percolate(pprimes1, add_constants = fix, remove_constants = False, copy=True) # type: ignore
        trs = compute_trap_spaces(pprimes3, "min") # type: ignore
        for tr in trs:
            for node in pprimes1.keys(): # type: ignore
                if node not in tr.keys(): # type: ignore
                    tr[node] = "X" # type: ignore
        
        if len(trs) == 1:
            print("There is 1 minimal trapspace for ", dict(sorted(fix.items())))
        else:
            print("There are " + str(len(trs)) + " minimal trapspaces for", dict(sorted(fix.items())))
        for tr in trs:
            print(dict(sorted(tr.items()))) # type: ignore
        print()

if __name__ == "__main__":
    get_detailed_report(SETTINGS, FINAL, START)