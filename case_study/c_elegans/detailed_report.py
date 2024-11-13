import json
import boolmore.config as config
import pystablemotifs as sm
import pyboolnet.trap_spaces as ts
import pyboolnet.prime_implicants as pi

from boolmore.model import Model
from boolmore.experiment import import_exps

SETTINGS = "boolmore/c_elegans/config_20231121.json"
MODEL = "boolmore/c_elegans/start_20231121.txt"
NAME = None

def get_detailed_report(json_file:str, model:str|None=None, name:str|None=None):

    f = open(json_file)
    json_dict = json.load(f)

    # take model specific data from the json file
    DEFAULT_SOURCES = json_dict["default_sources"]
    CONSTRAINTS = json_dict["constraints"]
    EDGE_POOL = json_dict["edge_pool"]

    DATA = json_dict["data"]
    BASE = json_dict["base"]

    if model != None:
        MODEL = model
    # if starting model is not given, take the base as the start
    else:
        MODEL = BASE

    if name != None:
        FILE = name
    else:
        FILE = MODEL.split("/")[-1][:-4]+"_score.tsv"

    print("Loading experimental data . . .")
    exps, pert = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = sm.format.import_primes(BASE)
    base = Model.import_model(base_primes,
                              constraints=CONSTRAINTS, edge_pool=EDGE_POOL,
                              default_sources=DEFAULT_SOURCES)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    print("Loading starting model . . .")
    primes = sm.format.import_primes(MODEL)
    n1 = Model.import_model(primes, config.id, 0, base)
    print("Starting model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps, report=True, file=FILE)
    n1.info()
    print("default_sources:", n1.default_sources)
    print()

    # ### check which modifications are significant
    # for node in base_primes:
    #     if base_primes[node] == primes[node]:
    #         continue
    #     else:
    #         base_copy = base.mutate(0,0)
    #         n1_copy = n1.mutate(0,0)

    #         base_copy.primes[node] = n1.primes[node]
    #         base_copy.regulators_dict[node] = n1.regulators_dict[node]
    #         base_copy.signs_dict[node] = n1.signs_dict[node]
    #         base_copy.rr_dict[node] = n1.rr_dict[node]

    #         n1_copy.primes[node] = base.primes[node]
    #         n1_copy.regulators_dict[node] = base.regulators_dict[node]
    #         n1_copy.signs_dict[node] = base.signs_dict[node]
    #         n1_copy.rr_dict[node] = base.rr_dict[node]

    #         base_copy.get_predictions(pert)
    #         base_copy.get_model_score(exps)

    #         n1_copy.get_predictions(pert)
    #         n1_copy.get_model_score(exps)

    #         print(f"{node}\t{n1_copy.score}\t{base_copy.score}")

    ### percolate the constants and reduce the model
    pprimes1 = pi.percolate(primes, remove_constants = True, copy=True)
    print("percolated rules: ")
    sm.format.pretty_print_prime_rules({k:pprimes1[k] for k in sorted(pprimes1)}) # type: ignore
    print()

    ### find nodes that have different functions
    pprimes0 = pi.percolate(base_primes, remove_constants = True, copy=True)
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

    ### find the edges that are deleted
    deleted_regulators = {}
    deleted_regulators_perc = {}
    for node in base_primes:
        base_regulators = pi.find_predecessors(base_primes, [node])
        regulators = pi.find_predecessors(primes, [node])
        deleted_regulator_list = [x for x in base_regulators if x not in regulators]
        if len(deleted_regulator_list) != 0:
            deleted_regulators[node] = deleted_regulator_list
        ### find extra edges that are deleted in the percolated rules
        if node in pprimes0 and node in pprimes1:
            base_regulators_perc = pi.find_predecessors(pprimes0, [node]) # type: ignore
            regulators_perc = pi.find_predecessors(pprimes1, [node]) # type: ignore
            deleted_regulator_list_perc = [x for x in base_regulators_perc if x not in regulators_perc and x not in deleted_regulator_list]
            if len(deleted_regulator_list_perc) != 0:
                deleted_regulators_perc[node] = deleted_regulator_list_perc
    print("deleted regulators: ",deleted_regulators)
    print("extra deleted regulators in the percolated rules: ",deleted_regulators_perc)
    print()

    ### find stable motifs
    ### TODO: not include the source nodes.
    stable_motifs = ts.compute_trap_spaces(pprimes1, "max") # type: ignore
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
        pprimes2 = pi.percolate(pprimes1, add_constants = fix, remove_constants = True, copy=True) # type: ignore
        # sm.format.pretty_print_prime_rules({k:primes2[k] for k in sorted(primes2)})
        conditinally_stable_motifs = ts.compute_trap_spaces(pprimes2, "max") # type: ignore
        for conditinally_stable_motif in conditinally_stable_motifs:
            if conditinally_stable_motif not in stable_motifs:
                print(conditinally_stable_motif)
    print()

    ### fix nodes to find minimal trapspaces and LDOIs
    ### fixing constant nodes are not allowed
    print("LDOI, DOI and minimal trapspaces:")
    fixes = [{"Food":0, "oxidative_stress":0},
             {"Food":0, "oxidative_stress":1},
             {"Food":1, "oxidative_stress":0},
             {"Food":1, "oxidative_stress":1}]
    for fix in fixes:
        print("- - - - - - - - - -")
        print("fix -", fix)
        LDOI = sm.drivers.logical_domain_of_influence(fix,pprimes1)[0]
        MPBN_DOI = sm.drivers.domain_of_influence(fix,pprimes1,MPBN_update=True)[0] # type: ignore

        print("LDOI: ",{k:LDOI[k] for k in sorted(LDOI)})
        print("only in MPBN_DOI: ",{k:MPBN_DOI[k] for k in MPBN_DOI if k not in LDOI})
        print()

        pprimes2 = pi.percolate(pprimes1, add_constants = fix, remove_constants = False, copy=True) # type: ignore
        trs = ts.compute_trap_spaces(pprimes2, "min") # type: ignore
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
    get_detailed_report(SETTINGS, MODEL, NAME)