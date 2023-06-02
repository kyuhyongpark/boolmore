from pyboolnet.external.bnet2primes import bnet_text2primes
from pystablemotifs.format import import_primes
from pystablemotifs.drivers import logical_domain_of_influence
import pyboolnet.prime_implicants
import itertools as it

BASE = 'networkmutation/baseline/ABA_full_20230407.txt'
MODEL1 = 'networkmutation/models/20230512_7790_gen125_mod.txt'
MODEL2 = 'networkmutation/models/osc_20230514_8025_gen128.txt'

def source_sets(primes, min_set_size=None, max_set_size=None, forbidden=None, fixed=None):
    """Short summary.
    Parameters
    ----------
    primes : pyboolnet primes dictionary
        Update rules.
    min_set_size : int
        Minimum size of source set to consider.
        If None, is set to the size of the fixed set + 1.
    max_drivers : int
        Maximum size of source set to consider.
        If None, is set to the number of all the nodes in the prime that are not in the forbidden - 1.
    forbidden : list of str variable names
        Variables to be considered uncontrollable (the default is None).
    fixed : partial state dictionaries
        Node set that should be included in all the source sets (the default is None).
    Returns
    -------
    source_sets : list of partial state dictionaries
        Each state dictionary in the list is a possible source set.
    """
    if forbidden is None:
        forbidden = []
    if fixed is None:
        fixed = {}

    # consistency check
    for node in forbidden:
        if node in fixed.keys():
            print("The forbidden nodes are inconsistent with the fixed nodes.")
            return
        if node not in primes.keys():
            print("The forbidden nodes are inconsistent with the primes.")
            return
    for node in fixed.keys():
        if node not in primes.keys():
            print("The fixed nodes are inconsistent with the primes.")
            return

    if min_set_size is None:
        min_set_size = len(fixed) + 1
    if max_set_size is None:
        max_set_size = len(primes) - len(forbidden) - 1

    # consistency check
    if min_set_size < len(fixed):
        print("min set size is inconsistent with the fixed nodes")
        return
    if max_set_size < min_set_size:
        print("min set size is inconsistent with the max set size")
        return

    nodes = []
    source_sets = []

    for node in primes.keys():
        if node not in forbidden:
            if node not in fixed.keys():
                nodes.append(node)

    for source_set_size in range(min_set_size-len(fixed), max_set_size+1-len(fixed)):
        for add_nodes in it.combinations(nodes,source_set_size):
            for s in it.product([0,1],repeat=len(add_nodes)):
                source_set = {k:s for k,s in zip(add_nodes,s)}
                source_set.update(fixed)
                source_set = dict(sorted(source_set.items()))
                source_sets.append(source_set)

    return source_sets

### import the model into primes
print("Loading networks . . .")
base_primes = import_primes(BASE)
model1_primes = import_primes(MODEL1)
model2_primes = import_primes(MODEL2)
print("Networks loaded.\n")

# percolate the constants and reduce the model
pyboolnet.prime_implicants.percolate(base_primes, remove_constants = True, copy=False)
pyboolnet.prime_implicants.percolate(model1_primes, remove_constants = True, copy=False)
pyboolnet.prime_implicants.percolate(model2_primes, remove_constants = True, copy=False)

base_fixes = source_sets(base_primes,min_set_size=0,max_set_size=1)
model1_fixes = source_sets(model1_primes,min_set_size=0,max_set_size=1)
model2_fixes = source_sets(model2_primes,min_set_size=0,max_set_size=1)

nodes_only_in_base = []

for fix in base_fixes:
    if fix not in model1_fixes or fix not in model2_fixes:
        nodes_only_in_base.append(fix)
        continue

    base_LDOI, base_LDOI_contra = logical_domain_of_influence(fix,base_primes)
    model1_LDOI, model1_LDOI_contra = logical_domain_of_influence(fix,model1_primes)
    model2_LDOI, model2_LDOI_contra = logical_domain_of_influence(fix,model2_primes)

    ### find the opposite value in the LDOI if it ever exists
    for k in base_LDOI:
        if k in model1_LDOI and base_LDOI[k] != model1_LDOI[k]:
            print("IMPORTANT * * * * * * * * * * IMPORTANT")
            print("fix: ",fix)
            print('contradiction in : ', k)
            print('base -', base_LDOI)
            print('model1 -', model1_LDOI)
            print("IMPORTANT * * * * * * * * * * IMPORTANT")

    ### find the opposite value in the LDOI if it ever exists
    for k in base_LDOI:
        if k in model2_LDOI and base_LDOI[k] != model2_LDOI[k]:
            print("IMPORTANT * * * * * * * * * * IMPORTANT")
            print("fix: ",fix)
            print('contradiction in : ', k)
            print('base -', base_LDOI)
            print('model2 -', model2_LDOI)
            print("IMPORTANT * * * * * * * * * * IMPORTANT")

    shared_in_models = {k:model1_LDOI[k] for k in model1_LDOI if k in model2_LDOI and model1_LDOI[k]==model2_LDOI[k]}
    only_in_models = {k:shared_in_models[k] for k in shared_in_models if k not in base_LDOI}

    if len(only_in_models) != 0:
        print("- - - - - - - - - -")
        print("fix: ",fix)
        print('only in GAs: ', {k:only_in_models[k] for k in sorted(only_in_models)})

print("\nEXCEPTIONS")

print("\nnodes only in the original:")
for fix in nodes_only_in_base:
    print("fix: ",fix)
    base_LDOI, base_LDOI_contra = logical_domain_of_influence(fix,base_primes)
    print('original LDOI: ', {k:base_LDOI[k] for k in sorted(base_LDOI)})
    print("- - - - - - - - - -")

print("\nnodes only in the GA1-A:")
for fix in model1_fixes:
    if fix not in base_fixes or fix not in model2_fixes:
        print("fix: ",fix)
        model1_LDOI, model1_LDOI_contra = logical_domain_of_influence(fix,model1_primes)
        print('GA1-A LDOI: ', {k:model1_LDOI[k] for k in sorted(model1_LDOI)})
        print("- - - - - - - - - -")

print("\nnodes only in the GA1-B:")
for fix in model2_fixes:
    if fix not in base_fixes or fix not in model1_fixes:
        print("fix: ",fix)
        model2_LDOI, model2_LDOI_contra = logical_domain_of_influence(fix,model2_primes)
        print('GA1-B LDOI: ', {k:model2_LDOI[k] for k in sorted(model2_LDOI)})
        print("- - - - - - - - - -")
