import random
import os
from pyboolnet.interaction_graphs import primes2igraph
from boolmore.model import Model


PrimeType = list[list[dict[str, int]]]
FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]


def generate_experiments(primes:dict[str, PrimeType], n_exps:int|None=None, export:bool=False,
                         file:str='artificial_experiments.tsv') -> tuple[list[ExpType], list[FixesType], float]:
    """
    Given the primes dictionary, generates and returns artificial experiments and interventions

    An intervention consists of fixing multiple nodes. This always includes all source nodes.
    Other nodes are included with a probability of 1/N_other.
    e.g. fixes = ((source A, 1), (source B, 0), (other C, 1))

    Observed nodes have sink nodes as priority.
    Other nodes are included with a probability of 1/N_other in a single iteration.
    Constant nodes are not observed.
    However, if the same fixes are generated multiple times, more nodes can be observed for that fixes.
    This is preferred as we usually have many observations for wildtypes or single perturbations.
    e.g. observed_nodes = {sink A, sink B, other C}

    Parameters
    ----------
    primes - pyboolnet primes dictionary        :length N dict[str, PrimeType]
             {node: prime}
    n_exps - number of experiments to generate  :int|None
             when None, generates 10*N

    Returns
    -------
    experiments - list of exp                           :list[ExpType]
        exp     - info of a single experiment           :ExpType = tuple[int, float, FixesType, str, str]
            exp[0] - id of the experiment               :int
            exp[1] - max_score for the experiment       :float
            exp[2] - fixes                              :FixesType = tuple[tuple[str, int]]
                     ((node A, value1), (node B, value2), ...)
            exp[3] - observed_node                      :str
            exp[4] - outcome_value                      :str
                     one of OFF, OFF/Some, Some, Some/ON, ON

    interventions - summarized list of fixes for convenience    :list[FixesType]
        fixes     - ((node A, value1), (node B, value2), ...)   :FixesType = tuple[tuple[str, int]]
    
    max_score   - possible max score        :float
        
    """
    if n_exps == None:
        n_exps = 10*len(primes)

    # find source nodes
    source_vars = []
    for node in primes:
        if primes[node] == [[{node:0}],[{node:1}]]:
            source_vars.append(node)

    # find constant nodes
    constants = []
    for node in primes:
        if primes[node] == [[{}],[]] or primes[node] == [[],[{}]]:
            constants.append(node)

    # find sink nodes
    IG = primes2igraph(primes)
    sinks = [node for node, out_degree in IG.out_degree if out_degree == 0]

    other_nodes = [node for node in primes if node not in source_vars and node not in sinks]
    N_other = len(other_nodes)

    # generate fixes-observations pairs
    interventions = []
    dct = {}
    n = 0
    while n < n_exps:
        # fixes should be all source nodes + 1 or 2 or 3 or 4 ... other nodes
        fixed_nodes = source_vars.copy()
        for node in other_nodes:
            if random.random() < 1/N_other:
                fixed_nodes.append(node)

        fixes = [(node, random.randrange(2)) for node in fixed_nodes]
        fixes = tuple(sorted(fixes))

        if fixes not in interventions:
            interventions.append(fixes)
            dct[fixes] = set()

        # observations should be all sink node + 1 or 2 or 3 or 4 ... other nodes
        observed_nodes = sinks.copy()
        for node in other_nodes:
            if random.random() < 1/N_other and node not in fixed_nodes and node not in constants:
                observed_nodes.append(node)

        dct[fixes].update(observed_nodes)

        n = 0
        for key in dct:
            n += len(dct[key])

    model = Model.import_model(primes)
    model.get_predictions(interventions)

    experiments = []
    exp_id = 0
    for fixes in interventions:
        for observed_node in dct[fixes]:
            exp_id += 1
            exp = [exp_id, 1.0, fixes, observed_node]

            value = model.predictions[fixes][observed_node]

            if value < 0.0001:
                exp.append('OFF')
            elif value < 0.33:
                exp.append('OFF/Some')
            elif value < 0.67:
                exp.append('Some')
            elif value < 0.9999:
                exp.append('Some/ON')
            elif value < 1.0001:
                exp.append('ON')
            else:
                print("Unexpected value", value)
                raise Exception("Unexpected experiment output")

            exp = tuple(exp)
            experiments.append(exp)

    experiments = experiments[0:n_exps]

    if export == True:
        fp = open(file, "w")
        fp.write('ID\tSCORE\tFIXES\tNODE\tVALUE\n')
        for exp in experiments:
            fp.write(str(exp[0]) + '\t')
            fp.write(str(exp[1]) + '\t')
            lst = [fix[0] + '=' + str(fix[1]) for fix in exp[2]]
            fp.write(','.join(lst) + '\t')
            fp.write(exp[3] + '\t')
            fp.write(exp[4] + '\n')

        print('Exported generated experiments to', os.path.abspath(file))

    return experiments, interventions, float(n_exps)