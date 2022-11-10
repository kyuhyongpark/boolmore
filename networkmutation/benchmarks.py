import random
from pyboolnet.interaction_graphs import primes2igraph
from Model import *

def generate_experiments(primes, n_exps):
    """
    Returns experiments and interventions

    Parameters
    ----------
    location : data location

    Returns
    -------
    experiments : list of expset (tuple)
        expset[0] : score (float) representing the score for the experiment set
        expset[1] : exp (dict) representing each perturbation/result pair
            exp key : fixes (tuple) ((node A, value1), (node B, value2), ...)
            exp value : result (tuple) (measured_node, values)

    interventions : list of tuples
        each element represents fixes (tuple)
    """
    # find source nodes
    source_vars = []
    for x in primes.keys():
        if primes[x] == [[{x:0}],[{x:1}]]:
            source_vars.append(x)

    # find sink nodes
    IG = primes2igraph(primes)
    sinks = [node for node, out_degree in IG.out_degree if out_degree == 0]

    other_nodes = [node for node in primes if node not in source_vars and node not in sinks]
    N = len(other_nodes)

    # generate fixes-observations pairs
    interventions = []
    dct = {}
    n = 0
    while n < n_exps:
        # perturbation should be source node + 1 or 2 or 3 or 4 ... perturbations
        fixes = []
        for node in other_nodes:
            if random.random() < 1/N:
                fixes.append((node, random.randrange(2)))
        for node in source_vars:
            fixes.append((node, random.randrange(2)))

        fixes = tuple(sorted(fixes))

        if fixes not in interventions:
            dct[fixes] = set()
            interventions.append(fixes)

        # observation should be sink node + 1 or 2 or 3 or 4 ... nodes
        # FIXME: nodes that were intervened should not be observed!
        observations = set()
        for node in sinks:
            observations.add(node)
        for node in other_nodes:
            if random.random() < 1/N:
                observations.add(node)

        dct[fixes].update(observations)

        n = 0
        for key in dct:
            n += len(dct[key])

    model = Model.import_model(primes)
    model.get_predictions(interventions)

    experiments = []
    for intervention in interventions:
        for observation in dct[intervention]:
            expset = [1.0]
            exp = {}
            exp[intervention] = []
            exp[intervention].append(observation)
            value = model.predictions[intervention][observation]
            if value < 0.1:
                exp[intervention].append('0')
            elif value < 0.2:
                exp[intervention].append('0')
                exp[intervention].append('lenient 1')
            elif value < 0.3:
                exp[intervention].append('0.5')
                exp[intervention].append('0')
            elif value < 0.4:
                exp[intervention].append('0.5')
                exp[intervention].append('lenient 0')
            elif value < 0.6:
                exp[intervention].append('0.5')
            elif value < 0.7:
                exp[intervention].append('0.5')
                exp[intervention].append('lenient 1')
            elif value < 0.8:
                exp[intervention].append('0.5')
                exp[intervention].append('1')
            elif value < 0.9:
                exp[intervention].append('1')
                exp[intervention].append('lenient 0')
            else:
                exp[intervention].append('1')

            exp[intervention] = tuple(exp[intervention])
            expset.append(exp)
            experiments.append(tuple(expset))

    experiments = experiments[0:n_exps]

    return experiments, interventions
