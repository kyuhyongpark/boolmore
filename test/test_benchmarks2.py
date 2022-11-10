import sys
sys.path.insert(0, 'C:/Users/danie/OneDrive/Documents/Codes/networkmutation/networkmutation')

from benchmarks import *
from mutation import *
from Model import *
import config
import pystablemotifs as sm
import cana
from cana.datasets.bio import load_all_cell_collective_models
import pystablemotifs as sm
import pyboolnet.trap_spaces
from pyboolnet.external.bnet2primes import bnet_text2primes
from cana.datasets.bio import THALIANA

def node_rule_from_cana(node: cana.boolean_node.BooleanNode,
                        int2name: dict[int, str] | None = None) -> str:
    """Transforms the prime implicants LUT of a Boolean Node from CANA to algebraic format.

    Parameters
    ----------
    node : BooleanNode
        CANA Boolean node. See: https://github.com/rionbr/CANA
    int2name : dict[int, str], optional
        Dictionary with the node ids as keys and node name as values, by default None

    Returns
    -------
    str
        Node rule in algebraic format.
        Ex.: A* = A|B&C
    """
    if int2name == None:
        int2name = {i: "x{}".format(i) for i in node.inputs}

    if node.constant:
        return "{name}* = {state}".format(name=int2name[node.id], state=node.state)

    node._check_compute_canalization_variables(prime_implicants=True)

    if node.bias() < 0.5:
        alg_rule = "{name}* = ".format(name=int2name[node.id])
        prime_rules = node._prime_implicants['1']
    else:
        alg_rule = "{name}* = !(".format(name=int2name[node.id])
        prime_rules = node._prime_implicants['0']

    for rule in prime_rules:
        for k, out in enumerate(rule):
            if out == '1':
                alg_rule += "{name}&".format(name=int2name[node.inputs[k]])
            elif out == '0':
                alg_rule += "!{name}&".format(name=int2name[node.inputs[k]])
        alg_rule = alg_rule[:-1]+"|"

    if node.bias() < 0.5:
        alg_rule = alg_rule[:-1]
    else:
        alg_rule = alg_rule[:-1]+")"

    return alg_rule

def name_adjustment(name: str) -> str:
    """Adjust the node name to fit proper formatting.

    Parameters
    ----------
    name : str
        Original name of the node.

    Returns
    -------
    str
        Adjusted name of the node.
    """

    not_allowed = ["-", ")", "(", "/"]

    for expression in not_allowed:
        name = name.replace(expression, "_")

    name = name.replace("\\", "")
    name = name.replace("+", "_plus_")

    if name[0].isdigit():
        name = 'number_' + name

    return name

def network_rules_from_cana(BN: cana.boolean_network.BooleanNetwork) -> str:
    """Transforms the prime implicants LUT of a Boolean Network from CANA to algebraic format.

    ----------
    BN : BooleanNetwork
        CANA Boolean network. See: https://github.com/rionbr/CANA

    Returns
    -------
    str
        Network rules in algebraic format.
        Ex.: A* = A|B&C\nB* = C\nC* = A|B
    """

    alg_rule = ""
    int2name = {v: name_adjustment(k) for k, v in BN.name2int.items()}
    for node in BN.nodes:
        alg_rule += node_rule_from_cana(node=node, int2name=int2name)
        alg_rule += "\n"

    return alg_rule[:-1]

nets = []
nets = load_all_cell_collective_models()
# nets.append(THALIANA())

for net in nets:

    # print(net.name)
    bnet = sm.format.booleannet2bnet(network_rules_from_cana(net))
    # print(bnet)

    # print("Loading network . . .")
    primes = bnet_text2primes(bnet)
    if primes == None:
        print(net.name, 'failed')
        continue

    N=0
    for node in primes:
        N+=1

    if N > 20:
        print(net.name, 'has more than 20 nodes')
        continue

    print(net.name)
    print('N=',N)
    # print("Network loaded.")
    # print()
    # print("RULES")
    # sm.format.pretty_print_prime_rules({k:primes[k] for k in sorted(primes)})
    # print(MODEL)

    N_EXPS = 5 * N

    STARTING_ID = 1
    STARTING_GEN = 1

    ITERATIONS = 100
    PER_ITERATION = 100
    KEEP = 20
    EXPORT_TOP = 1
    PROB = 0.01
    EDGE_PROB = 0

    config.id = STARTING_ID

    print("Primes loaded.\n")

    print("Generating artificial experimental data . . .")
    exps, pert = generate_experiments(primes, N_EXPS)
    print("Experimental data generated.\n")

    print("Loading base model . . .")
    base = Model.import_model(primes)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    starting_model = base.mutate(0.5, 0)
    starting_model.get_predictions(pert)
    starting_model.get_model_score(exps)
    starting_model.info()
    print()

    print('number of experiments:', N_EXPS)

    ### Genetic Algorithm
    print("- - - - - iteration ", 0, " - - - - -")

    iteration = []
    iteration.append(starting_model)
    for i in range(PER_ITERATION-1):
        new_model = starting_model.mutate(PROB, EDGE_PROB)
        new_model.get_predictions(pert)
        new_model.get_model_score(exps)
        iteration.append(new_model)

    iteration = sorted(iteration, key=lambda x: x.score, reverse=True)

    average = 0
    for i in range(EXPORT_TOP):
        average += iteration[i].score
    average /= EXPORT_TOP
    print("average score for top ", EXPORT_TOP, " : ", average)

    for i in range(ITERATIONS):
        print("- - - - - iteration ", i+1, " - - - - -")
        new_iteration = []
        # keep the best ones
        for j in range(KEEP):
            new_iteration.append(iteration[j])

        # mix the best ones
        to_be_mixed = sorted(new_iteration, key=lambda x: x.score, reverse=True)
        weights = list(range(1, KEEP+1))
        weights.reverse()
        for j in range(KEEP):
            mix = random.choices(to_be_mixed, weights=weights, k=2)
            mixed_model = mix_models(mix[0], mix[1])
            mixed_model.get_predictions(pert)
            mixed_model.get_model_score(exps)
            new_iteration.append(mixed_model)

        # mutate the best ones
        weights = list(range(1, 2*KEEP+1))
        weights.reverse()
        new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)
        targets = random.choices(new_iteration, weights=weights, k=PER_ITERATION-2*KEEP)
        for target in targets:
            new_model = target.mutate(PROB, EDGE_PROB)
            new_model.get_predictions(pert)
            new_model.get_model_score(exps)
            new_iteration.append(new_model)

        new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)

        average = 0
        for i in range(EXPORT_TOP):
            average += new_iteration[i].score
        average /= EXPORT_TOP
        print("average score for top ", EXPORT_TOP, " : ", average)

        iteration = new_iteration
