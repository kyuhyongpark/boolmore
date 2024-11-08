import random
import os

from pyboolnet.interaction_graphs import primes2igraph

from boolmore.model import Model


PrimeType = list[list[dict[str, int]]]
FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]


def generate_experiments(primes:dict[str, PrimeType], n_exps:int|None=None,
                         seed:int|None=None,
                         export:bool=False, file_name:str="artificial_experiments.tsv"
                         ) -> tuple[list[ExpType], list[FixesType]]:
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
    seed - random seed                                          :int|None
    
    export - if true, export the data to file_name              :bool
    file_name - location of the exported data in tsv format      :str

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

    fixes_list - summarized list of fixes for convenience    :list[FixesType]
        fixes     - ((node A, value1), (node B, value2), ...)   :FixesType = tuple[tuple[str, int]]
        
    """
    if n_exps == None:
        n_exps = 10*len(primes)

    if seed != None:
        random.seed(seed)

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
    fixes_list_temp = []
    dct = {}
    n = 0
    while n < n_exps:
        # fixes should be all source nodes + 1 or 2 or 3 or 4 ... other nodes
        fixed_nodes = source_vars.copy()
        for node in other_nodes:
            if random.random() < 1/N_other:
                fixed_nodes.append(node)

        lst = [(node, random.randrange(2)) for node in fixed_nodes]
        fixes = tuple(sorted(lst))

        # new fixes
        if fixes not in fixes_list_temp:
            fixes_list_temp.append(fixes)
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

    # there can be fixes that have no observations. remove them.
    for fixes in fixes_list_temp:
        if len(dct[fixes]) == 0:
            fixes_list_temp.remove(fixes)
            del dct[fixes]

    model = Model.import_model(primes)
    model.get_predictions(fixes_list_temp)

    experiments = []
    exp_id = 0
    for fixes in fixes_list_temp:
        for observed_node in sorted(list(dct[fixes])):
            exp_id += 1
            exp = [exp_id, 1.0, fixes, observed_node]

            value = model.predictions[fixes][observed_node]

            if value < 0.0001:
                exp.append("OFF")
            elif value < 0.33:
                exp.append("OFF/Some")
            elif value < 0.67:
                exp.append("Some")
            elif value < 0.9999:
                exp.append("Some/ON")
            elif value < 1.0001:
                exp.append("ON")
            else:
                print("Unexpected value", value)
                raise Exception("Unexpected experiment output")

            exp = tuple(exp)
            experiments.append(exp)

    # trim experiments and fixes_list to fit the required size
    while len(experiments) > n_exps:
        random_exp = random.choice(experiments)
        experiments.remove(random_exp)

    fixes_list = []
    for exp in experiments:
        if exp[2] not in fixes_list:
            fixes_list.append(exp[2])

    # export
    if export == True:
        export_exps(primes, experiments, file_name)

    return experiments, sorted(fixes_list)

def export_exps(primes:dict[str, PrimeType], experiments:list[ExpType], 
                file_name:str="artificial_experiments.tsv"):
    """
    Export the experiments in a tsv format.
    Primes are required to separate out the source nodes.

    Parameters
    ----------
    primes - pyboolnet primes dictionary        :length N dict[str, PrimeType]
             {node: prime}
    experiments - list of exp                   :list[ExpType]
    file_name - location of the exported data   :str

    Exports
    -------
    The tsv file has 6 columns
    ID     - e.g. 1
    SCORE  - e.g. 1.0
    SOURCE - e.g. A=1
    PERT   - e.g. B KO, C KO, D CA
    NODE   - the observed node
    VALUE  - one of OFF, OFF/Some, Some, Some/ON, ON

    """
    # find source nodes
    source_vars = []
    for node in primes:
        if primes[node] == [[{node:0}],[{node:1}]]:
            source_vars.append(node)

    fp = open(file_name, "w")
    fp.write("ID\tScore\tSource\tPerturbation\tObserved node\tCategorization\n")
    for exp in experiments:
        fp.write(str(exp[0]) + "\t")
        fp.write(str(exp[1]) + "\t")
        sources = []
        interv = []
        for fix in exp[2]:
            if fix[0] in source_vars:
                sources.append(fix[0] + "=" + str(fix[1]))
            else:
                if fix[1] == 1:
                    interv.append(fix[0] + " CA")
                else:
                    interv.append(fix[0] + " KO")
        fp.write(",".join(sources) + "\t")
        fp.write(",".join(interv) + "\t")
        fp.write(exp[3] + "\t")
        fp.write(exp[4] + "\n")

    print("Exported generated experiments to", os.path.abspath(file_name))

def train_and_valid(experiments:list[ExpType], ratio:float|None=None, valid_ids:list[int]|None=None,
                    seed:int|None = None,
                    ) -> tuple[list[ExpType],list[ExpType],list[int]]:
    """
    Given the artificial experiments, generate a training set and a validation set.
    If given the ratio, the validation set is randomly created with the given ratio.
    The training set is the rest.
    If given an ID list, the validation set is created with the given ids.

    Parameters
    ----------
    experiments - list of exp                           :list[ExpType]
        exp     - info of a single experiment           :ExpType = tuple[int, float, FixesType, str, str]
            exp[0] - id of the experiment               :int
            exp[1] - max_score for the experiment       :float
            exp[2] - fixes                              :FixesType = tuple[tuple[str, int]]
                     ((node A, value1), (node B, value2), ...)
            exp[3] - observed_node                      :str
            exp[4] - outcome_value                      :str
                     one of OFF, OFF/Some, Some, Some/ON, ON

    ratio - ratio of the validation set                         :float|None
    id_list - list of IDs of experments for the validation set  :list[int]|None
    seed - random seed                                          :int|None

    Returns
    -------
    training_exps - list of exp                          :list[ExpType]
                    contains (1-ratio) of experiments
                    or all experiments excluding the id_list

    validation_exps - list of exp                        :list[ExpType]
                      contains all of experiments.
                      score is set to 0 for (1-ratio) of experiments
                      or for all experiments excluding the id_list
                      It is necessary for the validation set to contain all experiments,
                      so that hierachy scoring works.

    valid_ids - same as the input id_list if it was given     :list[int]
                or newly created if given the ratio
    """

    if seed != None:
        random.seed(seed)

    n_exps = len(experiments)
    ids = [exp[0] for exp in experiments]
    if ratio != None:
        n_valid = int(n_exps * ratio)
        valid_ids = sorted(random.sample(ids,n_valid))
    else:
        assert valid_ids != None, "either ratio or id_list should be given"

    training_exps = [exp for exp in experiments if exp[0] not in valid_ids]

    validation_exps = []
    for exp in experiments:
        if exp[0] in valid_ids:
            validation_exps.append(exp)
        else:
            lst = list(exp)
            lst[1] = 0.0
            validation_exps.append(tuple(lst))

    return training_exps, validation_exps, sorted(valid_ids)