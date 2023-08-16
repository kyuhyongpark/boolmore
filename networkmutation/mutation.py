import random
import conversions as conv
import constraint as cons


def mutate_rr(rr, probability):
    '''
    Returns the mutated representation of that rule when given any representation of a rule.
    This function is biased to activate all the implicants.
    Use mutate_rr_bias for non-biased or specific biased mutations.

    Parameters
    ----------
    rr : length 2^N binary str
        representation of the rules
    probability : float between 0 and 1
        probability that each of the binary number in the representation is mutated.

    Returns
    -------
    mutated_rr : length 2^N binary str
        representation of the rules
    '''
    lst = []
    for num in rr:
        lst.append(int(num))
    for i in range(len(lst)):
        rnd = random.random()
        if rnd < probability:
            lst[i] += 1
    mutated_rr = ''
    for num in lst:
        mutated_rr += str(num % 2)
    return mutated_rr

def mutate_rr_bias(rr, probability, bias = 0.5):
    """
    When given any representation of a rule,
    returns the mutated representation of that rule
    with a certain inacitivity bias.

    Parameters
    ----------
    rr : length 2^N binary str
        representation of the rules
    probability : float between 0 and 1
        probability that each of the binary number in the representation is mutated.
    bias : float between 0 and 1
        bias to deactivate implicants

    Returns
    -------
    mutated_rr : length 2^N binary str
        representation of the rules
    """
    rnd = random.random()
    if rnd < bias:
        irr = conv.get_max_irr(rr)
        irr = mutate_rr(irr, probability)
        mutated_rr = conv.get_max_irr(irr)
    else:
        mutated_rr = mutate_rr(rr, probability)

    return mutated_rr


def mutate_rr_constraint(regulators:tuple[str], rr:str, base_rr:str, constraints:dict,
                         node:str, probability:float, bias:float=0.5) -> tuple[str, bool]:
    """
    When given any representation of a rule,
    returns the mutated representation of that rule
    with a certain inacitivity bias
    that follows all the constraints.

    Parameters
    ----------
    regulators  - the regulating nodes                      : tuple(str)
    rr          - representation of the rule                : length 2^N binary str
    base_rr     - representation of the the baseline rule   : length 2^N binary str
    constraints - represents 5 types of constraints         : dict{str: set or dict}
                  (fixed, regulate, necessary, group, possible_constant)
    node        - the target node                           : str
    probability - probability that each binary is mutated   : float between 0 and 1
    bias        - bias to deactivate implicants             : float between 0 and 1

    Returns
    -------
    mutated_rr  - representation of the mutated rule        : length 2^N binary str
    modified    - True if modified                          : bool

    """

    # constant nodes have no degree of freedom and should not be mutated
    if len(regulators) == 0:
        if rr == base_rr:
            return rr, False
        else:
            return base_rr, True

    # nodes with single regulators (including source nodes)
    # have no degree of freedom, and hence should not be mutated
    # UNLESS it was originally a constant node but gained an extra edge,
    # in which case it can become a constant node
    if len(regulators) == 1 and len(base_rr) != 1:
        if rr == base_rr:
            return rr, False
        else:
            return base_rr, True

    # impose constraints - fixed
    if node in constraints['fixed']:
        if rr == base_rr:
            return rr, False
        else:
            return base_rr, True

    mutated_rr = ''
    redo = True
    trial = 0
    while redo == True:
        if trial > 1000:
            raise Exception("too many trials. check mutation parameter")
        
        ### mutation module ###
        # impose constraint - group
        if node in constraints['group']:
            groups = constraints['group'][node]
            group_rr, group_regulators = cons.rr2group_rr(regulators, rr, groups)
            mutated_group_rr = mutate_rr_bias(group_rr, probability, bias)
            mutated_rr = cons.group_rr2rr(regulators, mutated_group_rr, groups, group_regulators)
        else:
            mutated_rr = mutate_rr_bias(rr, probability, bias)

        # impose constraint - necessary
        if node in constraints['necessary']: 
            for necc in constraints['necessary'][node]:
                mutated_rr = cons.impose_necessary(regulators, mutated_rr, necc)
        ### end of mutation ###

        redo = False
        # check constraint - regulate
        if node in constraints['regulate']:
            for reg in constraints['regulate'][node]:
                redo = redo or not cons.check_regulation(regulators, mutated_rr, reg)
                if redo == True:
                    # print('redo to ensure', reg, 'regulates', node)
                    break
        
        # node with a self loop should not become a source node
        elif node in regulators:
            redo = redo or cons.check_source(regulators, mutated_rr, node)
            # if redo == True:
                # print('redo to ensure', node, 'is not a source')
        
        # only nodes that were originally a constant node
        # or nodes in possible_constant can become constants
        if cons.check_constant(mutated_rr):
            max_rr = conv.get_uni_rr(mutated_rr, max = True)
            if node not in constraints['possible_constant'] and len(base_rr) != 1:
                redo = True
                # print('redo because', node, 'became a constant')
            # a constant node should not have its value changed
            elif base_rr == '1' and '0' in max_rr:
                redo = True
                # print('redo because', node, 'got opposite value')
            elif base_rr == '0' and '1' in max_rr:
                redo = True
                # print('redo because', node, 'got opposite value')

        if redo == True:
            trial += 1
            print(trial)

    max_original = conv.get_uni_rr(rr)
    max_mutated = conv.get_uni_rr(mutated_rr)
    if max_original == max_mutated:
        modified = False
    else:
        modified = True

    return mutated_rr, modified


def add_regulator(regulators:tuple[str], rr:str, signs:str, new_regulator:str, new_sign:str, bias:float=0.5):
    """
    Returns a new representation of a rule with an extra regulator.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    new_regulator : str
        the added regulating node
    new_sign : length 1 binary str
        represents sign of the new regulator. 0 means negation of the node

    Returns
    -------
    added_regulators : tuple of strings
        the regulating nodes
    added_rr : length 2^(N+1) binary str
        representation of the rules
    added_signs : length N+1 binary str
        represents signs of regulators. 0 means negation of the node

    """
    nodes = list(regulators)
    nodes.append(new_regulator)
    added_regulators = tuple(nodes)

    added_signs = signs + new_sign

    N = len(regulators)

    rnd = random.random()
    if rnd < bias:
        rr = conv.get_max_irr(rr)

    added_rr = ''
    for bi in rr:
        added_rr += str(random.randint(0,1))
        added_rr += bi

    if rnd < bias:
        added_rr = conv.get_max_irr(added_rr)

    return added_regulators, added_rr, added_signs


def delete_regulator(regulators, rr, signs, target_regulator, bias=0.5):
    """
    Returns a new representation of a rule with the target regulator deleted.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    target_regulator : str
        the regulatort to be deleted

    Returns
    -------
    deleted_regulators : tuple of strings
        the regulating nodes
    deleted_rr : length 2^(N-1) binary str
        representation of the rules
    deleted_signs : length N-1 binary str
        represents signs of regulators. 0 means negation of the node

    """
    N = len(regulators)
    n = regulators.index(target_regulator)

    nodes = list(regulators)
    del nodes[n]
    deleted_regulators = tuple(nodes)

    deleted_signs = signs[:n] + signs[n+1:]

    deleted_rr = ''

    rnd = random.random()
    if rnd < bias:
        rr = conv.get_max_irr(rr)

    bi = list(rr)
    for i in range(2**N):
        if i % 2**(N-n) > 2**(N-1-n)-1:
            deleted_rr += bi[i]

    if rnd < bias:
        deleted_rr = conv.get_max_irr(deleted_rr)

    return deleted_regulators, deleted_rr, deleted_signs
