import random
import math
import itertools as it
import constraint as cons
import Model
import config

def rr2bnet(regulators, rr, signs):
    """
    Returns rules when given the regulating nodes, the rule representation
    and the node signs.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        rule representation
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node

    Returns
    -------
    bnet : str
        bnet formatted rules

    """
    nodes = list(regulators)
    N = len(nodes)

    assert 2**N == len(rr), "The length of rr should be equal to 2^(number of nodes)"
    assert N == len(signs), "signs should be a list with the number of nodes as length"

    for i, num in enumerate(signs):
        if num == '0':
            nodes[i] = '!' + nodes[i]

    # start constructing bnet = (0 and A and B and C) or (1 and A and B) or ...
    bnet = '('
    for i, num in enumerate(rr):
        bnet += str(num)
        # bin = 111, 110, 101, ...
        bi = format(2**N-i-1, '0' + str(N) + 'b')
        for j, node in enumerate(nodes):
            if bi[j] == '1':
                bnet += ' & ' + node
        bnet += ') | ('
    bnet = bnet.removesuffix(' | (')

    return bnet

def prime2rr(prime, regulators = None, signs = None):
    """
    Returns the representation of the rule of a node
    when given the prime implicants of that rule.

    Parameters
    ----------
    prime : list of list of dictionaries
        prime implicants of the rule
    regulators : tuple of strings
        the regulating nodes
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node

    Returns
    -------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        rule representation
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    """
    # Get the tuple of the regulator nodes
    if regulators == None:
        nodes = set()
        for dct in prime[1]:
            nodes = nodes.union(set(dct.keys()))
        nodes = list(nodes)
        nodes.sort()
        regulators = tuple(nodes)

    # check if the node is a fixed node
    if len(regulators) == 0:
        signs = ''
        if bool(prime[1]): # 1 implicant is not empty
            rr = '1'
        else:
            rr = '0'
        return regulators, rr, signs

    # Get the rr and signs of the regulators
    rr = ['0'] * (2**len(regulators))
    if signs == None:
        signs = ['1'] * len(regulators)
        for implicant in prime[1]:
            # check whether certain node is inside
            for i, node in enumerate(regulators):
                if node in implicant:
                    if implicant[node] == 0:
                        signs[i] = '0'
        signs = ''.join(signs)
    # Get a binary number that gives us the position of the implicant on the rr
    # for every piece of rule that gives 1 to the regulated node,
    for implicant in prime[1]:
        bi = ''
        # check whether certain node is inside
        for i, node in enumerate(regulators):
            if node in implicant:
                # if it exists, put 1
                bi += '1'
            else:
                # if it does not exist, put 0
                bi += '0'
        # put 1 on that position of the rr
        rr[-int(bi, 2)-1] = '1'

    rr = ''.join(rr)

    return regulators, rr, signs

def get_uni_rr(rr, max = True):
    """
    Returns the unique representation of the rule of a node(max or min)
    when given any representation of that rule.

    Parameters
    ----------
    rr : length 2^N binary str
        represents the rules
    max : Boole
        if True, get maximal representation
        if False, get minimal representation

    Returns
    -------
    uni_rr : length 2^N binary str
        unique representation of the rules
        if maximal, it is equivalent to the truth table
        if minimal, it is in a Blake canonical form

    """
    # N = number of regulators
    n = len(rr)
    N = int(math.log2(n))

    # already minimal if the node is a fixed node
    if N == 0:
        return rr

    uni_rr = list(rr)

    modified = []

    # iterate through the string in reverse
    for i in range(n):
        # get the regulator values if there is '1'
        if uni_rr[-i-1] == '1':
            # bi represents the regulator values in binary string
            bi = format(i, '0' + str(N) + 'b')
        else:
            continue
        # find all the positions that need to be changed to 0
        if bi in modified:
            continue

        lst = []
        for num in bi:
            if num == '0':
                lst.append(['0', '1'])
            else:
                lst.append(['1'])
        positions = [''.join(p) for p in it.product(*lst)]
        # do not change the original position
        positions.remove(bi)
        modified.extend(positions)
        # change to '0' if min or '1' if max in the gained positions
        for position in positions:
            j = n - int(position,2) - 1
            uni_rr[j] = str(int(max))

    uni_rr = ''.join(uni_rr)

    return uni_rr

def get_max_irr(rr):
    """
    Returns the inveted maximal representation of the rule of a node
    when given any representation of that rule.

    Parameters
    ----------
    rr : length 2^N binary str
        represents the rule

    Returns
    -------
    max_irr : length 2^N binary str
        inverted maximal representation of the rule

    """
    max_rr = get_uni_rr(rr, max = True)
    max_rr = max_rr.replace('0', 'X')
    max_rr = max_rr.replace('1', '0')
    max_rr = max_rr.replace('X', '1')
    max_irr = max_rr[::-1]

    return(max_irr)

def rr2prime(regulators, rr, signs, inverted = False):
    """
    Returns the prime implicants of the rule of a node
    when given any representation of that rule.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    inverted : Boolean
        If True, get the prime assuming the input rr is inverted.
        Default is False.

    Returns
    -------
    prime : list of list of dictionaries
        prime implicants of the rule

    """
    N = len(regulators)
    assert 2**N == len(rr), "The length of rr should be equal to 2^(number of nodes)"
    assert N == len(signs), "signs should be a list with the numbe of nodes as length"

    inverted = int(inverted)
    min_rr = get_uni_rr(rr, max = False)
    max_irr = get_max_irr(rr)
    min_irr = get_uni_rr(max_irr, max = False)

    prime = [[],[]]
    for i in range(len(rr)):
        # get the regulator values if there is '1'
        if min_rr[i] == '1':
            # bi represents the regulator values in binary string
            bi = format(2**N - i - 1, '0' + str(N) + 'b')
        else:
            continue
        implicant = {}
        for i, num in enumerate(bi):
            if num == '1':
                implicant[regulators[i]] = (int(signs[i]) - inverted)%2
        prime[1-inverted].append(implicant)

    for i in range(len(rr)):
        # get the regulator values if there is '1'
        if min_irr[i] == '1':
            # bi represents the regulator values in binary string
            bi = format(2**N - i - 1, '0' + str(N) + 'b')
        else:
            continue
        implicant = {}
        for i, num in enumerate(bi):
            if num == '1':
                implicant[regulators[i]] = (1-int(signs[i]) - inverted)%2
        prime[0-inverted].append(implicant)

    return prime


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
        irr = get_max_irr(rr)
        irr = mutate_rr(irr, probability)
        mutated_rr = get_max_irr(irr)
    else:
        mutated_rr = mutate_rr(rr, probability)

    return mutated_rr


def mutate_rr_constraint(regulators:tuple[str], rr:str, base_rr:str, constraints:dict, node:str, probability:float, bias:float=0.5):
    """
    When given any representation of a rule,
    returns the mutated representation of that rule
    with a certain inacitivity bias
    that follows all the constraints.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rule
    base_rr : length 2^N binary str
        representation of the the baseline rule
    constraints : dictionary of dictionary of tuples
        represents 5 types of constraints
        fixed
        regulate
        necessary
        group
        possible_constant
    node : string
        the target node
    probability : float between 0 and 1
        probability that each of the binary number in the representation is mutated.
    bias : float between 0 and 1
        bias to deactivate implicants

    Returns
    -------
    mutated_rr : length 2^N binary str
        representation of the rules
    modified : Boole
        True if modified
    """
    # no need to mutate source nodes
    if len(regulators) == 1 and regulators[0] == node:
        return rr, False

    # in case of constant nodes, ensure that it has the correct value
    if len(regulators) == 0:
        if rr != base_rr:
            return base_rr, True
        else:
            return rr, False

    # impose constraints - fixed nodes
    if node in constraints['fixed']:
        if rr != base_rr:
            return base_rr, True
        else:
            return rr, False

    redo = True
    trial = 0
    while redo == True:
        if trial > 1000:
            raise Exception("too many trials. check mutation parameter")
        ### mutation module
        if node in constraints['group']: # impose group constraint
            groups = constraints['group'][node]
            group_rr, group_regulators = cons.rr2group_rr(regulators, rr, groups)
            mutated_group_rr = mutate_rr_bias(group_rr, probability, bias)
            mutated_rr = cons.group_rr2rr(regulators, mutated_group_rr, groups, group_regulators)
        else: # no group constraints
            mutated_rr = mutate_rr_bias(rr, probability, bias)

        if node in constraints['necessary']: # impose necessary constraint
            for necc in constraints['necessary'][node]:
                mutated_rr = cons.impose_necessary(regulators, mutated_rr, necc)
        ### end of mutation

        redo = False
        if node in constraints['regulate']:
            # check if the constrained nodes are regulating the target
            for reg in constraints['regulate'][node]:
                redo = not cons.check_regulation(regulators, mutated_rr, reg)
                if redo == True:
                    trial += 1
                    break
        # nodes that do not have a guaranteed regulator may become a source or a constant,
        # which may have to be prevented.
        else:
            # check if the node became a source node if it has a self loop.
            if node in regulators:
                redo = cons.check_source(regulators, mutated_rr, node)
            # need to check if a node became a constant node and mutate more if it did
            if node not in constraints['possible_constant']:
                redo = cons.check_constant(mutated_rr)
            if redo == True:
                trial += 1

    max_original = get_uni_rr(rr)
    max_mutated = get_uni_rr(mutated_rr)
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
        rr = get_max_irr(rr)

    added_rr = ''
    for bi in rr:
        added_rr += str(random.randint(0,1))
        added_rr += bi

    if rnd < bias:
        added_rr = get_max_irr(added_rr)

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
        rr = get_max_irr(rr)

    bi = list(rr)
    for i in range(2**N):
        if i % 2**(N-n) > 2**(N-1-n)-1:
            deleted_rr += bi[i]

    if rnd < bias:
        deleted_rr = get_max_irr(deleted_rr)

    return deleted_regulators, deleted_rr, deleted_signs
