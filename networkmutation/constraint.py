# TODO: enable check_node function to check constant nodes, source nodes,
# non-soure nodes, nodes with single regulators and fixed functions

import conversions as conv

def check_node(regulators, rr, base_rr, constraints, node):
    """
    Checks if the model follows regulate, necessary, possible_constant constraints.
    Checking fixed and group constraints are not implemented yet.

    Parameters
    ----------
    regulators  - the regulating nodes                      : tuple(str)
    rr          - representation of the rule                : length 2^N binary str
    base_rr     - representation of the the baseline rule   : length 2^N binary str
    constraints - represents 5 types of constraints         : dict{str: set or dict}
                  (fixed, regulate, necessary, group, possible_constant)
    node        - the target node                           : str

    Returns
    -------
    check       - True if follows constraints               : bool
    
    """

    # a constant node should not be mutated
    if len(regulators) == 0:
        if rr == base_rr:
            return True
        else:
            print("Constant node", node, "was mutated")
            return False

    # a source node should not be mutated
    if len(regulators) == 1 and regulators[0] == node:
        if rr == base_rr:
            return True
        else:
            print("Source node", node, "was mutated")
            return False

    # check constraint - fixed
    if node in constraints['fixed']:
        if rr == base_rr:
            return True
        else:    
            print(node, "with fixed function was mutated")
            return False

    if node in constraints['necessary']:
        for necc in constraints['necessary'][node]:
            if not check_necessary(regulators, rr, necc):
                print(necc, "should be necessary for", node)
                return False

    if node in constraints['regulate']:
        for reg in constraints['regulate'][node]:
            if not check_regulation(regulators, rr, reg):
                print(reg, "should regulate", node)
                return False

    # node with a self loop should not become a source node
    if node in regulators:
        if check_source(regulators, rr, node):
            print(node, "should not be a source")
            return False

    # only nodes that were originally a constant node
    # or nodes in possible_constant can become constants
    if check_constant(rr):
        if node not in constraints['possible_constant'] and len(base_rr) != 1:
            print(node, "should not be a constant")
            return False
        # a constant node should not have its value changed
        elif base_rr == '1' and '0' in rr:
            print(node, "has opposite constant value")
            return False
        elif base_rr == '0' and '1' in rr:
            print(node, "has opposite constant value")
            return False

    return True

def check_regulation(regulators, rr, node, is_min = False):
    """
    Check if a node does appear in the rule
    """
    assert node in regulators, "The node should be one of the regulators"

    check = False

    # N = number of regulators
    N = len(regulators)

    # bi = the minimal representation.
    if is_min:
        bi = rr
    else:
        bi = conv.get_uni_rr(rr, max = False)

    # n = nth regulator
    n = regulators.index(node)

    add = 0
    for i in range(2**N):
        if i % 2**(N-n) <= 2**(N-1-n)-1:
            add += int(bi[i])

    if add != 0:
        check = True
    return check

def group_bi2bi(group_bi, group_regulators):
    """
    Returns the binary position of the old representation
    when given a binary position of the group representation
    """
    num = iter(group_bi)
    implicant = {}
    for group in group_regulators:
        if type(group) == tuple:
            value = next(num)
            for node in group:
                if node not in implicant.keys():
                    implicant[node] = value
                elif implicant[node] == '0':
                    implicant[node] = value
                else:
                    pass
        else:
            implicant[group] = next(num)
    implicant = dict(sorted(implicant.items()))
    bi = ''.join(implicant.values())
    return bi

def rr2group_rr(regulators, rr, groups):

    for group in groups:
        for node in group:
            assert node in regulators, f"{node} is not one of the regulators {regulators}"

    group_regulators = set(regulators)
    for i, group in enumerate(groups):
        # print('group:', group)
        group_regulators.add(group)
        group_regulators -= set(group)
    group_regulators = list(group_regulators)
    # print('group_regulators:',group_regulators)

    N = len(group_regulators)
    group_rr = ['0']*(2**N)

    for i in range(2**N):
        group_bi = format(2**N - 1 - i, '0' + str(N) + 'b')
        bi = group_bi2bi(group_bi, group_regulators)
        if rr[-int(bi, 2)-1] == '1':
            group_rr[i] = '1'

    group_rr = ''.join(group_rr)
    return group_rr, group_regulators

def group_rr2rr(regulators, group_rr, groups, group_regulators):
    N = len(group_regulators)
    M = len(regulators)
    rr = ['0']*(2**M)

    for i in range(2**N):
        group_bi = format(2**N - 1 - i, '0' + str(N) + 'b')
        bi = group_bi2bi(group_bi, group_regulators)
        # print('bi:',bi)
        if group_rr[i] == '1':
            rr[-int(bi, 2)-1] = '1'

    rr = ''.join(rr)
    return rr

def check_necessary(regulators, rr, node):
    """
    Check if a node is necessary for the rule
    """
    assert node in regulators, "The node should be one of the regulators"

    check = False

    # N = number of regulators
    N = len(regulators)

    # n = nth regulator
    n = regulators.index(node)

    if check_regulation(regulators, rr, node) == False:
        return False

    add = 0
    for i in range(2**N):
        if i % 2**(N-n) > 2**(N-1-n)-1:
            add += int(rr[i])

    if add == 0:
        check = True
    return check

def impose_necessary(regulators, rr, node):
    """
    Impose the necessary condition for the rule
    """
    assert node in regulators, "The node should be one of the regulators"

    # N = number of regulators
    N = len(regulators)
    # n = nth regulator
    n = regulators.index(node)

    bi = list(rr)
    for i in range(2**N):
        if i % 2**(N-n) > 2**(N-1-n)-1:
            bi[i] = '0'

    necessary_rr = ''.join(bi)

    return necessary_rr

def check_constant(rr):
    """
    Check if the rule is fixed to 0 or 1
    """
    max_rr = conv.get_uni_rr(rr, max = True)
    if '0' in max_rr and '1' in max_rr:
        return False
    else:
        return True

def check_source(regulators, rr, node):
    """
    Check if the rule became a source
    """
    # N = number of regulators
    N = len(regulators)
    # n = nth regulator
    n = regulators.index(node)

    min_rr = conv.get_uni_rr(rr, max = False)
    add = 0
    for i in range(2**N):
        add += int(min_rr[i])

    if add == 1 and min_rr[2**N-2**(N-n-1)-1] == '1':
        return True
    else:
        return False