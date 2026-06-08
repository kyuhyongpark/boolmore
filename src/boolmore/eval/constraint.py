# TODO: enable check_node function to check group contraints


import boolmore.core.conversions as conv


def check_constant(rr:str) -> bool:
    """
    Checks if the rule is fixed to a constant.

    Parameters
    ----------
    rr - representation of the rule     :length 2^k binary str

    Returns
    -------
    check - True if the rule is constant    :bool

    """
    max_rr = conv.get_uni_rr(rr, max = True)
    if '0' in max_rr and '1' in max_rr:
        return False
    else:
        return True


def check_source(regulators:tuple[str, ...], rr:str, node:str) -> bool:
    """
    Check if the node became a source node.

    Parameters
    ----------
    regulators - the regulating nodes           :length k tuple[str]
    rr         - representation of the rule     :length 2^k binary str
    reg        - the target node to check       :str

    Returns
    -------
    check - True if the node became a source    :bool
    
    """
    # k = number of regulators
    k = len(regulators)
    # n = nth regulator
    n = regulators.index(node)

    min_rr = conv.get_uni_rr(rr, max = False)
    add = 0
    for i in range(2**k):
        add += int(min_rr[i])

    if add == 1 and min_rr[2**k-2**(k-n-1)-1] == '1':
        return True
    else:
        return False


def check_regulate(regulators:tuple[str, ...], rr:str, reg:str, is_min:bool=False) -> bool:
    """
    Checks if a regulator appears in the rule in a non-redundant manner.

    Parameters
    ----------
    regulators - the regulating nodes               :length k tuple[str]
    rr         - representation of the rule         :length 2^k binary str
    reg        - the regulating node to check       :str
    is_min     - True if minimal representation     :bool

    Returns
    -------
    check - True if the regulator appears   :bool

    """

    assert reg in regulators, "The node should be one of the regulators"

    check = False

    # k = number of regulators
    k = len(regulators)
    # n = nth regulator
    n = regulators.index(reg)

    # bi = the minimal representation.
    if is_min:
        bi = rr
    else:
        bi = conv.get_uni_rr(rr, max = False)

    add = 0
    for i in range(2**k):
        if i % 2**(k-n) <= 2**(k-1-n)-1:
            add += int(bi[i])

    if add != 0:
        check = True

    return check


def check_necessary(regulators:tuple[str, ...], rr:str, nec:str) -> bool:
    """
    Checks if the node is necessary for the rule.

    Parameters
    ----------
    regulators - the regulating nodes                   :length k tuple[str]
    rr         - representation of the rule             :length 2^k binary str
    nec        - the node to check whether necessary    :str

    Returns
    -------
    check - True if the regulator appears   :bool

    """
    assert nec in regulators, "The node should be one of the regulators"

    check = False

    # k = number of regulators
    k = len(regulators)
    # n = nth regulator
    n = regulators.index(nec)

    if check_regulate(regulators, rr, nec) == False:
        return False

    add = 0
    for i in range(2**k):
        if i % 2**(k-n) > 2**(k-1-n)-1:
            add += int(rr[i])

    if add == 0:
        check = True
    return check


def check_node(regulators:tuple[str, ...], rr:str, base_rr:str, constraints:dict, node:str) -> bool:
    """
    Checks if the model follows fixed, regulate, necessary, possible_constant constraints.
    Also checks if a non-source became a source, or if a non-constant became a constant.

    Checking group constraints are not implemented yet.

    Parameters
    ----------
    regulators  - the regulating nodes                      :length (k+a) tuple[str]
                  Note that there can be added regulators
    rr          - representation of the rule                :length 2^(k+a) binary str
    base_rr     - representation of the the baseline rule   :length 2^k binary str
    constraints - represents 5 types of constraints         :dict[str, list or dict]
                  (fixed, regulate, necessary, group, possible_constant)
    node        - the target node                           :str

    Returns
    -------
    check - True if follows constraints     :bool
    
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
            if not check_regulate(regulators, rr, reg):
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
        max_rr = conv.get_uni_rr(rr, max = True)
        if node not in constraints['possible_constant'] and len(base_rr) != 1:
            print(node, "should not be a constant")
            return False
        # a constant node should not have its value changed
        elif base_rr == '1' and '0' in max_rr:
            print(node, "has opposite constant value")
            return False
        elif base_rr == '0' and '1' in max_rr:
            print(node, "has opposite constant value")
            return False

    return True






