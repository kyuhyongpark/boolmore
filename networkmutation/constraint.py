import mutation as m

def check_node(regulators, rr, constraints, node):
    """
    Checks if the model follows regulate, necessary, possible_source constraints.
    Checking fixed and group constraints are not implemented yet.
    """

    check = True
    # constant nodes are not mutated
    if len(regulators) == 0:
        return check
    # source nodes are not mutated
    if len(regulators) == 1 and regulators[0] == node:
        return check

    if node in constraints['necessary']:
        for necc in constraints['necessary'][node]:
            check = check_necessary(regulators, rr, necc)
            if check == False:
                print(necc, "should be necessary for", node)
                return check

    if node in constraints['regulate']:
        # check if the constrained nodes are regulating the target
        for reg in constraints['regulate'][node]:
            check = check_regulation(regulators, rr, reg)
            if check == False:
                print(reg, "should regulate", node)
                return check
    elif node not in constraints['possible_source']:

        # need to check if a node became a source node and mutate more if it did
        # nodes that have a regulating node cannot be a source, and hence elif.
        check = not check_source(rr)
        if check == False:
            print(node, "should not be a source")

    return check

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
        bi = m.get_uni_rr(rr, max = False)

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
            assert node in regulators, "The node should be one of the regulators"

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

def check_source(rr):
    """
    Check if the rule is fixed to 0 or 1
    """
    max_rr = m.get_uni_rr(rr, max = True)
    if '0' in max_rr and '1' in max_rr:
        return False
    else:
        return True
