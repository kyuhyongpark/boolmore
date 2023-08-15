import math
import itertools as it


def prime2rr(prime:list[list[dict[str,int]]], regulators:tuple[str] | None = None, signs:str | None = None):
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


def rr2bnet(regulators:tuple[str], rr, signs):
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
