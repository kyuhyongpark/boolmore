import math
import itertools as it


PrimeType = list[list[dict[str, int]]]


def prime2rr(prime:PrimeType,
             regulators:tuple[str]|None=None, signs:str|None=None) -> tuple[tuple[str],str,str]:
    """
    Returns a binary representation, regulators and signs of a rule
    when given prime implicants of a rule.

    Parameters
    ----------
    prime      - prime implicants of the rule           :PrimeType = list[list[dict[str,int]]]
    regulators - the regulating nodes                   :length k tuple[str]
    signs      - represents the signs of regulators     :length k binary str
                 0 means negative regulation                                 

    Returns
    -------
    regulators - the regulating nodes                   :length k tuple[str]
    rr         - representation of the rule             :length 2^k binary str
    signs      - represents the signs of regulators     :length k binary str
                 0 means negative regulation
    
    """
    # Get the tuple of the regulator nodes
    if regulators == None:
        nodes = set()
        for dct in prime[1]:
            nodes = nodes.union(set(dct.keys()))
        nodes = list(nodes)
        nodes.sort()
        regulators = tuple(nodes) # type: ignore

    # check if the node is a fixed node
    if len(regulators) == 0: # type: ignore
        signs = ''
        if bool(prime[1]): # 1 implicant is not empty
            rr = '1'
        else:
            rr = '0'
        return regulators, rr, signs # type: ignore

    # Get the rr and signs of the regulators
    rr = ['0'] * (2**len(regulators)) # type: ignore
    if signs == None:
        signs_list = ['1'] * len(regulators) # type: ignore
        for implicant in prime[1]:
            # check whether certain node is inside
            for i, node in enumerate(regulators): # type: ignore
                if node in implicant:
                    if implicant[node] == 0:
                        signs_list[i] = '0'
        signs = ''.join(signs_list)
    # Get a binary number that gives us the position of the implicant on the rr
    # for every piece of rule that gives 1 to the regulated node,
    for implicant in prime[1]:
        bi = ''
        # check whether certain node is inside
        for i, node in enumerate(regulators): # type: ignore
            if node in implicant:
                # if it exists, put 1
                bi += '1'
            else:
                # if it does not exist, put 0
                bi += '0'
        # put 1 on that position of the rr
        rr[-int(bi, 2)-1] = '1'

    rr = ''.join(rr)

    return regulators, rr, signs # type: ignore


def rr2prime(regulators:tuple[str], rr:str, signs:str, inverted:bool=False) -> PrimeType:
    """
    Returns prime implicants of a rule
    when given regulators, signs and a binary representation of a rule.

    Parameters
    ----------
    regulators - the regulating nodes                       :length k tuple[str]
    rr         - representation of the rule                 :length 2^k binary str
    signs      - represents the signs of regulators         :length k binary str
                 0 means negative regulation
    inverted   - if true, input rr is taken as inverted     :bool
                 default is False

    Returns
    -------
    prime - prime implicants of the rule    :PrimeType = list[list[dict[str,int]]]
        

    """
    k = len(regulators)
    assert 2**k == len(rr), "The length of rr should be equal to 2^(number of nodes)"
    assert k == len(signs), "signs should be a list with the numbe of nodes as length"

    inv = int(inverted)
    min_rr = get_uni_rr(rr, max = False)
    max_irr = get_max_irr(rr)
    min_irr = get_uni_rr(max_irr, max = False)

    prime = [[],[]]
    for i in range(len(rr)):
        # get the regulator values if there is '1'
        if min_rr[i] == '1':
            # bi represents the regulator values in binary string
            bi = format(2**k - i - 1, '0' + str(k) + 'b')
        else:
            continue
        implicant = {}
        for i, num in enumerate(bi):
            if num == '1':
                implicant[regulators[i]] = (int(signs[i]) - inv)%2
        prime[1-inv].append(implicant)

    for i in range(len(rr)):
        # get the regulator values if there is '1'
        if min_irr[i] == '1':
            # bi represents the regulator values in binary string
            bi = format(2**k - i - 1, '0' + str(k) + 'b')
        else:
            continue
        implicant = {}
        for i, num in enumerate(bi):
            if num == '1':
                implicant[regulators[i]] = (1-int(signs[i]) - inv)%2
        prime[0-inv].append(implicant)

    return prime


def rr2bnet(regulators:tuple[str], rr:str, signs:str) -> str:
    """
    Returns a bnet string
    when given regulators, signs and a binary representation of a rule.

    Parameters
    ----------
    regulators - the regulating nodes                       :length k tuple[str]
    rr         - representation of the rule                 :length 2^k binary str
    signs      - represents the signs of regulators         :length k binary str
                 0 means negative regulation

    Returns
    -------
    bnet - bnet formatted rules     :str
        
    """
    nodes = list(regulators)
    k = len(nodes)

    assert 2**k == len(rr), "The length of rr should be equal to 2^(number of nodes)"
    assert k == len(signs), "signs should be a list with the number of nodes as length"

    for i, num in enumerate(signs):
        if num == '0':
            nodes[i] = '!' + nodes[i]

    # start constructing bnet = (0 and A and B and C) or (1 and A and B) or ...
    bnet = '('
    for i, num in enumerate(rr):
        bnet += str(num)
        # bin = 111, 110, 101, ...
        bi = format(2**k-i-1, '0' + str(k) + 'b')
        for j, node in enumerate(nodes):
            if bi[j] == '1':
                bnet += ' & ' + node
        bnet += ') | ('
    bnet = bnet.removesuffix(' | (')

    return bnet

def prime2bnet(node:str, prime:PrimeType) -> str:
    """
    Parameters
    ----------
    node  - the node for which the prime is given   :str
    prime - prime implicants of the rule            :PrimeType = list[list[dict[str,int]]]

    Returns
    -------
    bnet - bnet formatted rules     :str
    """
    k = node
    v = prime
    
    s = k + ",\t"
    sl = []
    for c in v[1]:
        sll = []
        for kk,vv in sorted(c.items()):
            if vv: sli = kk
            else: sli = '!'+kk
            sll.append(sli)
        if len(sll) > 0:
            sl.append(' & '.join(sll))
    if len(sl) > 0:
        s += ' | '.join(sl)
    if v[1]==[]:
        s = k + ",\t0"
    if v[1]==[{}]:
        s = k + ",\t1"
    
    return s

def get_uni_rr(rr:str, max:bool=True) -> str:
    """
    Returns the unique binary representation (max or min) of a rule
    when given any binary representation of a rule.

    Parameters
    ----------
    rr  - representation of the rule            :length 2^k binary str
    max - if True, get maximal representation   :bool
          if False, get minimal representation

    Returns
    -------
    uni_rr - unique representation of the rules                 :length 2^k binary str
             if maximal, it is equivalent to the truth table
             if minimal, it is in a Blake canonical form

    """
    # k = number of regulators
    n = len(rr)
    k = int(math.log2(n))

    # already minimal if the node is a fixed node
    if k == 0:
        return rr

    uni_rr = list(rr)

    modified = []

    # iterate through the string in reverse
    for i in range(n):
        # get the regulator values if there is '1'
        if uni_rr[-i-1] == '1':
            # bi represents the regulator values in binary string
            bi = format(i, '0' + str(k) + 'b')
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


def get_max_irr(rr:str) -> str:
    """
    Returns the inverted maximal representation of the rule of a node
    when given any representation of the rule.

    Parameters
    ----------
    rr - represents the rule    :length 2^k binary str

    Returns
    -------
    max_irr - inverted maximal representation of the rule   :length 2^k binary str
        
    """
    max_rr = get_uni_rr(rr, max = True)
    max_rr = max_rr.replace('0', 'X')
    max_rr = max_rr.replace('1', '0')
    max_rr = max_rr.replace('X', '1')
    max_irr = max_rr[::-1]

    return(max_irr)
