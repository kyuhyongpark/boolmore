from boolmore.types import Assignment, PrimeType

# RR bit-string convention (public / stored)
# ------------------------------------------
# ``rr`` is a length-``2**k`` binary string with ``index == mask``.
# For regulators ``(A, B)``:
#
#   index:  0   1   2   3
#   term:   1 | B | A | A&B
#
# Regulator at position ``n`` uses bit ``1 << (k - n - 1)`` (first = MSB).
# Newly appended regulators become the LSB.
#
# Note: ``get_max_irr`` still ends with ``[::-1]``. That reverse is semantic
# (input complement after flipping outputs), not an order adapter:
# ``max_irr[mask] = 1 - max_rr[(2**k - 1) ^ mask]``.


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
    rr         - rule in mask order (index == mask)     :length 2^k binary str
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

    # Build rr with index == mask.
    rr = ['0'] * (2**len(regulators)) # type: ignore
    if signs == None:
        signs_list = ['*'] * len(regulators) # type: ignore
        for implicant in prime[1]:
            # check whether certain node is inside
            for i, node in enumerate(regulators): # type: ignore
                if node in implicant:
                    if implicant[node] == 0 and signs_list[i] == '*':
                        signs_list[i] = '0'
                    elif implicant[node] == 1 and signs_list[i] == '*':
                        signs_list[i] = '1'
                    else:
                        if implicant[node] != int(signs_list[i]):
                            print(f"Warning: the sign of the regulator {node} is not consistent in the prime implicants.")
        assert '*' not in signs_list, "The signs of the regulators cannot be determined from the prime implicants. Please check the input prime implicants."
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
        rr[int(bi, 2)] = '1'

    return regulators, ''.join(rr), signs # type: ignore


def rr2prime(regulators:tuple[str], rr:str, signs:str, inverted:bool=False) -> PrimeType:
    """
    Returns prime implicants of a rule
    when given regulators, signs and a binary representation of a rule.

    Parameters
    ----------
    regulators - the regulating nodes                       :length k tuple[str]
    rr         - rule in mask order (index == mask)         :length 2^k binary str
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
    for i in range(len(min_rr)):
        if min_rr[i] != '1':
            continue
        bi = format(i, '0' + str(k) + 'b')
        implicant = {}
        for j, num in enumerate(bi):
            if num == '1':
                implicant[regulators[j]] = (int(signs[j]) - inv)%2
        prime[1-inv].append(implicant)

    for i in range(len(min_irr)):
        if min_irr[i] != '1':
            continue
        bi = format(i, '0' + str(k) + 'b')
        implicant = {}
        for j, num in enumerate(bi):
            if num == '1':
                implicant[regulators[j]] = (1-int(signs[j]) - inv)%2
        prime[0-inv].append(implicant)

    return prime


def rr2bnet(regulators:tuple[str], rr:str, signs:str) -> str:
    """
    Returns a bnet string
    when given regulators, signs and a binary representation of a rule.

    Terms are emitted in ascending mask order (``1 | B | A | A&B`` for two
    regulators). See the module header for the ``rr`` convention.

    Parameters
    ----------
    regulators - the regulating nodes                       :length k tuple[str]
    rr         - rule in mask order (index == mask)         :length 2^k binary str
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

    bnet = '('
    for i, num in enumerate(rr):
        bnet += str(num)
        bi = format(i, '0' + str(k) + 'b')
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

# Smallest regulator count ``k`` where the Cython kernel is used when available.
# Measured locally (py vs cy on dens 0.05/0.3, max/min); Cython wins from k=1
# with >=1.2x margin on this machine (see plan bake-off).
_UNI_RR_CYTHON_MIN_K = 1

try:
    from boolmore._get_uni_rr import get_uni_rr as _get_uni_rr_cy

    _UNI_RR_CYTHON = True
except ImportError:  # pragma: no cover - depends on local extension build
    _get_uni_rr_cy = None  # type: ignore[assignment]
    _UNI_RR_CYTHON = False


def _get_uni_rr_py(rr: str, max: bool = True) -> str:
    """Pure-Python canonical maximal / minimal rr (mask-order)."""
    n = len(rr)

    if n <= 1:
        return rr

    uni_rr = list(rr)

    modified = set()
    fill_value = "1" if max else "0"

    for i in range(n):
        if uni_rr[i] != "1":
            continue

        if i in modified:
            continue

        for position in range(i + 1, n):
            if (position & i) == i:
                modified.add(position)

                uni_rr[position] = fill_value

    return "".join(uni_rr)


def get_uni_rr(rr: str, max: bool = True) -> str:
    """
    Returns the unique canonical representation (maximal truth table or 
    minimal Blake canonical form) of a Boolean rule.
    
    Operates on mask-order ``rr`` (``index == mask``). Fills supersets of each
    active implicant mask via bitwise subset checks.

    Parameters
    ----------
    rr  - rule in mask order (index == mask)    : length 2^k binary str
    max - if True, get maximal representation   : bool
          if False, get minimal representation

    Returns
    -------
    uni_rr - unique representation of the rule  : length 2^k binary str
    """
    n = len(rr)
    if (
        _UNI_RR_CYTHON
        and _get_uni_rr_cy is not None
        and n >= (1 << _UNI_RR_CYTHON_MIN_K)
    ):
        return _get_uni_rr_cy(rr, max)
    return _get_uni_rr_py(rr, max)


def get_max_irr(rr:str) -> str:
    """
    Returns the inverted maximal representation of the rule of a node
    when given any representation of the rule.

    After taking the maximal form and complementing each output bit, the
    string is reversed. That reverse complements the regulator assignment
    mask (``max_irr[mask] = 1 - max_rr[(2**k - 1) ^ mask]``); it is not an
    rr order-adapter.

    Parameters
    ----------
    rr - rule in mask order (index == mask)    :length 2^k binary str

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

def merge_primes(primes_list:list[PrimeType]) -> tuple[tuple[str],str,str]:
    """
    Returns the prime implicants of the merged rule
    when given a list of prime implicants of rules to be merged.
    Note that here we only use the activation primes of the rules to be merged, and we do not use the inhibition primes.
    Opposite can be done by using th deactivation primes if needed.

    Parameters
    ----------
    list[PrimeType] - a list of prime implicants of rules to be merged

    Returns
    -------
    prime - prime implicants of the merged rule   :PrimeType = list[list[dict[str,int]]]
        
    """

    activation_primes = []
    for prime in primes_list:
        activation_primes.extend(prime[1])
    
    # get the regulators and signs of the merged rule
    regulators, rr, signs = prime2rr([[], activation_primes], regulators=None, signs=None)

    return regulators, rr, signs

def merge_rules(primes_list:list[dict[str, PrimeType]]) -> tuple[dict[str, PrimeType], dict[str, tuple[str]], dict[str, str], dict[str, str]]:
    """
    Returns the prime implicants of the merged rules
    when given a list of prime implicants of rules to be merged.
    Note that here we only use the activation primes of the rules to be merged, and we do not use the inhibition primes.
    Opposite can be done by using th deactivation primes if needed.

    Parameters
    ----------
    list[dict[str, PrimeType]] - a list of dictionaries of prime implicants of rules to be merged

    Returns
    -------
    dict[str, PrimeType] - a dictionary of prime implicants of the merged rules
                            keys are nodes and values are prime implicants of the rules for the nodes
        
    """

    nodes = set()
    for primes in primes_list:
        nodes = nodes.union(set(primes.keys()))
    nodes = sorted(nodes)

    merged_primes = {}
    merged_regulators_dict = {}
    merged_rr_dict = {}
    merged_signs_dict = {}
    for node in nodes:
        node_primes_list = []
        for primes in primes_list:
            if node in primes:
                node_primes_list.append(primes[node])
        regulators, rr, signs = merge_primes(node_primes_list)

        merged_primes[node] = rr2prime(regulators, rr, signs, inverted = False)
        merged_regulators_dict[node] = regulators
        merged_rr_dict[node] = rr
        merged_signs_dict[node] = signs

    return merged_primes, merged_regulators_dict, merged_rr_dict, merged_signs_dict


def assignment_to_dict(assignment: Assignment):
    """
    Convert an immutable assignment representation into a dictionary.

    Parameters
    ----------
    assignment : Assignment
        Tuple of ``(node, value)`` pairs, where ``value`` is typically
        ``0`` or ``1``.

    Returns
    -------
    dict[str, int]
        Dictionary mapping each node to its assigned value.

    Notes
    -----
    If the same node appears multiple times, the last occurrence
    overwrites any previous value.
    """
    result = {}
    for node, value in assignment:
        result[node] = value
    return result