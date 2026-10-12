def impose_necessary(regulators:tuple[str, ...], rr:str, nec:str) -> str:
    """
    Impose the necessary condition for the rule.

    Parameters
    ----------
    regulators - the regulating nodes           :length k tuple[str]
    rr         - representation of the rule     :length 2^k binary str
    nec        - the necessary node             :str

    Returns
    -------
    necessary_rr - modified so that nec is necessary    :str
        

    """
    assert nec in regulators, "The node should be one of the regulators"

    # k = number of regulators
    k = len(regulators)
    # n = nth regulator
    n = regulators.index(nec)

    bi = list(rr)
    for i in range(2**k):
        if i % 2**(k-n) > 2**(k-1-n)-1:
            bi[i] = '0'

    necessary_rr = ''.join(bi)

    return necessary_rr


def group_bi2bi(group_bi:str, group_regulators:tuple[str, tuple[str, ...]]) -> str:
    """
    Returns the binary position of the old representation
    when given a binary position of the group representation
    and the group regulators.

    Parameters
    ----------
    group_bi         - binary position of the rule representation with  :length m binary str
                       grouped regulators
                       e.g. 101 with regulators (A, B, (C,D)) means the
                            position in the rr that represents A&(C&D)
    group_regulators - e.g. (A, (B,C), (B,D))                           :length m tuple[str, tuple[str]]

    Returns
    -------
    bi - binary position of the rule representation with the original regulators    :length k binary str
         e.g. 101 with grouped regulators (A, B, (C,D))
              becomes 1011 with regulators (A, B, C, D)
         This transition is not onto, which enforces the group constraint.
         However, this transition is also not one-to-one which does cause bias.

    """
    num = iter(group_bi)
    bi_dict = {}
    for reg in group_regulators:
        if type(reg) == tuple:
            value = next(num)
            for node in reg:
                if node not in bi_dict.keys():
                    bi_dict[node] = value
                elif bi_dict[node] == '0':
                    bi_dict[node] = value
                else:
                    pass
        else:
            bi_dict[reg] = next(num)
    
    # sort by node names, which restores the order of nodes in the original regulators
    bi_dict = dict(sorted(bi_dict.items()))

    bi = ''.join(bi_dict.values())

    return bi


def rr2group_rr(regulators:tuple[str, ...], rr:str, groups:list[list[str]]) -> tuple[str, tuple[str, tuple[str, ...]]]:
    """
    Convert the representation of a rule into a representation in which
    certain nodes are grouped together.

    Parameters
    ----------
    regulators - the regulating nodes           :length k tuple[str]
    rr         - representation of the rule     :length 2^k binary str
    groups     - groups of nodes                :list[list[str]]
                 [[node A, node B], [node A, node C], ...]
    
    Returns
    -------
    group_rr         - modified so that nec is necessary            :length 2^m binary str
    group_regulators - the regulating nodes where some are grouped  :length m tuple[str, tuple[str]]
                       (node A, node B, (node C, node D), ...)

    """

    for group in groups:
        for node in group:
            assert node in regulators, f"{node} is not one of the regulators {regulators}"

    group_regulators = []
    individual_regulators = set(regulators)
    for i, group in enumerate(groups):
        group_regulators.append(tuple(group)) # type: ignore
        individual_regulators -= set(group)
    group_regulators.extend((sorted(list(individual_regulators))))
    group_regulators = tuple(group_regulators)

    m = len(group_regulators)
    group_rr = ['0']*(2**m)

    for i in range(2**m):
        group_bi = format(2**m - 1 - i, '0' + str(m) + 'b')
        bi = group_bi2bi(group_bi, group_regulators) # type: ignore
        if rr[-int(bi, 2)-1] == '1':
            group_rr[i] = '1'

    group_rr = ''.join(group_rr)
    return group_rr, group_regulators # type: ignore


def group_rr2rr(regulators:tuple[str, ...], group_rr:str, group_regulators:tuple[str, tuple[str, ...]]) -> str:
    """
    Convert the representation of a rule with grouped regulators
    into a representation with original regulators.

    Parameters
    ----------
    regulators       - the regulating nodes                         :length k tuple[str]
    group_rr         - modified so that nec is necessary            :length 2^m binary str
    group_regulators - the regulating nodes where some are grouped  :length m tuple[str, tuple[str]]
                       (node A, node B, (node C, node D), ...)

    Returns
    -------
    rr - representation of the rule     :length 2^k binary str

    """

    m = len(group_regulators)
    k = len(regulators)
    rr = ['0']*(2**k)

    for i in range(2**m):
        group_bi = format(2**m - 1 - i, '0' + str(m) + 'b')
        bi = group_bi2bi(group_bi, group_regulators)
        # print('bi:',bi)
        if group_rr[i] == '1':
            rr[-int(bi, 2)-1] = '1'

    rr = ''.join(rr)

    return rr