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
    max_rr = conv.get_uni_rr(rr, max=True)
    return not ('0' in max_rr and '1' in max_rr)


def check_source(
    regulators: tuple[str, ...],
    rr: str,
    node: str,
) -> bool:
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
    if node not in regulators:
        return False

    k = len(regulators)
    n = regulators.index(node)

    min_rr = conv.get_uni_rr(rr, max=False)

    add = 0
    for i in range(2**k):
        add += int(min_rr[i])

    return add == 1 and min_rr[2**k - 2**(k - n - 1) - 1] == '1'


def check_regulate(
    regulators: tuple[str, ...],
    rr: str,
    reg: str,
    is_min: bool = False,
) -> bool:
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
    if reg not in regulators:
        return False

    k = len(regulators)
    n = regulators.index(reg)

    bi = rr if is_min else conv.get_uni_rr(rr, max=False)

    add = 0
    for i in range(2**k):
        if i % 2**(k - n) <= 2**(k - 1 - n) - 1:
            add += int(bi[i])

    return add != 0


def check_necessary(
    regulators: tuple[str, ...],
    rr: str,
    nec: str,
) -> bool:
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
    if nec not in regulators:
        return False

    if not check_regulate(regulators, rr, nec):
        return False

    k = len(regulators)
    n = regulators.index(nec)

    add = 0
    for i in range(2**k):
        if i % 2**(k - n) > 2**(k - 1 - n) - 1:
            add += int(rr[i])

    return add == 0


def check_node(
    regulators: tuple[str, ...],
    rr: str,
    base_rr: str,
    constraints: dict,
    node: str,
) -> bool:
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
    bool
        True if all checks pass.
    """

    errors: list[str] = []

    # a constant node should not be mutated
    if len(base_rr) == 1:
        if rr != base_rr:
            errors.append(f"Constant node {node} was mutated")

    # a source node should not be mutated
    elif len(base_rr) == 2 and regulators[0] == node:
        if rr != base_rr:
            errors.append(f"Source node {node} was mutated")

    # fixed function
    elif node in constraints["fixed"]:
        if rr != base_rr:
            errors.append(f"{node} with fixed function was mutated")

    else:
        # necessary constraints
        if node in constraints["necessary"]:
            for necc in constraints["necessary"][node]:
                if necc not in regulators:
                    errors.append(
                        f"{node}: necessary regulator {necc} is completely absent"
                    )
                elif not check_necessary(regulators, rr, necc):
                    errors.append(
                        f"{necc} should be necessary for {node}"
                    )

        # regulate constraints
        if node in constraints["regulate"]:
            for reg in constraints["regulate"][node]:
                if reg not in regulators:
                    errors.append(
                        f"{node}: regulator {reg} is completely absent"
                    )
                elif not check_regulate(regulators, rr, reg):
                    errors.append(
                        f"{reg} should regulate {node}"
                    )

        # node with a self loop should not become a source node
        if node in regulators:
            if check_source(regulators, rr, node):
                errors.append(f"{node} should not be a source")

        # constant checks
        if check_constant(rr):
            max_rr = conv.get_uni_rr(rr, max=True)

            if (
                node not in constraints["possible_constant"]
                and len(base_rr) != 1
            ):
                errors.append(f"{node} should not be a constant")

            elif base_rr == "1" and "0" in max_rr:
                errors.append(f"{node} has opposite constant value")

            elif base_rr == "0" and "1" in max_rr:
                errors.append(f"{node} has opposite constant value")

    if errors:
        for error in errors:
            print(error)
        return False

    return True





