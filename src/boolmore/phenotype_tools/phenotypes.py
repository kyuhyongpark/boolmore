from pyboolnet.trap_spaces import compute_trap_spaces

def get_mintr_for_source_comb(primes, source_comb, DEBUG=False):
    """
    Fixes source node values in a Boolean network and computes minimal trap spaces.

    Parameters
    ----------
    primes : dict
        Boolean network in PyBoolNet prime format.
    source_comb : dict
        Dictionary mapping source node names to fixed values (0 or 1).
        May be partial unless `require_all_sources=True`.
    DEBUG : bool, optional
        If True, check if all source nodes in the network also appear in `source_comb`.
        Default is False.

    Returns
    -------
    list
        List of minimal trap spaces after applying the source constraints.
        Nodes not appearing in each trap space are considered free (can take either value).

    Raises
    ------
    ValueError
        - If a node in `source_comb` is not present in `primes`.
        - If `DEBUG=True` and some source nodes are missing
          from `source_comb`.

    Notes
    -----
    This function applies constant assignments to selected nodes by replacing
    their update rules with fixed Boolean states before computing trap spaces.

    Examples
    --------
    >>> primes = {
    ...     "A": [[{"A": 0}], [{"A": 1}]],
    ...     "B": [[{"B": 0}, {"A": 0}], [{"A": 1, "B": 1}]],
    ...     "C": [[{"C": 1}], [{"C": 0}]]
    ... }
    >>> source_comb = {"A": 1}
    >>> result = get_mintr_for_source_comb(primes, source_comb)
    >>> len(result)
    2
    >>> {"A": 1, "B": 1} in result
    True
    >>> {"A": 1, "B": 0} in result
    True
    """

    if DEBUG:
        for node in primes:
            if primes[node] == [[{node:0}],[{node:1}]]:
                if node not in source_comb:
                    raise ValueError(f"Source {node} is missing from the combination.")

    new_primes = primes.copy()

    for source, value in source_comb.items():
        if source not in new_primes:
            raise ValueError(f"Source {source} not found in the model.")

        if value == 0:
            new_primes[source] = [[{}], []]
        else:
            new_primes[source] = [[], [{}]]

    return compute_trap_spaces(new_primes, "min")


def find_matching_phenotype(answer_sheet, target_phenotype):
    """
    Search for a minimal trap space that matches a target phenotype.

    The function iterates over all minimal trap spaces grouped by source
    combinations (masks) and checks whether any trap space is consistent
    with the given target phenotype.

    A match is defined as: for every key-value pair in `target_phenotype`,
    the corresponding entry exists in the trap space and has the same value.

    Parameters
    ----------
    answer_sheet : dict
        Dictionary mapping a mask (source combination identifier) to a list
        of minimal trap spaces. Each trap space is a dictionary of node states.

    target_phenotype : dict
        Partial or full phenotype specification (node -> value).

    Returns
    -------
    tuple or None
        (mask, min_trap_space) if a matching phenotype is found, otherwise None.

        - mask : hashable identifier of the source combination
        - min_trap_space : dict representing the matching trap space

    Notes
    -----
    - Only the first matching trap space is returned.

    Examples
    --------
    >>> answer_sheet = {
    ... 1: [{"A": 0, "B": 1, "C": 0, "D": 1},
    ...     {"A": 0, "B": 1, "C": 0, "D": 0},],
    ... 3: [{"A": 1, "B": 1, "C": 1, "D": 1},
    ...     {"A": 1, "B": 1, "C": 1, "D": 0},],}
    >>> result = find_matching_phenotype(answer_sheet, {"C": 1, "D": 0})
    >>> result[0]
    3
    >>> result[1] == {"A": 1, "B": 1, "C": 1, "D": 0}
    True
    """
    for mask, min_traps in answer_sheet.items():
        for min_tr in min_traps:
            if all(min_tr.get(k) == v for k, v in target_phenotype.items()):
                return mask, min_tr
    return None