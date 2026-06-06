from pyboolnet.trap_spaces import compute_trap_spaces

def get_phenotypes_for_source_comb(primes, source_comb, DEBUG=False):
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
    ...     'A': [[{'A': 0}], [{'A': 1}]],
    ...     'B': [[{'B': 0}, {'A': 0}], [{'A': 1, 'B': 1}]],
    ...     'C': [[{'C': 1}], [{'C': 0}]]
    ... }
    >>> source_comb = {'A': 1}
    >>> get_phenotypes_for_source_comb(primes, source_comb)
    [{'A': 1, 'B': 1}, {'A': 1, 'B': 0}]
    """

    if DEBUG:
        for node in primes:
            if primes[node] == [[{node:0}],[{node:1}]]:
                if node not in source_comb:
                    raise ValueError(f"Source '{node}' is missing from the combination.")

    new_primes = primes.copy()

    for source, value in source_comb.items():
        if source not in new_primes:
            raise ValueError(f"Source '{source}' not found in the model.")

        if value == 0:
            new_primes[source] = [[{}], []]
        else:
            new_primes[source] = [[], [{}]]

    return compute_trap_spaces(new_primes, "min")