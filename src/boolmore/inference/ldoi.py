from pyboolnet.prime_implicants import create_constants, find_constants, percolate


def logical_domain_of_influence(
    primes: dict,
    seed: dict[str, int],
) -> dict[str, int]:
    """Return seed ∪ LDOI after rewriting seed nodes to constants and percolating.

    This is the pyboolnet path ``create_constants`` → ``percolate`` →
    ``find_constants``. Seed nodes are rewritten to constant update rules, so
    every seed assignment is always present in the result, together with any
    nodes forced by percolation (including preexisting network constants).

    This does **not** compute a contradiction boundary. By contrast,
    ``pystablemotifs.drivers.logical_domain_of_influence`` (Yang et al. 2018)
    returns ``(implied, contradicted)`` and may omit seed nodes whose own
    update rules do not force the seed value. When there is no contradiction,
    this function's result equals ``seed ∪ implied`` from that API.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network.
    seed : dict of str to int
        Seed (intervention) node values to fix (0 or 1).

    Returns
    -------
    dict of str to int
        Seed ∪ LDOI: all constant node values after fixing the seed and
        percolating. Not a contradiction-boundary pair.

    Raises
    ------
    ValueError
        If a seed node is not present in the model.
    """
    for node in seed:
        if node not in primes:
            raise ValueError(f"{node} is not in the model")

    fixed_primes = create_constants(primes, seed, copy=True)
    percolated = percolate(fixed_primes, copy=True)
    return find_constants(percolated)
