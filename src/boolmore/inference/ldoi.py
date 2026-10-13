"""Logical domain of influence (LDOI) via prime-implicant closure.

Adapted from ``pystablemotifs.drivers.logical_domain_of_influence`` (MIT;
Rozum et al.), which implements the LDOI of Yang et al. 2018. This module
returns only ``seed ∪ LDOI`` as a single ``dict[str, int]`` and does not
compute a contradiction boundary.

References
----------
Yang, G., Gómez Tejeda Zañudo, J., and Albert, R. (2018). Target control in
logical models using the domain of influence of nodes. *Frontiers in
Physiology* 9:454. https://doi.org/10.3389/fphys.2018.00454
"""

from __future__ import annotations


def _implicant_implied_by(fixed: dict[str, int], implicant: dict[str, int]) -> bool:
    """Return True iff every literal of ``implicant`` is present in ``fixed``."""
    for node, value in implicant.items():
        if fixed.get(node) != value:
            return False
    return True


def logical_domain_of_influence(
    primes: dict,
    seed: dict[str, int],
) -> dict[str, int]:
    """Return seed ∪ LDOI by iterative prime-implicant implication.

    Starting from ``seed``, repeatedly force any node whose update rule has a
    prime implicant covered by the growing fixed set. Seed nodes are always
    present in the result. Preexisting constant rules (empty implicants) are
    forced even when the seed is empty.

    Unlike ``pystablemotifs.drivers.logical_domain_of_influence``, this does
    **not** return a contradiction boundary: if a seed assignment disagrees
    with a forced value, the seed value is kept and search continues.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network (PyBoolNet format).
    seed : dict of str to int
        Seed (intervention) node values to fix (0 or 1).

    Returns
    -------
    dict of str to int
        Seed ∪ LDOI: all forced node values.

    Raises
    ------
    ValueError
        If a seed node is not present in the model.
    """
    for node in seed:
        if node not in primes:
            raise ValueError(f"{node} is not in the model")

    fixed = dict(seed)
    # Seed nodes need not be re-checked (no contradiction boundary).
    remaining = set(primes) - fixed.keys()

    while remaining:
        newly_forced: list[str] = []
        for node in remaining:
            node_primes = primes[node]
            forced_value: int | None = None
            # Match pystablemotifs: try value 0 then 1.
            for value in (0, 1):
                if forced_value is not None:
                    break
                for implicant in node_primes[value]:
                    if _implicant_implied_by(fixed, implicant):
                        forced_value = value
                        break
            if forced_value is not None:
                fixed[node] = forced_value
                newly_forced.append(node)

        if not newly_forced:
            break
        remaining.difference_update(newly_forced)

    return fixed
