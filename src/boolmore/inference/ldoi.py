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

from dataclasses import dataclass, field

# Under a matching base P, enumerate ``2^|delta|`` probe keys when the
# leftover literals (query \\ P) are this small or smaller; else scan probes.
_DELTA_ENUM_MAX = 20


def _implicant_implied_by(fixed: dict[str, int], implicant: dict[str, int]) -> bool:
    """Return True iff every literal of ``implicant`` is present in ``fixed``."""
    for node, value in implicant.items():
        if fixed.get(node) != value:
            return False
    return True


def _seed_key(seed: dict[str, int]) -> frozenset[tuple[str, int]]:
    """Canonical cache key for a seed assignment."""
    return frozenset(seed.items())


def _ldoi_conflicts(ldoi: dict[str, int], seed: dict[str, int]) -> bool:
    """True iff ``ldoi`` forces any seed node to the opposite value."""
    for node, value in seed.items():
        if node in ldoi and ldoi[node] != value:
            return True
    return False


def _merge_if_ok(
    merged: dict[str, int],
    ldoi: dict[str, int],
    seed: dict[str, int],
) -> None:
    if not _ldoi_conflicts(ldoi, seed):
        merged.update(ldoi)


@dataclass
class _BaseEntry:
    """LDOI of intervention ``P`` plus probes under ``P ∪ SC``."""

    ldoi: dict[str, int] | None = None
    probes: dict[frozenset[tuple[str, int]], dict[str, int]] = field(
        default_factory=dict
    )


class LdoiCache:
    """Nested LDOI cache: intervention ``P`` → optional SC-probe deltas.

    Structure::

        cache[P] = {ldoi: LDOI(P), probes: {SC: LDOI(P ∪ SC), ...}}

    Exact hits:

    - ``get(P)`` → ``LDOI(P)``
    - ``get(SC, base=P)`` → ``LDOI(P ∪ SC)``
    - ``get(P ∪ SC)`` as a top-level seed also finds the probe via
      ``delta = (P ∪ SC) \\ P`` when that probe was stored under ``P``

    Warm-start (``compatible_merge``): for every cached ``P ⊆ query``, merge
    ``LDOI(P)`` and every probe ``D ⊆ query \\ P`` that does not conflict
    with the query seed.
    """

    def __init__(self) -> None:
        self._bases: dict[frozenset[tuple[str, int]], _BaseEntry] = {}

    def __len__(self) -> int:
        n = 0
        for entry in self._bases.values():
            if entry.ldoi is not None:
                n += 1
            n += len(entry.probes)
        return n

    def _ensure_base(self, base_key: frozenset[tuple[str, int]]) -> _BaseEntry:
        entry = self._bases.get(base_key)
        if entry is None:
            entry = _BaseEntry()
            self._bases[base_key] = entry
        return entry

    def get(
        self,
        seed: dict[str, int],
        *,
        base: dict[str, int] | None = None,
    ) -> dict[str, int] | None:
        """Return a copy of a cached LDOI, or ``None``.

        If ``base`` is given, ``seed`` is an SC delta under that intervention.
        Otherwise ``seed`` is treated as a full intervention (and may match
        ``P ∪ SC`` for a previously stored probe).
        """
        if base is not None:
            entry = self._bases.get(_seed_key(base))
            if entry is None:
                return None
            cached = entry.probes.get(_seed_key(seed))
            return dict(cached) if cached is not None else None

        key = _seed_key(seed)
        entry = self._bases.get(key)
        if entry is not None and entry.ldoi is not None:
            return dict(entry.ldoi)

        # seed == P ∪ SC for some probe under P?
        for p_key, ent in self._bases.items():
            if not p_key < key:
                continue
            delta = key - p_key
            probe = ent.probes.get(delta)
            if probe is not None:
                return dict(probe)
        return None

    def store(
        self,
        seed: dict[str, int],
        ldoi: dict[str, int],
        *,
        base: dict[str, int] | None = None,
    ) -> None:
        """Store a copy of ``ldoi``.

        If ``base`` is given, store under ``base``'s probe map with key
        ``seed`` (SC). Otherwise store as the LDOI of intervention ``seed``.
        """
        if base is not None:
            entry = self._ensure_base(_seed_key(base))
            entry.probes[_seed_key(seed)] = dict(ldoi)
            return
        entry = self._ensure_base(_seed_key(seed))
        entry.ldoi = dict(ldoi)

    def compatible_merge(self, seed: dict[str, int]) -> dict[str, int]:
        """Blind-merge non-conflicting ``P`` / ``P ∪ D`` LDOIs for ``seed``.

        Does not overlay ``seed`` itself. For each cached intervention
        ``P ⊆ seed``, merges ``LDOI(P)`` and probes ``D ⊆ seed \\ P``.
        """
        key = _seed_key(seed)
        merged: dict[str, int] = {}
        for p_key, entry in self._bases.items():
            if not p_key <= key:
                continue
            if entry.ldoi is not None:
                _merge_if_ok(merged, entry.ldoi, seed)
            delta_space = key - p_key
            if not entry.probes:
                continue
            if len(delta_space) <= _DELTA_ENUM_MAX:
                items = list(delta_space)
                n = len(items)
                for mask in range(1 << n):
                    d_key = frozenset(
                        items[i] for i in range(n) if mask & (1 << i)
                    )
                    probe = entry.probes.get(d_key)
                    if probe is not None:
                        _merge_if_ok(merged, probe, seed)
            else:
                for d_key, probe in entry.probes.items():
                    if d_key <= delta_space:
                        _merge_if_ok(merged, probe, seed)
        return merged


def _ldoi_closure(primes: dict, fixed: dict[str, int]) -> dict[str, int]:
    """Grow ``fixed`` by iterative prime-implicant implication.

    Returns a new dict starting from ``fixed``. Nodes already in ``fixed``
    are not re-checked (no contradiction boundary).
    """
    fixed = dict(fixed)
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


def logical_domain_of_influence(
    primes: dict,
    seed: dict[str, int],
    *,
    cache: LdoiCache | None = None,
    base: dict[str, int] | None = None,
    implied: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return seed ∪ LDOI by iterative prime-implicant implication.

    Starting from ``seed``, repeatedly force any node whose update rule has a
    prime implicant covered by the growing fixed set. Seed nodes are always
    present in the result. Preexisting constant rules (empty implicants) are
    forced even when the seed is empty.

    Unlike ``pystablemotifs.drivers.logical_domain_of_influence``, this does
    **not** return a contradiction boundary: if a seed assignment disagrees
    with a forced value, the seed value is kept and search continues.

    When ``cache`` is provided, compatible subset LDOIs warm-start the
    closure and the result is stored. Pass ``base`` for an SC probe under
    intervention ``base`` (nested ``P → SC`` indexing). ``implied`` supplies
    an already-forced starting set (e.g. plain LDOI of ``base``) that is not
    part of the cache key.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network (PyBoolNet format).
    seed : dict of str to int
        Seed node values to fix (0 or 1). For probes, the SC assignment only.
    cache : LdoiCache, optional
        Shared nested result cache.
    base : dict of str to int, optional
        Parent intervention when ``seed`` is an SC delta under that
        intervention. Required for correct nested store/lookup of probes.
    implied : dict of str to int, optional
        Already-forced nodes included in the starting fixed set and result,
        but excluded from the cache key (typically ``LDOI(base)``).

    Returns
    -------
    dict of str to int
        Seed ∪ LDOI: all forced node values (including ``implied``).

    Raises
    ------
    ValueError
        If a seed, base, or implied node is not present in the model.
    """
    for node in seed:
        if node not in primes:
            raise ValueError(f"{node} is not in the model")
    if base:
        for node in base:
            if node not in primes:
                raise ValueError(f"{node} is not in the model")
    if implied:
        for node in implied:
            if node not in primes:
                raise ValueError(f"{node} is not in the model")

    if cache is not None:
        hit = cache.get(seed, base=base)
        if hit is not None:
            return hit

    # Full seed for subset merge: intervention ∪ SC (or just intervention).
    if base is not None:
        full_seed = dict(base)
        full_seed.update(seed)
    else:
        full_seed = seed

    fixed: dict[str, int] = dict(implied) if implied else {}
    if cache is not None:
        fixed.update(cache.compatible_merge(full_seed))
    fixed.update(full_seed)

    result = _ldoi_closure(primes, fixed)

    if cache is not None:
        cache.store(seed, result, base=base)

    return result
