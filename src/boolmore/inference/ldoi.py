"""Logical domain of influence (LDOI) via prime-implicant closure.

Adapted from ``pystablemotifs.drivers.logical_domain_of_influence`` (MIT;
Rozum et al.), which implements the LDOI of Yang et al. 2018. This module
returns only ``seed ∪ LDOI`` as a single ``dict[str, int]`` and does not
compute a contradiction boundary.

The default closure is a compiled event-driven propagator. The original
whole-network scan remains available as an equivalence oracle via
``set_ldoi_closure_engine("scan")``.

References
----------
Yang, G., Gómez Tejeda Zañudo, J., and Albert, R. (2018). Target control in
logical models using the domain of influence of nodes. *Frontiers in
Physiology* 9:454. https://doi.org/10.3389/fphys.2018.00454
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

# Under a matching base P, enumerate ``2^|delta|`` probe keys when the
# leftover literals (query \\ P) are this small or smaller; else scan probes.
_DELTA_ENUM_MAX = 20

# TODO: Cython/native ``_ldoi_closure_fast`` (flat arrays / CSR watchers).
# TODO: batch ``ldoi_many(primes, seeds)`` API for multi-intervention NAV.
# TODO: restrict NAV dicts to observed/related nodes in scoring.

LdoiClosureEngine = Literal["fast", "scan"]
_CLOSURE_ENGINE: LdoiClosureEngine = "fast"

# Unset sentinel for integer state arrays (-1 is not a valid Boolean value).
_UNSET = -1


def set_ldoi_closure_engine(engine: LdoiClosureEngine) -> None:
    """Select the LDOI closure implementation (``"fast"`` or ``"scan"``).

    Intended for tests and A/B benches only. Production callers should leave
    the default ``"fast"``.
    """
    global _CLOSURE_ENGINE
    if engine not in ("fast", "scan"):
        raise ValueError(f"unknown LDOI closure engine: {engine!r}")
    _CLOSURE_ENGINE = engine


def get_ldoi_closure_engine() -> LdoiClosureEngine:
    """Return the active LDOI closure engine."""
    return _CLOSURE_ENGINE


@contextmanager
def ldoi_closure_engine(engine: LdoiClosureEngine) -> Iterator[None]:
    """Temporarily select a closure engine; restore the previous on exit."""
    previous = _CLOSURE_ENGINE
    set_ldoi_closure_engine(engine)
    try:
        yield
    finally:
        set_ldoi_closure_engine(previous)


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


@dataclass(frozen=True, slots=True)
class CompiledPrimes:
    """Integer-indexed prime implicants for event-driven LDOI closure."""

    names: tuple[str, ...]
    name_to_id: dict[str, int]
    # Per-implicant: (target_id, value, number of literals)
    targets: tuple[int, ...]
    values: tuple[int, ...]
    lit_counts: tuple[int, ...]
    # watchers[node_id][value] -> implicant indices containing that literal
    watchers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    # Empty implicants in scan order: node order, then value 0 then 1.
    empty_implicants: tuple[int, ...]
    # Object identities of primes[node] at compile time (invalidation).
    fingerprint: tuple[tuple[str, int], ...]


# Plain dicts are not weak-referenceable; cache by id(primes) with a
# node-entry identity fingerprint and a bounded LRU to avoid leaks.
_COMPILED_CACHE_MAX = 128
_compiled_cache: OrderedDict[
    int, tuple[tuple[tuple[str, int], ...], CompiledPrimes]
] = OrderedDict()


def _primes_fingerprint(primes: dict) -> tuple[tuple[str, int], ...]:
    """Fingerprint node → primes[node] object identity for cache validity."""
    return tuple((node, id(primes[node])) for node in primes)


def _compile_primes(primes: dict) -> CompiledPrimes:
    """Compile PyBoolNet primes into integer-indexed watcher tables."""
    names = tuple(primes)
    name_to_id = {name: i for i, name in enumerate(names)}
    n = len(names)

    targets: list[int] = []
    values: list[int] = []
    lit_counts: list[int] = []
    empty_implicants: list[int] = []
    # Temporary mutable watchers: [node][value] -> list[implicant_idx]
    watchers_mut: list[list[list[int]]] = [[[], []] for _ in range(n)]

    for target_id, name in enumerate(names):
        node_primes = primes[name]
        if not isinstance(node_primes, (list, tuple)) or len(node_primes) != 2:
            raise ValueError(
                f"malformed primes for {name!r}: expected [neg, pos] lists"
            )
        # Match scan: try value 0 then 1; empty implicants fire in that order.
        for value in (0, 1):
            implicants = node_primes[value]
            if not isinstance(implicants, (list, tuple)):
                raise ValueError(
                    f"malformed primes for {name!r} value {value}: "
                    "expected a list of implicants"
                )
            for implicant in implicants:
                if not isinstance(implicant, dict):
                    raise ValueError(
                        f"malformed implicant for {name!r}={value}: "
                        f"expected dict, got {type(implicant).__name__}"
                    )
                imp_idx = len(targets)
                targets.append(target_id)
                values.append(value)
                lit_counts.append(len(implicant))
                if not implicant:
                    empty_implicants.append(imp_idx)
                    continue
                for lit_node, lit_value in implicant.items():
                    lit_id = name_to_id.get(lit_node)
                    if lit_id is None:
                        raise ValueError(
                            f"implicant for {name!r} references unknown "
                            f"node {lit_node!r}"
                        )
                    if lit_value not in (0, 1):
                        raise ValueError(
                            f"implicant for {name!r} has non-Boolean "
                            f"literal {lit_node!r}={lit_value!r}"
                        )
                    watchers_mut[lit_id][lit_value].append(imp_idx)

    watchers = tuple(
        (tuple(w[0]), tuple(w[1])) for w in watchers_mut
    )
    return CompiledPrimes(
        names=names,
        name_to_id=name_to_id,
        targets=tuple(targets),
        values=tuple(values),
        lit_counts=tuple(lit_counts),
        watchers=watchers,
        empty_implicants=tuple(empty_implicants),
        fingerprint=_primes_fingerprint(primes),
    )


def _get_compiled(primes: dict) -> CompiledPrimes:
    """Return a cached compilation of ``primes``, rebuilding if mutated."""
    key = id(primes)
    fingerprint = _primes_fingerprint(primes)
    cached = _compiled_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        _compiled_cache.move_to_end(key)
        return cached[1]
    compiled = _compile_primes(primes)
    _compiled_cache[key] = (fingerprint, compiled)
    _compiled_cache.move_to_end(key)
    while len(_compiled_cache) > _COMPILED_CACHE_MAX:
        _compiled_cache.popitem(last=False)
    return compiled


def _ldoi_closure_scan(primes: dict, fixed: dict[str, int]) -> dict[str, int]:
    """Grow ``fixed`` by iterative prime-implicant implication (scan oracle).

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


def _ldoi_closure_fast(
    compiled: CompiledPrimes,
    fixed: dict[str, int],
) -> dict[str, int]:
    """Grow ``fixed`` via event-driven prime-implicant propagation.

    Semantically equivalent to ``_ldoi_closure_scan``: seeds are never
    overwritten; empty implicants fire in node/value-0-then-1 order; already
    fixed targets are skipped.
    """
    n = len(compiled.names)
    state = [_UNSET] * n
    for name, value in fixed.items():
        node_id = compiled.name_to_id.get(name)
        if node_id is None:
            raise ValueError(f"{name} is not in the model")
        state[node_id] = value

    residuals = list(compiled.lit_counts)
    queue: list[int] = []

    def _force(node_id: int, value: int) -> None:
        if state[node_id] != _UNSET:
            return
        state[node_id] = value
        queue.append(node_id)

    # Constants / empty implicants first (same order as scan: nodes then 0,1).
    for imp_idx in compiled.empty_implicants:
        _force(compiled.targets[imp_idx], compiled.values[imp_idx])

    # Seed / warm-start nodes already fixed (may overlap constants).
    for node_id, value in enumerate(state):
        if value != _UNSET:
            queue.append(node_id)

    # Deduplicate initial queue while preserving first-occurrence order so
    # constant-forced nodes are processed before later seed echoes when both
    # appear; processing order of already-fixed nodes only affects watcher
    # decrement timing, not the final fixed point for consistent primes.
    seen_init = [False] * n
    deduped: list[int] = []
    for node_id in queue:
        if not seen_init[node_id]:
            seen_init[node_id] = True
            deduped.append(node_id)
    queue = deduped

    head = 0
    while head < len(queue):
        node_id = queue[head]
        head += 1
        value = state[node_id]
        for imp_idx in compiled.watchers[node_id][value]:
            residuals[imp_idx] -= 1
            if residuals[imp_idx] == 0:
                _force(compiled.targets[imp_idx], compiled.values[imp_idx])

    result: dict[str, int] = {}
    names = compiled.names
    for node_id, value in enumerate(state):
        if value != _UNSET:
            result[names[node_id]] = value
    return result


def _ldoi_closure(
    primes: dict,
    fixed: dict[str, int],
    *,
    compiled: CompiledPrimes | None = None,
) -> dict[str, int]:
    """Dispatch to the active closure engine."""
    if _CLOSURE_ENGINE == "scan":
        return _ldoi_closure_scan(primes, fixed)
    return _ldoi_closure_fast(
        compiled if compiled is not None else _get_compiled(primes),
        fixed,
    )


def logical_domain_of_influence(
    primes: dict,
    seed: dict[str, int],
    *,
    cache: LdoiCache | None = None,
    base: dict[str, int] | None = None,
    implied: dict[str, int] | None = None,
    compiled: CompiledPrimes | None = None,
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
    compiled : CompiledPrimes, optional
        Precompiled watcher tables for the fast engine. When omitted, primes
        are compiled (and fingerprint-validated) on each call. Pass one
        compilation across many seeds on the same primes object to avoid
        repeated fingerprint work. Ignored when the scan engine is active.

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

    result = _ldoi_closure(primes, fixed, compiled=compiled)

    if cache is not None:
        cache.store(seed, result, base=base)

    return result
