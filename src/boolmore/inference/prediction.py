import logging
from collections import defaultdict
from itertools import combinations, product
from time import perf_counter
from dataclasses import dataclass

import networkx as nx
from pyboolnet.trap_spaces import compute_trapspaces_within_subspace, compute_trap_spaces

from boolmore.experiment import PhenotypeExperiment, NAVExperiment
from boolmore.boolean_functions import assignment_to_dict
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.ldoi import _implicant_implied_by, logical_domain_of_influence
from boolmore.types import Assignment

NAV_METHODS = ("min_traps", "ldoi", "ldoi_enhanced")
LDOI_ENHANCED_CUTOFF_MAX = 3


@dataclass(frozen=True)
class PhenotypePrediction:
    """Prediction for a single phenotype experiment.

    Attributes
    ----------
    id : int
        Experiment identifier.
    perturbation : Assignment
        Fixed node interventions for the experiment.
    sources : Assignment
        Source-node assignments.
    phenotype : Assignment
        Target phenotype assignment.
    maximal_trap_spaces : list of dict of str to int
        Maximal trap spaces compatible with the phenotype constraints.
    predicted_exists : bool
        Whether a compatible maximal trap space was found.
    """

    id: int

    perturbation: Assignment
    sources: Assignment
    phenotype: Assignment

    maximal_trap_spaces: list[dict[str, int]]
    predicted_exists: bool


@dataclass(frozen=True)
class NAVPrediction:
    """Prediction for a unique NAV intervention.

    Attributes
    ----------
    fixes : Assignment
        Intervention assignment ``((node, value), ...)``.
    average_values : dict of str to float or None
        Average attractor value of each model node in ``[0, 1]``.
        ``None`` means unknown / free (``?``), used only by the LDOI
        upper-bound path. Synthetic ``*_avg`` observed names are resolved
        later by :func:`~boolmore.evaluation.agreement.get_agreements`.
    """

    fixes: Assignment
    average_values: dict[str, float | None]


def _fix_constants(primes: dict, fixed: dict[str, int]) -> dict:
    """Return a copy of ``primes`` with fixed nodes set to constant functions.

    This only replaces each fixed node's update rule with a constant; it does
    not percolate those constants through the rest of the network.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network.
    fixed : dict of str to int
        Nodes to fix and their constant values (0 or 1).

    Returns
    -------
    dict
        Prime implicants with the specified nodes forced to constants.

    Raises
    ------
    ValueError
        If a fixed node is not present in the model.
    """
    fixed_primes = primes.copy()
    for node, value in fixed.items():
        if node not in primes:
            raise ValueError(f"{node} is not in the model")

        if int(value) == 0:
            fixed_primes[node] = [[{}], []]
        else:
            fixed_primes[node] = [[], [{}]]
    return fixed_primes


def get_phenotype_prediction(
    primes,
    experiments: list[PhenotypeExperiment],
    debug: bool = False,
) -> list[PhenotypePrediction]:
    """Predict phenotype compatibility for each experiment.

    For each experiment, the function first checks whether a previously
    computed maximal trap space for the same perturbation already satisfies
    the required sources and phenotype. If so, that cached result is reused.

    Otherwise, the perturbed nodes are fixed to constant update functions,
    and maximal trap spaces contained within the combined source and
    phenotype subspace are searched. If such a trap space exists, the
    experiment is predicted to be feasible.

    Prime implicants with fixed constants are cached by perturbation, and
    discovered maximal trap spaces are also cached for reuse by later
    experiments with the same perturbation.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network in the format expected by
        PyBoolNet.
    experiments : list of PhenotypeExperiment
        Experiments to evaluate. Each experiment must define
        ``perturbation``, ``sources``, and ``phenotype`` assignments.
    debug : bool, default False
        If True, print per-experiment timing information.

    Returns
    -------
    list of PhenotypePrediction
        One prediction per experiment, containing whether a compatible
        trap space was found and, if so, the matching maximal trap spaces.

    Raises
    ------
    ValueError
        If a node appearing in a perturbation, source assignment, or
        phenotype assignment is not present in the model.
    RuntimeError
        If PyBoolNet's ASP backend (gringo/clasp) fails the probe check.
    """
    ensure_asp_backend()

    results = []
    traps_cache = defaultdict(list)
    primes_cache = defaultdict(list)

    for exp in experiments:

        if debug:
            print(f"\nExperiment {exp.id}")

        maximal_trap_spaces: list[dict[str, int]] = []
        predicted_exists = False

        # check cache
        t0 = perf_counter()

        sources = assignment_to_dict(exp.sources)
        phenotype = assignment_to_dict(exp.phenotype)

        if exp.perturbation in traps_cache:
            for max_trap in traps_cache[exp.perturbation]:
                if not all(node in max_trap for node in sources):
                    continue
                if not all(node in max_trap for node in phenotype):
                    continue
                if (
                    all(
                        sources[node] == max_trap[node]
                        for node in sources
                    )
                    and
                    all(
                        phenotype[node] == max_trap[node]
                        for node in phenotype
                    )
                ):
                    maximal_trap_spaces.append(max_trap)
                    predicted_exists = True
                    break

        if debug:
            print(
                f"  cache check: "
                f"{perf_counter() - t0:.6f} s"
            )

        if predicted_exists:
            if debug:
                print("  cache hit (maximal_trap_spaces)")

            result = PhenotypePrediction(
                id=exp.id,
                perturbation=exp.perturbation,
                sources=exp.sources,
                phenotype=exp.phenotype,
                maximal_trap_spaces=maximal_trap_spaces,
                predicted_exists=predicted_exists,
            )

            results.append(result)
            continue

        perturbation = assignment_to_dict(exp.perturbation)

        # fix perturbation nodes as constants
        t0 = perf_counter()

        if exp.perturbation in primes_cache:
            fixed_primes = primes_cache[exp.perturbation]
            if debug:
                print("  cache hit (fixed_primes)")
        else:
            fixed_primes = _fix_constants(primes, perturbation)
            primes_cache[exp.perturbation] = fixed_primes

        if debug:
            print(
                f"  get fixed_primes: "
                f"{perf_counter() - t0:.6f} s"
            )

        subspace = sources.copy()
        subspace.update(phenotype)

        for node in subspace:
            if node not in fixed_primes:
                raise ValueError(f"{node} not in the model.")

        # compute max traps
        t0 = perf_counter()

        logging.disable(logging.CRITICAL)
        max_traps = compute_trapspaces_within_subspace(
            fixed_primes,
            subspace=subspace,
            type_="max",
            max_output=1,
        )
        logging.disable(logging.NOTSET)

        if debug:
            print(
                f"  compute max_trap: "
                f"{perf_counter() - t0:.6f} s"
            )

        if max_traps:
            maximal_trap_spaces += max_traps
            predicted_exists = True
            traps_cache[exp.perturbation] += max_traps

        result = PhenotypePrediction(
            id=exp.id,
            perturbation=exp.perturbation,
            sources=exp.sources,
            phenotype=exp.phenotype,
            maximal_trap_spaces=maximal_trap_spaces,
            predicted_exists=predicted_exists,
        )

        results.append(result)

    return results


def _source_nodes(primes: dict) -> list[str]:
    """Return sorted names of syntactic source nodes (self-loop inputs)."""
    return sorted(
        node
        for node in primes
        if primes[node] == [[{node: 0}], [{node: 1}]]
    )


def _trap_node_value(trap: dict, node: str) -> float:
    """Map a trap-space entry to a numeric contribution (1 / 0 / 0.5)."""
    value = trap[node]
    if value == "1":
        return 1.0
    if value == "?":
        return 0.5
    return 0.0


def _nav_from_min_traps(
    primes,
    perturbation: dict[str, int],
) -> dict[str, float]:
    """Average node values over minimal trap spaces after fixing interventions.

    Unfixed source nodes in the intervened network define groups: traps that
    share the same source assignment are averaged first, then those group
    averages are averaged with equal weight. This prevents source combinations
    that yield many minimal traps from dominating the NAV vector.
    """
    ensure_asp_backend()
    new_primes = _fix_constants(primes, perturbation)
    sources = _source_nodes(new_primes)

    tr = compute_trap_spaces(new_primes, "min")

    for trap in tr:
        for node in primes:
            if node not in trap:  # type: ignore
                trap[node] = "?"  # type: ignore
            else:
                trap[node] = str(trap[node])  # type: ignore

    by_source: dict[tuple, list] = defaultdict(list)
    for trap in tr:
        key = tuple(trap[node] for node in sources)  # type: ignore
        by_source[key].append(trap)

    n_groups = len(by_source)
    average_values: dict[str, float] = {node: 0.0 for node in primes}
    for group in by_source.values():
        n_traps = len(group)
        for node in primes:
            group_avg = sum(
                _trap_node_value(trap, node) for trap in group
            ) / n_traps
            average_values[node] += group_avg / n_groups

    return average_values


def _nav_from_ldoi(
    primes,
    perturbation: dict[str, int],
) -> dict[str, float]:
    """Approximate NAV from LDOI (forced 0/1, else 0.5).

    This approximation is only credible when the model does not rely
    heavily on oscillatory behavior: free nodes are scored as 0.5 and
    cyclic attractors are not resolved.
    """
    constants = logical_domain_of_influence(primes, perturbation)
    return {
        node: float(constants[node]) if node in constants else 0.5
        for node in primes
    }


def _node_has_self_loop(primes: dict, node: str) -> bool:
    """Return True iff ``node`` appears as a literal in its own update rule."""
    for value in (0, 1):
        for implicant in primes[node][value]:
            if node in implicant:
                return True
    return False


def _is_self_sustaining(
    primes: dict,
    closure: dict[str, int],
    node: str,
    value: int,
) -> bool:
    """Return True iff some implicant of ``node`` at ``value`` is covered by ``closure``."""
    return any(
        _implicant_implied_by(closure, implicant)
        for implicant in primes[node][value]
    )


def _is_self_negating(
    primes: dict,
    closure: dict[str, int],
    node: str,
    value: int,
) -> bool:
    """Return True iff some implicant of ``node`` at ``1 - value`` is covered by ``closure``."""
    return any(
        _implicant_implied_by(closure, implicant)
        for implicant in primes[node][1 - value]
    )


def _interaction_digraph(primes: dict) -> nx.DiGraph:
    """Regulatory digraph: edge ``u→v`` if ``u`` appears in any implicant of ``v``."""
    digraph = nx.DiGraph()
    digraph.add_nodes_from(primes)
    for target, prime in primes.items():
        for value in (0, 1):
            for implicant in prime[value]:
                for regulator in implicant:
                    digraph.add_edge(regulator, target)
    return digraph


def _triple_strongly_connected(
    succ: dict[str, set[str]],
    a: str,
    b: str,
    c: str,
) -> bool:
    """True iff the induced digraph on ``{a,b,c}`` is strongly connected.

    Uses Floyd–Warshall on the three nodes with successor sets restricted
    to the ambient graph (edges leaving ``{a,b,c}`` are ignored).
    """
    nodes = (a, b, c)
    idx = {a: 0, b: 1, c: 2}
    reach = [[i == j for j in range(3)] for i in range(3)]
    for u in nodes:
        ui = idx[u]
        for v in succ[u]:
            if v in idx:
                reach[ui][idx[v]] = True
    for k in range(3):
        for i in range(3):
            if not reach[i][k]:
                continue
            for j in range(3):
                if reach[k][j]:
                    reach[i][j] = True
    return all(reach[i][j] for i in range(3) for j in range(3))


def _sc_triples_in_scc(
    digraph: nx.DiGraph,
    scc: list[str],
) -> list[tuple[str, ...]]:
    """Strongly connected 3-subsets inside one SCC.

    Every SC triple is weakly connected, so it has a hub of undirected
    degree 2 inside the triple. Enumerate hub + two undirected neighbors
    (``O(Σ deg²)``) and test directed strong connectivity with a 3-node
    Floyd check — far cheaper than ``C(|SCC|, 3)`` NetworkX subgraph tests.
    """
    scc_set = set(scc)
    succ = {
        u: {v for v in digraph.successors(u) if v in scc_set}
        for u in scc
    }
    undirected = {
        u: succ[u] | {v for v in digraph.predecessors(u) if v in scc_set}
        for u in scc
    }
    found: set[tuple[str, ...]] = set()
    for hub in scc:
        nbrs = sorted(undirected[hub])
        for i, b in enumerate(nbrs):
            for c in nbrs[i + 1 :]:
                triple = tuple(sorted((hub, b, c)))
                if triple in found:
                    continue
                if _triple_strongly_connected(succ, *triple):
                    found.add(triple)
    return sorted(found)


def _sc_subsets_by_size(
    primes: dict,
    cutoff: int,
) -> dict[int, list[tuple[str, ...]]]:
    """Strongly connected subsets of size ``1..cutoff``, keyed by size.

    Size 1: self-loop nodes. Size 2: mutual edge pairs. Size 3: SC triples
    inside each SCC (hub + neighbor enumeration; see
    :func:`_sc_triples_in_scc`).
    """
    if not 1 <= cutoff <= LDOI_ENHANCED_CUTOFF_MAX:
        raise ValueError(
            f"cutoff must be in 1..{LDOI_ENHANCED_CUTOFF_MAX}, got {cutoff}"
        )

    digraph = _interaction_digraph(primes)
    nodes = sorted(primes)
    by_size: dict[int, list[tuple[str, ...]]] = {
        1: [(node,) for node in nodes if _node_has_self_loop(primes, node)],
    }

    if cutoff >= 2:
        by_size[2] = [
            (a, b)
            for a, b in combinations(nodes, 2)
            if digraph.has_edge(a, b) and digraph.has_edge(b, a)
        ]

    if cutoff >= 3:
        triples: list[tuple[str, ...]] = []
        for component in nx.strongly_connected_components(digraph):
            if len(component) < 3:
                continue
            triples.extend(_sc_triples_in_scc(digraph, sorted(component)))
        by_size[3] = triples

    return by_size


def _enhanced_nav_from_forcers(
    forcers_by_component: dict[tuple[str, ...], dict[tuple[int, ...], int]],
    impossible_by_component: (
        dict[tuple[str, ...], set[tuple[int, ...]]] | None
    ) = None,
) -> float:
    """Expected NAV from self-sustaining SC-subset forcers of one free node.

    ``forcers_by_component[component][state] = forced_value`` where
    ``component`` is a sorted node tuple and ``state`` holds values in the
    same order. ``impossible_by_component[component]`` holds self-negating
    states that are dropped from the uniform measure (renormalize over the
    rest).

    Enumerate the uniform measure over the ``2^k`` assignments to the
    ``k`` distinct nodes appearing in any forcer. Matching impossible
    states whose component is covered by those nodes are skipped;
    remaining assignments take the forced value of every matching forcer
    (raises if they disagree); unmatched assignments contribute ``0.5``.
    Impossible components that mention nodes outside the forcer set are
    ignored (they would otherwise enlarge the joint measure with unrelated
    SC-subset nodes). If every assignment is impossible, return ``0.5``.
    """
    if impossible_by_component is None:
        impossible_by_component = {}

    if not forcers_by_component:
        return 0.5

    # Forcer nodes only — do not union in global impossible components.
    nodes = sorted({node for component in forcers_by_component for node in component})
    if not nodes:
        return 0.5

    index = {node: i for i, node in enumerate(nodes)}
    # Flattened matchers: (node_indices, state_tuple, forced_value)
    matchers: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for component, states in forcers_by_component.items():
        idxs = tuple(index[node] for node in component)
        for state, value in states.items():
            matchers.append((idxs, state, int(value)))

    if not matchers:
        return 0.5

    impossible_matchers: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for component, states in impossible_by_component.items():
        if any(node not in index for node in component):
            continue
        idxs = tuple(index[node] for node in component)
        for state in states:
            impossible_matchers.append((idxs, state))

    total = 0.0
    kept = 0
    for assignment in product((0, 1), repeat=len(nodes)):
        if any(
            all(assignment[i] == bit for i, bit in zip(idxs, state, strict=True))
            for idxs, state in impossible_matchers
        ):
            continue
        forced: int | None = None
        for idxs, state, value in matchers:
            if all(assignment[i] == bit for i, bit in zip(idxs, state, strict=True)):
                if forced is not None and forced != value:
                    raise ValueError(
                        f"conflicting forcers on assignment "
                        f"{dict(zip(nodes, assignment, strict=True))}: "
                        f"{forced} vs {value}"
                    )
                forced = value
        total += float(forced) if forced is not None else 0.5
        kept += 1

    if kept == 0:
        return 0.5
    return total / kept


def _nav_from_ldoi_enhanced(
    primes,
    perturbation: dict[str, int],
    *,
    cutoff: int = 1,
    sc_subsets: dict[int, list[tuple[str, ...]]] | None = None,
) -> dict[str, float]:
    """LDOI NAV refined by self-sustaining / self-negating SC-subset probes.

    Starts from plain LDOI (forced → 0/1, else 0.5). Free strongly
    connected subsets of size ``1..cutoff`` are probed over all ``2^k``
    joint assignments (combined with ``perturbation``). Assignments that
    self-sustain for every member contribute probability mass that can
    move downstream free-node NAV away from 0.5. Assignments where any
    member self-negates are treated as impossible and dropped from the
    uniform measure (renormalize). A member that both self-sustains and
    self-negates under the same closure raises ``ValueError``.

    Parameters
    ----------
    cutoff : int, default 1
        Maximum SC-subset size to probe (``1..3``). ``1`` is self-loops only.
    sc_subsets : optional
        Precomputed output of :func:`_sc_subsets_by_size`; computed from
        ``primes`` when omitted.
    """
    if not 1 <= cutoff <= LDOI_ENHANCED_CUTOFF_MAX:
        raise ValueError(
            f"cutoff must be in 1..{LDOI_ENHANCED_CUTOFF_MAX}, got {cutoff}"
        )

    fixed = logical_domain_of_influence(primes, perturbation)
    nav = {
        node: float(fixed[node]) if node in fixed else 0.5
        for node in primes
    }

    free_set = {node for node in primes if node not in fixed}
    if not free_set:
        return nav

    if sc_subsets is None:
        sc_subsets = _sc_subsets_by_size(primes, cutoff)

    # forcers_for_node[N][component][state] = forced_value
    forcers_for_node: dict[
        str, dict[tuple[str, ...], dict[tuple[int, ...], int]]
    ] = defaultdict(lambda: defaultdict(dict))
    impossible_by_component: dict[
        tuple[str, ...], set[tuple[int, ...]]
    ] = defaultdict(set)

    for k in range(1, cutoff + 1):
        for component in sc_subsets.get(k, ()):
            if not free_set.issuperset(component):
                continue
            for state in product((0, 1), repeat=k):
                seed = dict(perturbation)
                seed.update(zip(component, state, strict=True))
                closure = logical_domain_of_influence(primes, seed)
                members = list(zip(component, state, strict=True))
                sustains = [
                    _is_self_sustaining(primes, closure, node, value)
                    for node, value in members
                ]
                negates = [
                    _is_self_negating(primes, closure, node, value)
                    for node, value in members
                ]
                for (node, value), sust, neg in zip(
                    members, sustains, negates, strict=True
                ):
                    if sust and neg:
                        raise ValueError(
                            f"{node}={value} both self-sustains and "
                            f"self-negates under LDOI of "
                            f"{dict(zip(component, state, strict=True))} "
                            f"(component {component})"
                        )
                if any(negates):
                    impossible_by_component[component].add(state)
                    continue
                if not all(sustains):
                    continue
                for n, val in closure.items():
                    if n not in free_set:
                        continue
                    prev = forcers_for_node[n][component].get(state)
                    if prev is not None and prev != int(val):
                        raise ValueError(
                            f"{n}: same state {component}={state} claimed both "
                            f"{prev} and {val}"
                        )
                    forcers_for_node[n][component][state] = int(val)

    impossible = {
        comp: set(states) for comp, states in impossible_by_component.items()
    }
    for N, by_component in forcers_for_node.items():
        nav[N] = _enhanced_nav_from_forcers(
            {comp: dict(states) for comp, states in by_component.items()},
            impossible,
        )

    return nav


def _nav_from_ldoi_partial(
    primes,
    perturbation: dict[str, int],
) -> dict[str, float | None]:
    """LDOI prediction with free nodes left as ``None`` (``?``).

    Forced nodes map to ``0.0`` / ``1.0``; nodes outside the LDOI are
    ``None`` so an optimistic upper-bound score can treat them as a
    perfect match.
    """
    constants = logical_domain_of_influence(primes, perturbation)
    return {
        node: float(constants[node]) if node in constants else None
        for node in primes
    }


def get_NAV_upper_bound_prediction(
    primes,
    experiments: list[NAVExperiment],
) -> dict[Assignment, NAVPrediction]:
    """LDOI-based partial predictions for optimistic upper-bound scoring.

    Same unique-intervention loop as :func:`get_NAV_prediction`, but free
    (non-LDOI) nodes are ``None`` instead of ``0.5``.
    """
    interventions = sorted({exp.fixes for exp in experiments})

    predictions: dict[Assignment, NAVPrediction] = {}
    for fixes in interventions:
        perturbation = assignment_to_dict(fixes)
        predictions[fixes] = NAVPrediction(
            fixes=fixes,
            average_values=_nav_from_ldoi_partial(primes, perturbation),
        )

    return predictions


def get_NAV_prediction(
    primes,
    experiments: list[NAVExperiment],
    *,
    method: str = "min_traps",
    cutoff: int = 1,
) -> dict[Assignment, NAVPrediction]:
    """Predict average attractor values for each unique NAV intervention.

    Collects the distinct ``fixes`` assignments across experiments and
    returns node average values for each unique intervention.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network in the format expected by
        PyBoolNet.
    experiments : list of NAVExperiment
        NAV experiments whose ``fixes`` interventions are evaluated.
    method : {"min_traps", "ldoi", "ldoi_enhanced"}, default "min_traps"
        How to compute ``average_values``:

        - ``"min_traps"``: fix interventions, compute minimal trap spaces,
          group them by unfixed source-node assignment, average within each
          group, then average groups equally (fixed 1 → 1.0, fixed 0 → 0.0,
          free ``?`` → 0.5).
        - ``"ldoi"``: coarser approximation via logical domain of influence;
          forced constants map to 0.0/1.0 and all other nodes to 0.5.
          Only credible when the model does not rely heavily on
          oscillatory behavior (free nodes → 0.5; oscillations unresolved).
        - ``"ldoi_enhanced"``: plain LDOI plus probability-mass refinement
          from free strongly connected subsets of size ``1..cutoff`` that
          self-sustain; self-negating assignments are dropped and the
          measure is renormalized (opt-in; default remains ``"min_traps"``).
    cutoff : int, default 1
        Maximum SC-subset size for ``method="ldoi_enhanced"`` (``1..3``).
        Ignored for other methods.

    Returns
    -------
    dict of Assignment to NAVPrediction
        One prediction per unique intervention, keyed by ``fixes``.
        ``average_values`` maps each model node to a value in ``[0, 1]``.
        Observed names ending in ``*_avg`` are resolved later by
        :func:`~boolmore.evaluation.agreement.get_agreements`.

    Raises
    ------
    ValueError
        If a fixed node is not present in the model, ``method`` is not
        recognized, or (for ``ldoi_enhanced``) a probed SC-subset member
        both self-sustains and self-negates under the same LDOI closure.
    RuntimeError
        If ``method="min_traps"`` and PyBoolNet's ASP backend
        (gringo/clasp) fails the probe check.

    Notes
    -----
    ``method="ldoi"`` / ``"ldoi_enhanced"`` NAV scores should be interpreted
    with care for models whose phenotype depends on oscillations or complex
    cyclic attractors; prefer ``"min_traps"`` in those cases.
    """
    if method not in NAV_METHODS:
        raise ValueError(
            f"Unknown NAV prediction method {method!r}; "
            f"expected one of {NAV_METHODS}"
        )

    interventions = sorted({exp.fixes for exp in experiments})

    predictions: dict[Assignment, NAVPrediction] = {}
    if method == "min_traps":
        for fixes in interventions:
            perturbation = assignment_to_dict(fixes)
            predictions[fixes] = NAVPrediction(
                fixes=fixes,
                average_values=_nav_from_min_traps(primes, perturbation),
            )
    elif method == "ldoi_enhanced":
        sc_subsets = _sc_subsets_by_size(primes, cutoff)
        for fixes in interventions:
            perturbation = assignment_to_dict(fixes)
            predictions[fixes] = NAVPrediction(
                fixes=fixes,
                average_values=_nav_from_ldoi_enhanced(
                    primes,
                    perturbation,
                    cutoff=cutoff,
                    sc_subsets=sc_subsets,
                ),
            )
    else:
        for fixes in interventions:
            perturbation = assignment_to_dict(fixes)
            predictions[fixes] = NAVPrediction(
                fixes=fixes,
                average_values=_nav_from_ldoi(primes, perturbation),
            )

    return predictions
