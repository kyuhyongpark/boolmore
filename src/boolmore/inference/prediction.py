import logging
from collections import defaultdict
from time import perf_counter
from dataclasses import dataclass

from pyboolnet.trap_spaces import compute_trapspaces_within_subspace, compute_trap_spaces

from boolmore.experiment import PhenotypeExperiment, NAVExperiment
from boolmore.boolean_functions import assignment_to_dict
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.ldoi import _implicant_implied_by, logical_domain_of_influence
from boolmore.types import Assignment

NAV_METHODS = ("min_traps", "ldoi", "ldoi_enhanced")


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


def _enhanced_nav_from_forcers(
    forcers_by_ancestor: dict[str, dict[int, int]],
) -> float:
    """Expected NAV from self-sustaining self-loop forcers of one free node.

    ``forcers_by_ancestor[A][v] = forced_value_of_node`` for qualifying
    states. Ancestors that cover both values consume all remaining mass
    first (50/50 over ``A=0`` and ``A=1``); one-sided ancestors then each
    take half of what is left. Unknown remainder contributes ``0.5``.
    """
    if not forcers_by_ancestor:
        return 0.5

    both = sorted(
        (A, d) for A, d in forcers_by_ancestor.items() if 0 in d and 1 in d
    )
    one = sorted(
        (A, d) for A, d in forcers_by_ancestor.items() if len(d) == 1
    )

    remaining = 1.0
    expected = 0.0

    for _A, d in both:
        if remaining <= 0:
            break
        for v in (0, 1):
            mass = remaining * 0.5
            expected += mass * float(d[v])
        remaining = 0.0

    for _A, d in one:
        if remaining <= 0:
            break
        _v, val = next(iter(d.items()))
        mass = remaining * 0.5
        expected += mass * float(val)
        remaining *= 0.5

    if remaining > 0:
        expected += remaining * 0.5

    return float(expected)


def _nav_from_ldoi_enhanced(
    primes,
    perturbation: dict[str, int],
) -> dict[str, float]:
    """LDOI NAV refined by self-loop + self-sustaining free-node forcers.

    Starts from plain LDOI (forced → 0/1, else 0.5). Free nodes with a
    self-loop are probed at 0 and 1 (combined with ``perturbation``); those
    states that self-sustain contribute probability mass that can move
    downstream free-node NAV away from 0.5.
    """
    fixed = logical_domain_of_influence(primes, perturbation)
    nav = {
        node: float(fixed[node]) if node in fixed else 0.5
        for node in primes
    }

    free = [node for node in primes if node not in fixed]
    if not free:
        return nav

    candidates = [node for node in free if _node_has_self_loop(primes, node)]
    if not candidates:
        return nav

    # forcers_for_node[N] = list of (A, v, forced_value)
    forcers_for_node: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for A in candidates:
        for v in (0, 1):
            seed = dict(perturbation)
            seed[A] = v
            closure = logical_domain_of_influence(primes, seed)
            if not _is_self_sustaining(primes, closure, A, v):
                continue
            for n, val in closure.items():
                if n not in free:
                    continue
                forcers_for_node[n].append((A, v, int(val)))

    for N, forcers in forcers_for_node.items():
        by_anc: dict[str, dict[int, int]] = defaultdict(dict)
        for A, v, val in forcers:
            if v in by_anc[A] and by_anc[A][v] != val:
                raise ValueError(
                    f"{N}: same state {A}={v} claimed both "
                    f"{by_anc[A][v]} and {val}"
                )
            by_anc[A][v] = val
        nav[N] = _enhanced_nav_from_forcers(dict(by_anc))

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
          from free nodes that have a self-loop and self-sustain at 0
          and/or 1 (opt-in; default remains ``"min_traps"``).

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
        If a fixed node is not present in the model or ``method`` is not
        recognized.
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

    if method == "min_traps":
        compute_average = _nav_from_min_traps
    elif method == "ldoi_enhanced":
        compute_average = _nav_from_ldoi_enhanced
    else:
        compute_average = _nav_from_ldoi

    interventions = sorted({exp.fixes for exp in experiments})

    predictions: dict[Assignment, NAVPrediction] = {}
    for fixes in interventions:
        perturbation = assignment_to_dict(fixes)
        predictions[fixes] = NAVPrediction(
            fixes=fixes,
            average_values=compute_average(primes, perturbation),
        )

    return predictions
