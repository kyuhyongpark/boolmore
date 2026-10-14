import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from time import perf_counter
from dataclasses import dataclass

from pyboolnet.trap_spaces import compute_trapspaces_within_subspace, compute_trap_spaces

from boolmore.experiment import PhenotypeExperiment, NAVExperiment
from boolmore.boolean_functions import assignment_to_dict
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.ldoi import logical_domain_of_influence
from boolmore.types import Assignment

NAV_METHODS = ("min_traps", "ldoi")
_AVG_SUFFIX = "_avg"


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
        Average attractor value of each node in ``[0, 1]``.
        ``None`` means unknown / free (``?``), used only by the LDOI
        upper-bound path. May also include synthetic ``*_avg`` keys
        requested by experiments, equal to the mean of related sibling
        nodes (or ``None`` if any sibling is unknown).
    """

    fixes: Assignment
    average_values: dict[str, float | None]


def _related_node_names(base: str, node_names: Iterable[str]) -> list[str]:
    """Return names matching ``base``, ``base_<int>``, or ``base_<A-Z>``."""
    pattern = re.compile(rf"^(?:{re.escape(base)}|{re.escape(base)}_(?:\d+|[A-Z]))$")
    return [name for name in node_names if pattern.fullmatch(name)]


def _average_related_values(
    average_values: dict[str, float | None],
    observed_avg: str,
) -> float | None:
    """Mean of sibling node values for an ``*_avg`` observed name.

    Raises
    ------
    ValueError
        If ``observed_avg`` does not end with ``_avg``, the base is empty,
        or no related sibling nodes are present in ``average_values``.
    """
    if not observed_avg.endswith(_AVG_SUFFIX):
        raise ValueError(f"Expected observed name ending with {_AVG_SUFFIX!r}: {observed_avg!r}")
    base = observed_avg[: -len(_AVG_SUFFIX)]
    if not base:
        raise ValueError(f"Empty base for average observed node {observed_avg!r}")

    related = _related_node_names(base, average_values)
    if not related:
        raise ValueError(
            f"No related nodes found for {observed_avg!r} "
            f"(looked for {base!r}, {base}_<int>, or {base}_<A-Z>)"
        )

    floats: list[float] = []
    for name in related:
        value = average_values[name]
        if value is None:
            return None
        floats.append(value)
    return sum(floats) / len(floats)


def _inject_avg_values(
    average_values: dict[str, float | None],
    avg_observed: set[str],
) -> dict[str, float | None]:
    """Return a copy of ``average_values`` with synthetic ``*_avg`` keys filled."""
    if not avg_observed:
        return average_values
    result = dict(average_values)
    for observed_avg in sorted(avg_observed):
        result[observed_avg] = _average_related_values(average_values, observed_avg)
    return result


def _avg_observed_from_experiments(experiments: list[NAVExperiment]) -> set[str]:
    """Collect unique observed names ending with ``_avg``."""
    return {
        exp.observed
        for exp in experiments
        if exp.observed.endswith(_AVG_SUFFIX)
    }


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


def _nav_from_min_traps(
    primes,
    perturbation: dict[str, int],
) -> dict[str, float]:
    """Average node values over minimal trap spaces after fixing interventions."""
    ensure_asp_backend()
    new_primes = _fix_constants(primes, perturbation)

    tr = compute_trap_spaces(new_primes, "min")

    for trap in tr:
        for node in primes:
            if node not in trap:  # type: ignore
                trap[node] = "?"  # type: ignore
            else:
                trap[node] = str(trap[node])  # type: ignore

    average_values: dict[str, float] = {}
    for trap in tr:
        for node in primes:
            if node not in average_values:
                average_values[node] = 0.0
            if trap[node] == "1":  # type: ignore
                average_values[node] += 1.0 / len(tr)
            elif trap[node] == "?":  # type: ignore
                average_values[node] += 0.5 / len(tr)

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
    (non-LDOI) nodes are ``None`` instead of ``0.5``. Observed names ending
    in ``_avg`` are stored as the mean of related sibling nodes (or
    ``None`` if any sibling is free).
    """
    avg_observed = _avg_observed_from_experiments(experiments)
    interventions = sorted({exp.fixes for exp in experiments})

    predictions: dict[Assignment, NAVPrediction] = {}
    for fixes in interventions:
        perturbation = assignment_to_dict(fixes)
        average_values = _inject_avg_values(
            _nav_from_ldoi_partial(primes, perturbation),
            avg_observed,
        )
        predictions[fixes] = NAVPrediction(
            fixes=fixes,
            average_values=average_values,
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
        Observed names ending in ``_avg`` request a synthetic key equal
        to the mean of related sibling nodes (``base``, ``base_<int>``,
        or ``base_<A-Z>``).
    method : {"min_traps", "ldoi"}, default "min_traps"
        How to compute ``average_values``:

        - ``"min_traps"``: fix interventions, compute minimal trap spaces,
          and average node values over those trap spaces (fixed 1 → 1.0,
          fixed 0 → 0.0, free ``?`` → 0.5).
        - ``"ldoi"``: coarser approximation via logical domain of influence;
          forced constants map to 0.0/1.0 and all other nodes to 0.5.
          Only credible when the model does not rely heavily on
          oscillatory behavior (free nodes → 0.5; oscillations unresolved).

    Returns
    -------
    dict of Assignment to NAVPrediction
        One prediction per unique intervention, keyed by ``fixes``.
        ``average_values`` maps each model node (and any requested
        ``*_avg`` keys) to a value in ``[0, 1]``.

    Raises
    ------
    ValueError
        If a fixed node is not present in the model, ``method`` is not
        recognized, or an ``*_avg`` observed name has no related siblings.
    RuntimeError
        If ``method="min_traps"`` and PyBoolNet's ASP backend
        (gringo/clasp) fails the probe check.

    Notes
    -----
    ``method="ldoi"`` NAV scores should be interpreted with care for models
    whose phenotype depends on oscillations or complex cyclic attractors;
    prefer ``"min_traps"`` in those cases.
    """
    if method not in NAV_METHODS:
        raise ValueError(
            f"Unknown NAV prediction method {method!r}; "
            f"expected one of {NAV_METHODS}"
        )

    if method == "min_traps":
        compute_average = _nav_from_min_traps
    else:
        compute_average = _nav_from_ldoi

    avg_observed = _avg_observed_from_experiments(experiments)
    interventions = sorted({exp.fixes for exp in experiments})

    predictions: dict[Assignment, NAVPrediction] = {}
    for fixes in interventions:
        perturbation = assignment_to_dict(fixes)
        average_values = _inject_avg_values(
            compute_average(primes, perturbation),
            avg_observed,
        )
        predictions[fixes] = NAVPrediction(
            fixes=fixes,
            average_values=average_values,
        )

    return predictions
