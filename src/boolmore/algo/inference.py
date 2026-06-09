from collections import defaultdict
from time import perf_counter

from pyboolnet.trap_spaces import compute_trapspaces_within_subspace
from pyboolnet.prime_implicants import percolate

from boolmore.core.experiment import Experiment
from boolmore.core.prediction import Prediction

Assignment = tuple[tuple[str, int], ...]

def _assignment_to_dict(assignment: Assignment):
    """
    Convert an immutable assignment representation into a dictionary.

    Parameters
    ----------
    assignment : Assignment
        Tuple of ``(node, value)`` pairs, where ``value`` is typically
        ``0`` or ``1``.

    Returns
    -------
    dict[str, int]
        Dictionary mapping each node to its assigned value.

    Notes
    -----
    If the same node appears multiple times, the last occurrence
    overwrites any previous value.
    """
    result = {}
    for node, value in assignment:
        result[node] = value
    return result

def get_phenotype_prediction(
    primes,
    experiments: list[Experiment],
    debug: bool = False,
):
    """
    Predict whether each experiment's phenotype is compatible with the
    Boolean network under its perturbation and source assignments.

    For each experiment, the function first checks whether a previously
    computed maximal trap space for the same perturbation already satisfies
    the required sources and phenotype. If so, that cached result is reused.

    Otherwise, the network is percolated according to the perturbation, and
    maximal trap spaces contained within the combined source and phenotype
    subspace are searched. If such a trap space exists, the experiment is
    predicted to be feasible.

    Percolated prime implicants are cached by perturbation, and discovered
    maximal trap spaces are also cached for reuse by later experiments with
    the same perturbation.

    Parameters
    ----------
    primes
        Prime implicants of the Boolean network in the format expected by
        PyBoolNet.
    experiments : list[Experiment]
        Experiments to evaluate. Each experiment must define
        ``perturbation``, ``sources``, and ``phenotype`` assignments.

    Returns
    -------
    list[Prediction]
        One ``Prediction`` object for each experiment, containing whether a
        compatible trap space was found and, if so, the matching maximal
        trap spaces.

    Raises
    ------
    ValueError
        If a node appearing in a perturbation, source assignment, or
        phenotype assignment is not present in the model.
    """
    results = []
    traps_cache = defaultdict(list)
    primes_cache = defaultdict(list)

    for exp in experiments:

        if debug:
            print(f"\nExperiment {exp.id}")

        result = Prediction(
            id=exp.id,
            perturbation=exp.perturbation,
            sources=exp.sources,
            phenotype=exp.phenotype,
            found_phenotypes=[],
            predicted_exists=False,
            agreement=0.0,
            score=0.0,
        )

        # check cache
        t0 = perf_counter()

        sources = _assignment_to_dict(exp.sources)
        phenotype = _assignment_to_dict(exp.phenotype)

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
                    result.found_phenotypes.append(max_trap)
                    result.predicted_exists = True
                    break

        if debug:
            print(
                f"  cache check: "
                f"{perf_counter() - t0:.6f} s"
            )

        if result.predicted_exists:
            if debug:
                print("  cache hit (found_phenotypes)")
            results.append(result)
            continue

        perturbation = _assignment_to_dict(exp.perturbation)

        # get percolated primes
        t0 = perf_counter()

        if exp.perturbation in primes_cache:
            perc_primes = primes_cache[exp.perturbation]
            if debug:
                print("  cache hit (perc_primes)")
        else:
            perc_primes = primes.copy()
            for node in perturbation:
                if node not in primes:
                    raise ValueError(f"{node} is not in the model")

            # perc_primes = percolate(
            #     primes,
            #     add_constants=perturbation,
            #     remove_constants=False,
            #     copy=True,
            # )
                if perturbation[node] == 0:
                    perc_primes[node] = [[{}],[]]
                else:
                    perc_primes[node] = [[],[{}]]
            primes_cache[exp.perturbation] = perc_primes

        if debug:
            print(
                f"  get perc_primes: "
                f"{perf_counter() - t0:.6f} s"
            )

        subspace = sources.copy()
        subspace.update(phenotype)

        for node in subspace:
            if node not in perc_primes:
                raise ValueError(f"{node} not in the model.")

        # compute max traps
        t0 = perf_counter()

        max_traps = compute_trapspaces_within_subspace(
            perc_primes,
            subspace=subspace,
            type_="max",
            max_output=1,
        )

        if debug:
            print(
                f"  compute max_trap: "
                f"{perf_counter() - t0:.6f} s"
            )

        if max_traps:
            result.found_phenotypes += max_traps
            result.predicted_exists = True
            traps_cache[exp.perturbation] += max_traps

        results.append(result)

    return results