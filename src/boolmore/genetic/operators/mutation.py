import random
from contextlib import contextmanager
from typing import Iterator

import boolmore.boolean_functions as bf
from boolmore.evaluation import constraint as cons
from boolmore.genetic.operators import constraint_enforcement as ce
from boolmore.model import Model

# Regulate-failure handling after a mutation trial:
#   "remutate" — no try_add; full remutation only
#   "oneshot"  — at most one try_add_regulator, then remutate if still invalid
REGULATE_REPAIR_MODE = "oneshot"
# Backward-compatible gate used by older benches (False => remutate).
REGULATE_REPAIR = True

_REPAIR_STAT_KEYS = (
    "trials",
    "retries",
    "repair_attempts",
    "repair_noop",
    "repair_forward",
    "repair_irr_fallback",
    # irr fallback that fixed regulate (after flip, before necessary):
    "repair_irr_fallback_fixed",
    # After try_add_regulator, before re-imposing necessary:
    "repair_regulate_fixed",
    "repair_regulate_still_false",
    # Was fixed by the flip, then false again after impose_necessary:
    "repair_undone_by_necessary",
    "repair_accepted",
    "repair_regulate_fixed_but_other_fail",
)

_repair_stats: dict[str, int] = {k: 0 for k in _REPAIR_STAT_KEYS}


def reset_repair_stats() -> None:
    """Zero mutation / regulate-repair counters used by benches."""
    for key in _REPAIR_STAT_KEYS:
        _repair_stats[key] = 0


def get_repair_stats() -> dict[str, int]:
    """Return a copy of mutation / regulate-repair counters."""
    return dict(_repair_stats)


@contextmanager
def regulate_repair_enabled(enabled: bool) -> Iterator[None]:
    """Temporarily toggle repair (``oneshot`` if True, ``remutate`` if False)."""
    with regulate_repair_mode("oneshot" if enabled else "remutate"):
        yield


@contextmanager
def regulate_repair_mode(mode: str) -> Iterator[None]:
    """Temporarily set ``REGULATE_REPAIR_MODE`` to ``remutate`` or ``oneshot``."""
    global REGULATE_REPAIR_MODE, REGULATE_REPAIR
    if mode not in ("remutate", "oneshot"):
        raise ValueError(f"unknown regulate repair mode: {mode!r}")
    previous_mode = REGULATE_REPAIR_MODE
    previous_flag = REGULATE_REPAIR
    REGULATE_REPAIR_MODE = mode
    REGULATE_REPAIR = mode != "remutate"
    try:
        yield
    finally:
        REGULATE_REPAIR_MODE = previous_mode
        REGULATE_REPAIR = previous_flag


def mutate_rr(rr: str, probability: float) -> str:
    """
    Returns the mutated representation of that rule
    when given any representation of a rule.
    
    This function is biased to activate all the implicants.
    Use mutate_rr_bias for non-biased or specific biased mutations.

    Parameters
    ----------
    rr          - representation of the rule                    :length 2^k binary str
    probability - probability that each of the binary number    :float between 0 and 1
                  in the representation is mutated.

    Returns
    -------
    mutated_rr - representation of the mutated rule     :length 2^k binary str
    
    """
    out = bytearray(rr, "ascii")
    zero = 48  # '0'
    for i in range(len(out)):
        if random.random() < probability:
            out[i] = zero + ((out[i] - zero + 1) & 1)
    return out.decode("ascii")


def mutate_rr_bias(rr:str, probability:float, bias:float=0.5) -> str:
    """
    Returns the mutated representation of a rule
    when given any representation of a rule,
    with a bias to invert representation in the process.

    Parameters
    ----------
    rr          - representation of the rule                    :length 2^k binary str
    probability - probability that each of the binary number    :float between 0 and 1
                  in the representation is mutated.
    bias        - probability to invert rr when mutating        :float between 0 and 1

    Returns
    -------
    mutated_rr - representation of the mutated rule     :length 2^k binary str

    """
    rnd = random.random()
    if rnd < bias:
        irr = bf.get_max_irr(rr)
        irr = mutate_rr(irr, probability)
        mutated_rr = bf.get_max_irr(irr)
    else:
        mutated_rr = mutate_rr(rr, probability)

    return mutated_rr


def mutate_rr_constraint(regulators:tuple[str, ...], rr:str, base_rr:str, constraints:dict,
                         node:str, probability:float, bias:float=0.5,
                         ) -> tuple[str, bool, str | None, str | None]:
    """
    When given any representation of a rule,
    returns the mutated representation of that rule
    with a certain inacitivity bias
    that follows all the constraints.

    Parameters
    ----------
    regulators  - the regulating nodes                      :tuple[str]
    rr          - representation of the rule                :length 2^(k+a) binary str
    base_rr     - representation of the the baseline rule   :length 2^k binary str
    constraints - represents 5 types of constraints         :dict[str, list or dict]
                  (fixed, regulate, necessary, group, possible_constant)
    node        - the target node                           :str
    probability - probability that each binary is mutated   :float between 0 and 1
    bias        - probability to invert rr when mutating    :float between 0 and 1

    Returns
    -------
    mutated_rr  - representation of the mutated rule        :length 2^(k+a) binary str
    modified    - True if modified                          :bool
    min_rr      - minimal form of accepted ``mutated_rr``, or None on early return
    max_rr      - maximal form of accepted ``mutated_rr``, or None on early return

    """
    # constant nodes have no degree of freedom and should not be mutated
    if len(regulators) == 0:
        if rr == base_rr:
            return rr, False, None, None
        else:
            return base_rr, True, None, None

    # nodes with single regulators (including source nodes)
    # have no degree of freedom, and hence should not be mutated
    # UNLESS it was originally a constant node but gained an extra edge,
    # in which case it can become a constant node
    if len(regulators) == 1 and len(base_rr) != 1:
        if rr == base_rr:
            return rr, False, None, None
        else:
            return base_rr, True, None, None

    # impose constraints - fixed
    if node in constraints['fixed']:
        if rr == base_rr:
            return rr, False, None, None
        else:
            return base_rr, True, None, None

    max_original = bf.get_uni_rr(rr)
    mutated_rr = ''
    min_rr = ''
    max_rr = max_original
    redo = True
    trial = 0
    while redo == True:
        if trial > 10000:
            raise Exception("too many trials. check mutation parameter")

        _repair_stats["trials"] += 1

        ### mutation module ###
        # impose constraint - group
        if node in constraints['group']:
            groups = constraints['group'][node]
            group_rr, group_regulators = ce.rr2group_rr(regulators, rr, groups)
            mutated_group_rr = mutate_rr_bias(group_rr, probability, bias)
            mutated_rr = ce.group_rr2rr(regulators, mutated_group_rr, group_regulators)
        else:
            mutated_rr = mutate_rr_bias(rr, probability, bias)

        # impose constraint - necessary
        if node in constraints['necessary']:
            for necc in constraints['necessary'][node]:
                mutated_rr = ce.impose_necessary(regulators, mutated_rr, necc)
        ### end of mutation ###

        min_rr = bf.get_uni_rr(mutated_rr, max=False)
        max_rr = bf.get_uni_rr(mutated_rr, max=True)

        def _constraint_status() -> tuple[bool, str | None]:
            """Return (needs_redo, first_failing_regulate_or_None)."""
            needs_redo = False
            failing_reg: str | None = None
            if node in constraints['regulate']:
                for reg in constraints['regulate'][node]:
                    if not cons.check_regulate(
                        regulators, min_rr, reg, is_min=True
                    ):
                        needs_redo = True
                        failing_reg = reg
                        break
            if node in regulators:
                needs_redo = needs_redo or cons.check_source(
                    regulators, mutated_rr, node, min_rr=min_rr
                )
            if cons.check_constant(mutated_rr, max_rr=max_rr):
                if (
                    node not in constraints['possible_constant']
                    and len(base_rr) != 1
                ):
                    needs_redo = True
                elif base_rr == '1' and '0' in max_rr:
                    needs_redo = True
                elif base_rr == '0' and '1' in max_rr:
                    needs_redo = True
            return needs_redo, failing_reg

        redo, failed_reg = _constraint_status()

        # Oneshot regulate repair (skipped when mode is remutate).
        if (
            redo
            and failed_reg is not None
            and REGULATE_REPAIR
            and REGULATE_REPAIR_MODE == "oneshot"
        ):
            _repair_stats["repair_attempts"] += 1
            necessary = tuple(constraints['necessary'].get(node, ()))
            before = mutated_rr
            mutated_rr, _, _, path = ce.try_add_regulator(
                regulators,
                mutated_rr,
                failed_reg,
                max_rr=max_rr,
                necessary=necessary,
            )
            if path == "noop" or mutated_rr == before:
                _repair_stats["repair_noop"] += 1
            else:
                if path == "forward":
                    _repair_stats["repair_forward"] += 1
                elif path == "irr":
                    _repair_stats["repair_irr_fallback"] += 1

                min_after_flip = bf.get_uni_rr(mutated_rr, max=False)
                fixed_by_flip = cons.check_regulate(
                    regulators, min_after_flip, failed_reg, is_min=True
                )
                if fixed_by_flip:
                    _repair_stats["repair_regulate_fixed"] += 1
                    if path == "irr":
                        _repair_stats["repair_irr_fallback_fixed"] += 1
                else:
                    _repair_stats["repair_regulate_still_false"] += 1

                if node in constraints['necessary']:
                    for necc in constraints['necessary'][node]:
                        mutated_rr = ce.impose_necessary(
                            regulators, mutated_rr, necc
                        )

                min_rr = bf.get_uni_rr(mutated_rr, max=False)
                max_rr = bf.get_uni_rr(mutated_rr, max=True)

                regulate_ok = cons.check_regulate(
                    regulators, min_rr, failed_reg, is_min=True
                )
                if fixed_by_flip and not regulate_ok:
                    _repair_stats["repair_undone_by_necessary"] += 1

                redo, failed_reg = _constraint_status()
                if not redo:
                    _repair_stats["repair_accepted"] += 1
                elif failed_reg is None:
                    # Regulate holds; source/constant still force remutation.
                    _repair_stats["repair_regulate_fixed_but_other_fail"] += 1

        if redo == True:
            trial += 1
            _repair_stats["retries"] += 1

    modified = max_original != max_rr

    return mutated_rr, modified, min_rr, max_rr


def add_regulator(regulators:tuple[str, ...], rr:str, signs:str, new_regulator:str, new_sign:str,
                  bias:float=0.5) -> tuple[tuple[str, ...], str, str]:
    """
    Returns a new representation of a rule with an extra regulator.

    ``rr`` / ``added_rr`` use mask order (``index == mask``). The new regulator
    is appended and becomes the least-significant mask bit.

    Parameters
    ----------
    regulators    - the regulating nodes                    :length k tuple[str]
    rr            - rule in mask order                      :length 2^k binary str
    signs         - represents the signs of regulators      :length k binary str
                    0 means negative regulation
    new_regulator - the added regulating node               :str
    new_sign      - represents sign of the new regulator    :length 1 binary str

    Returns
    -------
    added_regulators - the regulating nodes             :length k+1 tuple[str]
    added_rr         - rule in mask order               :length 2^(k+1) binary str
    added_signs      - represents signs of regulators   :length k+1 binary str

    """
    nodes = list(regulators)
    nodes.append(new_regulator)
    added_regulators = tuple(nodes)

    added_signs = signs + new_sign

    rnd = random.random()
    if rnd < bias:
        rr = bf.get_max_irr(rr)

    # Old value at new-bit=0, random at new-bit=1; ascending old-mask order.
    n = len(rr)
    added = ['0'] * (2 * n)
    for old_mask in range(n):
        added[old_mask << 1] = rr[old_mask]
        added[(old_mask << 1) | 1] = str(random.randint(0, 1))
    added_rr = ''.join(added)

    if rnd < bias:
        added_rr = bf.get_max_irr(added_rr)

    return added_regulators, added_rr, added_signs


def delete_regulator(regulators:tuple[str, ...], rr:str, signs:str, target_regulator:str,
                     bias:float=0.5) -> tuple[tuple[str, ...], str, str]:
    """
    Returns a new representation of a rule with the target regulator deleted.

    ``rr`` / ``deleted_rr`` use mask order (``index == mask``).

    Parameters
    ----------
    regulators       - the regulating nodes                 :length k tuple[str]
    rr               - rule in mask order                   :length 2^k binary str
    signs            - represents the signs of regulators   :length k binary str
                       0 means negative regulation
    target_regulator - the regulator to be deleted          :str

    Returns
    -------
    deleted_regulators - the regulating nodes                   :length k-1 tuple[str]
    deleted_rr         - rule in mask order                     :length 2^(k-1) binary str
    deleted_signs      - represents the signs of regulators     :length k-1 binary str

    """
    k = len(regulators)
    n = regulators.index(target_regulator)

    nodes = list(regulators)
    del nodes[n]
    deleted_regulators = tuple(nodes)

    deleted_signs = signs[:n] + signs[n+1:]

    deleted_rr = ''

    rnd = random.random()
    if rnd < bias:
        rr = bf.get_max_irr(rr)

    bit = 1 << (k - n - 1)
    bi = list(rr)
    for i in range(2**k):
        # Keep masks where the deleted regulator's bit is unset.
        if not (i & bit):
            deleted_rr += bi[i]

    if rnd < bias:
        deleted_rr = bf.get_max_irr(deleted_rr)

    return deleted_regulators, deleted_rr, deleted_signs


def mutate_model(
        model:Model,
        probability:float,
        edge_prob:float,
        constraints:dict,
        bias:float=0.5,
        seed:int|None=None
    ) -> Model:
    """
    Returns a mutated model.

    Parameters
    ----------
    probability - probability that each binary is mutated               :float between 0 and 1
    edge_prob   - probability to add or delete an edge from the pool    :float between 0 and 1
    bias        - probability to invert rr when mutating                :float between 0 and 1
    seed        - random seed                                           :int|None

    Returns
    -------
    mutated_model - a new model with mutated functions  :Model class

    """
    mutated_model = Model()

    mutated_model.base = model.base
    mutated_model.edge_pool = model.edge_pool

    mutated_model.primes = model.primes.copy()

    mutated_model.regulators_dict = model.regulators_dict.copy()
    mutated_model.signs_dict = model.signs_dict.copy()        
    mutated_model.rr_dict = model.rr_dict.copy()

    if seed != None:
        random.seed(seed)

    rnd = random.random()
    if rnd < edge_prob and len(mutated_model.edge_pool)>0:
        new_edge = random.choice(mutated_model.edge_pool)

        new_edge_node = new_edge[1]
        new_regulator = new_edge[0]
        new_sign = new_edge[2]

        regulators = mutated_model.regulators_dict[new_edge_node]
        rr = mutated_model.rr_dict[new_edge_node]
        signs = mutated_model.signs_dict[new_edge_node]

        if new_regulator not in regulators:
            modified_regulators, modified_rr, modified_signs = add_regulator(regulators, rr, signs, new_regulator, new_sign)
        else:
            modified_regulators, modified_rr, modified_signs = delete_regulator(regulators, rr, signs, new_regulator)

        mutated_model.regulators_dict[new_edge_node] = modified_regulators
        mutated_model.rr_dict[new_edge_node] = modified_rr
        mutated_model.signs_dict[new_edge_node] = modified_signs
        prime1 = bf.rr2prime(modified_regulators, modified_rr, modified_signs, inverted = False)
        mutated_model.primes[new_edge_node] = prime1

    for node in mutated_model.rr_dict:
        # get mutated_rr from rr
        mutated_rr, modified, min_rr, max_rr = mutate_rr_constraint(
            mutated_model.regulators_dict[node],
            mutated_model.rr_dict[node],
            mutated_model.base.rr_dict[node],
            constraints,
            node,
            probability,
            bias)
        mutated_model.rr_dict[node] = mutated_rr

        # get primes from the mutated_rr
        # if the representations are equivalent, take the old prime
        if modified:
            prime1 = bf.rr2prime(
                mutated_model.regulators_dict[node],
                mutated_rr,
                mutated_model.signs_dict[node],
                inverted=False,
                min_rr=min_rr,
                max_rr=max_rr,
            )
            mutated_model.primes[node] = prime1
            # irr = get_max_irr(mutated_model.rr_dict[node])
            # prime2 = rr2prime(mutated_model.regulators_dict[node], irr, mutated_model.signs_dict[node], inverted = True)
            # assert prime1 == prime2, "rr and irr lead to different result!"
    
    mutated_model.edges = mutated_model._construct_edges()

    return mutated_model