# %% [markdown]
# LDOI vs shared minimal trap-space states
#
# For each demo experiment, fix the experiment `fixes`, then check that the
# LDOI (constants after percolating those fixes in the primes) matches the
# shared node states among all **minimal** trap spaces of the fixed network.
#
# Percolation / LDOI forces exactly the nodes that take the same value in every
# minimal trap space after the intervention.
#
# LDOI: pyboolnet `percolate` + `find_constants`.
# Trap spaces: pyboolnet `compute_trap_spaces(..., "min")` after
# `create_constants` for the experiment fixes.
#
# Note: `compute_trap_spaces` needs a working clingo/ASP binary (may fail in
# restricted sandboxes).

# %%
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pyboolnet.prime_implicants import create_constants, find_constants, percolate
from pyboolnet.trap_spaces import compute_trap_spaces

from boolmore.io.load import load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.types import Assignment

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
BNET = REPO / "demo" / "CAD.bnet"
DATA = REPO / "demo" / "CAD_data.csv"

# %% [markdown]
# ## Helpers


# %%
def fixes_to_dict(fixes: Assignment) -> dict[str, int]:
    return {node: int(value) for node, value in fixes}


def shared_trap_states(
    traps: list[dict[str, int]],
    fallback: dict[str, int],
) -> dict[str, int]:
    """Intersection of node states present with the same value in every trap.

    If there are no trap spaces, return ``fallback`` (typically the constants
    already present after applying the experiment fixes).
    """
    if not traps:
        return dict(fallback)

    shared = dict(traps[0])
    for trap in traps[1:]:
        shared = {
            node: value
            for node, value in shared.items()
            if trap.get(node) == value
        }
    return shared


def ldoi_from_fixes(primes: dict, fixes: dict[str, int]) -> dict[str, int]:
    fixed_primes = create_constants(primes, fixes, copy=True)
    percolated = percolate(fixed_primes, copy=True)
    return find_constants(percolated)


def min_traps_from_fixes(primes: dict, fixes: dict[str, int]) -> list[dict[str, int]]:
    fixed_primes = create_constants(primes, fixes, copy=True)
    return compute_trap_spaces(fixed_primes, "min")


def format_state(state: dict[str, int]) -> str:
    if not state:
        return "{}"
    return "{" + ", ".join(f"{n}={state[n]}" for n in sorted(state)) + "}"


# %% [markdown]
# ## Load model and experiments

# %%
primes = load_primes(BNET)
experiments = import_NAV_exps(DATA)

print(f"model: {BNET.name} ({len(primes)} nodes)")
print(f"experiments: {len(experiments)}")

# %% [markdown]
# ## Check LDOI == shared minimal trap states

# %%
ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    ids_by_fixes[exp.fixes].append(exp.id)

mismatches: list[Assignment] = []

for fixes, exp_ids in ids_by_fixes.items():
    fixes_dict = fixes_to_dict(fixes)
    fixed_primes = create_constants(primes, fixes_dict, copy=True)
    seed_constants = find_constants(fixed_primes)

    ldoi = ldoi_from_fixes(primes, fixes_dict)
    min_traps = min_traps_from_fixes(primes, fixes_dict)
    shared = shared_trap_states(min_traps, fallback=seed_constants)
    match = ldoi == shared

    fixes_label = format_state(fixes_dict) if fixes_dict else "(none)"
    print(f"fixes {fixes_label}")
    print(f"  experiment ids: {exp_ids}")
    print(f"  LDOI:           {format_state(ldoi)}")
    print(f"  min traps:      {len(min_traps)}")
    print(f"  shared:         {format_state(shared)}")
    print(f"  match:          {match}")
    if not match:
        mismatches.append(fixes)

print()
print(
    f"unique fixes checked: {len(ids_by_fixes)}; "
    f"mismatches: {len(mismatches)}"
)

if mismatches:
    raise AssertionError(
        f"LDOI did not match shared min-trap states for {len(mismatches)} "
        f"fix set(s)"
    )

print("All unique experiment fixes: LDOI == shared minimal trap states.")
