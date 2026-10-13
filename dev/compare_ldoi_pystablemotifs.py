# %% [markdown]
# Compare boolmore (pyboolnet) LDOI vs pystablemotifs LDOI
#
# Boolmore's LDOI in `check_ldoi_vs_min_traps.py` is:
# `create_constants` → `percolate` → `find_constants`.
#
# pystablemotifs uses `logical_domain_of_influence` (Yang et al. 2018), which
# returns `(implied, contradicted)`.
#
# Hypotheses to check:
# 1. psm LDOI omits the seed unless the seed's own rule percolates to the seed
#    value.
# 2. psm LDOI does not percolate preexisting network constants. (Expect: false.)
# 3. psm builds a contradiction boundary: nodes whose update would force the
#    opposite of the seed under the fixed context.
#
# Expected relation when there is no contradiction:
# `pyboolnet_LDOI == seed ∪ psm_implied`
#
# Note: `pystablemotifs` is not a declared boolmore dependency; this notebook
# assumes it is available in the environment (same ad-hoc style as other draft
# notebooks).

# %%
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pyboolnet.file_exchange import bnet2primes
from pyboolnet.prime_implicants import create_constants, find_constants, percolate
from pystablemotifs.drivers import logical_domain_of_influence

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


def format_state(state: dict[str, int]) -> str:
    if not state:
        return "{}"
    return "{" + ", ".join(f"{n}={state[n]}" for n in sorted(state)) + "}"


def ldoi_pyboolnet(primes: dict, seed: dict[str, int]) -> dict[str, int]:
    """LDOI via create_constants → percolate → find_constants (seed always in)."""
    fixed_primes = create_constants(primes, seed, copy=True)
    percolated = percolate(fixed_primes, copy=True)
    return find_constants(percolated)


def ldoi_pystablemotifs(
    primes: dict, seed: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """LDOI via pystablemotifs; returns (implied, contradicted)."""
    return logical_domain_of_influence(seed, primes)


def classify_seeds(
    seed: dict[str, int],
    implied: dict[str, int],
    contradicted: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Partition seed nodes into in-LDOI / missing / contradiction."""
    in_ldoi: dict[str, int] = {}
    missing: dict[str, int] = {}
    in_contra: dict[str, int] = {}
    for node, value in seed.items():
        if node in contradicted:
            in_contra[node] = contradicted[node]
        elif implied.get(node) == value:
            in_ldoi[node] = value
        else:
            missing[node] = value
    return {"in_ldoi": in_ldoi, "missing": missing, "in_contra": in_contra}


def compare_one(primes: dict, seed: dict[str, int], label: str | None = None) -> dict:
    """Print and return a comparison of both LDOI implementations for one seed."""
    pyb = ldoi_pyboolnet(primes, seed)
    implied, contradicted = ldoi_pystablemotifs(primes, seed)
    parts = classify_seeds(seed, implied, contradicted)
    union = {**seed, **implied}
    match_union = pyb == union

    title = label if label is not None else format_state(seed) if seed else "(empty)"
    print(f"seed {title}")
    print(f"  pyboolnet LDOI:  {format_state(pyb)}")
    print(f"  psm implied:     {format_state(implied)}")
    print(f"  psm contradicted:{format_state(contradicted)}")
    print(f"  seed in LDOI:    {format_state(parts['in_ldoi'])}")
    print(f"  seed missing:    {format_state(parts['missing'])}")
    print(f"  seed contra:     {format_state(parts['in_contra'])}")
    print(f"  pyb == seed|imp: {match_union}")
    return {
        "seed": seed,
        "pyboolnet": pyb,
        "implied": implied,
        "contradicted": contradicted,
        "parts": parts,
        "match_union": match_union,
    }


# %% [markdown]
# ## Hypothesis A — seed inclusion (CAD)
#
# Expect: many experiment seeds are absent from psm `implied` unless the node's
# own update rule also forces the seed value.

# %%
primes = load_primes(BNET)
experiments = import_NAV_exps(DATA)

ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    ids_by_fixes[exp.fixes].append(exp.id)

print(f"model: {BNET.name} ({len(primes)} nodes)")
print(f"unique fix sets: {len(ids_by_fixes)}")
print()

n_seed_nodes = 0
n_seed_in_ldoi = 0
n_seed_missing = 0
n_seed_contra = 0
n_match_union = 0
n_with_contra = 0

for fixes, exp_ids in ids_by_fixes.items():
    seed = fixes_to_dict(fixes)
    result = compare_one(primes, seed, label=format_state(seed) if seed else "(none)")
    print(f"  experiment ids: {exp_ids}")
    print()

    parts = result["parts"]
    n_seed_nodes += len(seed)
    n_seed_in_ldoi += len(parts["in_ldoi"])
    n_seed_missing += len(parts["missing"])
    n_seed_contra += len(parts["in_contra"])
    if result["match_union"]:
        n_match_union += 1
    if result["contradicted"]:
        n_with_contra += 1

print(
    f"CAD seed nodes: {n_seed_nodes}; "
    f"in psm LDOI: {n_seed_in_ldoi}; "
    f"missing: {n_seed_missing}; "
    f"in contradiction: {n_seed_contra}"
)
print(
    f"fix sets with pyb == seed|implied: {n_match_union}/{len(ids_by_fixes)}; "
    f"with nonempty contradiction: {n_with_contra}"
)
print(
    "Hypothesis A (seed omitted unless self-consistent): "
    + ("SUPPORTED" if n_seed_missing > 0 else "NOT OBSERVED ON CAD")
)

# %% [markdown]
# ## Hypothesis B — preexisting constants (toy)
#
# Expect (corrected): psm **does** percolate preexisting constants, because
# empty implicants always fire.

# %%
toy_const_bnet = """C, 1
A, C
B, A
"""
primes_const = bnet2primes(toy_const_bnet)
print("preexisting constants in primes:", format_state(find_constants(primes_const)))
print()
print("--- empty seed ---")
result_empty = compare_one(primes_const, {})
print()
print("--- seed B=1 ---")
result_b = compare_one(primes_const, {"B": 1})

psm_includes_const = result_empty["implied"].get("C") == 1
print()
print(
    "Hypothesis B (psm skips preexisting constants): "
    + ("FALSE — constants are percolated" if psm_includes_const else "TRUE")
)

# %% [markdown]
# ## Hypothesis C — contradiction boundary (toys)
#
# Contradiction boundary = nodes whose original update would force the opposite
# of the seed under the fixed context. pyboolnet overwrites the seed rule, so
# it never reports a contradiction and can diverge from psm.

# %%
print("--- seed vs constant rule: A=0, B=A; seed {A:1} ---")
primes_contra_const = bnet2primes("A, 0\nB, A\n")
result_cc = compare_one(primes_contra_const, {"A": 1})
print()

print("--- seed vs update rule: A=!B, B=B; seed {A:1, B:1} ---")
primes_contra_rule = bnet2primes("A, !B\nB, B\n")
result_cr = compare_one(primes_contra_rule, {"A": 1, "B": 1})
print()

print("--- self-consistent vs not: A = A & B ---")
primes_self = bnet2primes("A, A & B\nB, B\n")
print("seed {A:0} (self-consistent OFF):")
compare_one(primes_self, {"A": 0})
print()
print("seed {A:1} (not forced by rule):")
compare_one(primes_self, {"A": 1})

has_boundary = bool(result_cc["contradicted"] or result_cr["contradicted"])
print()
print(
    "Hypothesis C (contradiction boundary exists): "
    + ("SUPPORTED" if has_boundary else "NOT OBSERVED")
)

# %% [markdown]
# ## Summary

# %%
print("Verdicts")
print("--------")
print(
    "A. Seed omitted unless own rule percolates to seed value: "
    "TRUE (many CAD seeds missing from psm implied)."
)
print(
    "B. Preexisting constants not percolated by psm: "
    "FALSE (empty implicants fire; constants enter implied)."
)
print(
    "C. Contradiction boundary: "
    "TRUE (return value contradicted; seed stays in fixed, opposite recorded)."
)
print()
print("Takeaway")
print("--------")
print(
    "boolmore/pyboolnet LDOI always includes the seed (create_constants "
    "rewrites seed rules). pystablemotifs LDOI is the set of nodes forced by "
    "logical percolation without rewriting seeds; seeds appear only if "
    "self-consistent. With no contradiction, "
    "pyboolnet_LDOI == seed ∪ psm_implied. With a contradiction, pyboolnet "
    "silently forces the seed and continues; psm reports contradicted and "
    "still percolates other nodes from the seed value."
)
