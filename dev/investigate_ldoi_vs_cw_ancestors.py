# %% [markdown]
# LDOI vs CubeWalkers — NAV diff + ancestor probes
#
# For a configurable seed on the breast-cancer gen108 model:
#
# 1. Compute node / entity NAV via LDOI and via cubewalkers
# 2. Show the differences and pickle the results (esp. CW NAV) for reload
# 3. Percolate out constants under the seed → free-node interaction graph
# 4. Pick an entity of interest (default Proliferation), show its free ancestors
# 5. For each ancestor A∈{0,1}: self-sustaining check + whether EOI is fixed
# 6. Part 3: refine free-node / entity NAV using ancestor states that are
#    both self-sustaining **and** have a self-loop (halve remaining
#    probability per independent forcer); compare |Δ| vs cubewalkers
#
# Cubewalkers requires CUDA + CuPy in WSL2 (see ``cw_helpers.require_cubewalkers``).
# ``SMOKE=1`` uses a tiny horizon so the notebook can be checked end-to-end.

# %%
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyboolnet.interaction_graphs import primes2igraph
from pyboolnet.prime_implicants import create_constants, percolate

from boolmore.evaluation.agreement import _related_node_names
from boolmore.inference.ldoi import _implicant_implied_by, logical_domain_of_influence
from boolmore.inference.prediction import _nav_from_ldoi
from boolmore.io.load import load_primes
from boolmore.io.paths import project_root

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from cw_helpers import (
    DEFAULT_THREADS_PER_BLOCK,
    bnet_rules_for_cubewalkers,
    require_cubewalkers,
    simulate_mean_nav,
)
from nav_helpers import Tee, discover_full_entities, entity_nav_dict

# %% [markdown]
# ## Config

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "breast_cancer"
MODEL_STEM = (
    CASE
    / "20261016_173102"
    / "BreastCancer_primes_parsimonious_20261016_173102_9916_gen108"
)
PKL = MODEL_STEM.with_suffix(".pkl")
BNET = MODEL_STEM.with_suffix(".bnet")
LOG_DIR = Path(__file__).resolve().parent / "logs"

# Intervention seed; empty → preexisting constants only.
SEED: dict[str, int] = {
    "Alpelisib":1,
    "ER":1,
    "ES":1,
    "HER2":0,
    "PBX1":1,
    "PIM":1,
    "PTEN_T_2":0,
    "Rb_T":1}

ENTITY_OF_INTEREST = "Proliferation"
# "manual" → ENTITY_OF_INTEREST; "top_diff" → entity with max |cw − ldoi|.
EOI_MODE = "manual"

N_WALKERS = 2048
N_TIME_STEPS: int | None = None  # None → 55*N + 6000
T_WINDOW: int | None = None  # None → max(1000, T // 4)
THREADS_PER_BLOCK: tuple[int, int] = DEFAULT_THREADS_PER_BLOCK

FORCE_RECOMPUTE = False

SMOKE = os.environ.get("SMOKE") == "1"
OUT_TAG = "_smoke" if SMOKE else ""
if SMOKE:
    N_WALKERS = 64
    N_TIME_STEPS = 200
    T_WINDOW = 50


def _seed_hash(seed: dict[str, int]) -> str:
    payload = json.dumps(sorted(seed.items()), separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


CACHE_PATH = (
    LOG_DIR
    / f"investigate_ldoi_vs_cw_{MODEL_STEM.name}_seed-{_seed_hash(SEED)}{OUT_TAG}.pkl"
)
PLOT_ANCESTORS = (
    LOG_DIR
    / f"investigate_ldoi_vs_cw_ancestors_{MODEL_STEM.name}_seed-{_seed_hash(SEED)}{OUT_TAG}.png"
)

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast; skipped when loading cache)

# %%
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = (
    LOG_DIR
    / f"investigate_ldoi_vs_cw_{MODEL_STEM.name}_seed-{_seed_hash(SEED)}{OUT_TAG}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"cache:    {CACHE_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model pkl:  {PKL}")
print(f"model bnet: {BNET}")
print(f"SEED={SEED}")
print(f"ENTITY_OF_INTEREST={ENTITY_OF_INTEREST!r}  EOI_MODE={EOI_MODE!r}")
print(f"FORCE_RECOMPUTE={FORCE_RECOMPUTE}  SMOKE={SMOKE}")
print()

_load_cache = CACHE_PATH.is_file() and not FORCE_RECOMPUTE
if not _load_cache:
    cw, cp = require_cubewalkers()
    print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"cubewalkers: {getattr(cw, '__file__', cw)}")
else:
    print(f"cache hit — skipping cubewalkers gate ({CACHE_PATH.name})")
print()

# %% [markdown]
# ## Load model

# %%
primes = load_primes(PKL)
node_names = sorted(primes)
_n_nodes = len(primes)
if N_TIME_STEPS is None:
    N_TIME_STEPS = 55 * _n_nodes + 6000
if T_WINDOW is None:
    T_WINDOW = max(1000, N_TIME_STEPS // 4)
assert T_WINDOW <= N_TIME_STEPS

merged_bases, entity_keys = discover_full_entities(node_names, [])

print(f"Model nodes : {_n_nodes}")
print(f"Entities    : {len(entity_keys)} ({len(merged_bases)} merged)")
print(f"cubewalkers T={N_TIME_STEPS}  T_WINDOW={T_WINDOW}  walkers={N_WALKERS}")
print()

# %% [markdown]
# ## Part 1 — LDOI vs CW NAV (load cache if present)

# %%
cache: dict | None = None
if _load_cache:
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    print(f"loaded cache from {CACHE_PATH}")
    fixed = dict(cache["fixed"])
    nav_ldoi_nodes = dict(cache["nav_ldoi_nodes"])
    nav_ldoi_entities = dict(cache["nav_ldoi_entities"])
    nav_cw_nodes = dict(cache["nav_cw_nodes"])
    nav_cw_entities = dict(cache["nav_cw_entities"])
    delta_nodes = dict(cache["delta_nodes"])
    delta_entities = dict(cache["delta_entities"])
    cw_timings = dict(cache.get("cw_timings", {}))
else:
    fixed = logical_domain_of_influence(primes, SEED)
    nav_ldoi_nodes = _nav_from_ldoi(primes, SEED)
    nav_ldoi_entities = entity_nav_dict(nav_ldoi_nodes, entity_keys, merged_bases)

    BNET_RULES = bnet_rules_for_cubewalkers(BNET)
    clamps = list(SEED.items()) or None
    nav_cw_nodes, cw_timings = simulate_mean_nav(
        BNET_RULES,
        n_time_steps=N_TIME_STEPS,
        t_window=T_WINDOW,
        n_walkers=N_WALKERS,
        clamps=clamps,
        threads_per_block=THREADS_PER_BLOCK,
        model_name="investigate_ldoi_vs_cw",
    )
    # Align to model node set (CW uses bnet varname order).
    nav_cw_nodes = {n: float(nav_cw_nodes[n]) for n in node_names}
    nav_cw_entities = entity_nav_dict(nav_cw_nodes, entity_keys, merged_bases)

    delta_nodes = {
        n: nav_cw_nodes[n] - nav_ldoi_nodes[n] for n in node_names
    }
    delta_entities = {
        e: nav_cw_entities[e] - nav_ldoi_entities[e] for e in entity_keys
    }

print(f"LDOI fixed nodes: {len(fixed)} / {_n_nodes}")
if fixed:
    print("  " + ", ".join(f"{n}={v}" for n, v in sorted(fixed.items())))
else:
    print("  (none)")
if cw_timings:
    print(
        f"CW timings: total={cw_timings.get('t_total', float('nan')):.3f}s  "
        f"setup={cw_timings.get('t_setup', float('nan')):.3f}s  "
        f"sim={cw_timings.get('t_simulate', float('nan')):.3f}s"
    )
print()

# %% [markdown]
# ### Diff tables

# %%
node_diff_df = (
    pd.DataFrame(
        {
            "node": node_names,
            "ldoi": [nav_ldoi_nodes[n] for n in node_names],
            "cw": [nav_cw_nodes[n] for n in node_names],
            "delta_cw_minus_ldoi": [delta_nodes[n] for n in node_names],
        }
    )
    .assign(abs_delta=lambda d: d["delta_cw_minus_ldoi"].abs())
    .sort_values("abs_delta", ascending=False)
    .reset_index(drop=True)
)

entity_diff_df = (
    pd.DataFrame(
        {
            "entity": entity_keys,
            "ldoi": [nav_ldoi_entities[e] for e in entity_keys],
            "cw": [nav_cw_entities[e] for e in entity_keys],
            "delta_cw_minus_ldoi": [delta_entities[e] for e in entity_keys],
        }
    )
    .assign(abs_delta=lambda d: d["delta_cw_minus_ldoi"].abs())
    .sort_values("abs_delta", ascending=False)
    .reset_index(drop=True)
)

print("Top node |Δ| (cw − ldoi):")
print(node_diff_df.head(20).to_string(index=False))
print()
print("Top entity |Δ| (cw − ldoi):")
print(entity_diff_df.head(20).to_string(index=False))
print()

# %% [markdown]
# ### Pickle cache

# %%
if not _load_cache:
    cache = {
        "model_path": str(PKL),
        "bnet_path": str(BNET),
        "seed": dict(SEED),
        "cw_params": {
            "n_time_steps": N_TIME_STEPS,
            "t_window": T_WINDOW,
            "n_walkers": N_WALKERS,
            "threads_per_block": THREADS_PER_BLOCK,
            "smoke": SMOKE,
        },
        "cw_timings": cw_timings,
        "fixed": dict(fixed),
        "nav_ldoi_nodes": dict(nav_ldoi_nodes),
        "nav_ldoi_entities": dict(nav_ldoi_entities),
        "nav_cw_nodes": dict(nav_cw_nodes),
        "nav_cw_entities": dict(nav_cw_entities),
        "delta_nodes": dict(delta_nodes),
        "delta_entities": dict(delta_entities),
        "entity_keys": list(entity_keys),
        "merged_bases": list(merged_bases),
    }
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)
    print(f"wrote cache → {CACHE_PATH}")
else:
    print(f"cache unchanged → {CACHE_PATH}")
print()

# %% [markdown]
# ## Part 2 — Percolated free network + ancestor probes
#
# Absorb `SEED` once via pyboolnet (`create_constants` → `percolate` with
# constants removed). The interaction graph and all LDOI probes then use only
# the reduced primes (no re-merging of the original seed).

# %%
seeded = create_constants(primes, SEED, copy=True)
percolated_primes = percolate(seeded, remove_constants=True, copy=True)
H = primes2igraph(percolated_primes)

print(
    f"Percolated free nodes: {len(percolated_primes)} "
    f"(removed {_n_nodes - len(percolated_primes)})"
)
print(f"IG: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
print()

# %% [markdown]
# ### Resolve entity of interest

# %%
if EOI_MODE == "top_diff":
    eoi_name = max(delta_entities, key=lambda e: abs(delta_entities[e]))
    print(
        f"EOI_MODE=top_diff → {eoi_name!r} "
        f"(|Δ|={abs(delta_entities[eoi_name]):.4f})"
    )
elif EOI_MODE == "manual":
    eoi_name = ENTITY_OF_INTEREST
    print(f"EOI_MODE=manual → {eoi_name!r}")
else:
    raise ValueError(f"Unknown EOI_MODE={EOI_MODE!r}")

eoi_siblings = _related_node_names(eoi_name, node_names)
if not eoi_siblings and eoi_name in node_names:
    eoi_siblings = [eoi_name]
if not eoi_siblings:
    raise ValueError(f"No nodes found for entity of interest {eoi_name!r}")

eoi_free = [n for n in eoi_siblings if n in percolated_primes]
eoi_already_fixed = [n for n in eoi_siblings if n not in percolated_primes]

print(f"EOI siblings: {eoi_siblings}")
print(f"  free (in percolated): {eoi_free}")
print(f"  already fixed:        {eoi_already_fixed}")
if eoi_name in nav_ldoi_entities:
    print(
        f"  entity NAV — ldoi={nav_ldoi_entities[eoi_name]:.4f}  "
        f"cw={nav_cw_entities[eoi_name]:.4f}  "
        f"Δ={delta_entities[eoi_name]:+.4f}"
    )
print()

_skip_probes = False
if not eoi_free:
    print(
        f"All EOI siblings of {eoi_name!r} were removed as constants under "
        f"SEED — EOI is already fixed. Skipping ancestor graph and probes."
    )
    _skip_probes = True

# %% [markdown]
# ### Ancestor cone (free nodes only)

# %%
ancestors: set[str] = set()
if not _skip_probes:
    for n in eoi_free:
        ancestors |= set(nx.ancestors(H, n))
    # Self-loop = autoregulation edge A→A in the interaction graph.
    ancestor_self_loop = {A: H.has_edge(A, A) for A in ancestors}
    cone_nodes = set(eoi_free) | ancestors
    cone = H.subgraph(cone_nodes).copy()

    print(f"Ancestors of EOI free nodes ({len(ancestors)}):")
    if ancestors:
        for A in sorted(ancestors):
            loop = "self-loop" if ancestor_self_loop[A] else "no self-loop"
            print(f"  {A}  ({loop})")
        n_loop = sum(ancestor_self_loop.values())
        print(f"  → {n_loop}/{len(ancestors)} ancestors have a self-loop")
    else:
        print("  (none)")
    print(
        f"Cone subgraph: {cone.number_of_nodes()} nodes, "
        f"{cone.number_of_edges()} edges"
    )

    pos = nx.spring_layout(cone, seed=0, k=1.2 / max(cone.number_of_nodes(), 1) ** 0.5)
    colors = []
    for n in cone.nodes:
        if n in eoi_free:
            colors.append("#f4a261")  # EOI
        elif ancestor_self_loop.get(n, H.has_edge(n, n)):
            colors.append("#90be6d")  # ancestor / node with self-loop
        else:
            colors.append("#cde4f5")
    fig_w = max(8, min(18, 2 + cone.number_of_nodes() * 0.35))
    fig_h = max(6, min(14, 2 + cone.number_of_nodes() * 0.25))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    nx.draw_networkx_nodes(cone, pos, ax=ax, node_size=400, node_color=colors)
    nx.draw_networkx_edges(
        cone,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=12,
        edge_color="#555555",
        connectionstyle="arc3,rad=0.05",
        alpha=0.7,
    )
    nx.draw_networkx_labels(cone, pos, ax=ax, font_size=7)
    ax.set_axis_off()
    ax.set_title(
        f"Free-node ancestor cone → {eoi_name}  "
        f"(orange = EOI, green = self-loop)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PLOT_ANCESTORS, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote ancestor plot → {PLOT_ANCESTORS}")
print()

# %% [markdown]
# ### Per-ancestor LDOI probes on percolated primes
#
# For each ancestor A and value v∈{0,1}:
#
# 1. **Self-sustaining** — LDOI(`{A: v}`) on `percolated_primes`; A's update
#    rule has an implicant covered by that closure (including A=v for
#    positive autoregulation).
# 2. **EOI fixed** — every remaining EOI sibling appears in that same closure.
#
# ``self_loop`` is whether the interaction graph has an A→A edge
# (autoregulation), independent of the self-sustaining check.

# %%
probe_rows: list[dict] = []
if not _skip_probes:
    for A in sorted(ancestors):
        has_loop = bool(ancestor_self_loop[A])
        for v in (0, 1):
            closure = logical_domain_of_influence(percolated_primes, {A: v})
            sustains = any(
                _implicant_implied_by(closure, imp)
                for imp in percolated_primes[A][v]
            )
            eoi_values = {n: closure[n] for n in eoi_free if n in closure}
            eoi_fixed = bool(eoi_free) and len(eoi_values) == len(eoi_free)
            probe_rows.append(
                {
                    "A": A,
                    "v": v,
                    "self_loop": has_loop,
                    "self_sustaining": sustains,
                    "eoi_fixed": eoi_fixed,
                    "eoi_values": eoi_values if eoi_values else None,
                    "n_forced": len(closure),
                }
            )

    probe_df = pd.DataFrame(probe_rows)
    print("All ancestor probes:")
    print(probe_df.to_string(index=False))
    print()

    sustaining = probe_df[probe_df["self_sustaining"]]
    print(f"Self-sustaining node states: {len(sustaining)}")
    if not sustaining.empty:
        print(
            sustaining[
                ["A", "v", "self_loop", "eoi_fixed", "eoi_values", "n_forced"]
            ].to_string(index=False)
        )
    print()

    fixing = sustaining[sustaining["eoi_fixed"]]
    print(
        f"Self-sustaining states that also fix EOI {eoi_name!r}: "
        f"{len(fixing)}"
    )
    if not fixing.empty:
        print(
            fixing[["A", "v", "self_loop", "eoi_values", "n_forced"]].to_string(
                index=False
            )
        )
    else:
        print("  (none)")
else:
    probe_df = pd.DataFrame()
    ancestor_self_loop = {}

# %% [markdown]
# ## Part 3 — Enhance LDOI NAV with self-loop + self-sustaining ancestors
#
# Forcers are stricter than Part 2's self-sustaining set: an ancestor state
# `(A, v)` counts only if **both**
#
# 1. `(A, v)` is self-sustaining, and
# 2. `A` has a self-loop in the free interaction graph.
#
# For each free EOI sibling (and other free nodes forced by those states):
#
# - One such state → **50%** of the remaining probability mass is fixed to the
#   LDOI value; the other 50% stays unknown.
# - If the **same** ancestor also has the opposite state qualifying and fixing
#   the node → that ancestor alone determines the full remaining mass
#   (50/50 over `A=0` and `A=1`).
# - A **different** qualifying ancestor that fixes the node is treated as
#   independent and claims half of whatever mass is still unknown
#   (50% → 25% → 12.5% …).
# - Leftover mass uses the usual unknown value `0.5`.
#
# Start from plain LDOI NAV, overwrite refined free nodes, rebuild entity NAV,
# then compare `|cw − ·|` for plain vs enhanced.

# %%
def _enhanced_nav_from_forcers(
    forcers_by_ancestor: dict[str, dict[int, int]],
) -> tuple[float, list[dict[str, Any]]]:
    """Return (nav, allocation steps).

    ``forcers_by_ancestor[A][v] = forced_value_of_node`` for self-sustaining
    states that fix the node. Ancestors that cover both values consume all
    remaining mass first (50/50 over A=0 and A=1 — mutually exclusive states,
    not a combined seed); one-sided ancestors then each take half of what is
    left. Unknown remainder contributes ``0.5``.
    """
    steps: list[dict[str, Any]] = []
    if not forcers_by_ancestor:
        return 0.5, steps

    both = sorted(
        (A, d) for A, d in forcers_by_ancestor.items() if 0 in d and 1 in d
    )
    one = sorted(
        (A, d) for A, d in forcers_by_ancestor.items() if len(d) == 1
    )

    remaining = 1.0
    expected = 0.0

    for A, d in both:
        if remaining <= 0:
            break
        # Full remaining mass split across A=0 and A=1.
        for v in (0, 1):
            mass = remaining * 0.5
            expected += mass * float(d[v])
            steps.append(
                {
                    "A": A,
                    "v": v,
                    "forced_value": int(d[v]),
                    "mass": mass,
                    "kind": "both_states",
                }
            )
        remaining = 0.0

    for A, d in one:
        if remaining <= 0:
            break
        v, val = next(iter(d.items()))
        mass = remaining * 0.5
        expected += mass * float(val)
        steps.append(
            {
                "A": A,
                "v": int(v),
                "forced_value": int(val),
                "mass": mass,
                "kind": "one_state",
            }
        )
        remaining *= 0.5

    if remaining > 0:
        expected += remaining * 0.5
        steps.append(
            {
                "A": None,
                "v": None,
                "forced_value": 0.5,
                "mass": remaining,
                "kind": "unknown",
            }
        )

    return float(expected), steps


# %%
nav_enhanced_nodes = dict(nav_ldoi_nodes)
nav_enhanced_entities = dict(nav_ldoi_entities)
enhance_steps: dict[str, list[dict[str, Any]]] = {}

if _skip_probes or probe_df.empty:
    print("Part 3 skipped (no ancestor probes).")
else:
    # Strict forcers: self-sustaining AND self-loop.
    forcer_rows = probe_df[
        probe_df["self_sustaining"] & probe_df["self_loop"]
    ]
    n_ss = int(probe_df["self_sustaining"].sum())
    n_strict = len(forcer_rows)
    print(
        f"Strict forcers: {n_strict} states "
        f"(self-sustaining∩self-loop; {n_ss} self-sustaining total)"
    )
    if forcer_rows.empty:
        print("No qualifying ancestor states — enhanced NAV == plain LDOI.")
    print()

    # Recompute closures for qualifying (A, v); map node → forcers.
    forcers_for_node: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    closures_ss: dict[tuple[str, int], dict[str, int]] = {}
    for _, row in forcer_rows.iterrows():
        A = str(row["A"])
        v = int(row["v"])
        closure = logical_domain_of_influence(percolated_primes, {A: v})
        closures_ss[(A, v)] = closure
        for n, val in closure.items():
            if n not in percolated_primes:
                continue  # skip if somehow outside free set
            forcers_for_node[n].append((A, v, int(val)))

    # Focus walkthrough on EOI siblings; still refine every free node that has
    # at least one strict forcer from the EOI-ancestor set.
    targets = list(eoi_free)
    extra = sorted(
        n for n in forcers_for_node if n not in targets and n in percolated_primes
    )
    print(
        f"Refining {len(targets)} EOI free siblings "
        f"(+ {len(extra)} other free nodes forced by those states)"
    )
    print()

    for N in targets + extra:
        forcers = forcers_for_node.get(N, [])
        if not forcers:
            if N in targets:
                print(
                    f"{N}: no self-loop+self-sustaining forcer → keep LDOI NAV "
                    f"{nav_ldoi_nodes[N]}"
                )
            continue

        by_anc: dict[str, dict[int, int]] = defaultdict(dict)
        for A, v, val in forcers:
            if v in by_anc[A] and by_anc[A][v] != val:
                raise ValueError(
                    f"{N}: same state {A}={v} claimed both "
                    f"{by_anc[A][v]} and {val}"
                )
            by_anc[A][v] = val

        nav_new, steps = _enhanced_nav_from_forcers(dict(by_anc))
        nav_enhanced_nodes[N] = nav_new
        enhance_steps[N] = steps

        if N in targets:
            print(f"{N}: LDOI {nav_ldoi_nodes[N]:.4f} → enhanced {nav_new:.4f}")
            for s in steps:
                if s["kind"] == "unknown":
                    print(
                        f"  unknown mass={s['mass']:.4f} → value 0.5"
                    )
                else:
                    print(
                        f"  {s['A']}={s['v']} → {N}={s['forced_value']}  "
                        f"mass={s['mass']:.4f}  ({s['kind']})"
                    )

    nav_enhanced_entities = entity_nav_dict(
        nav_enhanced_nodes, entity_keys, merged_bases
    )

    # --- Compare plain LDOI vs enhanced vs cubewalkers ---
    print()
    print("EOI sibling node comparison (cw − method):")
    node_cmp_rows = []
    for N in targets:
        d_plain = nav_cw_nodes[N] - nav_ldoi_nodes[N]
        d_enh = nav_cw_nodes[N] - nav_enhanced_nodes[N]
        node_cmp_rows.append(
            {
                "node": N,
                "ldoi": nav_ldoi_nodes[N],
                "enhanced": nav_enhanced_nodes[N],
                "cw": nav_cw_nodes[N],
                "abs_delta_ldoi": abs(d_plain),
                "abs_delta_enhanced": abs(d_enh),
                "improved": abs(d_enh) < abs(d_plain) - 1e-12,
            }
        )
    node_cmp_df = pd.DataFrame(node_cmp_rows)
    print(node_cmp_df.to_string(index=False))
    print(
        f"siblings improved: "
        f"{int(node_cmp_df['improved'].sum())}/{len(node_cmp_df)}"
    )

    print()
    print(f"Entity {eoi_name!r} comparison:")
    e_ldoi = nav_ldoi_entities[eoi_name]
    e_enh = nav_enhanced_entities[eoi_name]
    e_cw = nav_cw_entities[eoi_name]
    d_plain_e = abs(e_cw - e_ldoi)
    d_enh_e = abs(e_cw - e_enh)
    print(
        f"  ldoi={e_ldoi:.4f}  enhanced={e_enh:.4f}  cw={e_cw:.4f}"
    )
    print(
        f"  |Δ| ldoi={d_plain_e:.4f}  |Δ| enhanced={d_enh_e:.4f}  "
        f"improved={d_enh_e < d_plain_e - 1e-12}"
    )

    # All entities that changed
    changed_entities = [
        e
        for e in entity_keys
        if abs(nav_enhanced_entities[e] - nav_ldoi_entities[e]) > 1e-12
    ]
    if changed_entities:
        print()
        print(f"Entities changed by enhancement ({len(changed_entities)}):")
        ent_cmp = pd.DataFrame(
            [
                {
                    "entity": e,
                    "ldoi": nav_ldoi_entities[e],
                    "enhanced": nav_enhanced_entities[e],
                    "cw": nav_cw_entities[e],
                    "abs_delta_ldoi": abs(nav_cw_entities[e] - nav_ldoi_entities[e]),
                    "abs_delta_enhanced": abs(
                        nav_cw_entities[e] - nav_enhanced_entities[e]
                    ),
                    "improved": abs(nav_cw_entities[e] - nav_enhanced_entities[e])
                    < abs(nav_cw_entities[e] - nav_ldoi_entities[e]) - 1e-12,
                }
                for e in changed_entities
            ]
        ).sort_values("abs_delta_ldoi", ascending=False)
        print(ent_cmp.to_string(index=False))
        print(
            f"entities improved: "
            f"{int(ent_cmp['improved'].sum())}/{len(ent_cmp)}"
        )

    # All refined free nodes vs CW
    refined_nodes = sorted(enhance_steps)
    if refined_nodes:
        print()
        print(f"All refined free nodes ({len(refined_nodes)}) vs CW:")
        all_node_cmp = pd.DataFrame(
            [
                {
                    "node": n,
                    "ldoi": nav_ldoi_nodes[n],
                    "enhanced": nav_enhanced_nodes[n],
                    "cw": nav_cw_nodes[n],
                    "abs_delta_ldoi": abs(nav_cw_nodes[n] - nav_ldoi_nodes[n]),
                    "abs_delta_enhanced": abs(
                        nav_cw_nodes[n] - nav_enhanced_nodes[n]
                    ),
                    "improved": abs(nav_cw_nodes[n] - nav_enhanced_nodes[n])
                    < abs(nav_cw_nodes[n] - nav_ldoi_nodes[n]) - 1e-12,
                }
                for n in refined_nodes
            ]
        ).sort_values("abs_delta_ldoi", ascending=False)
        print(all_node_cmp.head(30).to_string(index=False))
        print(
            f"refined nodes improved: "
            f"{int(all_node_cmp['improved'].sum())}/{len(all_node_cmp)}  "
            f"mean |Δ| ldoi={all_node_cmp['abs_delta_ldoi'].mean():.4f}  "
            f"enhanced={all_node_cmp['abs_delta_enhanced'].mean():.4f}"
        )

print()
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# %%
# Restore stdout and close log.
sys.stdout = _stdout
_log_file.close()
print(f"log written: {LOG_PATH}")
print(f"cache: {CACHE_PATH}")
if not _skip_probes:
    print(f"ancestor plot: {PLOT_ANCESTORS}")
