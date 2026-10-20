# %% [markdown]
# LDOI vs CubeWalkers — target-entity ancestor probes
#
# For a configurable seed on the breast-cancer gen108 model:
#
# 1. Load or compute per-walker cubewalkers NAV (pickled for reuse / bootstrap)
# 2. Compute plain LDOI NAV (always fresh; cheap) and show diffs vs CW means
# 3. Percolate out constants under the seed → print original vs reduced bnet
# 4. Pick a target entity (default Proliferation), plot its free-ancestor cone
# 5. For each ancestor ∈{0,1}: self-sustaining check + whether the target is fixed
#
# Cubewalkers requires CUDA + CuPy in WSL2 (see ``cw_helpers.require_cubewalkers``).
# ``SMOKE=1`` uses a tiny horizon so the notebook can be checked end-to-end.
#
# Split from ``investigate_ldoi_vs_cw_ancestors.py`` (Parts 1–2). Overall
# ``ldoi`` / ``ldoi_enhanced`` comparison lives in
# ``investigate_ldoi_vs_cw_enhanced.py``.

# %%
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyboolnet.file_exchange import primes2bnet
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
    align_walker_nav,
    bnet_rules_for_cubewalkers,
    require_cubewalkers,
    simulate_walker_nav,
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
    "Alpelisib": 1,
    "ER": 1,
    "ES": 1,
    "HER2": 0,
    "PBX1": 1,
    "PIM": 1,
    "PTEN_T_2": 0,
    "Rb_T": 1,
}

TARGET_ENTITY = "Proliferation"
# "manual" → TARGET_ENTITY; "top_diff" → entity with max |cw − ldoi|.
TARGET_MODE = "manual"

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


SEED_HASH = _seed_hash(SEED)
PLOT_ANCESTORS = (
    LOG_DIR
    / f"investigate_ldoi_vs_cw_target_ancestors_{MODEL_STEM.name}"
    f"_seed-{SEED_HASH}{OUT_TAG}.png"
)

# %% [markdown]
# ## Load model

# %%
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = (
    LOG_DIR
    / f"investigate_ldoi_vs_cw_target_{MODEL_STEM.name}_seed-{SEED_HASH}{OUT_TAG}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

primes = load_primes(PKL)
node_names = sorted(primes)
_n_nodes = len(primes)
if N_TIME_STEPS is None:
    N_TIME_STEPS = 55 * _n_nodes + 6000
if T_WINDOW is None:
    T_WINDOW = max(1000, N_TIME_STEPS // 4)
assert T_WINDOW <= N_TIME_STEPS

# Shared reusable CW cache (per-walker NAV; not LDOI). Same path as
# investigate_ldoi_vs_cw_enhanced.py so either script can populate it.
CW_CACHE_PATH = (
    LOG_DIR
    / (
        f"cw_walker_nav_{MODEL_STEM.name}_seed-{SEED_HASH}"
        f"_w{N_WALKERS}_T{N_TIME_STEPS}_tw{T_WINDOW}{OUT_TAG}.pkl"
    )
)

merged_bases, entity_keys = discover_full_entities(node_names, [])

print(f"log file: {LOG_PATH}")
print(f"CW cache: {CW_CACHE_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model pkl:  {PKL}")
print(f"model bnet: {BNET}")
print(f"SEED={SEED}")
print(f"TARGET_ENTITY={TARGET_ENTITY!r}  TARGET_MODE={TARGET_MODE!r}")
print(f"FORCE_RECOMPUTE={FORCE_RECOMPUTE}  SMOKE={SMOKE}")
print()
print(f"Model nodes : {_n_nodes}")
print(f"Entities    : {len(entity_keys)} ({len(merged_bases)} merged)")
print(f"cubewalkers T={N_TIME_STEPS}  T_WINDOW={T_WINDOW}  walkers={N_WALKERS}")
print()

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast; skipped when loading cache)

# %%
_load_cache = CW_CACHE_PATH.is_file() and not FORCE_RECOMPUTE
if not _load_cache:
    cw, cp = require_cubewalkers()
    print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"cubewalkers: {getattr(cw, '__file__', cw)}")
else:
    print(f"cache hit — skipping cubewalkers gate ({CW_CACHE_PATH.name})")
print()

# %% [markdown]
# ## Part 1 — Cubewalkers walker NAV (cache) + fresh LDOI

# %%
cw_timings: dict = {}
if _load_cache:
    with CW_CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    walker_nav = np.asarray(cache["walker_nav"], dtype=np.float64)
    cached_names = list(cache["node_names"])
    if cached_names != node_names:
        raise ValueError(
            f"CW cache node_names mismatch:\n"
            f"  cache: {cached_names[:5]}...\n"
            f"  model: {node_names[:5]}..."
        )
    cw_timings = dict(cache.get("cw_timings", {}))
    print(f"loaded CW walker_nav from {CW_CACHE_PATH}")
    print(f"  shape={walker_nav.shape}")
else:
    BNET_RULES = bnet_rules_for_cubewalkers(BNET)
    clamps = list(SEED.items()) or None
    walker_nav, varnames, cw_timings = simulate_walker_nav(
        BNET_RULES,
        n_time_steps=N_TIME_STEPS,
        t_window=T_WINDOW,
        n_walkers=N_WALKERS,
        clamps=clamps,
        threads_per_block=THREADS_PER_BLOCK,
        model_name="investigate_ldoi_vs_cw_target",
    )
    walker_nav = align_walker_nav(walker_nav, varnames, node_names)
    cache = {
        "model_path": str(PKL),
        "bnet_path": str(BNET),
        "seed": dict(SEED),
        "node_names": list(node_names),
        "walker_nav": walker_nav,
        "cw_params": {
            "n_time_steps": N_TIME_STEPS,
            "t_window": T_WINDOW,
            "n_walkers": N_WALKERS,
            "threads_per_block": THREADS_PER_BLOCK,
            "smoke": SMOKE,
        },
        "cw_timings": cw_timings,
    }
    with CW_CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)
    print(f"wrote CW walker_nav → {CW_CACHE_PATH}")
    print(f"  shape={walker_nav.shape}")

nav_cw_nodes = {
    n: float(walker_nav[i].mean()) for i, n in enumerate(node_names)
}
nav_cw_entities = entity_nav_dict(nav_cw_nodes, entity_keys, merged_bases)

# LDOI is cheap — always recompute; never cache.
fixed = logical_domain_of_influence(primes, SEED)
nav_ldoi_nodes = _nav_from_ldoi(primes, SEED)
nav_ldoi_entities = entity_nav_dict(nav_ldoi_nodes, entity_keys, merged_bases)
delta_nodes = {n: nav_cw_nodes[n] - nav_ldoi_nodes[n] for n in node_names}
delta_entities = {
    e: nav_cw_entities[e] - nav_ldoi_entities[e] for e in entity_keys
}

print(f"LDOI fixed nodes: {len(fixed)} / {_n_nodes}")
if fixed:
    print(
        "  "
        + ", ".join(f"{node}={value}" for node, value in sorted(fixed.items()))
    )
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
print("Original bnet:")
print(primes2bnet(primes, minimize=True))
print()
print("Reduced (percolated) bnet:")
print(primes2bnet(percolated_primes, minimize=True))
print()

# %% [markdown]
# ### Resolve target entity

# %%
if TARGET_MODE == "top_diff":
    target_entity = max(delta_entities, key=lambda e: abs(delta_entities[e]))
    print(
        f"TARGET_MODE=top_diff → {target_entity!r} "
        f"(|Δ|={abs(delta_entities[target_entity]):.4f})"
    )
elif TARGET_MODE == "manual":
    target_entity = TARGET_ENTITY
    print(f"TARGET_MODE=manual → {target_entity!r}")
else:
    raise ValueError(f"Unknown TARGET_MODE={TARGET_MODE!r}")

target_nodes = _related_node_names(target_entity, node_names)
if not target_nodes and target_entity in node_names:
    target_nodes = [target_entity]
if not target_nodes:
    raise ValueError(f"No nodes found for target entity {target_entity!r}")

target_free = [n for n in target_nodes if n in percolated_primes]
target_already_fixed = [n for n in target_nodes if n not in percolated_primes]

print(f"Target nodes: {target_nodes}")
print(f"  free (in percolated): {target_free}")
print(f"  already fixed:        {target_already_fixed}")
if target_entity in nav_ldoi_entities:
    print(
        f"  entity NAV — ldoi={nav_ldoi_entities[target_entity]:.4f}  "
        f"cw={nav_cw_entities[target_entity]:.4f}  "
        f"Δ={delta_entities[target_entity]:+.4f}"
    )
print()

_skip_probes = False
if not target_free:
    print(
        f"All target nodes of {target_entity!r} were removed as constants under "
        f"SEED — target is already fixed. Skipping ancestor graph and probes."
    )
    _skip_probes = True

# %% [markdown]
# ### Ancestor cone (free nodes only)

# %%
ancestors: set[str] = set()
if not _skip_probes:
    for node in target_free:
        ancestors |= set(nx.ancestors(H, node))
    # Self-loop = autoregulation edge in the interaction graph.
    ancestor_self_loop = {
        ancestor: H.has_edge(ancestor, ancestor) for ancestor in ancestors
    }
    cone_nodes = set(target_free) | ancestors
    cone = H.subgraph(cone_nodes).copy()

    print(
        f"Cone subgraph: {cone.number_of_nodes()} nodes, "
        f"{cone.number_of_edges()} edges "
        f"({len(ancestors)} ancestors)"
    )

    pos = nx.spring_layout(cone, seed=0, k=1.2 / max(cone.number_of_nodes(), 1) ** 0.5)
    colors = []
    for node in cone.nodes:
        if node in target_free:
            colors.append("#f4a261")  # target
        elif ancestor_self_loop.get(node, H.has_edge(node, node)):
            colors.append("#90be6d")  # ancestor with self-loop
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
        f"Free-node ancestor cone → {target_entity}  "
        f"(orange = target, green = self-loop)",
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
# For each ``ancestor`` and ``value`` ∈{0,1}:
#
# 1. **Self-sustaining** — LDOI(`{ancestor: value}`) on `percolated_primes`;
#    the ancestor's update rule has an implicant covered by that closure
#    (including ancestor=value for positive autoregulation).
# 2. **Target fixed** — every remaining free target node appears in that
#    same closure.
#
# ``self_loop`` is whether the interaction graph has an ancestor→ancestor edge
# (autoregulation), independent of the self-sustaining check.
# ``n_ldoi_fixed`` is the size of that LDOI closure.

# %%
probe_rows: list[dict] = []
if not _skip_probes:
    for ancestor in sorted(ancestors):
        has_loop = bool(ancestor_self_loop[ancestor])
        for value in (0, 1):
            ldoi_closure = logical_domain_of_influence(
                percolated_primes, {ancestor: value}
            )
            sustains = any(
                _implicant_implied_by(ldoi_closure, imp)
                for imp in percolated_primes[ancestor][value]
            )
            target_values = {
                node: ldoi_closure[node]
                for node in target_free
                if node in ldoi_closure
            }
            target_fixed = bool(target_free) and len(target_values) == len(
                target_free
            )
            probe_rows.append(
                {
                    "ancestor": ancestor,
                    "value": value,
                    "self_loop": has_loop,
                    "self_sustaining": sustains,
                    "target_fixed": target_fixed,
                    "target_values": target_values if target_values else None,
                    "n_ldoi_fixed": len(ldoi_closure),
                }
            )

    probe_df = pd.DataFrame(probe_rows)
    print("All ancestor probes:")
    print(probe_df.to_string(index=False))
    print()
else:
    probe_df = pd.DataFrame()
    ancestor_self_loop = {}

print()
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# %%
# Restore stdout and close log.
sys.stdout = _stdout
_log_file.close()
print(f"log written: {LOG_PATH}")
print(f"CW cache: {CW_CACHE_PATH}")
if not _skip_probes:
    print(f"ancestor plot: {PLOT_ANCESTORS}")
