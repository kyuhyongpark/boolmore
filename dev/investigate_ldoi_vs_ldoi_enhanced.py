# %% [markdown]
# LDOI vs LDOI-enhanced vs CubeWalkers — overall node NAV
#
# For a configurable seed on the breast-cancer gen108 model:
#
# 1. Load or compute per-walker cubewalkers NAV (pickled for reuse / bootstrap)
# 2. Compute plain ``_nav_from_ldoi`` and ``_nav_from_ldoi_enhanced`` (always fresh)
# 3. Compare overall node average values vs CW means (no entity-of-interest focus)
#
# Cubewalkers requires CUDA + CuPy in WSL2 (see ``cw_helpers.require_cubewalkers``).
# ``SMOKE=1`` uses a tiny horizon so the notebook can be checked end-to-end.
#
# Split from ``investigate_ldoi_vs_cw_ancestors.py`` (Part 3 refactor). EOI
# ancestor probes live in ``investigate_ldoi_vs_cw_target.py``.

# %%
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from boolmore.inference.ldoi import logical_domain_of_influence
from boolmore.inference.prediction import _nav_from_ldoi, _nav_from_ldoi_enhanced
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

# %% [markdown]
# ## Load model

# %%
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = (
    LOG_DIR
    / f"investigate_ldoi_vs_cw_enhanced_{MODEL_STEM.name}"
    f"_seed-{SEED_HASH}{OUT_TAG}.log"
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
# investigate_ldoi_vs_cw_target.py so either script can populate it.
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
# ## Cubewalkers walker NAV (cache) + fresh LDOI / LDOI-enhanced

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
        model_name="investigate_ldoi_vs_cw_enhanced",
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

# LDOI variants are cheap enough — always recompute; never cache.
fixed = logical_domain_of_influence(primes, SEED)
nav_ldoi_nodes = _nav_from_ldoi(primes, SEED)
nav_enhanced_nodes = _nav_from_ldoi_enhanced(primes, SEED)
nav_ldoi_entities = entity_nav_dict(nav_ldoi_nodes, entity_keys, merged_bases)
nav_enhanced_entities = entity_nav_dict(
    nav_enhanced_nodes, entity_keys, merged_bases
)

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
n_changed = sum(
    1
    for n in node_names
    if abs(nav_enhanced_nodes[n] - nav_ldoi_nodes[n]) > 1e-12
)
print(f"Nodes changed by ldoi_enhanced: {n_changed} / {_n_nodes}")
print()

# %% [markdown]
# ## Node comparison: plain LDOI vs enhanced vs CW

# %%
node_cmp_df = (
    pd.DataFrame(
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
            }
            for n in node_names
        ]
    )
    .assign(
        improved=lambda d: d["abs_delta_enhanced"]
        < d["abs_delta_ldoi"] - 1e-12,
        worsened=lambda d: d["abs_delta_enhanced"]
        > d["abs_delta_ldoi"] + 1e-12,
        changed=lambda d: (d["enhanced"] - d["ldoi"]).abs() > 1e-12,
    )
    .sort_values("abs_delta_ldoi", ascending=False)
    .reset_index(drop=True)
)

print("Top node |Δ| vs CW (sorted by plain LDOI error):")
print(
    node_cmp_df[
        [
            "node",
            "ldoi",
            "enhanced",
            "cw",
            "abs_delta_ldoi",
            "abs_delta_enhanced",
            "improved",
        ]
    ]
    .head(30)
    .to_string(index=False)
)
print()

n_imp = int(node_cmp_df["improved"].sum())
n_wors = int(node_cmp_df["worsened"].sum())
n_same = _n_nodes - n_imp - n_wors
print(
    f"All nodes vs CW: improved={n_imp}  worsened={n_wors}  "
    f"unchanged_error={n_same}  (of {_n_nodes})"
)
print(
    f"mean |Δ| ldoi={node_cmp_df['abs_delta_ldoi'].mean():.4f}  "
    f"enhanced={node_cmp_df['abs_delta_enhanced'].mean():.4f}"
)
print(
    f"median |Δ| ldoi={node_cmp_df['abs_delta_ldoi'].median():.4f}  "
    f"enhanced={node_cmp_df['abs_delta_enhanced'].median():.4f}"
)
print()

changed_df = node_cmp_df[node_cmp_df["changed"]].sort_values(
    "abs_delta_ldoi", ascending=False
)
if not changed_df.empty:
    print(f"Nodes changed by enhancement ({len(changed_df)}):")
    print(
        changed_df[
            [
                "node",
                "ldoi",
                "enhanced",
                "cw",
                "abs_delta_ldoi",
                "abs_delta_enhanced",
                "improved",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )
    print(
        f"changed nodes: improved={int(changed_df['improved'].sum())}/"
        f"{len(changed_df)}  "
        f"mean |Δ| ldoi={changed_df['abs_delta_ldoi'].mean():.4f}  "
        f"enhanced={changed_df['abs_delta_enhanced'].mean():.4f}"
    )
else:
    print("No nodes changed by ldoi_enhanced (identical to plain LDOI).")
print()

# %% [markdown]
# ### Optional entity rollup (no EOI selection)

# %%
ent_cmp_df = (
    pd.DataFrame(
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
            }
            for e in entity_keys
        ]
    )
    .assign(
        improved=lambda d: d["abs_delta_enhanced"]
        < d["abs_delta_ldoi"] - 1e-12,
        changed=lambda d: (d["enhanced"] - d["ldoi"]).abs() > 1e-12,
    )
    .sort_values("abs_delta_ldoi", ascending=False)
    .reset_index(drop=True)
)

print("Top entity |Δ| vs CW (sorted by plain LDOI error):")
print(
    ent_cmp_df[
        [
            "entity",
            "ldoi",
            "enhanced",
            "cw",
            "abs_delta_ldoi",
            "abs_delta_enhanced",
            "improved",
        ]
    ]
    .head(20)
    .to_string(index=False)
)
print()
print(
    f"Entities: mean |Δ| ldoi={ent_cmp_df['abs_delta_ldoi'].mean():.4f}  "
    f"enhanced={ent_cmp_df['abs_delta_enhanced'].mean():.4f}  "
    f"improved={int(ent_cmp_df['improved'].sum())}/{len(ent_cmp_df)}"
)

ent_changed = ent_cmp_df[ent_cmp_df["changed"]]
if not ent_changed.empty:
    print()
    print(f"Entities changed by enhancement ({len(ent_changed)}):")
    print(
        ent_changed[
            [
                "entity",
                "ldoi",
                "enhanced",
                "cw",
                "abs_delta_ldoi",
                "abs_delta_enhanced",
                "improved",
            ]
        ].to_string(index=False)
    )

print()
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# %%
# Restore stdout and close log.
sys.stdout = _stdout
_log_file.close()
print(f"log written: {LOG_PATH}")
print(f"CW cache: {CW_CACHE_PATH}")
