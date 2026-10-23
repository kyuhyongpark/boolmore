# %% [markdown]
# LDOI vs LDOI-enhanced (cutoffs 1–3) vs CubeWalkers — overall node NAV
#
# For a configurable seed on the breast-cancer gen108 model:
#
# 1. Load or compute per-walker cubewalkers NAV (pickled for reuse / bootstrap)
# 2. Compute plain ``_nav_from_ldoi`` and ``_nav_from_ldoi_enhanced`` for
#    cutoffs ``1..3`` (SC subsets built once at max cutoff, shared across calls).
#    With ``USE_JOINT_SELF_LOOPS``, also run enhanced with ``joint_self_loops=True``
#    (joint probes over already sustaining SC states) side-by-side.
# 3. Compare overall node average values vs CW means and plot |Δ| distributions
#    plus wall-clock compute time per method (LDOI / enhanced / CW)
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
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.inference.ldoi import logical_domain_of_influence
from boolmore.inference.prediction import (
    _nav_from_ldoi,
    _nav_from_ldoi_enhanced,
    _sc_subsets_by_size,
)
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
# Shared CW cache + sibling-script artifacts stay under logs/; this script's
# run outputs go in logs/investigate_ldoi_vs_ldoi_enhanced/{model}/seed-{hash}/.
LOGS_ROOT = Path(__file__).resolve().parent / "logs"
SCRIPT_NAME = Path(__file__).stem  # investigate_ldoi_vs_ldoi_enhanced

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

# ldoi_enhanced SC-subset size cutoffs to compare against plain LDOI / CW.
ENHANCED_CUTOFFS: tuple[int, ...] = (1, 2, 3)
# Also run enhanced with joint_self_loops=True (side-by-side columns).
USE_JOINT_SELF_LOOPS = True

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

METHOD_LDOI = "ldoi"
METHOD_ENH = {c: f"enhanced(c={c})" for c in ENHANCED_CUTOFFS}
METHOD_ENHJ = {c: f"enhanced+j(c={c})" for c in ENHANCED_CUTOFFS}
METHOD_ORDER = [METHOD_LDOI] + [METHOD_ENH[c] for c in ENHANCED_CUTOFFS]
if USE_JOINT_SELF_LOOPS:
    METHOD_ORDER = METHOD_ORDER + [METHOD_ENHJ[c] for c in ENHANCED_CUTOFFS]
# Shorter column keys for tables / CSV
ABS_COL_SHORT = {
    METHOD_LDOI: "abs_delta_ldoi",
    **{METHOD_ENH[c]: f"abs_delta_enh_c{c}" for c in ENHANCED_CUTOFFS},
}
if USE_JOINT_SELF_LOOPS:
    ABS_COL_SHORT.update(
        {METHOD_ENHJ[c]: f"abs_delta_enhj_c{c}" for c in ENHANCED_CUTOFFS}
    )

METHOD_CW = "cubewalkers"

# Distinct, readable colors (avoid default purple-bias cluster).
METHOD_COLORS = {
    METHOD_LDOI: "#2c3e50",
    METHOD_ENH[1]: "#1f77b4",
    METHOD_ENH[2]: "#2ca02c",
    METHOD_ENH[3]: "#d62728",
    METHOD_CW: "#7f7f7f",
}
if USE_JOINT_SELF_LOOPS:
    METHOD_COLORS.update(
        {
            METHOD_ENHJ[1]: "#17becf",
            METHOD_ENHJ[2]: "#bcbd22",
            METHOD_ENHJ[3]: "#e377c2",
        }
    )

# Timing bars: LDOI methods + CW total (when available).
TIMING_METHOD_ORDER = METHOD_ORDER + [METHOD_CW]

# %% [markdown]
# ## Load model

# %%
RUN_DIR = (
    LOGS_ROOT
    / SCRIPT_NAME
    / MODEL_STEM.name
    / f"seed-{SEED_HASH}{OUT_TAG}"
)
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = (
    RUN_DIR
    / f"{SCRIPT_NAME}_{MODEL_STEM.name}_seed-{SEED_HASH}{OUT_TAG}.log"
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
    LOGS_ROOT
    / (
        f"cw_walker_nav_{MODEL_STEM.name}_seed-{SEED_HASH}"
        f"_w{N_WALKERS}_T{N_TIME_STEPS}_tw{T_WINDOW}{OUT_TAG}.pkl"
    )
)

merged_bases, entity_keys = discover_full_entities(node_names, [])

print(f"run dir: {RUN_DIR}")
print(f"log file: {LOG_PATH}")
print(f"CW cache: {CW_CACHE_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model pkl:  {PKL}")
print(f"model bnet: {BNET}")
print(f"SEED={SEED}")
print(f"ENHANCED_CUTOFFS={ENHANCED_CUTOFFS}")
print(f"USE_JOINT_SELF_LOOPS={USE_JOINT_SELF_LOOPS}")
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
        model_name=SCRIPT_NAME,
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

_max_cutoff = max(ENHANCED_CUTOFFS)
sc_subsets = _sc_subsets_by_size(primes, _max_cutoff)
for k in range(1, _max_cutoff + 1):
    print(f"SC subsets size {k}: {len(sc_subsets.get(k, []))}")

fixed = logical_domain_of_influence(primes, SEED)

print("Computing ldoi…")
_t0 = time.perf_counter()
nav_ldoi_nodes = _nav_from_ldoi(primes, SEED)
t_ldoi = time.perf_counter() - _t0
nav_ldoi_entities = entity_nav_dict(nav_ldoi_nodes, entity_keys, merged_bases)
print(f"  ldoi done in {t_ldoi:.3f}s")

nav_enhanced_nodes: dict[int, dict[str, float]] = {}
nav_enhanced_entities: dict[int, dict[str, float]] = {}
t_enhanced: dict[int, float] = {}
for c in ENHANCED_CUTOFFS:
    print(f"Computing ldoi_enhanced (cutoff={c})…")
    _t0 = time.perf_counter()
    nav_enhanced_nodes[c] = _nav_from_ldoi_enhanced(
        primes, SEED, cutoff=c, sc_subsets=sc_subsets
    )
    t_enhanced[c] = time.perf_counter() - _t0
    nav_enhanced_entities[c] = entity_nav_dict(
        nav_enhanced_nodes[c], entity_keys, merged_bases
    )
    print(f"  enhanced(c={c}) done in {t_enhanced[c]:.3f}s")

nav_enhj_nodes: dict[int, dict[str, float]] = {}
nav_enhj_entities: dict[int, dict[str, float]] = {}
t_enhj: dict[int, float] = {}
joint_stats_by_cutoff: dict[int, dict] = {}
if USE_JOINT_SELF_LOOPS:
    for c in ENHANCED_CUTOFFS:
        print(f"Computing ldoi_enhanced+j (cutoff={c}, joint_self_loops)…")
        _stats: dict = {}
        _t0 = time.perf_counter()
        nav_enhj_nodes[c] = _nav_from_ldoi_enhanced(
            primes,
            SEED,
            cutoff=c,
            sc_subsets=sc_subsets,
            joint_self_loops=True,
            joint_stats=_stats,
        )
        t_enhj[c] = time.perf_counter() - _t0
        joint_stats_by_cutoff[c] = dict(_stats)
        nav_enhj_entities[c] = entity_nav_dict(
            nav_enhj_nodes[c], entity_keys, merged_bases
        )
        print(
            f"  enhanced+j(c={c}) done in {t_enhj[c]:.3f}s  "
            f"sustaining_size1={_stats.get('n_sustaining_size1', 0)}  "
            f"sustaining_size2={_stats.get('n_sustaining_size2', 0)}  "
            f"joint_probed={_stats.get('n_joint_probed', 0)}  "
            f"joint_kept={_stats.get('n_joint_kept', 0)}"
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

print("LDOI / enhanced compute times (excludes SC-subset prep):")
print(f"  ldoi              {t_ldoi:.3f}s")
for c in ENHANCED_CUTOFFS:
    print(f"  enhanced(c={c})     {t_enhanced[c]:.3f}s")
if USE_JOINT_SELF_LOOPS:
    for c in ENHANCED_CUTOFFS:
        print(f"  enhanced+j(c={c})   {t_enhj[c]:.3f}s")

for c in ENHANCED_CUTOFFS:
    n_changed = sum(
        1
        for n in node_names
        if abs(nav_enhanced_nodes[c][n] - nav_ldoi_nodes[n]) > 1e-12
    )
    print(f"Nodes changed by ldoi_enhanced(c={c}): {n_changed} / {_n_nodes}")
if USE_JOINT_SELF_LOOPS:
    for c in ENHANCED_CUTOFFS:
        n_changed = sum(
            1
            for n in node_names
            if abs(nav_enhj_nodes[c][n] - nav_ldoi_nodes[n]) > 1e-12
        )
        n_vs_enh = sum(
            1
            for n in node_names
            if abs(nav_enhj_nodes[c][n] - nav_enhanced_nodes[c][n]) > 1e-12
        )
        print(
            f"Nodes changed by ldoi_enhanced+j(c={c}): {n_changed} / {_n_nodes}  "
            f"(vs enhanced: {n_vs_enh})"
        )
print()

# %% [markdown]
# ## Node comparison: plain LDOI vs enhanced cutoffs vs CW


# %%
def _build_cmp_df(
    keys: list[str],
    key_col: str,
    nav_ldoi: dict[str, float],
    nav_enh: dict[int, dict[str, float]],
    nav_cw: dict[str, float],
    nav_enhj: dict[int, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for k in keys:
        row: dict = {
            key_col: k,
            "ldoi": nav_ldoi[k],
            "cw": nav_cw[k],
            ABS_COL_SHORT[METHOD_LDOI]: abs(nav_cw[k] - nav_ldoi[k]),
        }
        for c in ENHANCED_CUTOFFS:
            row[f"enh_c{c}"] = nav_enh[c][k]
            row[ABS_COL_SHORT[METHOD_ENH[c]]] = abs(nav_cw[k] - nav_enh[c][k])
        if nav_enhj is not None:
            for c in ENHANCED_CUTOFFS:
                row[f"enhj_c{c}"] = nav_enhj[c][k]
                row[ABS_COL_SHORT[METHOD_ENHJ[c]]] = abs(
                    nav_cw[k] - nav_enhj[c][k]
                )
        rows.append(row)
    df = pd.DataFrame(rows)
    # vs plain LDOI: did any enhanced (+joint) cutoff move NAV?
    enh_cols = [f"enh_c{c}" for c in ENHANCED_CUTOFFS]
    if nav_enhj is not None:
        enh_cols = enh_cols + [f"enhj_c{c}" for c in ENHANCED_CUTOFFS]
    df["changed"] = (df[enh_cols].sub(df["ldoi"], axis=0).abs() > 1e-12).any(
        axis=1
    )
    # Best (lowest |Δ|) among LDOI + all enhanced cutoffs.
    # Ties are shared: every method at the min is listed (idxmin would only
    # credit the leftmost column and under-count later cutoffs).
    abs_cols = [ABS_COL_SHORT[m] for m in METHOD_ORDER]
    short_to_method = {ABS_COL_SHORT[m]: m for m in METHOD_ORDER}
    best_vals = df[abs_cols].min(axis=1)
    df["abs_delta_best"] = best_vals
    tied = df[abs_cols].sub(best_vals, axis=0).abs() <= 1e-12
    df["best_method"] = tied.apply(
        lambda row: "|".join(short_to_method[c] for c in abs_cols if row[c]),
        axis=1,
    )
    return df.sort_values(
        ABS_COL_SHORT[METHOD_LDOI], ascending=False
    ).reset_index(drop=True)


node_cmp_df = _build_cmp_df(
    node_names,
    "node",
    nav_ldoi_nodes,
    nav_enhanced_nodes,
    nav_cw_nodes,
    nav_enhj_nodes if USE_JOINT_SELF_LOOPS else None,
)

_show_cols = (
    ["node", "ldoi"]
    + [f"enh_c{c}" for c in ENHANCED_CUTOFFS]
    + ([f"enhj_c{c}" for c in ENHANCED_CUTOFFS] if USE_JOINT_SELF_LOOPS else [])
    + ["cw", ABS_COL_SHORT[METHOD_LDOI]]
    + [ABS_COL_SHORT[METHOD_ENH[c]] for c in ENHANCED_CUTOFFS]
    + (
        [ABS_COL_SHORT[METHOD_ENHJ[c]] for c in ENHANCED_CUTOFFS]
        if USE_JOINT_SELF_LOOPS
        else []
    )
)
print("Top node |Δ| vs CW (sorted by plain LDOI error):")
print(node_cmp_df[_show_cols].head(30).to_string(index=False))
print()

print("All nodes vs CW — |Δ| summary by method:")
_summary_rows = []
for m in METHOD_ORDER:
    col = ABS_COL_SHORT[m]
    _summary_rows.append(
        {
            "method": m,
            "mean": node_cmp_df[col].mean(),
            "median": node_cmp_df[col].median(),
            "p75": node_cmp_df[col].quantile(0.75),
            "p90": node_cmp_df[col].quantile(0.90),
            "max": node_cmp_df[col].max(),
        }
    )
node_summary_df = pd.DataFrame(_summary_rows)
print(node_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

# Improvement vs plain LDOI for each cutoff
for c in ENHANCED_CUTOFFS:
    col = ABS_COL_SHORT[METHOD_ENH[c]]
    base = ABS_COL_SHORT[METHOD_LDOI]
    n_imp = int((node_cmp_df[col] < node_cmp_df[base] - 1e-12).sum())
    n_wors = int((node_cmp_df[col] > node_cmp_df[base] + 1e-12).sum())
    n_same = _n_nodes - n_imp - n_wors
    print(
        f"enhanced(c={c}) vs ldoi: improved={n_imp}  worsened={n_wors}  "
        f"unchanged_error={n_same}  (of {_n_nodes})"
    )
if USE_JOINT_SELF_LOOPS:
    for c in ENHANCED_CUTOFFS:
        col = ABS_COL_SHORT[METHOD_ENHJ[c]]
        base = ABS_COL_SHORT[METHOD_LDOI]
        n_imp = int((node_cmp_df[col] < node_cmp_df[base] - 1e-12).sum())
        n_wors = int((node_cmp_df[col] > node_cmp_df[base] + 1e-12).sum())
        n_same = _n_nodes - n_imp - n_wors
        print(
            f"enhanced+j(c={c}) vs ldoi: improved={n_imp}  worsened={n_wors}  "
            f"unchanged_error={n_same}  (of {_n_nodes})"
        )
print()

print(
    "Nodes where each method has the lowest |Δ| vs CW "
    "(ties counted for every tied method):"
)
_best_rows = []
for m in METHOD_ORDER:
    among = node_cmp_df["best_method"].str.split("|").apply(lambda ms: m in ms)
    n_among = int(among.sum())
    n_sole = int(
        (
            among & (node_cmp_df["best_method"] == m)
        ).sum()
    )
    _best_rows.append(
        {"method": m, "sole": n_sole, "tied": n_among - n_sole, "total": n_among}
    )
best_counts_df = pd.DataFrame(_best_rows)
print(best_counts_df.to_string(index=False))
print(
    f"  (n={_n_nodes} nodes; total can exceed n when methods tie)"
)
print()

changed_df = node_cmp_df[node_cmp_df["changed"]].sort_values(
    ABS_COL_SHORT[METHOD_LDOI], ascending=False
)
if not changed_df.empty:
    print(f"Nodes changed by any enhancement ({len(changed_df)}):")
    print(changed_df[_show_cols].head(40).to_string(index=False))
    print()
    print("Changed nodes — |Δ| summary:")
    for m in METHOD_ORDER:
        col = ABS_COL_SHORT[m]
        print(
            f"  {m:16s}  mean={changed_df[col].mean():.4f}  "
            f"median={changed_df[col].median():.4f}"
        )
else:
    print("No nodes changed by any ldoi_enhanced cutoff (identical to plain LDOI).")
print()

# %% [markdown]
# ### Optional entity rollup (no EOI selection)

# %%
ent_cmp_df = _build_cmp_df(
    list(entity_keys),
    "entity",
    nav_ldoi_entities,
    nav_enhanced_entities,
    nav_cw_entities,
    nav_enhj_entities if USE_JOINT_SELF_LOOPS else None,
)

_ent_show = (
    ["entity", "ldoi"]
    + [f"enh_c{c}" for c in ENHANCED_CUTOFFS]
    + ([f"enhj_c{c}" for c in ENHANCED_CUTOFFS] if USE_JOINT_SELF_LOOPS else [])
    + ["cw", ABS_COL_SHORT[METHOD_LDOI]]
    + [ABS_COL_SHORT[METHOD_ENH[c]] for c in ENHANCED_CUTOFFS]
    + (
        [ABS_COL_SHORT[METHOD_ENHJ[c]] for c in ENHANCED_CUTOFFS]
        if USE_JOINT_SELF_LOOPS
        else []
    )
)
print("Top entity |Δ| vs CW (sorted by plain LDOI error):")
print(ent_cmp_df[_ent_show].head(20).to_string(index=False))
print()

print("Entities vs CW — |Δ| summary by method:")
_ent_summary_rows = []
for m in METHOD_ORDER:
    col = ABS_COL_SHORT[m]
    _ent_summary_rows.append(
        {
            "method": m,
            "mean": ent_cmp_df[col].mean(),
            "median": ent_cmp_df[col].median(),
            "p75": ent_cmp_df[col].quantile(0.75),
            "p90": ent_cmp_df[col].quantile(0.90),
            "max": ent_cmp_df[col].max(),
        }
    )
ent_summary_df = pd.DataFrame(_ent_summary_rows)
print(ent_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

for c in ENHANCED_CUTOFFS:
    col = ABS_COL_SHORT[METHOD_ENH[c]]
    base = ABS_COL_SHORT[METHOD_LDOI]
    n_imp = int((ent_cmp_df[col] < ent_cmp_df[base] - 1e-12).sum())
    n_wors = int((ent_cmp_df[col] > ent_cmp_df[base] + 1e-12).sum())
    n_same = len(ent_cmp_df) - n_imp - n_wors
    print(
        f"Entities enhanced(c={c}) vs ldoi: improved={n_imp}  "
        f"worsened={n_wors}  unchanged_error={n_same}  "
        f"(of {len(ent_cmp_df)})"
    )
if USE_JOINT_SELF_LOOPS:
    for c in ENHANCED_CUTOFFS:
        col = ABS_COL_SHORT[METHOD_ENHJ[c]]
        base = ABS_COL_SHORT[METHOD_LDOI]
        n_imp = int((ent_cmp_df[col] < ent_cmp_df[base] - 1e-12).sum())
        n_wors = int((ent_cmp_df[col] > ent_cmp_df[base] + 1e-12).sum())
        n_same = len(ent_cmp_df) - n_imp - n_wors
        print(
            f"Entities enhanced+j(c={c}) vs ldoi: improved={n_imp}  "
            f"worsened={n_wors}  unchanged_error={n_same}  "
            f"(of {len(ent_cmp_df)})"
        )

ent_changed = ent_cmp_df[ent_cmp_df["changed"]]
if not ent_changed.empty:
    print()
    print(f"Entities changed by any enhancement ({len(ent_changed)}):")
    print(ent_changed[_ent_show].to_string(index=False))

# %% [markdown]
# ## Plots: |Δ| vs CW distributions + method compute times


# %%
def _method_timings_s() -> dict[str, float]:
    """Wall times (s) for LDOI / enhanced / CW total. Missing CW → omit."""
    out: dict[str, float] = {METHOD_LDOI: float(t_ldoi)}
    for c in ENHANCED_CUTOFFS:
        out[METHOD_ENH[c]] = float(t_enhanced[c])
    if USE_JOINT_SELF_LOOPS:
        for c in ENHANCED_CUTOFFS:
            out[METHOD_ENHJ[c]] = float(t_enhj[c])
    t_cw = cw_timings.get("t_total")
    if t_cw is not None and np.isfinite(t_cw):
        out[METHOD_CW] = float(t_cw)
    return out


def _plot_method_timings(out_dir: Path) -> Path | None:
    """Bar chart of wall-clock compute time per method."""
    timings = _method_timings_s()
    methods = [m for m in TIMING_METHOD_ORDER if m in timings]
    if not methods:
        return None
    vals = [timings[m] for m in methods]

    fig_w = max(8.0, 1.1 * len(methods) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        vals,
        color=[METHOD_COLORS[m] for m in methods],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Compute time by method")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    for bar, val in zip(bars, vals):
        label = f"{val:.2f}s" if val < 100 else f"{val:.1f}s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    note = "LDOI / enhanced exclude SC-subset prep"
    if METHOD_CW in timings:
        note += "; cubewalkers = t_total"
    ax.text(
        0.01,
        0.98,
        note,
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )
    fig.tight_layout()
    path = out_dir / "method_compute_times.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _abs_series_by_method(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        m: df[ABS_COL_SHORT[m]].to_numpy(dtype=np.float64) for m in METHOD_ORDER
    }


def _plot_abs_delta_distributions(
    df: pd.DataFrame,
    *,
    scope: str,
    out_dir: Path,
) -> list[Path]:
    """Boxplot + overlaid hist + ECDF + mean/median bars for |Δ| vs CW."""
    series = _abs_series_by_method(df)
    written: list[Path] = []
    all_vals = np.concatenate(list(series.values()))
    xmax = float(np.nanmax(all_vals)) if len(all_vals) else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    bins = np.linspace(0.0, xmax, 31)

    # --- 1) Boxplot + mean markers ---
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data = [series[m] for m in METHOD_ORDER]
    bp = ax.boxplot(
        data,
        tick_labels=METHOD_ORDER,
        showfliers=True,
        whis=(5, 95),
        widths=0.55,
        patch_artist=True,
    )
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.45)
        patch.set_edgecolor(METHOD_COLORS[m])
    for m, vals, xpos in zip(METHOD_ORDER, data, range(1, len(METHOD_ORDER) + 1)):
        ax.scatter(
            xpos,
            np.mean(vals),
            color=METHOD_COLORS[m],
            marker="D",
            s=36,
            zorder=4,
            edgecolors="white",
            linewidths=0.6,
            label="mean" if xpos == 1 else None,
        )
    ax.set_ylabel("|Δ| NAV vs CubeWalkers")
    ax.set_title(f"{scope}: |Δ| distribution by method")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    if bp["fliers"]:
        # single legend entry for mean diamond
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:1], labels[:1], loc="upper right", frameon=False)
    fig.tight_layout()
    path = out_dir / f"{scope}_abs_delta_box.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # --- 2) Overlaid histograms (density) ---
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for m in METHOD_ORDER:
        ax.hist(
            series[m],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=METHOD_COLORS[m],
            label=m,
            alpha=0.95,
        )
        ax.hist(
            series[m],
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=METHOD_COLORS[m],
            alpha=0.12,
        )
    ax.set_xlabel("|Δ| NAV vs CubeWalkers")
    ax.set_ylabel("Density")
    ax.set_title(f"{scope}: |Δ| histogram by method")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0.0, xmax * 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{scope}_abs_delta_hist.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # --- 3) ECDF (most informative for comparing error tails) ---
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for m in METHOD_ORDER:
        vals = np.sort(series[m])
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(
            vals,
            y,
            where="post",
            color=METHOD_COLORS[m],
            linewidth=2.0,
            label=m,
        )
    ax.set_xlabel("|Δ| NAV vs CubeWalkers")
    ax.set_ylabel("ECDF  (fraction of items ≤ x)")
    ax.set_title(f"{scope}: |Δ| empirical CDF by method")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0.0, xmax * 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{scope}_abs_delta_ecdf.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # --- 4) Mean / median summary bars ---
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(METHOD_ORDER))
    means = [float(np.mean(series[m])) for m in METHOD_ORDER]
    medians = [float(np.median(series[m])) for m in METHOD_ORDER]
    w = 0.36
    bars_mean = ax.bar(
        x - w / 2,
        means,
        width=w,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        alpha=0.85,
        label="mean",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_med = ax.bar(
        x + w / 2,
        medians,
        width=w,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        alpha=0.4,
        label="median",
        hatch="//",
        edgecolor=[METHOD_COLORS[m] for m in METHOD_ORDER],
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=15)
    ax.set_ylabel("|Δ| NAV vs CubeWalkers")
    ax.set_title(f"{scope}: mean / median |Δ| by method")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_mean, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, val in zip(bars_med, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#444444",
        )
    fig.tight_layout()
    path = out_dir / f"{scope}_abs_delta_summary.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # --- 5) Changed-only overlay (if any nodes moved) ---
    changed = df[df["changed"]] if "changed" in df.columns else df.iloc[0:0]
    if len(changed) >= 5:
        ch_series = _abs_series_by_method(changed)
        ch_all = np.concatenate(list(ch_series.values()))
        ch_xmax = float(np.nanmax(ch_all)) if len(ch_all) else xmax
        ch_bins = np.linspace(0.0, ch_xmax, 25)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

        ax = axes[0]
        for m in METHOD_ORDER:
            ax.hist(
                ch_series[m],
                bins=ch_bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=METHOD_COLORS[m],
                label=m,
            )
        ax.set_xlabel("|Δ| NAV vs CubeWalkers")
        ax.set_ylabel("Density")
        ax.set_title(f"{scope}: |Δ| hist (changed only, n={len(changed)})")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        for m in METHOD_ORDER:
            vals = np.sort(ch_series[m])
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(
                vals,
                y,
                where="post",
                color=METHOD_COLORS[m],
                linewidth=2.0,
                label=m,
            )
        ax.set_xlabel("|Δ| NAV vs CubeWalkers")
        ax.set_ylabel("ECDF")
        ax.set_title(f"{scope}: |Δ| ECDF (changed only)")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = out_dir / f"{scope}_abs_delta_changed.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    return written


print()
print("Writing |Δ| distribution plots…")
node_plots = _plot_abs_delta_distributions(
    node_cmp_df, scope="nodes", out_dir=RUN_DIR
)
ent_plots = _plot_abs_delta_distributions(
    ent_cmp_df, scope="entities", out_dir=RUN_DIR
)
for p in node_plots + ent_plots:
    print(f"  wrote {p}")

print("Writing method compute-time plot…")
timing_plot = _plot_method_timings(RUN_DIR)
if timing_plot is not None:
    print(f"  wrote {timing_plot}")
else:
    print("  (skipped — no timings available)")

# Persist comparison tables + timings for later inspection.
node_csv = RUN_DIR / "node_cmp.csv"
ent_csv = RUN_DIR / "entity_cmp.csv"
timings_csv = RUN_DIR / "method_timings.csv"
node_cmp_df.to_csv(node_csv, index=False)
ent_cmp_df.to_csv(ent_csv, index=False)
_timings = _method_timings_s()
pd.DataFrame(
    [{"method": m, "seconds": _timings[m]} for m in TIMING_METHOD_ORDER if m in _timings]
).to_csv(timings_csv, index=False)
print(f"  wrote {node_csv}")
print(f"  wrote {ent_csv}")
print(f"  wrote {timings_csv}")

print()
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# %%
# Restore stdout and close log.
sys.stdout = _stdout
_log_file.close()
print(f"log written: {LOG_PATH}")
print(f"CW cache: {CW_CACHE_PATH}")
print(f"outputs under: {RUN_DIR}")
