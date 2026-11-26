# %% [markdown]
# LDOI / LDOI-enhanced vs CubeWalkers (synchronous) — RA 243 source combos
#
# On ``case_study/RA/243.bnet``:
#
# 1. Import primes once and pickle them next to the ``.bnet`` for reuse
# 2. Count syntactic source nodes (dangling regulators become sources on load)
# 3. Sample 30 full 0/1 source assignments (seeded) and persist them
# 4. For each combo: LDOI + LDOI-enhanced (cutoffs 1–3, optional joint) NAV
# 5. For each combo: CubeWalkers synchronous NAV (``T = 5*N``)
# 6. Cache LDOI / CW results; compare prediction quality and wall times
#
# Cubewalkers requires CUDA + CuPy in WSL2 (see ``cw_helpers.require_cubewalkers``).
# ``SMOKE=1`` uses a tiny horizon / fewer combos for an end-to-end check.
#
# Sibling of ``investigate_ldoi_vs_ldoi_enhanced.py`` (breast-cancer single seed).

# %%
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyboolnet.file_exchange import primes2bnet

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
    bnet_rules_for_cubewalkers,
    require_cubewalkers,
    simulate_mean_nav,
)
from nav_helpers import (
    Tee,
    decode_source_index,
    sample_source_indices,
    source_nodes_from_primes,
)

# %% [markdown]
# ## Config

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "RA"
MODEL_STEM = CASE / "243"
BNET = MODEL_STEM.with_suffix(".bnet")
PKL = MODEL_STEM.with_suffix(".pkl")

LOGS_ROOT = Path(__file__).resolve().parent / "logs"
SCRIPT_NAME = Path(__file__).stem  # investigate_ldoi_ra_243_source_combos

COMBO_RNG_SEED = 0
N_COMBOS = 30

ENHANCED_CUTOFFS: tuple[int, ...] = (1, 2, 3)
USE_JOINT_SELF_LOOPS = True

N_WALKERS = 2048
# Resolved after N is known: T = 5*N, T_WINDOW = max(N, T//4)
N_TIME_STEPS: int | None = None
T_WINDOW: int | None = None
THREADS_PER_BLOCK: tuple[int, int] = DEFAULT_THREADS_PER_BLOCK
UPDATE_SCHEME = "synchronous"

FORCE_RECOMPUTE = False

SMOKE = os.environ.get("SMOKE") == "1"
OUT_TAG = "_smoke" if SMOKE else ""
if SMOKE:
    N_COMBOS = 2
    N_WALKERS = 64
    # T / window overridden after N is known (still a few×N but capped)

METHOD_LDOI = "ldoi"
METHOD_ENH = {c: f"enhanced(c={c})" for c in ENHANCED_CUTOFFS}
METHOD_ENHJ = {c: f"enhanced+j(c={c})" for c in ENHANCED_CUTOFFS}
METHOD_ORDER = [METHOD_LDOI] + [METHOD_ENH[c] for c in ENHANCED_CUTOFFS]
if USE_JOINT_SELF_LOOPS:
    METHOD_ORDER = METHOD_ORDER + [METHOD_ENHJ[c] for c in ENHANCED_CUTOFFS]
ABS_COL_SHORT = {
    METHOD_LDOI: "abs_delta_ldoi",
    **{METHOD_ENH[c]: f"abs_delta_enh_c{c}" for c in ENHANCED_CUTOFFS},
}
if USE_JOINT_SELF_LOOPS:
    ABS_COL_SHORT.update(
        {METHOD_ENHJ[c]: f"abs_delta_enhj_c{c}" for c in ENHANCED_CUTOFFS}
    )

METHOD_CW = "cubewalkers"
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
TIMING_METHOD_ORDER = METHOD_ORDER + [METHOD_CW]


def _assignment_key(assignment: dict[str, int]) -> str:
    return json.dumps(sorted(assignment.items()), separators=(",", ":"))


# %% [markdown]
# ## Load / pickle primes

# %%
RUN_DIR = (
    LOGS_ROOT
    / SCRIPT_NAME
    / MODEL_STEM.name
    / f"combo-seed-{COMBO_RNG_SEED}{OUT_TAG}"
)
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = (
    RUN_DIR
    / f"{SCRIPT_NAME}_{MODEL_STEM.name}_combo-seed-{COMBO_RNG_SEED}{OUT_TAG}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"run dir: {RUN_DIR}")
print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model bnet: {BNET}")
print(f"model pkl:  {PKL}")
print(f"COMBO_RNG_SEED={COMBO_RNG_SEED}  N_COMBOS={N_COMBOS}")
print(f"ENHANCED_CUTOFFS={ENHANCED_CUTOFFS}")
print(f"USE_JOINT_SELF_LOOPS={USE_JOINT_SELF_LOOPS}")
print(f"FORCE_RECOMPUTE={FORCE_RECOMPUTE}  SMOKE={SMOKE}")
print(f"UPDATE_SCHEME={UPDATE_SCHEME}")
print()

if PKL.is_file() and not FORCE_RECOMPUTE:
    print(f"loading primes from {PKL}")
    primes = load_primes(PKL)
else:
    if not BNET.is_file():
        raise SystemExit(f"missing bnet: {BNET}")
    print(f"importing primes from {BNET} (may take a while)…")
    _t0 = time.perf_counter()
    primes = load_primes(BNET)
    print(f"  load_primes done in {time.perf_counter() - _t0:.3f}s")
    with PKL.open("wb") as f:
        pickle.dump(primes, f)
    print(f"wrote primes → {PKL}")

node_names = sorted(primes)
_n_nodes = len(primes)
sources = source_nodes_from_primes(primes)
n_sources = len(sources)
print(f"Model nodes : {_n_nodes}")
print(f"Source nodes: {n_sources}")
if n_sources == 0:
    raise SystemExit("no syntactic source nodes found after import")
print(f"  first sources: {', '.join(sources[:8])}{'…' if n_sources > 8 else ''}")
print()

if N_TIME_STEPS is None:
    N_TIME_STEPS = 5 * _n_nodes
if T_WINDOW is None:
    T_WINDOW = max(_n_nodes, N_TIME_STEPS // 4)
if SMOKE:
    # Tiny absolute horizon for end-to-end checks (not scaled to N).
    N_TIME_STEPS = 80
    T_WINDOW = 16
assert T_WINDOW <= N_TIME_STEPS

COMPLETE_BNET = RUN_DIR / "complete_rules.bnet"
if COMPLETE_BNET.is_file() and not FORCE_RECOMPUTE:
    print(f"complete rules cache hit: {COMPLETE_BNET.name}")
else:
    print("writing complete rules via primes2bnet (includes auto-sources)…")
    _t0 = time.perf_counter()
    bnet_text = primes2bnet(primes, fname_bnet=None, minimize=False, header=True)
    COMPLETE_BNET.write_text(bnet_text, encoding="utf-8")
    print(f"  primes2bnet done in {time.perf_counter() - _t0:.3f}s → {COMPLETE_BNET}")
BNET_RULES = bnet_rules_for_cubewalkers(COMPLETE_BNET)

print(
    f"cubewalkers sync T={N_TIME_STEPS}  T_WINDOW={T_WINDOW}  "
    f"walkers={N_WALKERS}"
)
print()

# %% [markdown]
# ## Sample 30 full source assignments

# %%
COMBOS_PATH = RUN_DIR / "combos.json"
if COMBOS_PATH.is_file() and not FORCE_RECOMPUTE:
    combos_payload = json.loads(COMBOS_PATH.read_text(encoding="utf-8"))
    combo_indices = [int(x) for x in combos_payload["indices"]]
    print(f"loaded {len(combo_indices)} combo indices from {COMBOS_PATH.name}")
else:
    rng = random.Random(COMBO_RNG_SEED)
    combo_indices = sample_source_indices(n_sources, N_COMBOS, rng)
    combos_payload = {
        "combo_rng_seed": COMBO_RNG_SEED,
        "n_sources": n_sources,
        "sources": list(sources),
        "n_combos_requested": N_COMBOS,
        "indices": combo_indices,
        "smoke": SMOKE,
    }
    COMBOS_PATH.write_text(
        json.dumps(combos_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"sampled {len(combo_indices)} combo indices → {COMBOS_PATH}")

# Truncate if smoke reuses a full-run combos.json somehow mismatched.
combo_indices = combo_indices[:N_COMBOS]
assignments: list[dict[str, int]] = [
    decode_source_index(idx, sources, {}) for idx in combo_indices
]
print(f"using {len(assignments)} source assignments")
print(
    f"  combo[0] index={combo_indices[0]}  "
    f"n1={sum(assignments[0].values())}/{n_sources}"
)
print()

# %% [markdown]
# ## LDOI / LDOI-enhanced NAV (cached)

# %%
LDOI_CACHE_PATH = RUN_DIR / f"ldoi_nav_cache{OUT_TAG}.pkl"
_max_cutoff = max(ENHANCED_CUTOFFS)
print(f"building SC subsets (max cutoff={_max_cutoff})…")
_t0 = time.perf_counter()
sc_subsets = _sc_subsets_by_size(primes, _max_cutoff)
t_sc = time.perf_counter() - _t0
for k in range(1, _max_cutoff + 1):
    print(f"  SC subsets size {k}: {len(sc_subsets.get(k, []))}")
print(f"  SC-subset prep: {t_sc:.3f}s")
print()

ldoi_cache: dict = {}
if LDOI_CACHE_PATH.is_file() and not FORCE_RECOMPUTE:
    with LDOI_CACHE_PATH.open("rb") as f:
        ldoi_cache = pickle.load(f)
    print(f"loaded LDOI cache ({len(ldoi_cache.get('combos', {}))} combos) "
          f"from {LDOI_CACHE_PATH.name}")

ldoi_combos: dict[str, dict] = dict(ldoi_cache.get("combos", {}))
for i, (idx, assignment) in enumerate(zip(combo_indices, assignments)):
    key = _assignment_key(assignment)
    cached = ldoi_combos.get(key)
    if cached is not None and not FORCE_RECOMPUTE:
        print(f"[{i + 1}/{len(assignments)}] LDOI cache hit  index={idx}")
        continue

    print(f"[{i + 1}/{len(assignments)}] Computing LDOI methods  index={idx}…")
    entry: dict = {
        "combo_index": int(idx),
        "assignment": dict(assignment),
        "timings_s": {},
        "nav": {},
        "joint_stats": {},
    }

    _t0 = time.perf_counter()
    nav_ldoi = _nav_from_ldoi(primes, assignment)
    entry["timings_s"][METHOD_LDOI] = time.perf_counter() - _t0
    entry["nav"][METHOD_LDOI] = nav_ldoi
    print(f"  ldoi              {entry['timings_s'][METHOD_LDOI]:.3f}s")

    for c in ENHANCED_CUTOFFS:
        m = METHOD_ENH[c]
        _t0 = time.perf_counter()
        entry["nav"][m] = _nav_from_ldoi_enhanced(
            primes, assignment, cutoff=c, sc_subsets=sc_subsets
        )
        entry["timings_s"][m] = time.perf_counter() - _t0
        print(f"  {m:18s} {entry['timings_s'][m]:.3f}s")

    if USE_JOINT_SELF_LOOPS:
        for c in ENHANCED_CUTOFFS:
            m = METHOD_ENHJ[c]
            _stats: dict = {}
            _t0 = time.perf_counter()
            entry["nav"][m] = _nav_from_ldoi_enhanced(
                primes,
                assignment,
                cutoff=c,
                sc_subsets=sc_subsets,
                joint_self_loops=True,
                joint_stats=_stats,
            )
            entry["timings_s"][m] = time.perf_counter() - _t0
            entry["joint_stats"][m] = dict(_stats)
            print(
                f"  {m:18s} {entry['timings_s'][m]:.3f}s  "
                f"joint_probed={_stats.get('n_joint_probed', 0)}  "
                f"joint_kept={_stats.get('n_joint_kept', 0)}"
            )

    ldoi_combos[key] = entry
    # Incremental save so long runs survive interruption.
    with LDOI_CACHE_PATH.open("wb") as f:
        pickle.dump(
            {
                "model_pkl": str(PKL),
                "n_nodes": _n_nodes,
                "sources": list(sources),
                "enhanced_cutoffs": list(ENHANCED_CUTOFFS),
                "use_joint_self_loops": USE_JOINT_SELF_LOOPS,
                "t_sc_subsets_s": t_sc,
                "smoke": SMOKE,
                "combos": ldoi_combos,
            },
            f,
        )

print(f"LDOI cache → {LDOI_CACHE_PATH}")
print()

# %% [markdown]
# ## CubeWalkers synchronous NAV (cached)

# %%
CW_CACHE_PATH = (
    RUN_DIR
    / (
        f"cw_sync_nav_cache_w{N_WALKERS}_T{N_TIME_STEPS}"
        f"_tw{T_WINDOW}{OUT_TAG}.pkl"
    )
)

cw_cache: dict = {}
if CW_CACHE_PATH.is_file() and not FORCE_RECOMPUTE:
    with CW_CACHE_PATH.open("rb") as f:
        cw_cache = pickle.load(f)
    print(
        f"loaded CW sync cache ({len(cw_cache.get('combos', {}))} combos) "
        f"from {CW_CACHE_PATH.name}"
    )

cw_combos: dict[str, dict] = dict(cw_cache.get("combos", {}))
_need_cw = any(
    _assignment_key(a) not in cw_combos or FORCE_RECOMPUTE for a in assignments
)
if _need_cw:
    cw, cp = require_cubewalkers()
    print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"cubewalkers: {getattr(cw, '__file__', cw)}")
    sync_mask = cw.update_schemes.synchronous
else:
    cw = None
    sync_mask = None
    print(f"all CW combos cached — skipping cubewalkers gate ({CW_CACHE_PATH.name})")
print()

for i, (idx, assignment) in enumerate(zip(combo_indices, assignments)):
    key = _assignment_key(assignment)
    cached = cw_combos.get(key)
    if cached is not None and not FORCE_RECOMPUTE:
        print(f"[{i + 1}/{len(assignments)}] CW cache hit  index={idx}")
        continue

    assert cw is not None and sync_mask is not None
    print(f"[{i + 1}/{len(assignments)}] CW synchronous NAV  index={idx}…")
    clamps = list(assignment.items())
    nav_cw_raw, cw_timings = simulate_mean_nav(
        BNET_RULES,
        n_time_steps=N_TIME_STEPS,
        t_window=T_WINDOW,
        n_walkers=N_WALKERS,
        clamps=clamps,
        threads_per_block=THREADS_PER_BLOCK,
        model_name=f"{SCRIPT_NAME}_{i}",
        maskfunction=sync_mask,
    )
    # simulate_mean_nav returns cubewalkers varname keys; align to primes order.
    missing = [n for n in node_names if n not in nav_cw_raw]
    if missing:
        raise ValueError(
            f"CW NAV missing {len(missing)} nodes e.g. {missing[:5]}"
        )
    nav_cw = {n: float(nav_cw_raw[n]) for n in node_names}
    entry = {
        "combo_index": int(idx),
        "assignment": dict(assignment),
        "nav": nav_cw,
        "timings_s": dict(cw_timings),
    }
    cw_combos[key] = entry
    print(
        f"  CW total={cw_timings.get('t_total', float('nan')):.3f}s  "
        f"setup={cw_timings.get('t_setup', float('nan')):.3f}s  "
        f"sim={cw_timings.get('t_simulate', float('nan')):.3f}s"
    )
    with CW_CACHE_PATH.open("wb") as f:
        pickle.dump(
            {
                "model_pkl": str(PKL),
                "complete_bnet": str(COMPLETE_BNET),
                "update_scheme": UPDATE_SCHEME,
                "n_nodes": _n_nodes,
                "sources": list(sources),
                "cw_params": {
                    "n_time_steps": N_TIME_STEPS,
                    "t_window": T_WINDOW,
                    "n_walkers": N_WALKERS,
                    "threads_per_block": THREADS_PER_BLOCK,
                    "smoke": SMOKE,
                },
                "combos": cw_combos,
            },
            f,
        )

print(f"CW sync cache → {CW_CACHE_PATH}")
print()

# %% [markdown]
# ## Compare LDOI methods vs CW across combos


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
    enh_cols = [f"enh_c{c}" for c in ENHANCED_CUTOFFS]
    if nav_enhj is not None:
        enh_cols = enh_cols + [f"enhj_c{c}" for c in ENHANCED_CUTOFFS]
    df["changed"] = (df[enh_cols].sub(df["ldoi"], axis=0).abs() > 1e-12).any(
        axis=1
    )
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


per_combo_rows: list[dict] = []
timing_rows: list[dict] = []
pooled_abs: dict[str, list[float]] = {m: [] for m in METHOD_ORDER}
all_node_cmp: list[pd.DataFrame] = []

for i, (idx, assignment) in enumerate(zip(combo_indices, assignments)):
    key = _assignment_key(assignment)
    ldoi_entry = ldoi_combos[key]
    cw_entry = cw_combos[key]
    nav_ldoi = ldoi_entry["nav"][METHOD_LDOI]
    nav_enh = {c: ldoi_entry["nav"][METHOD_ENH[c]] for c in ENHANCED_CUTOFFS}
    nav_enhj = (
        {c: ldoi_entry["nav"][METHOD_ENHJ[c]] for c in ENHANCED_CUTOFFS}
        if USE_JOINT_SELF_LOOPS
        else None
    )
    nav_cw = cw_entry["nav"]

    cmp_df = _build_cmp_df(
        node_names, "node", nav_ldoi, nav_enh, nav_cw, nav_enhj
    )
    cmp_df.insert(0, "combo_i", i)
    cmp_df.insert(1, "combo_index", int(idx))
    all_node_cmp.append(cmp_df)

    row: dict = {
        "combo_i": i,
        "combo_index": int(idx),
        "n_sources_on": int(sum(assignment.values())),
    }
    for m in METHOD_ORDER:
        col = ABS_COL_SHORT[m]
        row[f"mean_{col}"] = float(cmp_df[col].mean())
        row[f"median_{col}"] = float(cmp_df[col].median())
        row[f"p90_{col}"] = float(cmp_df[col].quantile(0.90))
        row[f"max_{col}"] = float(cmp_df[col].max())
        pooled_abs[m].extend(cmp_df[col].tolist())
    row["n_changed"] = int(cmp_df["changed"].sum())
    per_combo_rows.append(row)

    for m in METHOD_ORDER:
        timing_rows.append(
            {
                "combo_i": i,
                "combo_index": int(idx),
                "method": m,
                "seconds": float(ldoi_entry["timings_s"][m]),
            }
        )
    timing_rows.append(
        {
            "combo_i": i,
            "combo_index": int(idx),
            "method": METHOD_CW,
            "seconds": float(cw_entry["timings_s"].get("t_total", float("nan"))),
        }
    )

    print(
        f"combo {i} index={idx}: "
        f"mean|Δ|ldoi={row['mean_abs_delta_ldoi']:.4f}  "
        f"enh_c3={row.get('mean_abs_delta_enh_c3', float('nan')):.4f}  "
        f"changed={row['n_changed']}"
    )

node_cmp_all = pd.concat(all_node_cmp, ignore_index=True)
per_combo_df = pd.DataFrame(per_combo_rows)
timings_df = pd.DataFrame(timing_rows)

print()
print("Pooled over all combos × nodes — |Δ| summary by method:")
_summary_rows = []
for m in METHOD_ORDER:
    vals = np.asarray(pooled_abs[m], dtype=np.float64)
    _summary_rows.append(
        {
            "method": m,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p75": float(np.quantile(vals, 0.75)),
            "p90": float(np.quantile(vals, 0.90)),
            "max": float(vals.max()),
            "n": int(vals.size),
        }
    )
pooled_summary_df = pd.DataFrame(_summary_rows)
print(pooled_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

print("Per-combo mean |Δ| (mean across combos):")
for m in METHOD_ORDER:
    col = f"mean_{ABS_COL_SHORT[m]}"
    print(
        f"  {m:18s}  "
        f"mean={per_combo_df[col].mean():.4f}  "
        f"std={per_combo_df[col].std(ddof=0):.4f}"
    )
print()

print("Method wall times across combos:")
timing_summary_rows = []
for m in TIMING_METHOD_ORDER:
    sub = timings_df[timings_df["method"] == m]["seconds"]
    if sub.empty:
        continue
    timing_summary_rows.append(
        {
            "method": m,
            "total_s": float(sub.sum()),
            "mean_s": float(sub.mean()),
            "std_s": float(sub.std(ddof=0)),
            "n": int(len(sub)),
        }
    )
    print(
        f"  {m:18s}  total={sub.sum():.3f}s  "
        f"mean={sub.mean():.3f}s  std={sub.std(ddof=0):.3f}s"
    )
timing_summary_df = pd.DataFrame(timing_summary_rows)
print()

print(
    "Nodes where each method has the lowest |Δ| vs CW "
    "(pooled; ties counted for every tied method):"
)
_best_rows = []
for m in METHOD_ORDER:
    among = node_cmp_all["best_method"].str.split("|").apply(lambda ms: m in ms)
    n_among = int(among.sum())
    n_sole = int((among & (node_cmp_all["best_method"] == m)).sum())
    _best_rows.append(
        {"method": m, "sole": n_sole, "tied": n_among - n_sole, "total": n_among}
    )
best_counts_df = pd.DataFrame(_best_rows)
print(best_counts_df.to_string(index=False))
print(f"  (n={len(node_cmp_all)} node×combo rows)")
print()

# %% [markdown]
# ## Plots + persist artifacts


# %%
def _plot_method_timings(out_dir: Path) -> Path | None:
    if timing_summary_df.empty:
        return None
    methods = [
        m
        for m in TIMING_METHOD_ORDER
        if m in set(timing_summary_df["method"])
    ]
    means = [
        float(timing_summary_df.loc[timing_summary_df["method"] == m, "mean_s"].iloc[0])
        for m in methods
    ]
    fig_w = max(8.0, 1.1 * len(methods) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        color=[METHOD_COLORS.get(m, "#888888") for m in methods],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Mean wall time per combo (s)")
    ax.set_title("Compute time by method (mean over combos)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    for bar, val in zip(bars, means):
        label = f"{val:.2f}s" if val < 100 else f"{val:.1f}s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.text(
        0.01,
        0.98,
        "LDOI / enhanced exclude SC-subset prep; CW = t_total",
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


def _abs_series_by_method_pooled() -> dict[str, np.ndarray]:
    return {
        m: np.asarray(pooled_abs[m], dtype=np.float64) for m in METHOD_ORDER
    }


def _plot_abs_delta_distributions(out_dir: Path) -> list[Path]:
    series = _abs_series_by_method_pooled()
    written: list[Path] = []
    all_vals = np.concatenate(list(series.values()))
    xmax = float(np.nanmax(all_vals)) if len(all_vals) else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    bins = np.linspace(0.0, xmax, 31)
    scope = "nodes_pooled"

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
    ax.set_ylabel("|Δ| NAV vs CubeWalkers (sync)")
    ax.set_title(f"{scope}: |Δ| distribution by method")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], loc="upper right", frameon=False)
    fig.tight_layout()
    path = out_dir / f"{scope}_abs_delta_box.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

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
    ax.set_xlabel("|Δ| NAV vs CubeWalkers (sync)")
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
    ax.set_xlabel("|Δ| NAV vs CubeWalkers (sync)")
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
    ax.bar(
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
    ax.set_ylabel("|Δ| NAV vs CubeWalkers (sync)")
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
    fig.tight_layout()
    path = out_dir / f"{scope}_abs_delta_summary.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    return written


print("Writing |Δ| distribution plots…")
for p in _plot_abs_delta_distributions(RUN_DIR):
    print(f"  wrote {p}")

print("Writing method compute-time plot…")
timing_plot = _plot_method_timings(RUN_DIR)
if timing_plot is not None:
    print(f"  wrote {timing_plot}")
else:
    print("  (skipped — no timings available)")

node_csv = RUN_DIR / "node_cmp.csv"
per_combo_csv = RUN_DIR / "per_combo_summary.csv"
timings_csv = RUN_DIR / "method_timings.csv"
pooled_csv = RUN_DIR / "pooled_abs_delta_summary.csv"
best_csv = RUN_DIR / "best_method_counts.csv"
timing_summary_csv = RUN_DIR / "method_timings_summary.csv"

node_cmp_all.to_csv(node_csv, index=False)
per_combo_df.to_csv(per_combo_csv, index=False)
timings_df.to_csv(timings_csv, index=False)
pooled_summary_df.to_csv(pooled_csv, index=False)
best_counts_df.to_csv(best_csv, index=False)
timing_summary_df.to_csv(timing_summary_csv, index=False)

print(f"  wrote {node_csv}")
print(f"  wrote {per_combo_csv}")
print(f"  wrote {timings_csv}")
print(f"  wrote {pooled_csv}")
print(f"  wrote {best_csv}")
print(f"  wrote {timing_summary_csv}")

print()
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# %%
sys.stdout = _stdout
_log_file.close()
print(f"log written: {LOG_PATH}")
print(f"primes pkl: {PKL}")
print(f"LDOI cache: {LDOI_CACHE_PATH}")
print(f"CW cache: {CW_CACHE_PATH}")
print(f"outputs under: {RUN_DIR}")
