# %% [markdown]
# LDOI NAV vs cubewalkers on scrambled breast-cancer models
#
# Scramble ``BreastCancer_primes_parsimonious`` with mutation probability 0.5
# (``edge_prob=0``), 30 times. For each scramble, compare entity-level NAV from:
#
# 1. **cubewalkers** — clamp interventions, sample walkers, async ensemble,
#    average node values over a late time window (ground truth)
# 2. **LDOI** — logical domain of influence on interventions only
#    (no source sampling)
#
# Aggregates over all scrambles (entity-level only; no node-by-node plots):
#
# - Mean total wall time (cubewalkers vs LDOI)
# - Per-scramble mean entity RMSE (LDOI vs cubewalkers)
# - Pooled (scramble × fix × boot) entity Δ boxplots
# - Flat-score scatter: cubewalkers vs LDOI (Pearson *r*)
#
# Prints to stdout and writes a log (+ plots) under `dev/logs/` (overwritten
# each full run). ``SMOKE=1`` writes separate ``*_smoke`` artifacts so it does
# not clobber full-run outputs.
#
# ---
#
# ## WSL CUDA setup (required for cubewalkers)
#
# 1. Windows: NVIDIA driver with WSL2 GPU support (`nvidia-smi` in PowerShell)
# 2. WSL: CUDA toolkit matching the driver major version
# 3. `pip install cupy-cuda12x` (or `cupy-cuda11x`) and
#    `pip install git+https://github.com/jcrozum/cubewalkers`
#
# This notebook **fails fast** if CuPy / cubewalkers / a CUDA device is missing
# (see ``cw_helpers.require_cubewalkers``).

# %%
from __future__ import annotations

import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.boolean_functions import assignment_to_dict
from boolmore.evaluation.nav_score import get_NAV_scores
from boolmore.evaluation.score import get_model_score
from boolmore.genetic.operators.mutation import mutate_model
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.prediction import _nav_from_ldoi
from boolmore.io.export import export_model
from boolmore.io.load import load_json_config, load_model
from boolmore.io.nav import import_NAV_exps
from boolmore.io.paths import project_root
from boolmore.types import Assignment

# Sibling helpers are not on the editable install; keep ``dev/`` importable.
_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from cw_helpers import (
    DEFAULT_BOOT_CHUNK,
    DEFAULT_N_BOOT,
    DEFAULT_THREADS_PER_BLOCK,
    align_walker_nav,
    bnet_rules_for_cubewalkers,
    bootstrap_se,
    require_cubewalkers,
    simulate_walker_nav,
)
from nav_helpers import (
    Tee,
    analyze_fixes,
    discover_full_entities,
    entity_from_node_array,
    entity_nav_dict,
    format_fixes,
    nav_vector_metrics,
    n_unfixed_sources,
    pooled_delta_entity_stats,
    source_nodes_from_primes,
    summarize_across_entities,
    timed,
    to_nav_predictions,
)

# %% [markdown]
# ## Config

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "breast_cancer"
PKL = CASE / "BreastCancer_primes_parsimonious.pkl"
CONFIG_JSON = CASE / "BreastCancer_config.json"
DATA = CASE / "breast_cancer_data_trial.csv"
LOG_DIR = Path(__file__).resolve().parent / "logs"

ID_COL = "id"
WEIGHT_COL = "Weight"
OBSERVED_COL = "Observed node"
OUTCOME_COL = "Value"
FIXES_COLS = (
    "Signals",
    "Drug treatment/intervention",
    "Cell line characteristic profile",
    "Additional features of the cell population",
)

N_SCRAMBLES = 30
MUT_PROB = 0.5
EDGE_PROB = 0.0

N_WALKERS = 2048
# Paper-style horizon; set explicitly below after N is known, or override here.
N_TIME_STEPS: int | None = None  # None → 55*N + 6000
T_WINDOW: int | None = None  # None → max(1000, T // 4)
# Small GPUs (e.g. MX150) hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at (32, 32).
THREADS_PER_BLOCK: tuple[int, int] = DEFAULT_THREADS_PER_BLOCK

N_BOOT = DEFAULT_N_BOOT
BOOT_CHUNK = DEFAULT_BOOT_CHUNK

SEED = 0
MAX_FIXES: int | None = None

# Optional: `SMOKE=1` for a tiny end-to-end check (separate output filenames).
SMOKE = os.environ.get("SMOKE") == "1"
OUT_TAG = "_smoke" if SMOKE else ""
if SMOKE:
    N_SCRAMBLES = 2
    MAX_FIXES = 2
    N_WALKERS = 64
    N_TIME_STEPS = 200
    T_WINDOW = 50
    N_BOOT = 40
    BOOT_CHUNK = 20

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast)

# %%
cw, cp = require_cubewalkers()
print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
print(f"cubewalkers: {getattr(cw, '__file__', cw)}")

# %% [markdown]
# ## Logging

# %%
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"compare_nav_ldoi_vs_cw_scrambled{OUT_TAG}.log"
PLOT_TIME = LOG_DIR / f"compare_nav_ldoi_vs_cw_scrambled_timing{OUT_TAG}.png"
PLOT_RMSE = LOG_DIR / f"compare_nav_ldoi_vs_cw_scrambled_entity_rmse{OUT_TAG}.png"
PLOT_DELTA_ENTITY = LOG_DIR / f"compare_nav_ldoi_vs_cw_scrambled_delta_box_entity{OUT_TAG}.png"
PLOT_SCORE_SCATTER = LOG_DIR / f"compare_nav_ldoi_vs_cw_scrambled_flat_score_scatter{OUT_TAG}.png"

_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model pkl:  {PKL}")
print(f"config:     {CONFIG_JSON}")
print(f"data:       {DATA}")
print(f"N_SCRAMBLES={N_SCRAMBLES}  MUT_PROB={MUT_PROB}  EDGE_PROB={EDGE_PROB}")
print(f"N_WALKERS={N_WALKERS}  N_BOOT={N_BOOT}  BOOT_CHUNK={BOOT_CHUNK}")
print(f"THREADS_PER_BLOCK={THREADS_PER_BLOCK}")
print(f"SEED={SEED}  MAX_FIXES={MAX_FIXES}  SMOKE={SMOKE}")
print()

# %% [markdown]
# ## Load base model, constraints, and experiments

# %%
config = load_json_config(CONFIG_JSON)
constraints = config["constraints"]

base_model = load_model(PKL)
primes_base = base_model.primes
experiments = import_NAV_exps(
    DATA,
    id_col=ID_COL,
    weight_col=WEIGHT_COL,
    observed_col=OBSERVED_COL,
    outcome_col=OUTCOME_COL,
    fixes_cols=FIXES_COLS,
)

source_nodes = source_nodes_from_primes(primes_base)
source_set = set(source_nodes)
node_names = sorted(primes_base)

_n_nodes = len(primes_base)
if N_TIME_STEPS is None:
    N_TIME_STEPS = 55 * _n_nodes + 6000
if T_WINDOW is None:
    T_WINDOW = max(1000, N_TIME_STEPS // 4)
assert T_WINDOW <= N_TIME_STEPS

_all_unique = list({exp.fixes for exp in experiments})
unique_fixes = sorted(
    _all_unique,
    key=lambda f: (n_unfixed_sources(f, source_set), len(f), f),
)
if MAX_FIXES is not None:
    unique_fixes = unique_fixes[:MAX_FIXES]

unique_fixes_set = set(unique_fixes)
ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    if exp.fixes not in unique_fixes_set:
        continue
    ids_by_fixes[exp.fixes].append(exp.id)

print(f"Model nodes : {len(primes_base)}")
print(f"Source nodes: {len(source_nodes)}")
print(f"  {source_nodes}")
print(f"Experiments : {len(experiments)}")
print(f"Unique fixes: {len(unique_fixes)}" + (f" (capped by MAX_FIXES)" if MAX_FIXES else ""))
print(f"cubewalkers T={N_TIME_STEPS}  T_WINDOW={T_WINDOW}  walkers={N_WALKERS}")
print()

ensure_asp_backend()

# %% [markdown]
# ## Unfixed-source analysis + entities

# %%
meta_rows = [analyze_fixes(fixes, source_set) for fixes in unique_fixes]
meta_df = pd.DataFrame(meta_rows)
meta_df["exp_ids"] = [ids_by_fixes[f] for f in meta_df["fixes"]]
meta_df["n_exps"] = meta_df["exp_ids"].map(len)
meta_by_fixes = {row["fixes"]: row for row in meta_rows}

hist = Counter(meta_df["n_unfixed_sources"])
hist_df = pd.DataFrame(
    [
        {
            "n_unfixed_sources (x)": x,
            "n_distinct_fixes": hist[x],
            "min_trap_lower_bound (2**x)": 2**x,
        }
        for x in sorted(hist)
    ]
)
print("Unfixed source histogram:")
print(hist_df.to_string(index=False))
print(
    f"Unfixed sources — min {meta_df['n_unfixed_sources'].min()}, "
    f"median {meta_df['n_unfixed_sources'].median():.0f}, "
    f"max {meta_df['n_unfixed_sources'].max()}"
)
print()

MERGED_BASES, ENTITY_KEYS = discover_full_entities(node_names, experiments)
MERGED_SET = set(MERGED_BASES)
print(
    f"Entities ({len(ENTITY_KEYS)}): "
    f"{len(MERGED_BASES)} merged + "
    f"{len(ENTITY_KEYS) - len(MERGED_BASES)} singletons"
)
print(f"  merged bases: {MERGED_BASES}")
print()

all_data_fixes = {exp.fixes for exp in experiments}
can_score = MAX_FIXES is None and set(unique_fixes) == all_data_fixes

# %% [markdown]
# ## Scramble loop — cubewalkers / LDOI per fix
#
# Peak memory stays per-fix: walker arrays are discarded after each
# intervention's entity Δ cloud block is stored.

# %%
rng_boot = np.random.default_rng(SEED + 1)

delta_blocks_ent: list[np.ndarray] = []
scramble_rows: list[dict] = []
score_rows: list[dict] = []

print(
    f"Running {N_SCRAMBLES} scrambles × {len(unique_fixes)} fixes "
    f"(cubewalkers + LDOI, N_BOOT={N_BOOT}, BOOT_CHUNK={BOOT_CHUNK})..."
)

for s in range(N_SCRAMBLES):
    scrambled = mutate_model(
        base_model, MUT_PROB, EDGE_PROB, constraints, seed=SEED + s
    )
    primes = scrambled.primes

    with tempfile.TemporaryDirectory(prefix=f"scramble_{s}_") as td:
        stem = str(Path(td) / "scrambled")
        # Avoid model.info() headers with commas — cubewalkers would treat them as rules.
        bnet_path = export_model(scrambled, stem, header=f"# scramble {s}")
        rules = "\n".join(
            ln
            for ln in bnet_rules_for_cubewalkers(bnet_path).splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ) + "\n"

        preds_m1: dict[Assignment, dict[str, float]] = {}
        preds_m4: dict[Assignment, dict[str, float]] = {}
        times_m1: dict[Assignment, float] = {}
        times_m4: dict[Assignment, float] = {}
        entity_rmse_fixes: list[float] = []

        for i, fixes in enumerate(unique_fixes, 1):
            m = meta_by_fixes[fixes]
            x = m["n_unfixed_sources"]

            # --- cubewalkers walkers + bootstrap ---
            walker_nav, varnames, timings_m1 = simulate_walker_nav(
                rules,
                clamps=fixes,
                n_time_steps=N_TIME_STEPS,
                t_window=T_WINDOW,
                n_walkers=N_WALKERS,
                threads_per_block=THREADS_PER_BLOCK,
            )
            walker_nav = align_walker_nav(walker_nav, varnames, node_names)
            means_m1, _se_m1, boot_m1, t_boot_m1 = bootstrap_se(
                walker_nav, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng_boot
            )
            del walker_nav
            preds_m1[fixes] = {n: float(means_m1[j]) for j, n in enumerate(node_names)}
            times_m1[fixes] = timings_m1["t_total"] + t_boot_m1

            means_m1_ent = entity_from_node_array(
                means_m1, node_names, ENTITY_KEYS, MERGED_SET
            )
            boot_m1_ent = entity_from_node_array(
                boot_m1, node_names, ENTITY_KEYS, MERGED_SET
            )

            # --- deterministic LDOI ---
            pert = assignment_to_dict(fixes)
            nav_m4, dt_m4 = timed(_nav_from_ldoi, primes, pert)
            times_m4[fixes] = dt_m4
            preds_m4[fixes] = {n: float(nav_m4[n]) for n in node_names}
            means_m4 = np.asarray(
                [preds_m4[fixes][n] for n in node_names], dtype=np.float64
            )
            boot_m4 = np.broadcast_to(
                means_m4[:, None], (len(node_names), N_BOOT)
            ).copy()
            means_m4_ent = entity_from_node_array(
                means_m4, node_names, ENTITY_KEYS, MERGED_SET
            )
            boot_m4_ent = entity_from_node_array(
                boot_m4, node_names, ENTITY_KEYS, MERGED_SET
            )

            delta_blocks_ent.append(boot_m4_ent - boot_m1_ent)

            ent_gt = entity_nav_dict(preds_m1[fixes], ENTITY_KEYS, MERGED_SET)
            ent_pred = entity_nav_dict(preds_m4[fixes], ENTITY_KEYS, MERGED_SET)
            em = nav_vector_metrics(ent_gt, ent_pred, nodes=ENTITY_KEYS)
            entity_rmse_fixes.append(em["rmse"])

            del (
                means_m1,
                boot_m1,
                means_m1_ent,
                boot_m1_ent,
                means_m4,
                boot_m4,
                means_m4_ent,
                boot_m4_ent,
            )

            if i % 10 == 0 or i == len(unique_fixes):
                print(
                    f"  scramble {s + 1}/{N_SCRAMBLES}  "
                    f"[{i}/{len(unique_fixes)}] x={x}  "
                    f"cum cw={sum(times_m1.values()):.1f}s  "
                    f"ldoi={sum(times_m4.values()):.1f}s  "
                    f"{format_fixes(fixes)}"
                )

    t_m1 = sum(times_m1.values())
    t_m4 = sum(times_m4.values())
    mean_ent_rmse = float(np.mean(entity_rmse_fixes))
    scramble_rows.append(
        {
            "scramble": s,
            "seed": SEED + s,
            "t_cubewalkers": t_m1,
            "t_ldoi": t_m4,
            "mean_entity_rmse": mean_ent_rmse,
        }
    )
    print(
        f"scramble {s + 1}/{N_SCRAMBLES} done: "
        f"cw={t_m1:.2f}s  ldoi={t_m4:.2f}s  "
        f"mean entity RMSE={mean_ent_rmse:.4f}"
    )

    if can_score:
        score_by_method: dict[str, tuple[float, float]] = {}
        for name, preds in (("cubewalkers", preds_m1), ("ldoi", preds_m4)):
            items = get_NAV_scores(
                experiments,
                to_nav_predictions(preds),
                default_sources={},
                hierarchy=False,
            )
            max_score, score = get_model_score(items)
            score_by_method[name] = (score, max_score)
        s1, mx = score_by_method["cubewalkers"]
        s4, _ = score_by_method["ldoi"]
        score_rows.append(
            {
                "scramble": s,
                "score_cubewalkers": s1,
                "score_ldoi": s4,
                "max_score": mx,
                "frac_cubewalkers": s1 / mx if mx else math.nan,
                "frac_ldoi": s4 / mx if mx else math.nan,
            }
        )
        print(
            f"  flat scores: cubewalkers={s1:.4f}/{mx:.4f}  "
            f"ldoi={s4:.4f}/{mx:.4f}"
        )
    else:
        print(
            "  Skipping flat scores "
            f"(need all fixes; MAX_FIXES={MAX_FIXES})"
        )
    print()

scramble_df = pd.DataFrame(scramble_rows)
score_df = pd.DataFrame(score_rows)

delta_cloud_ent = np.concatenate(delta_blocks_ent, axis=1)
del delta_blocks_ent

print(
    f"All scrambles done. "
    f"mean cubewalkers={scramble_df['t_cubewalkers'].mean():.2f}s  "
    f"mean ldoi={scramble_df['t_ldoi'].mean():.2f}s  "
    f"mean entity RMSE={scramble_df['mean_entity_rmse'].mean():.4f}"
)
print()

# %% [markdown]
# ## Pooled entity Δ stats (scramble × fix × boot)

# %%
per_ent = pooled_delta_entity_stats(delta_cloud_ent)
entity_stat_df = pd.DataFrame({"entity": ENTITY_KEYS, **per_ent})
entity_stat_df = entity_stat_df.sort_values(
    by=["median_abs", "mean_abs", "max_abs", "rmse", "entity"],
    ascending=[False, False, False, False, True],
    kind="mergesort",
).reset_index(drop=True)
glob_ent = summarize_across_entities(per_ent)
glob_ent["n_items"] = len(ENTITY_KEYS)
glob_ent["n_pool_points"] = int(delta_cloud_ent.shape[1])

print("=== Pooled (scramble × fix × boot) entity Δ = LDOI − cubewalkers ===")
print(f"  n_pool={glob_ent['n_pool_points']} per entity:")
print(
    f"    per-entity median_|Δ|: "
    f"mean={glob_ent['median_abs_mean']:.4f}  "
    f"median={glob_ent['median_abs_median']:.4f}  "
    f"max={glob_ent['median_abs_max']:.4f}"
)
print(
    f"    per-entity mean_|Δ|:   "
    f"mean={glob_ent['mean_abs_mean']:.4f}  "
    f"median={glob_ent['mean_abs_median']:.4f}  "
    f"max={glob_ent['mean_abs_max']:.4f}"
)
print(
    f"    per-entity rmse:      "
    f"mean={glob_ent['rmse_mean']:.4f}  "
    f"median={glob_ent['rmse_median']:.4f}  "
    f"max={glob_ent['rmse_max']:.4f}"
)
print("    top-5 entities by median_|Δ|:")
for _, r in entity_stat_df.head(5).iterrows():
    print(
        f"      {r['entity']}: "
        f"median_|Δ|={r['median_abs']:.4f}  "
        f"mean_|Δ|={r['mean_abs']:.4f}  "
        f"rmse={r['rmse']:.4f}"
    )
print()

# %% [markdown]
# ## Plots

# %%
colors = ["#4C72B0", "#C44E52"]

# Timing: mean total wall time across scrambles
fig, ax = plt.subplots(figsize=(5, 4))
method_order = ("cubewalkers", "ldoi")
method_labels = ("cubewalkers", "LDOI")
means_t = [float(scramble_df[f"t_{lab}"].mean()) for lab in method_order]
ax.bar(method_labels, means_t, color=colors)
ax.set_ylabel("Mean total wall time (s)")
ax.set_title(f"Prediction time by method (n={N_SCRAMBLES} scrambles)")
for i, v in enumerate(means_t):
    ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_TIME, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_TIME}")

# Entity RMSE distribution across scrambles
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.boxplot(
    [scramble_df["mean_entity_rmse"].to_numpy()],
    tick_labels=["LDOI vs cubewalkers"],
    showfliers=True,
    widths=0.5,
)
ax.scatter(
    np.ones(len(scramble_df))
    + np.random.default_rng(SEED).uniform(-0.06, 0.06, len(scramble_df)),
    scramble_df["mean_entity_rmse"],
    color=colors[1],
    alpha=0.7,
    s=28,
    zorder=3,
)
ax.set_ylabel("Mean entity RMSE vs cubewalkers")
ax.set_title(f"Entity NAV accuracy across {N_SCRAMBLES} scrambles")
fig.tight_layout()
fig.savefig(PLOT_RMSE, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_RMSE}")

# Entity Δ boxplots (signed / |Δ|)
order = list(entity_stat_df["entity"])
key_to_i = {k: i for i, k in enumerate(ENTITY_KEYS)}
signed = [delta_cloud_ent[key_to_i[k]] for k in order]
absolute = [np.abs(v) for v in signed]
n_items = len(order)
fig_h = max(6.0, 0.18 * n_items + 1.5)
fig, axes = plt.subplots(1, 2, figsize=(12, fig_h), sharey=True)
for ax, data, xlabel, panel in (
    (axes[0], signed, "Δ NAV (LDOI − cubewalkers)", "signed Δ"),
    (axes[1], absolute, "|Δ| NAV (LDOI − cubewalkers)", "|Δ|"),
):
    ax.boxplot(
        data,
        vert=False,
        tick_labels=order,
        showfliers=False,
        whis=(5, 95),
        widths=0.7,
    )
    ax.axvline(0.0, color="gray", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(panel)
    ax.tick_params(axis="y", labelsize=6)
fig.suptitle("LDOI − cubewalkers entity Δ (pooled scrambles × fixes × boots)")
fig.tight_layout()
fig.savefig(PLOT_DELTA_ENTITY, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_DELTA_ENTITY}")

# Flat-score scatter
if not score_df.empty:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    x = score_df["score_cubewalkers"].to_numpy()
    y = score_df["score_ldoi"].to_numpy()
    ax.scatter(x, y, color=colors[0], alpha=0.8, s=40, zorder=3)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="gray",
        lw=0.9,
        ls="--",
        label="y = x",
    )
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"Flat score: cubewalkers vs LDOI  (r = {r:.3f}, n = {len(x)})")
    else:
        r = float("nan")
        ax.set_title(f"Flat score: cubewalkers vs LDOI  (n = {len(x)})")
    ax.set_xlabel("Flat score (cubewalkers)")
    ax.set_ylabel("Flat score (LDOI)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_SCORE_SCATTER, dpi=120)
    plt.close(fig)
    print(f"Wrote {PLOT_SCORE_SCATTER}")
else:
    r = float("nan")
    print("Skipped flat-score scatter (no scores)")
print()

# %% [markdown]
# ## Summary

# %%
print("=== Summary ===")
print(f"  Scrambles: {N_SCRAMBLES}  (MUT_PROB={MUT_PROB}, EDGE_PROB={EDGE_PROB})")
print(f"  Unique fixes: {len(unique_fixes)}")
print(f"  Entities: {len(ENTITY_KEYS)} ({len(MERGED_BASES)} merged)")
print(f"  cubewalkers: walkers={N_WALKERS}  T={N_TIME_STEPS}  window={T_WINDOW}")
print(f"  bootstrap: N_BOOT={N_BOOT}  BOOT_CHUNK={BOOT_CHUNK}")
print(
    f"  Mean times (s): "
    f"cubewalkers={scramble_df['t_cubewalkers'].mean():.2f}  "
    f"ldoi={scramble_df['t_ldoi'].mean():.2f}"
)
print(
    f"  Mean entity RMSE LDOI vs cubewalkers: "
    f"{scramble_df['mean_entity_rmse'].mean():.4f}  "
    f"(median={scramble_df['mean_entity_rmse'].median():.4f})"
)
print(
    f"  Pooled entity Δ: mean(median_|Δ|)={glob_ent['median_abs_mean']:.4f}  "
    f"mean(rmse)={glob_ent['rmse_mean']:.4f}  "
    f"max(max_|Δ|)={glob_ent['max_abs_max']:.4f}"
)
if not score_df.empty:
    print(
        f"  Flat-score correlation cubewalkers vs LDOI: r={r:+.4f}  "
        f"(n={len(score_df)})"
    )
    print(
        f"  Mean flat frac: cubewalkers={score_df['frac_cubewalkers'].mean():.4f}  "
        f"ldoi={score_df['frac_ldoi'].mean():.4f}"
    )
print(f"  Log: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
