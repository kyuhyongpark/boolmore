# %% [markdown]
# NAV: min-traps vs LDOI vs source-combination sampling
#
# Compare four NAV predictors on the breast-cancer gen57 model and trial data:
#
# 1. **min_traps** — fix interventions only → all minimal trap spaces → NAV
# 2. **min_traps + sample sources** — sample free source assignments → min-traps
#    NAV each → average over samples
# 3. **ldoi** — LDOI on interventions only → 0/1/0.5 NAV
# 4. **ldoi + sample sources** — sample free sources → LDOI NAV each → average
#
# Method 1 is skipped when there are too many unfixed source nodes
# (`MAX_UNFIXED_M1`). Sampling stability for methods 2 and 4 is probed on a
# stratified subset; methods 2 and 4 then run on all unique fixes at their
# respective `N_SAMPLES_FINAL_*` counts.
#
# Prints to stdout and writes a timestamped log (+ plots) under `dev/logs/`.

# %%
from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.boolean_functions import assignment_to_dict
from boolmore.evaluation.nav_score import get_NAV_scores
from boolmore.evaluation.score import get_model_score
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.prediction import _nav_from_ldoi, _nav_from_min_traps
from boolmore.io.nav import import_NAV_exps
from boolmore.io.load import load_primes
from boolmore.io.paths import project_root
from boolmore.types import Assignment

# Sibling helpers are not on the editable install; keep ``dev/`` importable.
_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from nav_helpers import (
    Tee,
    analyze_fixes,
    format_fixes,
    nav_from_sampled_sources,
    nav_vector_metrics,
    n_unfixed_sources,
    sample_size_grid,
    sample_source_indices,
    source_nodes_from_primes,
    timed,
    timing_summary_line,
    to_nav_predictions,
)

# %% [markdown]
# ## Config

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "breast_cancer"
PKL = (
    CASE
    / "20261015_164622"
    / "BreastCancer_primes_parsimonious_20261015_164622_2404_gen57.pkl"
)
DATA = CASE / "breast_cancer_data_trial.csv"
LOG_DIR = Path(__file__).resolve().parent / "logs"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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

# Skip full min-traps when unfixed sources exceed this (2**8 = 256 lower bound).
MAX_UNFIXED_M1 = 9

# --- Sampling knobs (methods 2 and 4 are independent) ---
#
# SAMPLE_SIZES_M*: grid used only in the *stability* study (convergence curves).
#   Sizes > 2**x are dropped. If max(grid) >= 2**x, 2**x is included automatically
#   — that last point *is* full enumeration (no separate FULL_ENUM_MAX knob).
#   Δfull compares earlier prefixes to that full-enum NAV.
#
# N_SAMPLES_FINAL_M*: how many source combinations to average when running
#   method 2 / 4 on *all* unique fixes for the main comparison & scores.
#   Per fix: n = min(N_SAMPLES_FINAL_M*, 2**x).
#
SAMPLE_SIZES_M2 = [8, 16, 32, 64, 128]
SAMPLE_SIZES_M4 = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_SAMPLES_FINAL_M2 = 32
N_SAMPLES_FINAL_M4 = 2048

SEED = 0
# Set to an int to limit unique fixes (for quick iteration); None = all.
MAX_FIXES: int | None = None
# How many fixes per unfixed-source bin for the sampling-stability study.
N_STABILITY_PER_BIN = 2
STABILITY_BINS = [(6, 8), (9, 12), (13, 20)]
STABILITY_DELTA_TOL = 0.01
HIERARCHY = True

# Optional: `SMOKE=1` for a tiny end-to-end check.
if os.environ.get("SMOKE") == "1":
    MAX_FIXES = 3
    SAMPLE_SIZES_M2 = [2, 4]
    SAMPLE_SIZES_M4 = [2, 4, 8]
    N_SAMPLES_FINAL_M2 = 4
    N_SAMPLES_FINAL_M4 = 8
    N_STABILITY_PER_BIN = 1
    STABILITY_BINS = [(0, 24)]
    MAX_UNFIXED_M1 = 6

# %% [markdown]
# ## Logging

# %%
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"compare_nav_4methods_{RUN_ID}.log"
PLOT_CONV = LOG_DIR / f"compare_nav_4methods_convergence_{RUN_ID}.png"
PLOT_TIME = LOG_DIR / f"compare_nav_4methods_timing_{RUN_ID}.png"
PLOT_DIFF = LOG_DIR / f"compare_nav_4methods_diff_heatmap_{RUN_ID}.png"

_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model: {PKL}")
print(f"data:  {DATA}")
print(f"MAX_UNFIXED_M1={MAX_UNFIXED_M1}  SEED={SEED}")
print(f"  SAMPLE_SIZES_M2 (stability) = {SAMPLE_SIZES_M2}")
print(f"  SAMPLE_SIZES_M4 (stability) = {SAMPLE_SIZES_M4}")
print(f"  N_SAMPLES_FINAL_M2 (all fixes) = {N_SAMPLES_FINAL_M2}")
print(f"  N_SAMPLES_FINAL_M4 (all fixes) = {N_SAMPLES_FINAL_M4}")
print(
    f"  full enum in stability when 2**x <= max(grid) "
    f"(m2≤{max(SAMPLE_SIZES_M2)}, m4≤{max(SAMPLE_SIZES_M4)})"
)
print()

# %% [markdown]
# ## Load model and experiments

# %%
primes = load_primes(PKL)
experiments = import_NAV_exps(
    DATA,
    id_col=ID_COL,
    weight_col=WEIGHT_COL,
    observed_col=OBSERVED_COL,
    outcome_col=OUTCOME_COL,
    fixes_cols=FIXES_COLS,
)

source_nodes = source_nodes_from_primes(primes)
source_set = set(source_nodes)

_all_unique = list({exp.fixes for exp in experiments})
# Prefer fewer free sources first (method 1 cheaper / more likely eligible).
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

print(f"Model nodes : {len(primes)}")
print(f"Source nodes: {len(source_nodes)}")
print(f"  {source_nodes}")
print(f"Experiments : {len(experiments)}")
print(f"Unique fixes: {len(unique_fixes)}" + (f" (capped by MAX_FIXES)" if MAX_FIXES else ""))
print()

ensure_asp_backend()

# %% [markdown]
# ## Unfixed-source analysis and method-1 eligibility

# %%
meta_rows = [
    analyze_fixes(fixes, source_set, max_unfixed_m1=MAX_UNFIXED_M1)
    for fixes in unique_fixes
]
meta_df = pd.DataFrame(meta_rows)
meta_df["exp_ids"] = [ids_by_fixes[f] for f in meta_df["fixes"]]
meta_df["n_exps"] = meta_df["exp_ids"].map(len)

hist = Counter(meta_df["n_unfixed_sources"])
hist_df = pd.DataFrame(
    [
        {
            "n_unfixed_sources (x)": x,
            "n_distinct_fixes": hist[x],
            "min_trap_lower_bound (2**x)": 2**x,
            "method1_eligible": x <= MAX_UNFIXED_M1,
        }
        for x in sorted(hist)
    ]
)
print("Unfixed source histogram:")
print(hist_df.to_string(index=False))
print()
n_eligible = int(meta_df["run_method1"].sum())
print(
    f"Method 1 eligible: {n_eligible}/{len(meta_df)} "
    f"(MAX_UNFIXED_M1={MAX_UNFIXED_M1})"
)
print(
    f"Unfixed sources — min {meta_df['n_unfixed_sources'].min()}, "
    f"median {meta_df['n_unfixed_sources'].median():.0f}, "
    f"max {meta_df['n_unfixed_sources'].max()}"
)
print()

display_cols = [
    "n_exps",
    "n_fixed_sources",
    "n_unfixed_sources",
    "min_trap_lower_bound",
    "run_method1",
]
print(meta_df[display_cols].sort_values(
    ["n_unfixed_sources", "n_exps"], ascending=[False, False]
).head(20).to_string())
print()

meta_by_fixes = {row["fixes"]: row for row in meta_rows}

# %% [markdown]
# ## Method 1 — full min-traps (eligible fixes only)

# %%
preds_m1: dict[Assignment, dict[str, float]] = {}
times_m1: dict[Assignment, float] = {}

eligible_fixes = [r["fixes"] for r in meta_rows if r["run_method1"]]
print(f"Running method 1 on {len(eligible_fixes)} fixes...")
for i, fixes in enumerate(eligible_fixes, 1):
    pert = assignment_to_dict(fixes)
    nav, dt = timed(_nav_from_min_traps, primes, pert)
    preds_m1[fixes] = nav
    times_m1[fixes] = dt
    x = meta_by_fixes[fixes]["n_unfixed_sources"]
    print(
        f"  [{i}/{len(eligible_fixes)}] x={x}  "
        f"t={dt:.3f}s  {format_fixes(fixes)}"
    )

print(
    f"Method 1 done: n={len(preds_m1)}, "
    f"total={sum(times_m1.values()):.3f}s, "
    f"mean={np.mean(list(times_m1.values())) if times_m1 else 0:.3f}s"
)
print()

# %% [markdown]
# ## Method 3 — LDOI on fixes (all)

# %%
preds_m3: dict[Assignment, dict[str, float]] = {}
times_m3: dict[Assignment, float] = {}

print(f"Running method 3 (LDOI) on {len(unique_fixes)} fixes...")
for i, fixes in enumerate(unique_fixes, 1):
    pert = assignment_to_dict(fixes)
    nav, dt = timed(_nav_from_ldoi, primes, pert)
    preds_m3[fixes] = nav
    times_m3[fixes] = dt
    if i % 25 == 0 or i == len(unique_fixes):
        print(f"  [{i}/{len(unique_fixes)}] cumulative={sum(times_m3.values()):.3f}s")

print(
    f"Method 3 done: n={len(preds_m3)}, "
    f"total={sum(times_m3.values()):.3f}s, "
    f"mean={np.mean(list(times_m3.values())):.4f}s"
)
print()

# %% [markdown]
# ## Sampling-stability study (methods 2 and 4)
#
# Stratified subset across unfixed-source bins. Sample-size grids differ by
# method. When `2**x <= max(SAMPLE_SIZES_M*)`, the grid includes `2**x` and
# that last point is full enumeration; Δfull is vs that NAV (no extra knob).

# %%
rng_stab = random.Random(SEED)

method_grids = {
    "m2": ("min_traps", SAMPLE_SIZES_M2),
    "m4": ("ldoi", SAMPLE_SIZES_M4),
}
# A method can fully enumerate a fix when 2**x fits under its stability grid.
FULL_ENUM_REACH_M2 = max(SAMPLE_SIZES_M2) if SAMPLE_SIZES_M2 else 0
FULL_ENUM_REACH_M4 = max(SAMPLE_SIZES_M4) if SAMPLE_SIZES_M4 else 0
FULL_ENUM_REACH = max(FULL_ENUM_REACH_M2, FULL_ENUM_REACH_M4)


def pick_stability_fixes(meta_df: pd.DataFrame) -> list[Assignment]:
    chosen: list[Assignment] = []
    seen: set[Assignment] = set()

    def take_from(subset: pd.DataFrame, n: int) -> None:
        taken = 0
        for _, row in subset.iterrows():
            f = row["fixes"]
            if f in seen:
                continue
            chosen.append(f)
            seen.add(f)
            taken += 1
            if taken >= n:
                break

    # Prefer a few fixes whose space fits in at least one method's grid
    # (so Δfull is available), then stratified bins.
    full_enum = meta_df[
        meta_df["min_trap_lower_bound"] <= FULL_ENUM_REACH
    ].sort_values(["n_unfixed_sources", "n_exps"], ascending=[True, False])
    take_from(full_enum, N_STABILITY_PER_BIN)

    for lo, hi in STABILITY_BINS:
        subset = meta_df[
            (meta_df["n_unfixed_sources"] >= lo)
            & (meta_df["n_unfixed_sources"] <= hi)
        ].sort_values(["n_unfixed_sources", "n_exps"], ascending=[True, False])
        take_from(subset, N_STABILITY_PER_BIN)

    return chosen


stability_fixes = pick_stability_fixes(meta_df)
print(f"Stability study fixes: {len(stability_fixes)}")
for f in stability_fixes:
    m = meta_by_fixes[f]
    x = m["n_unfixed_sources"]
    reach = []
    if 2**x <= FULL_ENUM_REACH_M2:
        reach.append("m2")
    if 2**x <= FULL_ENUM_REACH_M4:
        reach.append("m4")
    print(
        f"  x={x:2d}  2^x={m['min_trap_lower_bound']:7d}  "
        f"full_enum={','.join(reach) or 'no'}  "
        f"exps={len(ids_by_fixes[f]):3d}  {format_fixes(f)}"
    )
print()

stability_rows: list[dict] = []
# NAV at n=2**x when that point is on the grid (used for Δfull / plots)
full_truth: dict[str, dict[Assignment, dict[str, float]]] = {
    "m2": {},
    "m4": {},
}

for fixes in stability_fixes:
    m = meta_by_fixes[fixes]
    unfixed = m["unfixed_sources"]
    x = m["n_unfixed_sources"]
    fixed_src = {
        n: v for n, v in assignment_to_dict(fixes).items() if n in source_set
    }
    sizes_by_method = {
        label: sample_size_grid(grid, x) for label, (_, grid) in method_grids.items()
    }
    max_n = max(
        (max(sz) for sz in sizes_by_method.values() if sz),
        default=0,
    )
    order = sample_source_indices(x, max_n, rng_stab) if max_n else []

    print(
        f"Stability: x={x} (2^x={2**x})  "
        f"m2_sizes={sizes_by_method['m2']}  "
        f"m4_sizes={sizes_by_method['m4']}  "
        f"{format_fixes(fixes)}"
    )

    for label, (nav_method, _) in method_grids.items():
        sizes = sizes_by_method[label]
        if not sizes:
            print(f"  {label}: no sample sizes <= 2^{x}={2**x}, skip")
            continue
        prev = None
        nav_by_n: dict[int, dict[str, float]] = {}
        for n in sizes:
            idxs = order[:n]
            t0 = perf_counter()
            nav = nav_from_sampled_sources(
                primes, fixes, unfixed, fixed_src, idxs, nav_method
            )
            dt = perf_counter() - t0
            nav_by_n[n] = nav
            row: dict = {
                "fixes": fixes,
                "method": label,
                "n_unfixed": x,
                "n_samples": n,
                "t": dt,
            }
            if prev is not None:
                row.update(
                    {f"vs_prev_{k}": v for k, v in nav_vector_metrics(prev, nav).items()}
                )
            stability_rows.append(row)
            prev = nav
            msg = f"  {label} n={n:4d}/{2**x}  t={dt:.3f}s"
            if "vs_prev_max_abs" in row:
                msg += f"  Δprev={row['vs_prev_max_abs']:.4f}"
            if n == 2**x:
                msg += "  [full enum]"
            print(msg)

        # Δfull: earlier sizes vs the full-enum grid point (same shuffled order)
        if 2**x in nav_by_n:
            full_truth[label][fixes] = nav_by_n[2**x]
            for row in stability_rows:
                if row["fixes"] != fixes or row["method"] != label:
                    continue
                if row["n_samples"] == 2**x:
                    continue
                d = nav_vector_metrics(nav_by_n[2**x], nav_by_n[row["n_samples"]])
                row.update({f"vs_full_{k}": v for k, v in d.items()})
            print(
                f"  {label} Δfull vs n={2**x}: "
                + ", ".join(
                    f"n={r['n_samples']}→{r['vs_full_max_abs']:.4f}"
                    for r in stability_rows
                    if r["fixes"] == fixes
                    and r["method"] == label
                    and "vs_full_max_abs" in r
                )
            )

stability_df = pd.DataFrame(stability_rows)
print()

print(f"First n_samples where max|Δ| vs previous size < {STABILITY_DELTA_TOL}:")
for fixes in stability_fixes:
    x = meta_by_fixes[fixes]["n_unfixed_sources"]
    for label in ("m2", "m4"):
        sub = stability_df[
            (stability_df["fixes"] == fixes) & (stability_df["method"] == label)
        ].sort_values("n_samples")
        stable_n = None
        for _, row in sub.iterrows():
            if "vs_prev_max_abs" not in row or pd.isna(row.get("vs_prev_max_abs")):
                continue
            if row["vs_prev_max_abs"] < STABILITY_DELTA_TOL:
                stable_n = int(row["n_samples"])
                break
        print(
            f"  x={x:2d} (2^x={2**x}) {label}: "
            f"{'stable by n=' + str(stable_n) if stable_n else 'not within grid'}  "
            f"{format_fixes(fixes, 40)}"
        )
print()

# %% [markdown]
# ### Convergence plots
#
# Each series is labeled `x=… (≤2^x=…)`. Points never exceed that fix's `2^x`.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, label, title in (
    (axes[0], "m2", "Method 2 (min_traps + sample)"),
    (axes[1], "m4", "Method 4 (ldoi + sample)"),
):
    sub_m = stability_df[stability_df["method"] == label]
    for fixes in stability_fixes:
        sub = sub_m[sub_m["fixes"] == fixes].dropna(subset=["vs_prev_max_abs"])
        if sub.empty:
            continue
        x = meta_by_fixes[fixes]["n_unfixed_sources"]
        assert sub["n_samples"].max() <= 2**x
        ax.plot(
            sub["n_samples"],
            sub["vs_prev_max_abs"],
            marker="o",
            label=f"x={x} (≤{2**x})",
        )
    ax.axhline(STABILITY_DELTA_TOL, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("n_samples (capped at 2^x)")
    ax.set_ylabel("max |Δ| vs previous size")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
fig.suptitle("Sampling convergence (max abs NAV change vs previous sample size)")
fig.tight_layout()
fig.savefig(PLOT_CONV, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_CONV}")

if any(full_truth[lab] for lab in ("m2", "m4")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, label, title in (
        (axes[0], "m2", "Method 2 vs full enum"),
        (axes[1], "m4", "Method 4 vs full enum"),
    ):
        sub_m = stability_df[stability_df["method"] == label]
        plotted = False
        for fixes in stability_fixes:
            if fixes not in full_truth[label]:
                continue
            sub = sub_m[sub_m["fixes"] == fixes].dropna(subset=["vs_full_max_abs"])
            if sub.empty:
                continue
            x = meta_by_fixes[fixes]["n_unfixed_sources"]
            ax.plot(
                sub["n_samples"],
                sub["vs_full_max_abs"],
                marker="o",
                label=f"x={x} (≤{2**x})",
            )
            plotted = True
        ax.axhline(STABILITY_DELTA_TOL, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("n_samples (capped at 2^x)")
        ax.set_ylabel("max |Δ| vs full enumeration")
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        if plotted:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Approximation error vs full source enumeration")
    fig.tight_layout()
    plot_full = LOG_DIR / f"compare_nav_4methods_vs_full_{RUN_ID}.png"
    fig.savefig(plot_full, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_full}")
print()

# %% [markdown]
# ## Methods 2 and 4 on all fixes
#
# Uses `N_SAMPLES_FINAL_M2` / `N_SAMPLES_FINAL_M4` (each capped at `2**x`).

# %%
rng_final = random.Random(SEED + 1)
preds_m2: dict[Assignment, dict[str, float]] = {}
preds_m4: dict[Assignment, dict[str, float]] = {}
times_m2: dict[Assignment, float] = {}
times_m4: dict[Assignment, float] = {}
n_used_m2: dict[Assignment, int] = {}
n_used_m4: dict[Assignment, int] = {}

print(
    f"Running methods 2 & 4 on {len(unique_fixes)} fixes "
    f"(m2: min({N_SAMPLES_FINAL_M2}, 2**x); "
    f"m4: min({N_SAMPLES_FINAL_M4}, 2**x))..."
)
for i, fixes in enumerate(unique_fixes, 1):
    m = meta_by_fixes[fixes]
    unfixed = m["unfixed_sources"]
    x = m["n_unfixed_sources"]
    fixed_src = {
        n: v for n, v in assignment_to_dict(fixes).items() if n in source_set
    }
    n2 = min(N_SAMPLES_FINAL_M2, 2**x)
    n4 = min(N_SAMPLES_FINAL_M4, 2**x)
    # Shared RNG stream: draw the larger set once, reuse prefix for the smaller.
    n_max = max(n2, n4)
    idxs_max = sample_source_indices(x, n_max, rng_final)
    idxs2 = idxs_max[:n2]
    idxs4 = idxs_max[:n4]
    n_used_m2[fixes] = len(idxs2)
    n_used_m4[fixes] = len(idxs4)

    t0 = perf_counter()
    preds_m2[fixes] = nav_from_sampled_sources(
        primes, fixes, unfixed, fixed_src, idxs2, "min_traps"
    )
    times_m2[fixes] = perf_counter() - t0

    t0 = perf_counter()
    preds_m4[fixes] = nav_from_sampled_sources(
        primes, fixes, unfixed, fixed_src, idxs4, "ldoi"
    )
    times_m4[fixes] = perf_counter() - t0

    if i % 10 == 0 or i == len(unique_fixes):
        print(
            f"  [{i}/{len(unique_fixes)}] "
            f"cum m2={sum(times_m2.values()):.1f}s  "
            f"m4={sum(times_m4.values()):.1f}s"
        )

print(
    f"Method 2 done: total={sum(times_m2.values()):.3f}s  "
    f"mean={np.mean(list(times_m2.values())):.3f}s"
)
print(
    f"Method 4 done: total={sum(times_m4.values()):.3f}s  "
    f"mean={np.mean(list(times_m4.values())):.4f}s"
)
print()

# %% [markdown]
# ## Cross-method comparison

# %%
method_preds = {
    "m1": preds_m1,
    "m2": preds_m2,
    "m3": preds_m3,
    "m4": preds_m4,
}
method_times = {
    "m1": times_m1,
    "m2": times_m2,
    "m3": times_m3,
    "m4": times_m4,
}
pairs = [("m1", "m2"), ("m1", "m3"), ("m1", "m4"), ("m2", "m3"), ("m2", "m4"), ("m3", "m4")]

cmp_rows: list[dict] = []
for fixes in unique_fixes:
    m = meta_by_fixes[fixes]
    row: dict = {
        "fixes": fixes,
        "n_unfixed": m["n_unfixed_sources"],
        "n_samples_m2": n_used_m2[fixes],
        "n_samples_m4": n_used_m4[fixes],
        "run_m1": m["run_method1"],
        "t_m1": times_m1.get(fixes, math.nan),
        "t_m2": times_m2[fixes],
        "t_m3": times_m3[fixes],
        "t_m4": times_m4[fixes],
    }
    for a, b in pairs:
        if fixes not in method_preds[a] or fixes not in method_preds[b]:
            continue
        all_m = nav_vector_metrics(method_preds[a][fixes], method_preds[b][fixes])
        row.update({f"{a}_vs_{b}_{k}": v for k, v in all_m.items()})
    cmp_rows.append(row)

cmp_df = pd.DataFrame(cmp_rows)

print("=== Timing summary (all unique fixes where each method ran) ===")
for label, tdict in method_times.items():
    print(timing_summary_line(label, list(tdict.values())))
print()

# Apples-to-apples: only the fixes where method 1 succeeded
m1_fixes = list(preds_m1.keys())
print(
    f"=== Timing summary (method-1-eligible only, n={len(m1_fixes)}) ==="
)
for label, tdict in method_times.items():
    vals = [tdict[f] for f in m1_fixes if f in tdict]
    print(timing_summary_line(label, vals))
if m1_fixes and times_m1:
    t1 = sum(times_m1[f] for f in m1_fixes)
    t2 = sum(times_m2[f] for f in m1_fixes)
    t3 = sum(times_m3[f] for f in m1_fixes)
    t4 = sum(times_m4[f] for f in m1_fixes)
    print(
        f"  ratios vs m1 on same {len(m1_fixes)} fixes: "
        f"m2/m1={t2 / t1:.2f}x  m3/m1={t3 / t1:.3f}x  m4/m1={t4 / t1:.2f}x"
    )
print()

print("=== Pairwise NAV agreement (all nodes, mean over fixes with both) ===")
for a, b in pairs:
    col = f"{a}_vs_{b}_rmse"
    if col not in cmp_df.columns:
        continue
    sub = cmp_df.dropna(subset=[col])
    if sub.empty:
        print(f"  {a} vs {b}: no overlap")
        continue
    print(
        f"  {a} vs {b}: n={len(sub)}  "
        f"RMSE mean={sub[col].mean():.4f}  "
        f"MAE mean={sub[f'{a}_vs_{b}_mae'].mean():.4f}  "
        f"max|Δ| mean={sub[f'{a}_vs_{b}_max_abs'].mean():.4f}"
    )
print()

# Breakdown by unfixed-source bins
print("=== RMSE by unfixed-source bin (m2 vs m3, m3 vs m4, m2 vs m4) ===")
bin_edges = [0, 8, 12, 16, 24]
cmp_df["x_bin"] = pd.cut(cmp_df["n_unfixed"], bins=bin_edges, right=True)
for col in ("m2_vs_m3_rmse", "m3_vs_m4_rmse", "m2_vs_m4_rmse"):
    if col not in cmp_df.columns:
        continue
    print(f"  {col}:")
    print(
        cmp_df.groupby("x_bin", observed=True)[col]
        .agg(["count", "mean", "median"])
        .to_string()
    )
    print()

# %% [markdown]
# ### Timing and difference plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
method_order = ("m1", "m2", "m3", "m4")

# Left: all fixes each method covered
ax = axes[0]
totals_all = [
    sum(method_times[lab].values()) if method_times[lab] else 0.0
    for lab in method_order
]
ax.bar(method_order, totals_all, color=colors)
ax.set_ylabel("Total wall time (s)")
ax.set_title("All fixes each method ran")
for i, v in enumerate(totals_all):
    ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)

# Right: same 80 (or whatever) method-1-eligible fixes
ax = axes[1]
totals_m1set = [
    sum(method_times[lab][f] for f in m1_fixes if f in method_times[lab])
    for lab in method_order
]
ax.bar(method_order, totals_m1set, color=colors)
ax.set_ylabel("Total wall time (s)")
ax.set_title(f"Method-1-eligible only (n={len(m1_fixes)})")
for i, v in enumerate(totals_m1set):
    ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)

fig.suptitle("Prediction time by method")
fig.tight_layout()
fig.savefig(PLOT_TIME, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_TIME}")

# Heatmap-like summary: mean RMSE per pair
pair_labels = []
pair_rmses = []
for a, b in pairs:
    col = f"{a}_vs_{b}_rmse"
    if col in cmp_df.columns and cmp_df[col].notna().any():
        pair_labels.append(f"{a}/{b}")
        pair_rmses.append(float(cmp_df[col].mean()))

if pair_rmses:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ypos = np.arange(len(pair_labels))
    ax.barh(ypos, pair_rmses, color="#4C72B0")
    ax.set_yticks(ypos)
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("Mean RMSE (all nodes)")
    ax.set_title("Cross-method NAV difference")
    fig.tight_layout()
    fig.savefig(PLOT_DIFF, dpi=120)
    plt.close(fig)
    print(f"Wrote {PLOT_DIFF}")
print()

# %% [markdown]
# ## Optional model scores
#
# Score methods that cover **all** interventions in the loaded experiment table
# (needed for comparative outcomes that cross-reference experiment ids).
# Method 1 only if every fix was eligible. Skipped when `MAX_FIXES` truncates.

# %%
all_data_fixes = {exp.fixes for exp in experiments}
can_score = MAX_FIXES is None and set(unique_fixes) == all_data_fixes

if not can_score:
    print(
        "Skipping model scores "
        f"(need all fixes; MAX_FIXES={MAX_FIXES}, "
        f"predicted={len(unique_fixes)}, data={len(all_data_fixes)})"
    )
else:
    score_methods: list[tuple[str, dict[Assignment, dict[str, float]]]] = [
        ("m2_sample", preds_m2),
        ("m3_ldoi", preds_m3),
        ("m4_ldoi_sample", preds_m4),
    ]
    if len(preds_m1) == len(unique_fixes):
        score_methods.insert(0, ("m1_min_traps", preds_m1))
    else:
        print(
            f"Skipping full-model score for m1 "
            f"(covered {len(preds_m1)}/{len(unique_fixes)} fixes)"
        )

    print("Model scores (hierarchy={}):".format(HIERARCHY))
    for name, preds in score_methods:
        if len(preds) < len(unique_fixes):
            print(f"  {name}: incomplete predictions, skip")
            continue
        items = get_NAV_scores(
            experiments,
            to_nav_predictions(preds),
            default_sources={},
            hierarchy=HIERARCHY,
        )
        max_score, score = get_model_score(items)
        print(f"  {name}: {score:.4f} / {max_score:.4f}")
print()

# %% [markdown]
# ## Summary

# %%
print("=== Summary ===")
print(f"  Unique fixes: {len(unique_fixes)}")
print(
    f"  Method 1 (min_traps): ran on {len(preds_m1)} "
    f"(skipped {len(unique_fixes) - len(preds_m1)} with x>{MAX_UNFIXED_M1})"
)
print(
    f"  Method 2 (min_traps+sample): "
    f"N_SAMPLES_FINAL_M2={N_SAMPLES_FINAL_M2}"
)
print(f"  Method 3 (ldoi): all fixes")
print(
    f"  Method 4 (ldoi+sample): "
    f"N_SAMPLES_FINAL_M4={N_SAMPLES_FINAL_M4}"
)
print(
    f"  Total times all fixes (s): "
    f"m1={sum(times_m1.values()):.2f}  "
    f"m2={sum(times_m2.values()):.2f}  "
    f"m3={sum(times_m3.values()):.2f}  "
    f"m4={sum(times_m4.values()):.2f}"
)
if m1_fixes:
    print(
        f"  Total times m1-eligible n={len(m1_fixes)} (s): "
        f"m1={sum(times_m1[f] for f in m1_fixes):.2f}  "
        f"m2={sum(times_m2[f] for f in m1_fixes):.2f}  "
        f"m3={sum(times_m3[f] for f in m1_fixes):.2f}  "
        f"m4={sum(times_m4[f] for f in m1_fixes):.2f}"
    )
if "m2_vs_m3_rmse" in cmp_df.columns:
    print(
        f"  Mean RMSE m2 vs m3 (sample min_traps vs plain ldoi): "
        f"{cmp_df['m2_vs_m3_rmse'].mean():.4f}"
    )
if "m3_vs_m4_rmse" in cmp_df.columns:
    print(
        f"  Mean RMSE m3 vs m4 (ldoi vs sampled ldoi): "
        f"{cmp_df['m3_vs_m4_rmse'].mean():.4f}"
    )
if "m1_vs_m2_rmse" in cmp_df.columns and cmp_df["m1_vs_m2_rmse"].notna().any():
    print(
        f"  Mean RMSE m1 vs m2 (eligible only): "
        f"{cmp_df['m1_vs_m2_rmse'].mean():.4f}"
    )
if "m1_vs_m3_rmse" in cmp_df.columns and cmp_df["m1_vs_m3_rmse"].notna().any():
    print(
        f"  Mean RMSE m1 vs m3 (eligible only): "
        f"{cmp_df['m1_vs_m3_rmse'].mean():.4f}"
    )
print(f"  Log: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
