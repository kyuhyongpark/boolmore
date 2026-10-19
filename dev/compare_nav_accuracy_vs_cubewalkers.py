# %% [markdown]
# NAV accuracy vs cubewalkers ground truth
#
# Compare three approximate NAV predictors against cubewalkers simulation NAV
# on the breast-cancer gen108 model and trial data.
#
# **Methods** (per unique intervention `fixes`):
#
# 1. **m1 — cubewalkers GT** — clamp interventions, sample walkers,
#    async ensemble, average node values over a late time window
# 2. **m2 — min_traps + sample sources** — sample up to 32 free-source
#    combinations → minimal trap spaces NAV → equal-weight average
# 3. **m3 — ldoi + sample sources** — sample up to 1024 free-source
#    combinations → LDOI NAV → equal-weight average
# 4. **m4 — ldoi** — LDOI on interventions only (no source sampling)
#
# **Bootstrap** (per fix, then discarded — peak memory stays per-fix):
#
# - m1: walker resample → SE and boot replicates
# - m2 / m3: resample over drawn source-sample NAVs
# - m4: deterministic (SE = 0; SE(Δ) = SE(m1))
# - Δ = method − m1 via independent two-sample bootstrap
#   (`std(boot_method − boot_m1)`)
#
# **Accuracy metrics** (m2 / m3 / m4 vs m1):
#
# 1. Per-node NAV (RMSE / MAE / max|Δ| over all model nodes)
# 2. Full-entity NAV (merged sibling means + singleton nodes)
# 3. Hierarchy score closeness to m1 (`Δscore`, `Δ(score/max)`)
# 4. Pooled (fix × bootstrap) Δ clouds → separate node-by-node and
#    entity-by-entity box plots + per-item / global |Δ| / RMSE / bias stats
#
# Prints to stdout and writes a log (+ plots) under `dev/logs/` (overwritten
# each full run). ``SMOKE=1`` writes separate ``*_smoke`` artifacts so it does
# not clobber full-run outputs.
#
# ---
#
# ## WSL CUDA setup (required for method 1)
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
from boolmore.inference.prediction import _nav_from_ldoi
from boolmore.io.load import load_primes
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
    diff_from_boot,
    require_cubewalkers,
    simulate_walker_nav,
)
from nav_helpers import (
    Tee,
    analyze_fixes,
    bootstrap_sample_mean,
    discover_full_entities,
    entity_from_node_array,
    entity_nav_dict,
    format_fixes,
    nav_samples_from_sources,
    nav_vector_metrics,
    n_unfixed_sources,
    pooled_delta_entity_stats,
    sample_source_indices,
    source_nodes_from_primes,
    summarize_across_entities,
    timed,
    timing_summary_line,
    to_nav_predictions,
)

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

N_WALKERS = 2048
N_SAMPLES_M2 = 32
N_SAMPLES_M3 = 1024
# Paper-style horizon; set explicitly below after N is known, or override here.
N_TIME_STEPS: int | None = None  # None → 55*N + 6000
T_WINDOW: int | None = None  # None → max(1000, T // 4)
# Small GPUs (e.g. MX150) hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at (32, 32).
THREADS_PER_BLOCK: tuple[int, int] = DEFAULT_THREADS_PER_BLOCK

N_BOOT = DEFAULT_N_BOOT
BOOT_CHUNK = DEFAULT_BOOT_CHUNK

SEED = 0
MAX_FIXES: int | None = None
HIERARCHY = True

# Optional: `SMOKE=1` for a tiny end-to-end check (separate output filenames).
SMOKE = os.environ.get("SMOKE") == "1"
OUT_TAG = "_smoke" if SMOKE else ""
if SMOKE:
    MAX_FIXES = 2
    N_WALKERS = 64
    N_SAMPLES_M2 = 4
    N_SAMPLES_M3 = 8
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
LOG_PATH = LOG_DIR / f"compare_nav_accuracy_vs_cw{OUT_TAG}.log"
PLOT_TIME = LOG_DIR / f"compare_nav_accuracy_vs_cw_timing{OUT_TAG}.png"
PLOT_RMSE = LOG_DIR / f"compare_nav_accuracy_vs_cw_rmse{OUT_TAG}.png"
PLOT_SCORE = LOG_DIR / f"compare_nav_accuracy_vs_cw_score_gap{OUT_TAG}.png"
PLOT_DELTA_NODE = {
    lab: LOG_DIR / f"compare_nav_accuracy_vs_cw_delta_box_node_{lab}{OUT_TAG}.png"
    for lab in ("m2", "m3", "m4")
}
PLOT_DELTA_ENTITY = {
    lab: LOG_DIR / f"compare_nav_accuracy_vs_cw_delta_box_entity_{lab}{OUT_TAG}.png"
    for lab in ("m2", "m3", "m4")
}

_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model pkl:  {PKL}")
print(f"model bnet: {BNET}")
print(f"data:       {DATA}")
print(f"N_WALKERS={N_WALKERS}  N_SAMPLES_M2={N_SAMPLES_M2}  N_SAMPLES_M3={N_SAMPLES_M3}")
print(f"N_BOOT={N_BOOT}  BOOT_CHUNK={BOOT_CHUNK}")
print(f"THREADS_PER_BLOCK={THREADS_PER_BLOCK}")
print(f"SEED={SEED}  MAX_FIXES={MAX_FIXES}  HIERARCHY={HIERARCHY}  SMOKE={SMOKE}")
print()

# %% [markdown]
# ## Load model and experiments

# %%
primes = load_primes(PKL)
BNET_RULES = bnet_rules_for_cubewalkers(BNET)
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
node_names = sorted(primes)

_n_nodes = len(primes)
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

print(f"Model nodes : {len(primes)}")
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

# %% [markdown]
# ## Per-fix methods + bootstrap (m1 / m2 / m3 / m4)
#
# One loop keeps peak memory per-fix: walker arrays and source-sample
# matrices are discarded after each intervention's Δ cloud block is stored.

# %%
rng_sources = random.Random(SEED)
rng_boot = np.random.default_rng(SEED + 1)

preds_m1: dict[Assignment, dict[str, float]] = {}
preds_m2: dict[Assignment, dict[str, float]] = {}
preds_m3: dict[Assignment, dict[str, float]] = {}
preds_m4: dict[Assignment, dict[str, float]] = {}
times_m1: dict[Assignment, float] = {}
times_m2: dict[Assignment, float] = {}
times_m3: dict[Assignment, float] = {}
times_m4: dict[Assignment, float] = {}
n_used_m2: dict[Assignment, int] = {}
n_used_m3: dict[Assignment, int] = {}

approx_methods = ("m2", "m3", "m4")
delta_blocks_node: dict[str, list[np.ndarray]] = {lab: [] for lab in approx_methods}
delta_blocks_ent: dict[str, list[np.ndarray]] = {lab: [] for lab in approx_methods}
se_rows: list[dict] = []

print(
    f"Running m1–m4 + bootstrap on {len(unique_fixes)} fixes "
    f"(N_BOOT={N_BOOT}, BOOT_CHUNK={BOOT_CHUNK})..."
)
for i, fixes in enumerate(unique_fixes, 1):
    m = meta_by_fixes[fixes]
    unfixed = m["unfixed_sources"]
    x = m["n_unfixed_sources"]
    fixed_src = {
        n: v for n, v in assignment_to_dict(fixes).items() if n in source_set
    }

    # --- m1: cubewalkers walkers + bootstrap ---
    walker_nav, varnames, timings_m1 = simulate_walker_nav(
        BNET_RULES,
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
    se_m1_ent = boot_m1_ent.std(axis=1, ddof=1)

    # --- m2 / m3: source samples + bootstrap ---
    n2 = min(N_SAMPLES_M2, 2**x)
    n3 = min(N_SAMPLES_M3, 2**x)
    n_max = max(n2, n3)
    idxs_max = sample_source_indices(x, n_max, rng_sources)
    idxs2 = idxs_max[:n2]
    idxs3 = idxs_max[:n3]
    n_used_m2[fixes] = len(idxs2)
    n_used_m3[fixes] = len(idxs3)

    t0 = perf_counter()
    samples_m2 = nav_samples_from_sources(
        primes, fixes, unfixed, fixed_src, idxs2, "min_traps", node_names
    )
    means_m2, _se_m2, boot_m2, _t_boot_m2 = bootstrap_sample_mean(
        samples_m2, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng_boot
    )
    del samples_m2
    times_m2[fixes] = perf_counter() - t0
    preds_m2[fixes] = {n: float(means_m2[j]) for j, n in enumerate(node_names)}
    means_m2_ent = entity_from_node_array(
        means_m2, node_names, ENTITY_KEYS, MERGED_SET
    )
    boot_m2_ent = entity_from_node_array(
        boot_m2, node_names, ENTITY_KEYS, MERGED_SET
    )

    t0 = perf_counter()
    samples_m3 = nav_samples_from_sources(
        primes, fixes, unfixed, fixed_src, idxs3, "ldoi", node_names
    )
    means_m3, _se_m3, boot_m3, _t_boot_m3 = bootstrap_sample_mean(
        samples_m3, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng_boot
    )
    del samples_m3
    times_m3[fixes] = perf_counter() - t0
    preds_m3[fixes] = {n: float(means_m3[j]) for j, n in enumerate(node_names)}
    means_m3_ent = entity_from_node_array(
        means_m3, node_names, ENTITY_KEYS, MERGED_SET
    )
    boot_m3_ent = entity_from_node_array(
        boot_m3, node_names, ENTITY_KEYS, MERGED_SET
    )

    # --- m4: deterministic LDOI ---
    pert = assignment_to_dict(fixes)
    nav_m4, dt_m4 = timed(_nav_from_ldoi, primes, pert)
    times_m4[fixes] = dt_m4
    preds_m4[fixes] = {n: float(nav_m4[n]) for n in node_names}
    means_m4 = np.asarray([preds_m4[fixes][n] for n in node_names], dtype=np.float64)
    boot_m4 = np.broadcast_to(means_m4[:, None], (len(node_names), N_BOOT)).copy()
    means_m4_ent = entity_from_node_array(
        means_m4, node_names, ENTITY_KEYS, MERGED_SET
    )
    boot_m4_ent = entity_from_node_array(
        boot_m4, node_names, ENTITY_KEYS, MERGED_SET
    )

    method_node = {
        "m2": (means_m2, boot_m2),
        "m3": (means_m3, boot_m3),
        "m4": (means_m4, boot_m4),
    }
    method_ent = {
        "m2": (means_m2_ent, boot_m2_ent),
        "m3": (means_m3_ent, boot_m3_ent),
        "m4": (means_m4_ent, boot_m4_ent),
    }
    se_row: dict = {
        "fixes": fixes,
        "n_unfixed": x,
        "median_se_m1": float(np.median(se_m1_ent)),
        "mean_se_m1": float(se_m1_ent.mean()),
    }
    for lab in approx_methods:
        means_n, boot_n = method_node[lab]
        means_e, boot_e = method_ent[lab]
        delta_blocks_node[lab].append(boot_n - boot_m1)
        delta_blocks_ent[lab].append(boot_e - boot_m1_ent)
        diff, se_diff = diff_from_boot(
            means_m1_ent, boot_m1_ent, means_e, boot_e
        )
        se_k = boot_e.std(axis=1, ddof=1)
        se_row[f"median_se_{lab}"] = float(np.median(se_k))
        se_row[f"mean_se_{lab}"] = float(se_k.mean())
        se_row[f"median_se_delta_{lab}"] = float(np.median(se_diff))
        se_row[f"mean_se_delta_{lab}"] = float(se_diff.mean())
        se_row[f"mean_abs_delta_{lab}"] = float(np.mean(np.abs(diff)))
    se_rows.append(se_row)

    del (
        means_m1,
        boot_m1,
        means_m1_ent,
        boot_m1_ent,
        se_m1_ent,
        means_m2,
        boot_m2,
        means_m2_ent,
        boot_m2_ent,
        means_m3,
        boot_m3,
        means_m3_ent,
        boot_m3_ent,
        means_m4,
        boot_m4,
        means_m4_ent,
        boot_m4_ent,
        method_node,
        method_ent,
    )

    if i % 10 == 0 or i == len(unique_fixes):
        print(
            f"  [{i}/{len(unique_fixes)}] x={x}  "
            f"cum m1={sum(times_m1.values()):.1f}s  "
            f"m2={sum(times_m2.values()):.1f}s  "
            f"m3={sum(times_m3.values()):.1f}s  "
            f"m4={sum(times_m4.values()):.1f}s  "
            f"{format_fixes(fixes)}"
        )

print(
    f"Methods done: "
    f"m1={sum(times_m1.values()):.3f}s  "
    f"m2={sum(times_m2.values()):.3f}s  "
    f"m3={sum(times_m3.values()):.3f}s  "
    f"m4={sum(times_m4.values()):.3f}s"
)
print()

delta_cloud_node: dict[str, np.ndarray] = {
    lab: np.concatenate(blocks, axis=1) for lab, blocks in delta_blocks_node.items()
}
delta_cloud_ent: dict[str, np.ndarray] = {
    lab: np.concatenate(blocks, axis=1) for lab, blocks in delta_blocks_ent.items()
}
del delta_blocks_node, delta_blocks_ent
se_df = pd.DataFrame(se_rows)

# %% [markdown]
# ## Metric 1 — per-node NAV vs m1

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

cmp_rows: list[dict] = []
for fixes in unique_fixes:
    m = meta_by_fixes[fixes]
    row: dict = {
        "fixes": fixes,
        "n_unfixed": m["n_unfixed_sources"],
        "n_samples_m2": n_used_m2[fixes],
        "n_samples_m3": n_used_m3[fixes],
        "t_m1": times_m1[fixes],
        "t_m2": times_m2[fixes],
        "t_m3": times_m3[fixes],
        "t_m4": times_m4[fixes],
    }
    gt = preds_m1[fixes]
    for lab in approx_methods:
        metrics = nav_vector_metrics(gt, method_preds[lab][fixes], nodes=node_names)
        row.update({f"nav_{lab}_{k}": v for k, v in metrics.items()})

        ent_gt = entity_nav_dict(gt, ENTITY_KEYS, MERGED_SET)
        ent_pred = entity_nav_dict(method_preds[lab][fixes], ENTITY_KEYS, MERGED_SET)
        em = nav_vector_metrics(ent_gt, ent_pred, nodes=ENTITY_KEYS)
        row.update({f"entity_{lab}_{k}": v for k, v in em.items()})
    cmp_rows.append(row)

cmp_df = pd.DataFrame(cmp_rows)

print("=== Timing summary ===")
for label, tdict in method_times.items():
    print(timing_summary_line(label, list(tdict.values())))
print()

print("=== Metric 1: per-node NAV vs m1 (mean over fixes) ===")
for lab in approx_methods:
    print(
        f"  {lab}: "
        f"RMSE={cmp_df[f'nav_{lab}_rmse'].mean():.4f}  "
        f"MAE={cmp_df[f'nav_{lab}_mae'].mean():.4f}  "
        f"max|Δ|={cmp_df[f'nav_{lab}_max_abs'].mean():.4f}"
    )
print()

print("=== Metric 1 by unfixed-source bin ===")
bin_edges = [0, 8, 12, 16, 24, 40]
cmp_df["x_bin"] = pd.cut(cmp_df["n_unfixed"], bins=bin_edges, right=True)
for lab in approx_methods:
    col = f"nav_{lab}_rmse"
    print(f"  {col}:")
    print(
        cmp_df.groupby("x_bin", observed=True)[col]
        .agg(["count", "mean", "median"])
        .to_string()
    )
    print()

# %% [markdown]
# ## Metric 2 — full-entity NAV vs m1

# %%
print("=== Metric 2: full-entity NAV vs m1 (mean over fixes) ===")
print(
    f"  entities ({len(ENTITY_KEYS)}): "
    f"{len(MERGED_BASES)} merged + "
    f"{len(ENTITY_KEYS) - len(MERGED_BASES)} singletons"
)
for lab in approx_methods:
    print(
        f"  {lab}: "
        f"RMSE={cmp_df[f'entity_{lab}_rmse'].mean():.4f}  "
        f"MAE={cmp_df[f'entity_{lab}_mae'].mean():.4f}  "
        f"max|Δ|={cmp_df[f'entity_{lab}_max_abs'].mean():.4f}"
    )
print()

print("=== Bootstrap SE (over entities, then mean over fixes) ===")
for lab in ("m1",) + approx_methods:
    for kind in ("median_se", "mean_se"):
        col = f"{kind}_{lab}"
        if col in se_df.columns:
            print(
                f"  {col}: mean={se_df[col].mean():.5f}  "
                f"median={se_df[col].median():.5f}"
            )
for lab in approx_methods:
    for kind in ("median_se_delta", "mean_se_delta"):
        col = f"{kind}_{lab}"
        print(
            f"  {col}: mean={se_df[col].mean():.5f}  "
            f"median={se_df[col].median():.5f}"
        )
print()

# %% [markdown]
# ## Metric 3 — hierarchy score closeness to m1

# %%
all_data_fixes = {exp.fixes for exp in experiments}
can_score = MAX_FIXES is None and set(unique_fixes) == all_data_fixes

score_rows: list[dict] = []
if not can_score:
    print(
        "Skipping hierarchy scores "
        f"(need all fixes; MAX_FIXES={MAX_FIXES}, "
        f"predicted={len(unique_fixes)}, data={len(all_data_fixes)})"
    )
else:
    print(f"Hierarchy scores (hierarchy={HIERARCHY}), Δ vs m1:")
    score_by_method: dict[str, tuple[float, float]] = {}
    for name, preds in (
        ("m1", preds_m1),
        ("m2", preds_m2),
        ("m3", preds_m3),
        ("m4", preds_m4),
    ):
        items = get_NAV_scores(
            experiments,
            to_nav_predictions(preds),
            default_sources={},
            hierarchy=HIERARCHY,
        )
        max_score, score = get_model_score(items)
        score_by_method[name] = (score, max_score)
        frac = score / max_score if max_score else math.nan
        print(f"  {name}: {score:.4f} / {max_score:.4f}  ({frac:.4f})")

    s1, m1max = score_by_method["m1"]
    frac1 = s1 / m1max if m1max else math.nan
    for lab in approx_methods:
        s, mx = score_by_method[lab]
        frac = s / mx if mx else math.nan
        d_score = s - s1
        d_frac = frac - frac1
        score_rows.append(
            {
                "method": lab,
                "score": s,
                "max_score": mx,
                "frac": frac,
                "delta_score_vs_m1": d_score,
                "delta_frac_vs_m1": d_frac,
            }
        )
        print(
            f"  {lab} vs m1: Δscore={d_score:+.4f}  "
            f"Δ(score/max)={d_frac:+.4f}"
        )
print()

score_df = pd.DataFrame(score_rows)

# %% [markdown]
# ## Bootstrap Δ clouds — per-node / per-entity + global stats

# %%
def _print_pooled_delta_stats(
    *,
    label: str,
    item_name: str,
    cloud: np.ndarray,
    keys: list[str],
) -> tuple[pd.DataFrame, dict]:
    per_item = pooled_delta_entity_stats(cloud)
    df = pd.DataFrame({item_name: keys, **per_item})
    # Primary: median |Δ|; ties → mean |Δ|, max |Δ|, rmse; then name.
    df = df.sort_values(
        by=["median_abs", "mean_abs", "max_abs", "rmse", item_name],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    glob = summarize_across_entities(per_item)
    glob["n_items"] = len(keys)
    glob["n_pool_points"] = int(cloud.shape[1])

    print(f"  {label}  (n_pool={cloud.shape[1]} per {item_name}):")
    print(
        f"    per-{item_name} median_|Δ|: "
        f"mean={glob['median_abs_mean']:.4f}  "
        f"median={glob['median_abs_median']:.4f}  "
        f"max={glob['median_abs_max']:.4f}"
    )
    print(
        f"    per-{item_name} mean_|Δ|:   "
        f"mean={glob['mean_abs_mean']:.4f}  "
        f"median={glob['mean_abs_median']:.4f}  "
        f"max={glob['mean_abs_max']:.4f}"
    )
    print(
        f"    per-{item_name} max_|Δ|:    "
        f"mean={glob['max_abs_mean']:.4f}  "
        f"median={glob['max_abs_median']:.4f}  "
        f"max={glob['max_abs_max']:.4f}"
    )
    print(
        f"    per-{item_name} rmse:      "
        f"mean={glob['rmse_mean']:.4f}  "
        f"median={glob['rmse_median']:.4f}  "
        f"max={glob['rmse_max']:.4f}"
    )
    print(
        f"    per-{item_name} mean_signed Δ: "
        f"mean={glob['mean_signed_mean']:+.4f}  "
        f"median={glob['mean_signed_median']:+.4f}"
    )
    plural = "entities" if item_name == "entity" else f"{item_name}s"
    print(f"    top-5 {plural} by median_|Δ|:")
    for _, r in df.head(5).iterrows():
        print(
            f"      {r[item_name]}: "
            f"median_|Δ|={r['median_abs']:.4f}  "
            f"mean_|Δ|={r['mean_abs']:.4f}  "
            f"max_|Δ|={r['max_abs']:.4f}  "
            f"rmse={r['rmse']:.4f}"
        )
    print()
    return df, glob


node_stat_dfs: dict[str, pd.DataFrame] = {}
entity_stat_dfs: dict[str, pd.DataFrame] = {}
global_node_rows: list[dict] = []
global_ent_rows: list[dict] = []

print("=== Pooled (fix × boot) node Δ = method − m1 ===")
for lab in approx_methods:
    ndf, nglob = _print_pooled_delta_stats(
        label=f"{lab} vs m1 (nodes)",
        item_name="node",
        cloud=delta_cloud_node[lab],
        keys=list(node_names),
    )
    node_stat_dfs[lab] = ndf
    nglob["method"] = lab
    global_node_rows.append(nglob)

print("=== Pooled (fix × boot) entity Δ = method − m1 ===")
for lab in approx_methods:
    edf, eglob = _print_pooled_delta_stats(
        label=f"{lab} vs m1 (entities)",
        item_name="entity",
        cloud=delta_cloud_ent[lab],
        keys=list(ENTITY_KEYS),
    )
    entity_stat_dfs[lab] = edf
    eglob["method"] = lab
    global_ent_rows.append(eglob)

global_node_df = pd.DataFrame(global_node_rows)
global_ent_df = pd.DataFrame(global_ent_rows)

# %% [markdown]
# ## Plots

# %%
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
method_order = ("m1", "m2", "m3", "m4")
totals = [sum(method_times[lab].values()) for lab in method_order]
ax.bar(method_order, totals, color=colors)
ax.set_ylabel("Total wall time (s)")
ax.set_title("Prediction time by method")
for i, v in enumerate(totals):
    ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_TIME, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_TIME}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
labels = list(approx_methods)
nav_rmse = [float(cmp_df[f"nav_{lab}_rmse"].mean()) for lab in labels]
ent_rmse = [float(cmp_df[f"entity_{lab}_rmse"].mean()) for lab in labels]
axes[0].bar(labels, nav_rmse, color=colors[1:])
axes[0].set_ylabel("Mean RMSE vs m1")
axes[0].set_title("Per-node NAV")
axes[1].bar(labels, ent_rmse, color=colors[1:])
axes[1].set_ylabel("Mean RMSE vs m1")
axes[1].set_title("Full-entity NAV")
fig.suptitle("Accuracy vs cubewalkers GT")
fig.tight_layout()
fig.savefig(PLOT_RMSE, dpi=120)
plt.close(fig)
print(f"Wrote {PLOT_RMSE}")

if not score_df.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        score_df["method"],
        score_df["delta_frac_vs_m1"],
        color=colors[1 : 1 + len(score_df)],
    )
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylabel("Δ(score / max_score) vs m1")
    ax.set_title("Hierarchy score gap vs cubewalkers GT")
    fig.tight_layout()
    fig.savefig(PLOT_SCORE, dpi=120)
    plt.close(fig)
    print(f"Wrote {PLOT_SCORE}")


def _plot_delta_boxes(
    cloud: np.ndarray,
    keys: list[str],
    order_by_median_abs: pd.DataFrame,
    key_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Side-by-side horizontal boxplots: signed Δ and |Δ|, sorted by median |Δ|."""
    order = list(order_by_median_abs[key_col])
    key_to_i = {k: i for i, k in enumerate(keys)}
    signed = [cloud[key_to_i[k]] for k in order]
    absolute = [np.abs(v) for v in signed]
    n_items = len(order)
    fig_h = max(6.0, 0.18 * n_items + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_h), sharey=True)
    for ax, data, xlabel, panel in (
        (axes[0], signed, "Δ NAV (method − m1)", "signed Δ"),
        (axes[1], absolute, "|Δ| NAV (method − m1)", "|Δ|"),
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
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


for lab in approx_methods:
    path_n = PLOT_DELTA_NODE[lab]
    _plot_delta_boxes(
        delta_cloud_node[lab],
        list(node_names),
        node_stat_dfs[lab],
        "node",
        title=f"{lab} − m1 node Δ (pooled fixes × boots)",
        out_path=path_n,
    )
    print(f"Wrote {path_n}")

    path_e = PLOT_DELTA_ENTITY[lab]
    _plot_delta_boxes(
        delta_cloud_ent[lab],
        list(ENTITY_KEYS),
        entity_stat_dfs[lab],
        "entity",
        title=f"{lab} − m1 entity Δ (pooled fixes × boots)",
        out_path=path_e,
    )
    print(f"Wrote {path_e}")
print()

# %% [markdown]
# ## Summary

# %%
print("=== Summary ===")
print(f"  Unique fixes: {len(unique_fixes)}")
print(f"  Entities: {len(ENTITY_KEYS)} ({len(MERGED_BASES)} merged)")
print(f"  m1 cubewalkers: walkers={N_WALKERS}  T={N_TIME_STEPS}  window={T_WINDOW}")
print(f"  m2 min_traps+sample: N_SAMPLES_M2={N_SAMPLES_M2}")
print(f"  m3 ldoi+sample: N_SAMPLES_M3={N_SAMPLES_M3}")
print(f"  bootstrap: N_BOOT={N_BOOT}  BOOT_CHUNK={BOOT_CHUNK}")
print("  m4 ldoi: all fixes")
print(
    f"  Total times (s): "
    f"m1={sum(times_m1.values()):.2f}  "
    f"m2={sum(times_m2.values()):.2f}  "
    f"m3={sum(times_m3.values()):.2f}  "
    f"m4={sum(times_m4.values()):.2f}"
)
for lab in approx_methods:
    print(
        f"  Mean NAV RMSE {lab} vs m1: {cmp_df[f'nav_{lab}_rmse'].mean():.4f}"
    )
    print(
        f"  Mean entity RMSE {lab} vs m1: {cmp_df[f'entity_{lab}_rmse'].mean():.4f}"
    )
    gn = global_node_df.loc[global_node_df["method"] == lab].iloc[0]
    ge = global_ent_df.loc[global_ent_df["method"] == lab].iloc[0]
    print(
        f"  Pooled node Δ {lab}: mean(median_|Δ|)={gn['median_abs_mean']:.4f}  "
        f"mean(rmse)={gn['rmse_mean']:.4f}  "
        f"max(max_|Δ|)={gn['max_abs_max']:.4f}"
    )
    print(
        f"  Pooled entity Δ {lab}: mean(median_|Δ|)={ge['median_abs_mean']:.4f}  "
        f"mean(rmse)={ge['rmse_mean']:.4f}  "
        f"max(max_|Δ|)={ge['max_abs_max']:.4f}"
    )
if not score_df.empty:
    for _, r in score_df.iterrows():
        print(
            f"  Hierarchy {r['method']} vs m1: "
            f"Δfrac={r['delta_frac_vs_m1']:+.4f}"
        )
print(f"  Log: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
