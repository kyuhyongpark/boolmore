# %% [markdown]
# NAV accuracy vs cubewalkers ground truth
#
# Compare three approximate NAV predictors against cubewalkers simulation NAV
# on the breast-cancer gen108 model and trial data.
#
# **Methods** (per unique intervention `fixes`):
#
# 1. **m1 — cubewalkers GT** — clamp interventions, sample 2000 initial states,
#    async ensemble, average node values over a late time window
# 2. **m2 — min_traps + sample sources** — sample up to 32 free-source
#    combinations → minimal trap spaces NAV → equal-weight average
# 3. **m3 — ldoi + sample sources** — sample up to 1024 free-source
#    combinations → LDOI NAV → equal-weight average
# 4. **m4 — ldoi** — LDOI on interventions only (no source sampling)
#
# **Accuracy metrics** (m2 / m3 / m4 vs m1):
#
# 1. Per-node NAV (RMSE / MAE / max|Δ| over all model nodes)
# 2. Entity-averaged NAV (sibling means, e.g. Proliferation / `_2` / `_3`)
# 3. Hierarchy score closeness to m1 (`Δscore`, `Δ(score/max)`)
#
# Prints to stdout and writes a timestamped log (+ plots) under `dev/logs/`.
#
# ---
#
# ## WSL CUDA setup (required for method 1)
#
# Cubewalkers needs an NVIDIA GPU visible inside WSL2, plus CuPy.
#
# 1. **Windows host**: install a current NVIDIA Game Ready / Studio driver with
#    WSL2 GPU support. Confirm `nvidia-smi` works in PowerShell.
# 2. **WSL2 Ubuntu**: install a CUDA toolkit that matches the driver major
#    version (NVIDIA's WSL CUDA packages are fine). Example for CUDA 12:
#    follow https://docs.nvidia.com/cuda/wsl-user-guide/
# 3. **Python env** (same env as boolmore):
#    ```bash
#    pip install cupy-cuda12x   # or cupy-cuda11x — match your CUDA major
#    pip install git+https://github.com/jcrozum/cubewalkers
#    ```
# 4. **Sanity check** in WSL:
#    ```bash
#    nvidia-smi
#    python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
#    python -c "import cubewalkers as cw; m=cw.Model('A, !A\nB, A'); m.n_walkers=8; m.n_time_steps=5; m.simulate_ensemble(); print(m.trajectories.shape)"
#    ```
#
# This notebook **fails fast** if CuPy / cubewalkers / a CUDA device is missing.

# %%
from __future__ import annotations

import math
import os
import random
import re
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
from boolmore.evaluation.agreement import (
    _AVG_SUFFIX,
    _average_related_values,
    _related_node_names,
)
from boolmore.evaluation.nav_score import get_NAV_scores
from boolmore.evaluation.score import get_model_score
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.prediction import (
    NAVPrediction,
    _nav_from_ldoi,
    _nav_from_min_traps,
)
from boolmore.io.load import load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.io.paths import project_root
from boolmore.types import Assignment

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

N_WALKERS = 2000
N_SAMPLES_M2 = 32
N_SAMPLES_M3 = 1024
# Paper-style horizon; set explicitly below after N is known, or override here.
N_TIME_STEPS: int | None = None  # None → 55*N + 6000
T_WINDOW: int | None = None  # None → max(1000, T // 4)
# Small GPUs (e.g. MX150) hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at (32, 32).
THREADS_PER_BLOCK: tuple[int, int] = (8, 8)

SEED = 0
MAX_FIXES: int | None = None
HIERARCHY = True

# Optional: `SMOKE=1` for a tiny end-to-end check.
if os.environ.get("SMOKE") == "1":
    MAX_FIXES = 2
    N_WALKERS = 64
    N_SAMPLES_M2 = 4
    N_SAMPLES_M3 = 8
    N_TIME_STEPS = 200
    T_WINDOW = 50

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast)

# %%
_CUDA_HELP = """
Cubewalkers ground truth requires CUDA + CuPy inside WSL2.

Setup checklist:
  1. Windows: NVIDIA driver with WSL2 GPU support (`nvidia-smi` in PowerShell)
  2. WSL: CUDA toolkit matching the driver major version
  3. pip install cupy-cuda12x   # or cupy-cuda11x
  4. pip install git+https://github.com/jcrozum/cubewalkers
  5. Re-run nvidia-smi / cupy device count / tiny cubewalkers smoke test
""".strip()


def require_cubewalkers():
    """Import cubewalkers and verify a CUDA device is available."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit(
            f"CuPy is not installed.\n\n{_CUDA_HELP}\n\nOriginal error: {exc}"
        ) from exc

    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 — surface any CUDA init failure
        raise SystemExit(
            f"CUDA runtime failed.\n\n{_CUDA_HELP}\n\nOriginal error: {exc}"
        ) from exc

    if n_dev < 1:
        raise SystemExit(f"No CUDA devices visible (count={n_dev}).\n\n{_CUDA_HELP}")

    try:
        import cubewalkers as cw
    except ImportError as exc:
        raise SystemExit(
            "cubewalkers is not installed.\n"
            "  pip install git+https://github.com/jcrozum/cubewalkers\n\n"
            f"{_CUDA_HELP}\n\nOriginal error: {exc}"
        ) from exc

    # Tiny kernel compile + simulate to catch broken installs early.
    try:
        probe = cw.Model("A, !A\nB, A", n_time_steps=2, n_walkers=4)
        probe.simulate_ensemble(
            T_window=1,
            averages_only=True,
            maskfunction=cw.update_schemes.asynchronous,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"cubewalkers smoke simulation failed.\n\n{_CUDA_HELP}\n\n"
            f"Original error: {exc}"
        ) from exc

    return cw, cp


cw, cp = require_cubewalkers()
print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
print(f"cubewalkers: {getattr(cw, '__file__', cw)}")

# %% [markdown]
# ## Logging

# %%
class Tee:
    """Write to stdout and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"compare_nav_accuracy_vs_cw_{RUN_ID}.log"
PLOT_TIME = LOG_DIR / f"compare_nav_accuracy_vs_cw_timing_{RUN_ID}.png"
PLOT_RMSE = LOG_DIR / f"compare_nav_accuracy_vs_cw_rmse_{RUN_ID}.png"
PLOT_SCORE = LOG_DIR / f"compare_nav_accuracy_vs_cw_score_gap_{RUN_ID}.png"

_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model pkl:  {PKL}")
print(f"model bnet: {BNET}")
print(f"data:       {DATA}")
print(f"N_WALKERS={N_WALKERS}  N_SAMPLES_M2={N_SAMPLES_M2}  N_SAMPLES_M3={N_SAMPLES_M3}")
print(f"THREADS_PER_BLOCK={THREADS_PER_BLOCK}")
print(f"SEED={SEED}  MAX_FIXES={MAX_FIXES}  HIERARCHY={HIERARCHY}")
print()

# %% [markdown]
# ## Load model and experiments

# %%
primes = load_primes(PKL)


def _bnet_rules_for_cubewalkers(path: Path) -> str:
    """Load a BoolNet ``.bnet``, dropping the ``targets, factors`` header row.

    Cubewalkers would otherwise treat ``targets`` as a node whose rule is
    the undefined identifier ``factors``.
    """
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().replace("\t", " ").replace(" ", "") in {
            "targets,factors",
            "targets,factors,",
        }:
            continue
        # Also match "targets,\tfactors" style loosely
        left = stripped.split(",", 1)[0].strip().lower()
        if left == "targets":
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


BNET_RULES = _bnet_rules_for_cubewalkers(BNET)
experiments = import_NAV_exps(
    DATA,
    id_col=ID_COL,
    weight_col=WEIGHT_COL,
    observed_col=OBSERVED_COL,
    outcome_col=OUTCOME_COL,
    fixes_cols=FIXES_COLS,
)

source_nodes = sorted(
    node for node in primes if primes[node] == [[{node: 0}], [{node: 1}]]
)
source_set = set(source_nodes)
node_names = sorted(primes)

_n_nodes = len(primes)
if N_TIME_STEPS is None:
    N_TIME_STEPS = 55 * _n_nodes + 6000
if T_WINDOW is None:
    T_WINDOW = max(1000, N_TIME_STEPS // 4)
assert T_WINDOW <= N_TIME_STEPS

_all_unique = list({exp.fixes for exp in experiments})


def _n_unfixed(fixes: Assignment) -> int:
    return len(source_set - {node for node, _ in fixes})


unique_fixes = sorted(_all_unique, key=lambda f: (_n_unfixed(f), len(f), f))
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
# ## Helpers

# %%
def analyze_fixes(fixes: Assignment, source_set: set[str]) -> dict:
    fixed_nodes = {node for node, _ in fixes}
    fixed_sources = sorted(fixed_nodes & source_set)
    unfixed_sources = sorted(source_set - fixed_nodes)
    x = len(unfixed_sources)
    return {
        "fixes": fixes,
        "n_fixes": len(fixes),
        "n_fixed_sources": len(fixed_sources),
        "n_unfixed_sources": x,
        "min_trap_lower_bound": 2**x,
        "fixed_sources": fixed_sources,
        "unfixed_sources": unfixed_sources,
    }


def format_fixes(fixes: Assignment, max_len: int = 60) -> str:
    if not fixes:
        s = "(none)"
    else:
        s = "{" + ", ".join(f"{n}={v}" for n, v in fixes) + "}"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def decode_source_index(
    index: int,
    unfixed: list[str],
    fixed_sources: dict[str, int],
) -> dict[str, int]:
    """Map an integer in ``[0, 2**x)`` to a full source assignment."""
    assignment = dict(fixed_sources)
    x = len(unfixed)
    for j, name in enumerate(unfixed):
        bit = (index >> (x - 1 - j)) & 1
        assignment[name] = bit
    return assignment


def sample_source_indices(x: int, n_samples: int, rng: random.Random) -> list[int]:
    """Without-replacement sample of source-combination indices."""
    total = 2**x
    n = min(n_samples, total)
    return rng.sample(range(total), n)


def merge_perturbation(
    fixes: Assignment,
    source_assignment: dict[str, int],
) -> dict[str, int]:
    """Intervention dict plus full source assignment (sources override)."""
    pert = assignment_to_dict(fixes)
    pert.update(source_assignment)
    return pert


def average_nav_dicts(navs: list[dict[str, float]]) -> dict[str, float]:
    if not navs:
        raise ValueError("no NAV dicts to average")
    nodes = navs[0].keys()
    n = len(navs)
    return {node: sum(d[node] for d in navs) / n for node in nodes}


def nav_from_sampled_sources(
    primes: dict,
    fixes: Assignment,
    unfixed: list[str],
    fixed_sources: dict[str, int],
    indices: list[int],
    method: str,
) -> dict[str, float]:
    """Average NAV over source combinations (method: 'min_traps' | 'ldoi')."""
    nav_fn = _nav_from_min_traps if method == "min_traps" else _nav_from_ldoi
    navs: list[dict[str, float]] = []
    for idx in indices:
        sources = decode_source_index(idx, unfixed, fixed_sources)
        pert = merge_perturbation(fixes, sources)
        navs.append(nav_fn(primes, pert))
    return average_nav_dicts(navs)


def nav_vector_metrics(
    a: dict[str, float],
    b: dict[str, float],
    nodes: list[str] | None = None,
) -> dict[str, float]:
    if nodes is None:
        nodes = sorted(a.keys())
    diffs = [abs(a[n] - b[n]) for n in nodes]
    if not diffs:
        return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0}
    arr = np.asarray(diffs, dtype=float)
    return {
        "mae": float(arr.mean()),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "max_abs": float(arr.max()),
    }


def timed(fn, *args, **kwargs):
    t0 = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - t0


def fixes_to_experiment(fixes: Assignment):
    """Build a cubewalkers Experiment that permanently clamps each fix."""
    if not fixes:
        return None
    lines = [f"{node},0,inf,{int(value)}" for node, value in fixes]
    return cw.Experiment("\n".join(lines))


def nav_from_cubewalkers(
    rules: str,
    fixes: Assignment,
    *,
    n_time_steps: int,
    t_window: int,
    n_walkers: int,
    threads_per_block: tuple[int, int] = THREADS_PER_BLOCK,
) -> dict[str, float]:
    """Simulate ensemble NAV under clamped interventions."""
    experiment = fixes_to_experiment(fixes)
    model = cw.Model(
        rules,
        experiment=experiment,
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
    )
    model.simulate_ensemble(
        T_window=t_window,
        averages_only=True,
        maskfunction=cw.update_schemes.asynchronous,
        threads_per_block=threads_per_block,
    )
    traj = model.trajectories
    # averages_only → shape (T_window, n_variables); mean over time window
    if hasattr(traj, "get"):
        traj = traj.get()
    traj = np.asarray(traj, dtype=float)
    means = traj.mean(axis=0)
    return {name: float(means[i]) for i, name in enumerate(model.varnames)}


def discover_entity_bases(node_names: list[str], experiments) -> list[str]:
    """Bases with ≥2 siblings, plus bases behind observed ``*_avg`` names."""
    bases: set[str] = set()
    for exp in experiments:
        obs = exp.observed
        if obs.endswith(_AVG_SUFFIX):
            base = obs[: -len(_AVG_SUFFIX)]
            if base and _related_node_names(base, node_names):
                bases.add(base)

    # Also any multi-sibling group in the model (Base / Base_2 / …)
    suffix_re = re.compile(r"^(.+?)_(?:\d+|[X-Z])$")
    candidates: set[str] = set(node_names)
    for name in node_names:
        m = suffix_re.match(name)
        if m:
            candidates.add(m.group(1))
    for base in candidates:
        related = _related_node_names(base, node_names)
        if len(related) >= 2:
            bases.add(base)

    return sorted(bases)


def entity_average_nav(
    nav: dict[str, float],
    bases: list[str],
) -> dict[str, float]:
    """Map each entity base → mean of sibling node NAVs."""
    out: dict[str, float] = {}
    for base in bases:
        avg_name = f"{base}{_AVG_SUFFIX}"
        val = _average_related_values(nav, avg_name)
        if val is None:
            raise ValueError(f"entity average for {base!r} hit a None sibling NAV")
        out[base] = float(val)
    return out


def to_nav_predictions(
    preds: dict[Assignment, dict[str, float]],
) -> dict[Assignment, NAVPrediction]:
    return {
        fixes: NAVPrediction(fixes=fixes, average_values=avg)
        for fixes, avg in preds.items()
    }

# %% [markdown]
# ## Unfixed-source analysis

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

ENTITY_BASES = discover_entity_bases(node_names, experiments)
print(f"Entity bases ({len(ENTITY_BASES)}): {ENTITY_BASES}")
print()

# %% [markdown]
# ## Method 1 — cubewalkers ground truth

# %%
preds_m1: dict[Assignment, dict[str, float]] = {}
times_m1: dict[Assignment, float] = {}

print(f"Running method 1 (cubewalkers) on {len(unique_fixes)} fixes...")
for i, fixes in enumerate(unique_fixes, 1):
    nav, dt = timed(
        nav_from_cubewalkers,
        BNET_RULES,
        fixes,
        n_time_steps=N_TIME_STEPS,
        t_window=T_WINDOW,
        n_walkers=N_WALKERS,
    )
    # Align keys to primes node set (cubewalkers varname order may differ)
    preds_m1[fixes] = {n: float(nav[n]) for n in node_names}
    times_m1[fixes] = dt
    x = meta_by_fixes[fixes]["n_unfixed_sources"]
    print(
        f"  [{i}/{len(unique_fixes)}] x={x}  "
        f"t={dt:.3f}s  {format_fixes(fixes)}"
    )

print(
    f"Method 1 done: n={len(preds_m1)}, "
    f"total={sum(times_m1.values()):.3f}s, "
    f"mean={np.mean(list(times_m1.values())):.3f}s"
)
print()

# %% [markdown]
# ## Method 4 — plain LDOI

# %%
preds_m4: dict[Assignment, dict[str, float]] = {}
times_m4: dict[Assignment, float] = {}

print(f"Running method 4 (LDOI) on {len(unique_fixes)} fixes...")
for i, fixes in enumerate(unique_fixes, 1):
    pert = assignment_to_dict(fixes)
    nav, dt = timed(_nav_from_ldoi, primes, pert)
    preds_m4[fixes] = nav
    times_m4[fixes] = dt
    if i % 25 == 0 or i == len(unique_fixes):
        print(f"  [{i}/{len(unique_fixes)}] cumulative={sum(times_m4.values()):.3f}s")

print(
    f"Method 4 done: n={len(preds_m4)}, "
    f"total={sum(times_m4.values()):.3f}s, "
    f"mean={np.mean(list(times_m4.values())):.4f}s"
)
print()

# %% [markdown]
# ## Methods 2 and 3 — sampled sources

# %%
rng_final = random.Random(SEED)
preds_m2: dict[Assignment, dict[str, float]] = {}
preds_m3: dict[Assignment, dict[str, float]] = {}
times_m2: dict[Assignment, float] = {}
times_m3: dict[Assignment, float] = {}
n_used_m2: dict[Assignment, int] = {}
n_used_m3: dict[Assignment, int] = {}

print(
    f"Running methods 2 & 3 on {len(unique_fixes)} fixes "
    f"(m2: min({N_SAMPLES_M2}, 2**x); "
    f"m3: min({N_SAMPLES_M3}, 2**x))..."
)
for i, fixes in enumerate(unique_fixes, 1):
    m = meta_by_fixes[fixes]
    unfixed = m["unfixed_sources"]
    x = m["n_unfixed_sources"]
    fixed_src = {
        n: v for n, v in assignment_to_dict(fixes).items() if n in source_set
    }
    n2 = min(N_SAMPLES_M2, 2**x)
    n3 = min(N_SAMPLES_M3, 2**x)
    n_max = max(n2, n3)
    idxs_max = sample_source_indices(x, n_max, rng_final)
    idxs2 = idxs_max[:n2]
    idxs3 = idxs_max[:n3]
    n_used_m2[fixes] = len(idxs2)
    n_used_m3[fixes] = len(idxs3)

    t0 = perf_counter()
    preds_m2[fixes] = nav_from_sampled_sources(
        primes, fixes, unfixed, fixed_src, idxs2, "min_traps"
    )
    times_m2[fixes] = perf_counter() - t0

    t0 = perf_counter()
    preds_m3[fixes] = nav_from_sampled_sources(
        primes, fixes, unfixed, fixed_src, idxs3, "ldoi"
    )
    times_m3[fixes] = perf_counter() - t0

    if i % 10 == 0 or i == len(unique_fixes):
        print(
            f"  [{i}/{len(unique_fixes)}] "
            f"cum m2={sum(times_m2.values()):.1f}s  "
            f"m3={sum(times_m3.values()):.1f}s"
        )

print(
    f"Method 2 done: total={sum(times_m2.values()):.3f}s  "
    f"mean={np.mean(list(times_m2.values())):.3f}s"
)
print(
    f"Method 3 done: total={sum(times_m3.values()):.3f}s  "
    f"mean={np.mean(list(times_m3.values())):.4f}s"
)
print()

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
approx_methods = ("m2", "m3", "m4")

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

        ent_gt = entity_average_nav(gt, ENTITY_BASES)
        ent_pred = entity_average_nav(method_preds[lab][fixes], ENTITY_BASES)
        em = nav_vector_metrics(ent_gt, ent_pred, nodes=ENTITY_BASES)
        row.update({f"entity_{lab}_{k}": v for k, v in em.items()})
    cmp_rows.append(row)

cmp_df = pd.DataFrame(cmp_rows)


def _timing_line(label: str, vals: list[float]) -> str:
    if not vals:
        return f"  {label}: skipped / empty"
    return (
        f"  {label}: n={len(vals)}  total={sum(vals):.3f}s  "
        f"mean={np.mean(vals):.4f}s  median={np.median(vals):.4f}s  "
        f"max={max(vals):.3f}s"
    )


print("=== Timing summary ===")
for label, tdict in method_times.items():
    print(_timing_line(label, list(tdict.values())))
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
# ## Metric 2 — entity-averaged NAV vs m1

# %%
print("=== Metric 2: entity-averaged NAV vs m1 (mean over fixes) ===")
print(f"  entities ({len(ENTITY_BASES)}): {ENTITY_BASES}")
for lab in approx_methods:
    print(
        f"  {lab}: "
        f"RMSE={cmp_df[f'entity_{lab}_rmse'].mean():.4f}  "
        f"MAE={cmp_df[f'entity_{lab}_mae'].mean():.4f}  "
        f"max|Δ|={cmp_df[f'entity_{lab}_max_abs'].mean():.4f}"
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
axes[1].set_title("Entity-averaged NAV")
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
print()

# %% [markdown]
# ## Summary

# %%
print("=== Summary ===")
print(f"  Unique fixes: {len(unique_fixes)}")
print(f"  m1 cubewalkers: walkers={N_WALKERS}  T={N_TIME_STEPS}  window={T_WINDOW}")
print(f"  m2 min_traps+sample: N_SAMPLES_M2={N_SAMPLES_M2}")
print(f"  m3 ldoi+sample: N_SAMPLES_M3={N_SAMPLES_M3}")
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
