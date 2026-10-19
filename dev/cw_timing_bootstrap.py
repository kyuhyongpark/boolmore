# %% [markdown]
# Cubewalkers timing + bootstrap SE
#
# Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
# Measures split wall time (`t_setup` / `t_simulate` / `t_reduce`) and
# per-node bootstrap SE of the time-window NAV, then sweeps
# `n_walkers`, `N_TIME_STEPS`, and `T_WINDOW`.
#
# Each sweep reports consecutive setting ΔNAV (B−A) by reusing per-setting
# bootstrap mean replicates (no second walker resample), a third panel of
# those summaries, and an overlay NAV±SE figure (one color per setting;
# per-setting SE, not SE of Δ).
#
# Default baseline/sweep sizes are moderate (W=512, T=2000, window=500).
# Set `RUN_PAPER_BASELINE = True` for the paper formula
# (`T=55*N+6000`, window=`max(1000, T//4)`, W=2000). Use `SMOKE=1` for a
# tiny end-to-end check.
#
# **Timing notes**
#
# - `t_setup` covers `cw.Model(...)` (rule parse + `RawKernel` construction +
#   initial states). NVRTC compile usually hits on the **first** kernel launch
#   inside `simulate_ensemble`, so compile often lands in `t_simulate`.
# - Each Model gets a unique kernel name, so every rebuild can recompile.
#   A discarded warm-up warms CuPy/driver but does not remove that cost.
# - GPU is synchronized before closing `t_simulate` / `t_reduce`.
#
# ---
#
# ## WSL CUDA setup
#
# 1. Windows: NVIDIA driver with WSL2 GPU support (`nvidia-smi` in PowerShell)
# 2. WSL: CUDA toolkit matching the driver major version
# 3. `pip install cupy-cuda12x` (or `cupy-cuda11x`) and
#    `pip install git+https://github.com/jcrozum/cubewalkers`
#
# This notebook **fails fast** if CuPy / cubewalkers / a CUDA device is missing.

# %%
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.io.paths import project_root

# %% [markdown]
# ## Config

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "breast_cancer"
BNET = (
    CASE
    / "20261016_173102"
    / "BreastCancer_primes_parsimonious_20261016_173102_9916_gen108.bnet"
)
PLOT_DIR = Path(__file__).resolve().parent / "logs"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Practical baseline / sweep base (paper-scale is optional — see RUN_PAPER_BASELINE).
# Full traj is (T_window, N, W) bools; paper defaults can be ~1–2 GB on device.
N_WALKERS = 2048
N_TIME_STEPS = 16505
T_WINDOW = 4126

# Sweep base (held fixed while varying one axis).
SWEEP_WALKERS = 2048
SWEEP_T = 16505
SWEEP_WINDOW = 4126

WALKERS_GRID = [64, 128, 256, 512, 1024, 2048]
T_GRID = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16505]
WINDOW_GRID = [50, 100, 250, 500, 1000, 2000, 4126]

THREADS_PER_BLOCK: tuple[int, int] = (16, 16)
SEED = 0
N_BOOT = 1000
BOOT_CHUNK = 50  # bootstrap batch size to limit (N, chunk, W) memory
RUN_PAPER_BASELINE = False

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_WALKERS = 64
    N_TIME_STEPS = 200
    T_WINDOW = 50
    SWEEP_WALKERS = 64
    SWEEP_T = 200
    SWEEP_WINDOW = 50
    WALKERS_GRID = [32, 64, 128]
    T_GRID = [100, 200, 400]
    WINDOW_GRID = [25, 50, 100]
    N_BOOT = 200
    BOOT_CHUNK = 40
    RUN_PAPER_BASELINE = False

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast)

# %%
_CUDA_HELP = """
Cubewalkers requires CUDA + CuPy inside WSL2.

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
    except Exception as exc:  # noqa: BLE001
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

    try:
        probe = cw.Model("A, !A\nB, A", n_time_steps=2, n_walkers=4)
        probe.simulate_ensemble(
            T_window=1,
            averages_only=True,
            maskfunction=cw.update_schemes.asynchronous,
        )
        cp.cuda.Device().synchronize()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"cubewalkers smoke simulation failed.\n\n{_CUDA_HELP}\n\n"
            f"Original error: {exc}"
        ) from exc

    return cw, cp


cw, cp = require_cubewalkers()
print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
print(f"cubewalkers: {getattr(cw, '__file__', cw)}")
print(f"SMOKE={SMOKE}  RUN_PAPER_BASELINE={RUN_PAPER_BASELINE}")

# %% [markdown]
# ## Load bnet (no experiment data)

# %%
def _bnet_rules_for_cubewalkers(path: Path) -> str:
    """Load a BoolNet ``.bnet``, dropping the ``targets, factors`` header row."""
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
        left = stripped.split(",", 1)[0].strip().lower()
        if left == "targets":
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


BNET_RULES = _bnet_rules_for_cubewalkers(BNET)
_n_nodes = sum(
    1
    for line in BNET_RULES.splitlines()
    if line.strip() and not line.strip().startswith("#")
)

assert T_WINDOW <= N_TIME_STEPS

PAPER_T = 55 * _n_nodes + 6000
PAPER_WINDOW = max(1000, PAPER_T // 4)

print(f"bnet: {BNET}")
print(f"nodes: {_n_nodes}")
print(
    f"baseline config: walkers={N_WALKERS}  T={N_TIME_STEPS}  "
    f"T_WINDOW={T_WINDOW}  N_BOOT={N_BOOT}"
)
print(f"paper formula: T={PAPER_T}  T_WINDOW={PAPER_WINDOW}  walkers=2000")
print(
    f"sweep base: walkers={SWEEP_WALKERS}  T={SWEEP_T}  "
    f"T_WINDOW={SWEEP_WINDOW}"
)

_model_name_counter = 0


def _next_model_name(prefix: str) -> str:
    global _model_name_counter
    _model_name_counter += 1
    # CuPy RawKernel names must be unique; keep them C-identifier-safe.
    return f"{prefix}_{_model_name_counter}"

# %% [markdown]
# ## Helpers

# %%
def simulate_walker_nav(
    rules: str,
    *,
    n_time_steps: int,
    t_window: int,
    n_walkers: int,
    threads_per_block: tuple[int, int] = THREADS_PER_BLOCK,
    model_name: str | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Run an empty-clamp ensemble; return per-walker NAV and split timings.

    Returns
    -------
    walker_nav : ndarray, shape (n_nodes, n_walkers)
        Time-window mean for each node and walker.
    varnames : list[str]
    timings : dict with t_setup, t_simulate, t_reduce, t_total

    Notes
    -----
    ``cw.Model`` builds a ``RawKernel`` during setup, but NVRTC compile is often
    deferred until the first kernel launch inside ``simulate_ensemble``.
    """
    if t_window > n_time_steps or t_window < 1:
        raise ValueError(f"T_WINDOW={t_window} must be in [1, T={n_time_steps}]")

    t0 = perf_counter()
    model_kwargs: dict = {
        "n_time_steps": n_time_steps,
        "n_walkers": n_walkers,
    }
    if model_name is not None:
        model_kwargs["model_name"] = model_name
    model = cw.Model(rules, experiment=None, **model_kwargs)
    cp.cuda.Device().synchronize()
    t_setup = perf_counter() - t0

    t0 = perf_counter()
    model.simulate_ensemble(
        T_window=t_window,
        averages_only=False,
        maskfunction=cw.update_schemes.asynchronous,
        threads_per_block=threads_per_block,
    )
    cp.cuda.Device().synchronize()
    t_simulate = perf_counter() - t0

    t0 = perf_counter()
    traj = model.trajectories  # (T_window, N, W)
    walker_nav_gpu = traj.mean(axis=0)  # (N, W)
    walker_nav = cp.asnumpy(walker_nav_gpu).astype(np.float64, copy=False)
    varnames = list(model.varnames)
    del traj, walker_nav_gpu, model
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
    t_reduce = perf_counter() - t0

    timings = {
        "t_setup": float(t_setup),
        "t_simulate": float(t_simulate),
        "t_reduce": float(t_reduce),
        "t_total": float(t_setup + t_simulate + t_reduce),
    }
    return walker_nav, varnames, timings


def bootstrap_se(
    walker_nav: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    chunk: int = BOOT_CHUNK,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bootstrap SE of the walker-mean NAV for each node.

    Parameters
    ----------
    walker_nav : (n_nodes, n_walkers)

    Returns
    -------
    means : (n_nodes,)
    se : (n_nodes,)
    boot_means : (n_nodes, n_boot)
        Walker-mean replicates; reuse for independent B−A diffs.
    t_boot : float

    Notes
    -----
    Resamples are processed in chunks so we never materialize a full
    ``(n_nodes, n_boot, n_walkers)`` array (can be multi-GB at paper scale).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_nodes, n_walkers = walker_nav.shape
    t0 = perf_counter()
    means = walker_nav.mean(axis=1)
    boot_means = np.empty((n_nodes, n_boot), dtype=np.float64)
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        k = end - start
        idx = rng.integers(0, n_walkers, size=(k, n_walkers))
        # walker_nav[:, idx] → (n_nodes, k, n_walkers)
        boot_means[:, start:end] = walker_nav[:, idx].mean(axis=2)
    se = boot_means.std(axis=1, ddof=1)
    t_boot = perf_counter() - t0
    return means, se, boot_means, float(t_boot)


def diff_from_boot(
    means_a: np.ndarray,
    boot_a: np.ndarray,
    means_b: np.ndarray,
    boot_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node Δ = mean_B − mean_A and bootstrap SE from stored replicates.

    Pairing independent setting-A and setting-B replicates is the same
    two-sample bootstrap as resampling both walker sets again.
    """
    if boot_a.shape != boot_b.shape:
        raise ValueError(f"boot shape mismatch: {boot_a.shape} vs {boot_b.shape}")
    diff = means_b - means_a
    se = (boot_b - boot_a).std(axis=1, ddof=1)
    return diff, se


def _diff_summary(diff: np.ndarray, se: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(diff)
    return {
        "vs_prev_median_abs": float(np.median(abs_diff)),
        "vs_prev_mean_abs": float(abs_diff.mean()),
        "vs_prev_max_abs": float(abs_diff.max()),
        "vs_prev_median_se": float(np.median(se)),
        "vs_prev_mean_se": float(se.mean()),
        "vs_prev_max_se": float(se.max()),
    }


def _format_timings(timings: dict[str, float]) -> str:
    return (
        f"setup={timings['t_setup']:.3f}s  "
        f"simulate={timings['t_simulate']:.3f}s  "
        f"reduce={timings['t_reduce']:.3f}s  "
        f"total={timings['t_total']:.3f}s"
    )


# %% [markdown]
# ## Warm-up (discarded)
#
# One Model+simulate on the real rules so later timed setups are not the first
# process-wide compile of this network size. NVRTC for *this* kernel still lands
# in the warm-up's first launch; timed runs rebuild Models (new kernel names)
# but CuPy/driver are warm.

# %%
print("Warm-up (discarded)...")
_wu_nav, _wu_names, _wu_t = simulate_walker_nav(
    BNET_RULES,
    n_time_steps=min(50, N_TIME_STEPS),
    t_window=min(10, T_WINDOW),
    n_walkers=min(32, N_WALKERS),
    model_name=_next_model_name("cw_warmup"),
)
print(f"  warm-up done: {_format_timings(_wu_t)}  nodes={len(_wu_names)}")
del _wu_nav

# %% [markdown]
# ## Baseline run

# %%
rng = np.random.default_rng(SEED)

print("Baseline simulation...")
walker_nav, varnames, timings = simulate_walker_nav(
    BNET_RULES,
    n_time_steps=N_TIME_STEPS,
    t_window=T_WINDOW,
    n_walkers=N_WALKERS,
    model_name=_next_model_name("cw_baseline"),
)
means, se, boot_means, t_boot = bootstrap_se(walker_nav, n_boot=N_BOOT, rng=rng)
del walker_nav, boot_means

print(f"  {_format_timings(timings)}")
print(f"  bootstrap: t_boot={t_boot:.3f}s  N_BOOT={N_BOOT}")
print(
    f"  NAV: mean of means={means.mean():.4f}  "
    f"median SE={np.median(se):.5f}  max SE={se.max():.5f}"
)

baseline_df = pd.DataFrame(
    {"node": varnames, "mean": means, "se": se}
).sort_values("mean", ascending=False)
print(baseline_df.head(10).to_string(index=False))

# Optional paper-scale baseline (expensive; large device traj).
if RUN_PAPER_BASELINE:
    print("\nPaper-scale baseline...")
    paper_nav, paper_names, paper_t = simulate_walker_nav(
        BNET_RULES,
        n_time_steps=PAPER_T,
        t_window=PAPER_WINDOW,
        n_walkers=2000,
        model_name=_next_model_name("cw_paper"),
    )
    paper_means, paper_se, paper_boot_means, paper_boot = bootstrap_se(
        paper_nav, n_boot=N_BOOT, rng=rng
    )
    print(f"  {_format_timings(paper_t)}  t_boot={paper_boot:.3f}s")
    print(
        f"  median SE={np.median(paper_se):.5f}  max SE={paper_se.max():.5f}"
    )
    del paper_nav, paper_boot_means

# %% [markdown]
# ## Plot — baseline node means ± bootstrap SE

# %%
plot_df = baseline_df.sort_values("mean")
fig, ax = plt.subplots(figsize=(10, max(4, 0.12 * len(plot_df))))
y = np.arange(len(plot_df))
ax.errorbar(
    plot_df["mean"],
    y,
    xerr=plot_df["se"],
    fmt="o",
    markersize=3,
    elinewidth=0.8,
    capsize=1.5,
    color="#2C5F7C",
    ecolor="#7A9EAD",
)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["node"], fontsize=6)
ax.set_xlabel("Node average value (NAV)")
ax.set_title(
    f"Baseline NAV ± bootstrap SE\n"
    f"W={N_WALKERS}, T={N_TIME_STEPS}, window={T_WINDOW}, B={N_BOOT}"
)
ax.set_xlim(-0.05, 1.05)
fig.tight_layout()
plot_baseline = PLOT_DIR / "cw_timing_bootstrap_baseline_nav.png"
fig.savefig(plot_baseline, dpi=120)
plt.close(fig)
print(f"Wrote {plot_baseline}")

# %% [markdown]
# ## Parameter sweeps
#
# Hold two factors at the sweep base; vary one. Each point rebuilds a Model and
# reports split timings plus median / max per-node bootstrap SE.

# %%
def run_sweep_point(
    *,
    n_walkers: int,
    n_time_steps: int,
    t_window: int,
    label: str,
    rng: np.random.Generator,
    prev: dict | None = None,
) -> dict:
    tw = min(t_window, n_time_steps)
    walker_nav, varnames, timings = simulate_walker_nav(
        BNET_RULES,
        n_time_steps=n_time_steps,
        t_window=tw,
        n_walkers=n_walkers,
        model_name=_next_model_name(f"cw_sweep_{label}"),
    )
    means, se, boot_means, t_boot = bootstrap_se(
        walker_nav, n_boot=N_BOOT, rng=rng
    )
    del walker_nav
    row: dict = {
        "label": label,
        "n_walkers": n_walkers,
        "n_time_steps": n_time_steps,
        "t_window": tw,
        "n_nodes": len(varnames),
        "varnames": varnames,
        "means": means,
        "se": se,
        "boot_means": boot_means,
        "median_se": float(np.median(se)),
        "mean_se": float(se.mean()),
        "max_se": float(se.max()),
        "t_boot": t_boot,
        **timings,
    }
    msg = (
        f"  [{label}] W={n_walkers} T={n_time_steps} win={tw}  "
        f"{_format_timings(timings)}  "
        f"medSE={row['median_se']:.5f} maxSE={row['max_se']:.5f}"
    )
    if prev is not None:
        if prev["varnames"] != varnames:
            raise RuntimeError(
                f"varname mismatch between {prev['label']} and {label}"
            )
        diff, se_diff = diff_from_boot(
            prev["means"], prev["boot_means"], means, boot_means
        )
        summary = _diff_summary(diff, se_diff)
        row.update(summary)
        row["vs_prev"] = prev["label"]
        # Drop previous replicates once the consecutive pair is done.
        prev.pop("boot_means", None)
        msg += (
            f"  vs {prev['label']}: "
            f"med|Δ|={summary['vs_prev_median_abs']:.5f} "
            f"max|Δ|={summary['vs_prev_max_abs']:.5f} "
            f"medSE(Δ)={summary['vs_prev_median_se']:.5f}"
        )
    print(msg)
    return row


sweep_rows: list[dict] = []

print("Sweep: n_walkers...")
_prev = None
for w in WALKERS_GRID:
    _prev = run_sweep_point(
        n_walkers=w,
        n_time_steps=SWEEP_T,
        t_window=SWEEP_WINDOW,
        label=f"W{w}",
        rng=rng,
        prev=_prev,
    )
    sweep_rows.append(_prev)
sweep_rows[-1].pop("boot_means", None)

print("Sweep: N_TIME_STEPS...")
_prev = None
for t in T_GRID:
    win = min(SWEEP_WINDOW, t)
    _prev = run_sweep_point(
        n_walkers=SWEEP_WALKERS,
        n_time_steps=t,
        t_window=win,
        label=f"T{t}",
        rng=rng,
        prev=_prev,
    )
    sweep_rows.append(_prev)
sweep_rows[-1].pop("boot_means", None)

print("Sweep: T_WINDOW...")
# Need T >= max window in the grid.
t_for_window = max(SWEEP_T, max(WINDOW_GRID))
_prev = None
for win in WINDOW_GRID:
    if win > t_for_window:
        continue
    _prev = run_sweep_point(
        n_walkers=SWEEP_WALKERS,
        n_time_steps=t_for_window,
        t_window=win,
        label=f"win{win}",
        rng=rng,
        prev=_prev,
    )
    sweep_rows.append(_prev)
sweep_rows[-1].pop("boot_means", None)

sweep_df = pd.DataFrame(
    [
        {
            k: v
            for k, v in row.items()
            if k not in {"means", "se", "varnames", "boot_means"}
        }
        for row in sweep_rows
    ]
)
print("\nSweep table:")
_table_cols = [
    "label",
    "n_walkers",
    "n_time_steps",
    "t_window",
    "t_setup",
    "t_simulate",
    "t_reduce",
    "t_total",
    "median_se",
    "max_se",
    "vs_prev",
    "vs_prev_median_abs",
    "vs_prev_max_abs",
    "vs_prev_median_se",
]
print(sweep_df[[c for c in _table_cols if c in sweep_df.columns]].to_string(index=False))

# %% [markdown]
# ## Sweep plots
#
# Three panels per factor: timing, per-setting bootstrap SE, and consecutive
# setting ΔNAV summary (B−A from stored bootstrap mean replicates). Separate
# overlay figures show per-node NAV ± SE for every setting on one axis.

# %%
_SWEEP_COLORS = [
    "#4C72B0",
    "#C44E52",
    "#55A868",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#E67E22",
    "#8C564B",
]


def _plot_factor(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(df[x_col], df["t_setup"], "o-", label="t_setup", color="#C44E52")
    ax.plot(df[x_col], df["t_simulate"], "s-", label="t_simulate", color="#4C72B0")
    ax.plot(df[x_col], df["t_reduce"], "^-", label="t_reduce", color="#55A868")
    ax.plot(df[x_col], df["t_total"], "D--", label="t_total", color="#8172B2", alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Timing")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)

    ax = axes[1]
    ax.plot(df[x_col], df["median_se"], "o-", label="median SE", color="#4C72B0")
    ax.plot(df[x_col], df["max_se"], "s-", label="max SE", color="#C44E52")
    ax.plot(df[x_col], df["mean_se"], "^-", label="mean SE", color="#55A868", alpha=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Bootstrap SE")
    ax.set_title("Standard error")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)

    ax = axes[2]
    has_diff = (
        "vs_prev_median_abs" in df.columns and df["vs_prev_median_abs"].notna().any()
    )
    if has_diff:
        ax.plot(
            df[x_col],
            df["vs_prev_median_abs"],
            "o-",
            label="median |Δ| vs prev",
            color="#4C72B0",
        )
        ax.plot(
            df[x_col],
            df["vs_prev_max_abs"],
            "s-",
            label="max |Δ| vs prev",
            color="#C44E52",
        )
        ax.plot(
            df[x_col],
            df["vs_prev_mean_abs"],
            "^-",
            label="mean |Δ| vs prev",
            color="#55A868",
            alpha=0.8,
        )
        ax.plot(
            df[x_col],
            df["vs_prev_median_se"],
            "D--",
            label="median SE(Δ)",
            color="#8172B2",
            alpha=0.8,
        )
        ax.legend(fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "no consecutive pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel("NAV difference vs previous setting")
    ax.set_title("Δ vs previous (bootstrapped)")
    ax.set_xscale("log", base=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_sweep_nav_overlay(
    points: list[dict],
    *,
    title: str,
    path: Path,
    sort_by: str = "last",
) -> None:
    """Horizontal NAV ± SE for every sweep setting, colored together.

    Each point is that setting's own node mean and bootstrap SE (not Δ).
    Nodes are ordered by the reference setting's means (default: last / largest).
    """
    if not points:
        return
    ref = points[-1] if sort_by == "last" else points[0]
    order = np.argsort(ref["means"])
    varnames = [ref["varnames"][i] for i in order]
    y = np.arange(len(varnames))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.12 * len(varnames))))
    n = len(points)
    # Small vertical jitter so overlapping settings remain visible.
    jitter = np.zeros(1) if n == 1 else 0.55 * np.linspace(-0.5, 0.5, n)
    for i, pt in enumerate(points):
        color = _SWEEP_COLORS[i % len(_SWEEP_COLORS)]
        means = np.asarray(pt["means"], dtype=np.float64)[order]
        se = np.asarray(pt["se"], dtype=np.float64)[order]
        ax.errorbar(
            means,
            y + jitter[i],
            xerr=se,
            fmt="o",
            markersize=2.5,
            elinewidth=0.6,
            capsize=1.0,
            color=color,
            ecolor=color,
            alpha=0.85,
            label=pt["label"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(varnames, fontsize=6)
    ax.set_xlabel("Node average value (NAV)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


def _rows_for_prefix(prefix: str) -> list[dict]:
    return [r for r in sweep_rows if str(r["label"]).startswith(prefix)]


walkers_df = sweep_df[sweep_df["label"].str.startswith("W")].sort_values("n_walkers")
t_df = sweep_df[sweep_df["label"].str.startswith("T")].sort_values("n_time_steps")
win_df = sweep_df[sweep_df["label"].str.startswith("win")].sort_values("t_window")

_plot_factor(
    walkers_df,
    "n_walkers",
    f"Vary n_walkers (T={SWEEP_T}, window={SWEEP_WINDOW})",
    PLOT_DIR / "cw_timing_bootstrap_sweep_walkers.png",
)
_plot_factor(
    t_df,
    "n_time_steps",
    f"Vary N_TIME_STEPS (W={SWEEP_WALKERS}, window≤{SWEEP_WINDOW})",
    PLOT_DIR / "cw_timing_bootstrap_sweep_T.png",
)
_plot_factor(
    win_df,
    "t_window",
    f"Vary T_WINDOW (W={SWEEP_WALKERS}, T={t_for_window})",
    PLOT_DIR / "cw_timing_bootstrap_sweep_window.png",
)

walkers_pts = sorted(_rows_for_prefix("W"), key=lambda r: r["n_walkers"])
t_pts = sorted(_rows_for_prefix("T"), key=lambda r: r["n_time_steps"])
win_pts = sorted(_rows_for_prefix("win"), key=lambda r: r["t_window"])

_plot_sweep_nav_overlay(
    walkers_pts,
    title=(
        f"Sweep n_walkers — NAV ± bootstrap SE\n"
        f"T={SWEEP_T}, window={SWEEP_WINDOW}, B={N_BOOT}"
    ),
    path=PLOT_DIR / "cw_timing_bootstrap_sweep_walkers_nav.png",
)
_plot_sweep_nav_overlay(
    t_pts,
    title=(
        f"Sweep N_TIME_STEPS — NAV ± bootstrap SE\n"
        f"W={SWEEP_WALKERS}, window≤{SWEEP_WINDOW}, B={N_BOOT}"
    ),
    path=PLOT_DIR / "cw_timing_bootstrap_sweep_T_nav.png",
)
_plot_sweep_nav_overlay(
    win_pts,
    title=(
        f"Sweep T_WINDOW — NAV ± bootstrap SE\n"
        f"W={SWEEP_WALKERS}, T={t_for_window}, B={N_BOOT}"
    ),
    path=PLOT_DIR / "cw_timing_bootstrap_sweep_window_nav.png",
)

# %% [markdown]
# ## Takeaway

# %%
def _dominant(df: pd.DataFrame, x_col: str) -> str:
    if len(df) < 2:
        return "n/a"
    sim_ratio = df["t_simulate"].iloc[-1] / max(df["t_simulate"].iloc[0], 1e-9)
    se_ratio = df["median_se"].iloc[0] / max(df["median_se"].iloc[-1], 1e-12)
    setup_share = df["t_setup"].mean() / max(df["t_total"].mean(), 1e-9)
    msg = (
        f"{x_col}: simulate scales ~{sim_ratio:.1f}x over grid; "
        f"median SE shrinks ~{se_ratio:.1f}x; "
        f"mean setup share of total={setup_share:.0%}"
    )
    if "vs_prev_median_abs" in df.columns:
        last = df.dropna(subset=["vs_prev_median_abs"]).iloc[-1]
        msg += (
            f"; last step vs {last.get('vs_prev', '?')}: "
            f"med|Δ|={last['vs_prev_median_abs']:.5f} "
            f"max|Δ|={last['vs_prev_max_abs']:.5f}"
        )
    return msg


print("=== Takeaway ===")
print(
    f"Baseline: {_format_timings(timings)}  "
    f"medSE={np.median(se):.5f} maxSE={se.max():.5f}"
)
print(_dominant(walkers_df, "n_walkers"))
print(_dominant(t_df, "n_time_steps"))
print(_dominant(win_df, "t_window"))
print(
    "Note: first launch after Model build may include NVRTC compile in "
    "t_simulate; warm-up reduces but does not eliminate per-Model name rebuild cost."
)
print(f"Plots under: {PLOT_DIR}")
