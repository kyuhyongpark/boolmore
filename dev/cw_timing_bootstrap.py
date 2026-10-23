# %% [markdown]
# Cubewalkers timing + bootstrap SE
#
# Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
# Measures split wall time (`t_setup` / `t_simulate` / `t_reduce`) and
# per-node bootstrap SE of the time-window NAV, then sweeps
# `n_walkers`, `N_TIME_STEPS`, and `T_WINDOW`.
#
# Update scheme is selected by `UPDATE_SCHEME` near the top
# (`asynchronous` | `asynchronous_set` | `synchronous`). Outputs (log,
# README, plots) overwrite under `dev/logs/cw_timing_bootstrap/<scheme>/`.
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
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.io.paths import project_root

# Sibling helpers are not on the editable install; keep ``dev/`` importable.
_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from cw_helpers import (
    bnet_rules_for_cubewalkers,
    bootstrap_se,
    diff_from_boot,
    diff_summary,
    format_timings,
    make_model_namer,
    require_cubewalkers,
    simulate_walker_nav,
)


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

# Cubewalkers update scheme: which nodes update each timestep.
#   asynchronous      — one random node per walker
#   asynchronous_set  — each node updates with prob SET_UPDATE_PROB
#   synchronous       — all nodes every step
UPDATE_SCHEME = "asynchronous"  # or "asynchronous_set" | "synchronous"
SET_UPDATE_PROB = 0.5  # used only when UPDATE_SCHEME == "asynchronous_set"

_ALLOWED_SCHEMES = ("asynchronous", "asynchronous_set", "synchronous")
if UPDATE_SCHEME not in _ALLOWED_SCHEMES:
    raise SystemExit(
        f"UPDATE_SCHEME={UPDATE_SCHEME!r} not in {_ALLOWED_SCHEMES}"
    )

OUT_DIR = (
    Path(__file__).resolve().parent
    / "logs"
    / "cw_timing_bootstrap"
    / UPDATE_SCHEME
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUT_DIR  # plots share the scheme output dir
LOG_PATH = OUT_DIR / "cw_timing_bootstrap.log"
README_PATH = OUT_DIR / "README.md"
STARTED_UTC = datetime.now(timezone.utc)
saved_plots: list[Path] = []

# Practical baseline / sweep base (paper-scale is optional — see RUN_PAPER_BASELINE).
# Full traj is (T_window, N, W) bools; paper defaults can be ~1–2 GB on device.
N_WALKERS = 2048
N_TIME_STEPS = 8000
T_WINDOW = 2000

# Sweep base (held fixed while varying one axis).
SWEEP_WALKERS = 2048
SWEEP_T = 8000
SWEEP_WINDOW = 2000

WALKERS_GRID = [64, 128, 256, 512, 1024, 2048, 4096]
T_GRID = [2000, 4000, 6000, 8000, 10000]
WINDOW_GRID = [50, 100, 250, 500, 1000, 2000, 4000]

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
# ## Logging

# %%
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"UPDATE_SCHEME={UPDATE_SCHEME}")
if UPDATE_SCHEME == "asynchronous_set":
    print(f"SET_UPDATE_PROB={SET_UPDATE_PROB}")
print(f"out dir:  {OUT_DIR}")
print(f"log file: {LOG_PATH}")
print(f"started (UTC): {STARTED_UTC.isoformat()}")
print(f"SMOKE={SMOKE}  RUN_PAPER_BASELINE={RUN_PAPER_BASELINE}")
print()

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast)

# %%
cw, cp = require_cubewalkers()
MASKFUNCTION = {
    "asynchronous": cw.update_schemes.asynchronous,
    "asynchronous_set": cw.update_schemes.asynchronous_set,
    "synchronous": cw.update_schemes.synchronous,
}[UPDATE_SCHEME]
print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
print(f"cubewalkers: {getattr(cw, '__file__', cw)}")

# %% [markdown]
# ## Load bnet (no experiment data)

# %%
BNET_RULES = bnet_rules_for_cubewalkers(BNET)
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

_next_model_name = make_model_namer()

# %% [markdown]
# ## Plot helpers (script-local)

# %%
_NAV_GUIDES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _draw_nav_guidelines(ax) -> None:
    """Light vertical guides at common NAV reference values."""
    for x in _NAV_GUIDES:
        ax.axvline(
            x,
            color="#B0B0B0",
            linestyle=":",
            linewidth=0.8,
            zorder=0,
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
    threads_per_block=THREADS_PER_BLOCK,
    model_name=_next_model_name("cw_warmup"),
    maskfunction=MASKFUNCTION,
    set_update_prob=SET_UPDATE_PROB,
)
print(f"  warm-up done: {format_timings(_wu_t)}  nodes={len(_wu_names)}")
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
    threads_per_block=THREADS_PER_BLOCK,
    model_name=_next_model_name("cw_baseline"),
    maskfunction=MASKFUNCTION,
    set_update_prob=SET_UPDATE_PROB,
)
means, se, boot_means, t_boot = bootstrap_se(
    walker_nav, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng
)
del walker_nav, boot_means

print(f"  {format_timings(timings)}")
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
        threads_per_block=THREADS_PER_BLOCK,
        model_name=_next_model_name("cw_paper"),
        maskfunction=MASKFUNCTION,
        set_update_prob=SET_UPDATE_PROB,
    )
    paper_means, paper_se, paper_boot_means, paper_boot = bootstrap_se(
        paper_nav, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng
    )
    print(f"  {format_timings(paper_t)}  t_boot={paper_boot:.3f}s")
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
_draw_nav_guidelines(ax)
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
    zorder=2,
)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["node"], fontsize=6)
ax.set_xlabel("Node average value (NAV)")
ax.set_title(
    f"Baseline NAV ± bootstrap SE  [{UPDATE_SCHEME}]\n"
    f"W={N_WALKERS}, T={N_TIME_STEPS}, window={T_WINDOW}, B={N_BOOT}"
)
ax.set_xlim(-0.05, 1.05)
fig.tight_layout()
plot_baseline = PLOT_DIR / "cw_timing_bootstrap_baseline_nav.png"
fig.savefig(plot_baseline, dpi=120)
plt.close(fig)
saved_plots.append(plot_baseline)
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
        threads_per_block=THREADS_PER_BLOCK,
        model_name=_next_model_name(f"cw_sweep_{label}"),
        maskfunction=MASKFUNCTION,
        set_update_prob=SET_UPDATE_PROB,
    )
    means, se, boot_means, t_boot = bootstrap_se(
        walker_nav, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng
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
        f"{format_timings(timings)}  "
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
        summary = diff_summary(diff, se_diff)
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
    saved_plots.append(path)
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
    _draw_nav_guidelines(ax)
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
            zorder=2,
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
    saved_plots.append(path)
    print(f"Wrote {path}")


def _rows_for_prefix(prefix: str) -> list[dict]:
    return [r for r in sweep_rows if str(r["label"]).startswith(prefix)]


walkers_df = sweep_df[sweep_df["label"].str.startswith("W")].sort_values("n_walkers")
t_df = sweep_df[sweep_df["label"].str.startswith("T")].sort_values("n_time_steps")
win_df = sweep_df[sweep_df["label"].str.startswith("win")].sort_values("t_window")

_plot_factor(
    walkers_df,
    "n_walkers",
    f"Vary n_walkers [{UPDATE_SCHEME}] (T={SWEEP_T}, window={SWEEP_WINDOW})",
    PLOT_DIR / "cw_timing_bootstrap_sweep_walkers.png",
)
_plot_factor(
    t_df,
    "n_time_steps",
    f"Vary N_TIME_STEPS [{UPDATE_SCHEME}] (W={SWEEP_WALKERS}, window≤{SWEEP_WINDOW})",
    PLOT_DIR / "cw_timing_bootstrap_sweep_T.png",
)
_plot_factor(
    win_df,
    "t_window",
    f"Vary T_WINDOW [{UPDATE_SCHEME}] (W={SWEEP_WALKERS}, T={t_for_window})",
    PLOT_DIR / "cw_timing_bootstrap_sweep_window.png",
)

walkers_pts = sorted(_rows_for_prefix("W"), key=lambda r: r["n_walkers"])
t_pts = sorted(_rows_for_prefix("T"), key=lambda r: r["n_time_steps"])
win_pts = sorted(_rows_for_prefix("win"), key=lambda r: r["t_window"])

_plot_sweep_nav_overlay(
    walkers_pts,
    title=(
        f"Sweep n_walkers — NAV ± bootstrap SE  [{UPDATE_SCHEME}]\n"
        f"T={SWEEP_T}, window={SWEEP_WINDOW}, B={N_BOOT}"
    ),
    path=PLOT_DIR / "cw_timing_bootstrap_sweep_walkers_nav.png",
)
_plot_sweep_nav_overlay(
    t_pts,
    title=(
        f"Sweep N_TIME_STEPS — NAV ± bootstrap SE  [{UPDATE_SCHEME}]\n"
        f"W={SWEEP_WALKERS}, window≤{SWEEP_WINDOW}, B={N_BOOT}"
    ),
    path=PLOT_DIR / "cw_timing_bootstrap_sweep_T_nav.png",
)
_plot_sweep_nav_overlay(
    win_pts,
    title=(
        f"Sweep T_WINDOW — NAV ± bootstrap SE  [{UPDATE_SCHEME}]\n"
        f"W={SWEEP_WALKERS}, T={t_for_window}, B={N_BOOT}"
    ),
    path=PLOT_DIR / "cw_timing_bootstrap_sweep_window_nav.png",
)

# %% [markdown]
# ## Takeaway + README

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


finished_utc = datetime.now(timezone.utc)
wall_s = (finished_utc - STARTED_UTC).total_seconds()

takeaway_walkers = _dominant(walkers_df, "n_walkers")
takeaway_t = _dominant(t_df, "n_time_steps")
takeaway_win = _dominant(win_df, "t_window")
baseline_med_se = float(np.median(se))
baseline_max_se = float(se.max())

print("=== Takeaway ===")
print(f"UPDATE_SCHEME={UPDATE_SCHEME}")
print(
    f"Baseline: {format_timings(timings)}  "
    f"medSE={baseline_med_se:.5f} maxSE={baseline_max_se:.5f}"
)
print(takeaway_walkers)
print(takeaway_t)
print(takeaway_win)
print(
    "Note: first launch after Model build may include NVRTC compile in "
    "t_simulate; warm-up reduces but does not eliminate per-Model name rebuild cost."
)
print(f"Wall clock: {wall_s:.1f}s")
print(f"Outputs under: {OUT_DIR}")

readme_lines = [
    f"# Cubewalkers timing + bootstrap SE (`{UPDATE_SCHEME}`)",
    "",
    "Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.",
    "Split wall time (`t_setup` / `t_simulate` / `t_reduce`) and per-node",
    "bootstrap SE of the time-window NAV; sweeps `n_walkers`, `N_TIME_STEPS`,",
    "and `T_WINDOW`.",
    "",
    f"- **Update scheme:** `{UPDATE_SCHEME}`",
]
if UPDATE_SCHEME == "asynchronous_set":
    readme_lines.append(f"- **set_update_prob:** `{SET_UPDATE_PROB}`")
readme_lines.extend(
    [
        f"- **Started (UTC):** {STARTED_UTC.isoformat()}",
        f"- **Finished (UTC):** {finished_utc.isoformat()}",
        f"- **Wall clock:** {wall_s:.1f}s",
        f"- **SMOKE:** `{SMOKE}`",
        f"- **bnet:** `{BNET.name}`",
        f"- **nodes:** {_n_nodes}",
        f"- **Baseline:** W={N_WALKERS}, T={N_TIME_STEPS}, "
        f"window={T_WINDOW}, N_BOOT={N_BOOT}, seed={SEED}",
        f"- **Sweep base:** W={SWEEP_WALKERS}, T={SWEEP_T}, "
        f"window={SWEEP_WINDOW}",
        f"- **Grids:** walkers={WALKERS_GRID}, T={T_GRID}, "
        f"window={WINDOW_GRID}",
        "",
        "## Baseline",
        "",
        f"- Timings: `{format_timings(timings)}`",
        f"- median SE: `{baseline_med_se:.5f}`",
        f"- max SE: `{baseline_max_se:.5f}`",
        f"- mean of node means: `{float(means.mean()):.4f}`",
        "",
        "## Sweep takeaways",
        "",
        f"- {takeaway_walkers}",
        f"- {takeaway_t}",
        f"- {takeaway_win}",
        "",
        "Note: first launch after Model build may include NVRTC compile in",
        "`t_simulate`; warm-up reduces but does not eliminate per-Model name",
        "rebuild cost.",
        "",
        "## Artifacts",
        "",
        "Stable filenames (overwritten on each run):",
        "",
        f"- [`{LOG_PATH.name}`]({LOG_PATH.name}) — full run log",
        f"- [`{README_PATH.name}`]({README_PATH.name}) — this summary",
    ]
)
for p in saved_plots:
    readme_lines.append(f"- [`{p.name}`]({p.name})")
readme_lines.append("")

README_PATH.write_text("\n".join(readme_lines), encoding="utf-8")
print(f"saved: {README_PATH}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
print(f"readme: {README_PATH}")
for p in saved_plots:
    print(f"plot saved: {p}")
