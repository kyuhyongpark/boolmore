# %% [markdown]
# Cubewalkers: compare update schemes (NAV + timing)
#
# Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
# Runs **synchronous**, **asynchronous_set**, and **asynchronous** on the
# same model with scheme-specific horizons, then compares:
#
# - per-node NAV ± bootstrap SE
# - pairwise ΔNAV (B−A) from stored bootstrap mean replicates
# - split wall time (`t_setup` / `t_simulate` / `t_reduce`)
#
# Defaults (paper-ish horizons at fixed W):
#
# | scheme            | W    | T     | T_window |
# |--------------------|------|-------|----------|
# | synchronous        | 2048 | 800   | 160      |
# | asynchronous_set   | 2048 | 1200  | 24       |
# | asynchronous       | 2048 | 16000 | 3200     |
#
# Outputs overwrite under `dev/logs/cw_compare_update_schemes/`.
# Use `SMOKE=1` for a tiny end-to-end check.
#
# **Timing notes**
#
# - `t_setup` covers `cw.Model(...)` (rule parse + `RawKernel` construction).
# - NVRTC compile usually hits on the first kernel launch inside
#   `simulate_ensemble`, so compile often lands in `t_simulate`.
# - A discarded warm-up warms CuPy/driver but each Model still gets a unique
#   kernel name (possible recompile).
# - GPU is synchronized before closing `t_simulate` / `t_reduce`.

# %%
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.io.paths import project_root

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

OUT_DIR = Path(__file__).resolve().parent / "logs" / "cw_compare_update_schemes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUT_DIR
LOG_PATH = OUT_DIR / "cw_compare_update_schemes.log"
README_PATH = OUT_DIR / "README.md"
CSV_PATH = OUT_DIR / "scheme_nav.csv"
TIMING_CSV_PATH = OUT_DIR / "scheme_timings.csv"
STARTED_UTC = datetime.now(timezone.utc)
saved_plots: list[Path] = []

N_WALKERS = 2048
SET_UPDATE_PROB = 0.5
THREADS_PER_BLOCK: tuple[int, int] = (16, 16)
SEED = 0
N_BOOT = 1000
BOOT_CHUNK = 50

# Scheme-specific horizons (same W for all).
SCHEME_CONFIG: dict[str, dict[str, int]] = {
    "synchronous": {"n_time_steps": 800, "t_window": 160},
    "asynchronous_set": {"n_time_steps": 1200, "t_window": 240},
    "asynchronous": {"n_time_steps": 16000, "t_window": 3200},
}
SCHEME_ORDER = ("synchronous", "asynchronous_set", "asynchronous")

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_WALKERS = 64
    N_BOOT = 200
    BOOT_CHUNK = 40
    SCHEME_CONFIG = {
        "synchronous": {"n_time_steps": 80, "t_window": 16},
        "asynchronous_set": {"n_time_steps": 120, "t_window": 12},
        "asynchronous": {"n_time_steps": 400, "t_window": 80},
    }

SCHEME_COLORS = {
    "synchronous": "#4C72B0",
    "asynchronous_set": "#C44E52",
    "asynchronous": "#55A868",
}
SCHEME_SHORT = {
    "synchronous": "synch",
    "asynchronous_set": "asynch_set",
    "asynchronous": "asynch",
}

# %% [markdown]
# ## Logging

# %%
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print("Compare update schemes: NAV + bootstrap SE + timing")
print(f"out dir:  {OUT_DIR}")
print(f"log file: {LOG_PATH}")
print(f"started (UTC): {STARTED_UTC.isoformat()}")
print(f"SMOKE={SMOKE}")
print(f"W={N_WALKERS}  N_BOOT={N_BOOT}  SET_UPDATE_PROB={SET_UPDATE_PROB}")
for scheme in SCHEME_ORDER:
    cfg = SCHEME_CONFIG[scheme]
    print(f"  {scheme}: T={cfg['n_time_steps']}  T_window={cfg['t_window']}")
print()

# %% [markdown]
# ## CUDA / cubewalkers gate (fail-fast)

# %%
cw, cp = require_cubewalkers()
MASKFUNCTIONS = {
    "asynchronous": cw.update_schemes.asynchronous,
    "asynchronous_set": cw.update_schemes.asynchronous_set,
    "synchronous": cw.update_schemes.synchronous,
}
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

for scheme, cfg in SCHEME_CONFIG.items():
    if cfg["t_window"] > cfg["n_time_steps"]:
        raise SystemExit(
            f"{scheme}: T_window={cfg['t_window']} > T={cfg['n_time_steps']}"
        )

print(f"bnet: {BNET}")
print(f"nodes: {_n_nodes}")

_next_model_name = make_model_namer()

# %% [markdown]
# ## Plot helpers (script-local)

# %%
_NAV_GUIDES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _draw_nav_guidelines(ax) -> None:
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
# One Model+simulate so later timed setups are not the first process-wide
# compile of this network size.

# %%
print("Warm-up (discarded)...")
_wu_cfg = SCHEME_CONFIG["synchronous"]
_wu_nav, _wu_names, _wu_t = simulate_walker_nav(
    BNET_RULES,
    n_time_steps=min(50, _wu_cfg["n_time_steps"]),
    t_window=min(10, _wu_cfg["t_window"]),
    n_walkers=min(32, N_WALKERS),
    threads_per_block=THREADS_PER_BLOCK,
    model_name=_next_model_name("cw_warmup"),
    maskfunction=MASKFUNCTIONS["synchronous"],
    set_update_prob=SET_UPDATE_PROB,
)
print(f"  warm-up done: {format_timings(_wu_t)}  nodes={len(_wu_names)}")
del _wu_nav

# %% [markdown]
# ## Run all three schemes

# %%
rng = np.random.default_rng(SEED)
results: dict[str, dict] = {}

for scheme in SCHEME_ORDER:
    cfg = SCHEME_CONFIG[scheme]
    print(
        f"\n=== {scheme}  W={N_WALKERS}  "
        f"T={cfg['n_time_steps']}  window={cfg['t_window']} ==="
    )
    walker_nav, varnames, timings = simulate_walker_nav(
        BNET_RULES,
        n_time_steps=cfg["n_time_steps"],
        t_window=cfg["t_window"],
        n_walkers=N_WALKERS,
        threads_per_block=THREADS_PER_BLOCK,
        model_name=_next_model_name(f"cw_{SCHEME_SHORT[scheme]}"),
        maskfunction=MASKFUNCTIONS[scheme],
        set_update_prob=SET_UPDATE_PROB,
    )
    means, se, boot_means, t_boot = bootstrap_se(
        walker_nav, n_boot=N_BOOT, chunk=BOOT_CHUNK, rng=rng
    )
    del walker_nav

    row = {
        "scheme": scheme,
        "short": SCHEME_SHORT[scheme],
        "n_walkers": N_WALKERS,
        "n_time_steps": cfg["n_time_steps"],
        "t_window": cfg["t_window"],
        "varnames": varnames,
        "means": means,
        "se": se,
        "boot_means": boot_means,
        "median_se": float(np.median(se)),
        "mean_se": float(se.mean()),
        "max_se": float(se.max()),
        "mean_of_means": float(means.mean()),
        "t_boot": t_boot,
        **timings,
    }
    results[scheme] = row
    print(f"  {format_timings(timings)}")
    print(f"  bootstrap: t_boot={t_boot:.3f}s  N_BOOT={N_BOOT}")
    print(
        f"  NAV: mean of means={row['mean_of_means']:.4f}  "
        f"median SE={row['median_se']:.5f}  max SE={row['max_se']:.5f}"
    )

# Align on first scheme's varname order (cubewalkers should match).
ref_names = results[SCHEME_ORDER[0]]["varnames"]
for scheme in SCHEME_ORDER[1:]:
    if results[scheme]["varnames"] != ref_names:
        raise RuntimeError(
            f"varname mismatch: {SCHEME_ORDER[0]} vs {scheme}"
        )

# %% [markdown]
# ## Pairwise ΔNAV (bootstrap)

# %%
pair_diffs: dict[tuple[str, str], dict] = {}

print("\n=== Pairwise NAV differences (B − A) ===")
for scheme_a, scheme_b in combinations(SCHEME_ORDER, 2):
    a = results[scheme_a]
    b = results[scheme_b]
    diff, se_diff = diff_from_boot(
        a["means"], a["boot_means"], b["means"], b["boot_means"]
    )
    summary = diff_summary(diff, se_diff)
    # Rename keys from vs_prev_* → clearer pair labels for tables.
    pair_key = (scheme_a, scheme_b)
    pair_diffs[pair_key] = {
        "scheme_a": scheme_a,
        "scheme_b": scheme_b,
        "label": f"{SCHEME_SHORT[scheme_b]} − {SCHEME_SHORT[scheme_a]}",
        "diff": diff,
        "se_diff": se_diff,
        "median_abs": summary["vs_prev_median_abs"],
        "mean_abs": summary["vs_prev_mean_abs"],
        "max_abs": summary["vs_prev_max_abs"],
        "median_se": summary["vs_prev_median_se"],
        "mean_se": summary["vs_prev_mean_se"],
        "max_se": summary["vs_prev_max_se"],
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": float(np.corrcoef(a["means"], b["means"])[0, 1]),
    }
    p = pair_diffs[pair_key]
    print(
        f"  {p['label']}: med|Δ|={p['median_abs']:.5f}  "
        f"mean|Δ|={p['mean_abs']:.5f}  max|Δ|={p['max_abs']:.5f}  "
        f"RMSE={p['rmse']:.5f}  corr={p['corr']:.4f}  "
        f"medSE(Δ)={p['median_se']:.5f}"
    )

# Free bootstrap replicates once pairwise diffs are done.
for scheme in SCHEME_ORDER:
    results[scheme].pop("boot_means", None)

# %% [markdown]
# ## Timing table

# %%
timing_rows = []
for scheme in SCHEME_ORDER:
    r = results[scheme]
    timing_rows.append(
        {
            "scheme": scheme,
            "short": r["short"],
            "n_walkers": r["n_walkers"],
            "n_time_steps": r["n_time_steps"],
            "t_window": r["t_window"],
            "t_setup": r["t_setup"],
            "t_simulate": r["t_simulate"],
            "t_reduce": r["t_reduce"],
            "t_total": r["t_total"],
            "t_boot": r["t_boot"],
            "median_se": r["median_se"],
            "max_se": r["max_se"],
            "mean_of_means": r["mean_of_means"],
        }
    )
timing_df = pd.DataFrame(timing_rows)
print("\nTiming / SE summary:")
print(timing_df.to_string(index=False))
timing_df.to_csv(TIMING_CSV_PATH, index=False)
print(f"Wrote {TIMING_CSV_PATH}")

# Relative to synchronous total time.
t_ref = float(results["synchronous"]["t_total"])
print("\nTiming relative to synchronous t_total:")
for scheme in SCHEME_ORDER:
    t = float(results[scheme]["t_total"])
    print(
        f"  {SCHEME_SHORT[scheme]}: t_total={t:.3f}s  "
        f"ratio={t / max(t_ref, 1e-12):.2f}x  "
        f"Δ={t - t_ref:+.3f}s"
    )

# %% [markdown]
# ## Per-node NAV table

# %%
nav_df = pd.DataFrame({"node": ref_names})
for scheme in SCHEME_ORDER:
    short = SCHEME_SHORT[scheme]
    nav_df[f"{short}_mean"] = results[scheme]["means"]
    nav_df[f"{short}_se"] = results[scheme]["se"]
for (scheme_a, scheme_b), p in pair_diffs.items():
    col = f"d_{SCHEME_SHORT[scheme_b]}_minus_{SCHEME_SHORT[scheme_a]}"
    nav_df[col] = p["diff"]
    nav_df[f"se_{col}"] = p["se_diff"]

nav_df = nav_df.sort_values("synch_mean", ascending=False)
nav_df.to_csv(CSV_PATH, index=False)
print(f"\nWrote {CSV_PATH}")
print("Top 10 nodes by synchronous NAV:")
_show = [
    "node",
    "synch_mean",
    "asynch_set_mean",
    "asynch_mean",
]
print(nav_df[_show].head(10).to_string(index=False))

# %% [markdown]
# ## Plot — overlay NAV ± bootstrap SE

# %%
# Order nodes by synchronous mean for a stable y-axis.
order = np.argsort(results["synchronous"]["means"])
ordered_names = [ref_names[i] for i in order]
y = np.arange(len(ordered_names))
n_schemes = len(SCHEME_ORDER)
jitter = 0.55 * np.linspace(-0.5, 0.5, n_schemes)

fig, ax = plt.subplots(figsize=(10, max(4, 0.12 * len(ordered_names))))
_draw_nav_guidelines(ax)
for i, scheme in enumerate(SCHEME_ORDER):
    r = results[scheme]
    color = SCHEME_COLORS[scheme]
    means = np.asarray(r["means"], dtype=np.float64)[order]
    se = np.asarray(r["se"], dtype=np.float64)[order]
    cfg = SCHEME_CONFIG[scheme]
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
        label=(
            f"{SCHEME_SHORT[scheme]}  "
            f"T={cfg['n_time_steps']}, win={cfg['t_window']}"
        ),
        zorder=2,
    )
ax.set_yticks(y)
ax.set_yticklabels(ordered_names, fontsize=6)
ax.set_xlabel("Node average value (NAV)")
ax.set_title(
    f"Update-scheme NAV ± bootstrap SE\n"
    f"W={N_WALKERS}, B={N_BOOT}"
)
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
fig.tight_layout()
plot_overlay = PLOT_DIR / "cw_compare_schemes_nav_overlay.png"
fig.savefig(plot_overlay, dpi=120)
plt.close(fig)
saved_plots.append(plot_overlay)
print(f"Wrote {plot_overlay}")

# %% [markdown]
# ## Plot — pairwise ΔNAV

# %%
n_pairs = len(pair_diffs)
fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, max(4, 0.12 * len(ordered_names))))
if n_pairs == 1:
    axes = [axes]

for ax, ((scheme_a, scheme_b), p) in zip(axes, pair_diffs.items()):
    diff = np.asarray(p["diff"], dtype=np.float64)[order]
    se_d = np.asarray(p["se_diff"], dtype=np.float64)[order]
    ax.axvline(0.0, color="#B0B0B0", linestyle=":", linewidth=0.8, zorder=0)
    ax.errorbar(
        diff,
        y,
        xerr=se_d,
        fmt="o",
        markersize=2.5,
        elinewidth=0.6,
        capsize=1.0,
        color="#2C5F7C",
        ecolor="#7A9EAD",
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_names, fontsize=5)
    ax.set_xlabel(f"ΔNAV ({p['label']})")
    ax.set_title(
        f"{p['label']}\n"
        f"med|Δ|={p['median_abs']:.4f}  max|Δ|={p['max_abs']:.4f}"
    )
fig.suptitle("Pairwise scheme differences (bootstrap SE of Δ)", y=1.01)
fig.tight_layout()
plot_diff = PLOT_DIR / "cw_compare_schemes_nav_diff.png"
fig.savefig(plot_diff, dpi=120, bbox_inches="tight")
plt.close(fig)
saved_plots.append(plot_diff)
print(f"Wrote {plot_diff}")

# %% [markdown]
# ## Plot — |Δ| distribution + timing bars

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
for (scheme_a, scheme_b), p in pair_diffs.items():
    abs_diff = np.abs(p["diff"])
    ax.hist(
        abs_diff,
        bins=30,
        range=(0, max(0.05, float(abs_diff.max()) * 1.05)),
        histtype="step",
        linewidth=1.5,
        label=p["label"],
        color=SCHEME_COLORS[scheme_b],
    )
ax.set_xlabel("|ΔNAV|")
ax.set_ylabel("Node count")
ax.set_title("Distribution of |ΔNAV| across nodes")
ax.legend(fontsize=7)

ax = axes[1]
x = np.arange(len(SCHEME_ORDER))
width = 0.2
for j, key in enumerate(("t_setup", "t_simulate", "t_reduce", "t_total")):
    vals = [results[s][key] for s in SCHEME_ORDER]
    ax.bar(
        x + (j - 1.5) * width,
        vals,
        width,
        label=key,
    )
ax.set_xticks(x)
ax.set_xticklabels([SCHEME_SHORT[s] for s in SCHEME_ORDER])
ax.set_ylabel("Wall time (s)")
ax.set_title("Split timings by scheme")
ax.legend(fontsize=7)

fig.tight_layout()
plot_summary = PLOT_DIR / "cw_compare_schemes_summary.png"
fig.savefig(plot_summary, dpi=120)
plt.close(fig)
saved_plots.append(plot_summary)
print(f"Wrote {plot_summary}")

# %% [markdown]
# ## Scatter — scheme vs scheme NAV

# %%
fig, axes = plt.subplots(1, n_pairs, figsize=(4.2 * n_pairs, 4))
if n_pairs == 1:
    axes = [axes]

for ax, ((scheme_a, scheme_b), p) in zip(axes, pair_diffs.items()):
    xa = results[scheme_a]["means"]
    xb = results[scheme_b]["means"]
    ax.plot([0, 1], [0, 1], ":", color="#B0B0B0", zorder=0)
    ax.scatter(
        xa,
        xb,
        s=12,
        alpha=0.75,
        color="#2C5F7C",
        edgecolors="none",
        zorder=2,
    )
    ax.set_xlabel(f"NAV ({SCHEME_SHORT[scheme_a]})")
    ax.set_ylabel(f"NAV ({SCHEME_SHORT[scheme_b]})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"corr={p['corr']:.4f}  RMSE={p['rmse']:.4f}")

fig.suptitle("Per-node NAV agreement across schemes", y=1.02)
fig.tight_layout()
plot_scatter = PLOT_DIR / "cw_compare_schemes_nav_scatter.png"
fig.savefig(plot_scatter, dpi=120, bbox_inches="tight")
plt.close(fig)
saved_plots.append(plot_scatter)
print(f"Wrote {plot_scatter}")

# %% [markdown]
# ## Takeaway + README

# %%
finished_utc = datetime.now(timezone.utc)
wall_s = (finished_utc - STARTED_UTC).total_seconds()

print("\n=== Takeaway ===")
for scheme in SCHEME_ORDER:
    r = results[scheme]
    print(
        f"{SCHEME_SHORT[scheme]}: {format_timings(r)}  "
        f"medSE={r['median_se']:.5f}  meanNAV={r['mean_of_means']:.4f}"
    )
for p in pair_diffs.values():
    print(
        f"{p['label']}: med|Δ|={p['median_abs']:.5f}  "
        f"max|Δ|={p['max_abs']:.5f}  RMSE={p['rmse']:.5f}  "
        f"corr={p['corr']:.4f}"
    )
print(f"Wall clock: {wall_s:.1f}s")
print(f"Outputs under: {OUT_DIR}")

readme_lines = [
    "# Cubewalkers: compare update schemes (NAV + timing)",
    "",
    "Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.",
    "Compares `synchronous`, `asynchronous_set`, and `asynchronous` NAV",
    "(bootstrap SE) and split wall times at scheme-specific horizons.",
    "",
    f"- **Started (UTC):** {STARTED_UTC.isoformat()}",
    f"- **Finished (UTC):** {finished_utc.isoformat()}",
    f"- **Wall clock:** {wall_s:.1f}s",
    f"- **SMOKE:** `{SMOKE}`",
    f"- **bnet:** `{BNET.name}`",
    f"- **nodes:** {_n_nodes}",
    f"- **W:** {N_WALKERS}",
    f"- **N_BOOT:** {N_BOOT}",
    f"- **seed:** {SEED}",
    f"- **set_update_prob:** {SET_UPDATE_PROB}",
    "",
    "## Horizons",
    "",
]
for scheme in SCHEME_ORDER:
    cfg = SCHEME_CONFIG[scheme]
    readme_lines.append(
        f"- **{scheme}:** T={cfg['n_time_steps']}, "
        f"T_window={cfg['t_window']}"
    )
readme_lines.extend(["", "## Per-scheme summary", ""])
for scheme in SCHEME_ORDER:
    r = results[scheme]
    readme_lines.extend(
        [
            f"### `{scheme}`",
            "",
            f"- Timings: `{format_timings(r)}`",
            f"- mean of node means: `{r['mean_of_means']:.4f}`",
            f"- median SE: `{r['median_se']:.5f}`",
            f"- max SE: `{r['max_se']:.5f}`",
            "",
        ]
    )
readme_lines.extend(
    [
        "## Timing vs synchronous",
        "",
    ]
)
for scheme in SCHEME_ORDER:
    t = float(results[scheme]["t_total"])
    readme_lines.append(
        f"- **{SCHEME_SHORT[scheme]}:** t_total=`{t:.3f}s`  "
        f"ratio=`{t / max(t_ref, 1e-12):.2f}x`  "
        f"Δ=`{t - t_ref:+.3f}s`"
    )
readme_lines.extend(["", "## Pairwise ΔNAV", ""])
for p in pair_diffs.values():
    readme_lines.append(
        f"- **{p['label']}:** med|Δ|=`{p['median_abs']:.5f}`  "
        f"mean|Δ|=`{p['mean_abs']:.5f}`  max|Δ|=`{p['max_abs']:.5f}`  "
        f"RMSE=`{p['rmse']:.5f}`  corr=`{p['corr']:.4f}`  "
        f"medSE(Δ)=`{p['median_se']:.5f}`"
    )
readme_lines.extend(
    [
        "",
        "## Artifacts",
        "",
        "Stable filenames (overwritten on each run):",
        "",
        f"- [`{LOG_PATH.name}`]({LOG_PATH.name}) — full run log",
        f"- [`{README_PATH.name}`]({README_PATH.name}) — this summary",
        f"- [`{CSV_PATH.name}`]({CSV_PATH.name}) — per-node NAV / Δ",
        f"- [`{TIMING_CSV_PATH.name}`]({TIMING_CSV_PATH.name}) — timings",
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
print(f"csv: {CSV_PATH}")
print(f"timings: {TIMING_CSV_PATH}")
for p in saved_plots:
    print(f"plot saved: {p}")
