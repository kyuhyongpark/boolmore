# %% [markdown]
# Focused plots from ``investigate_ldoi_ra_243_source_combos`` artifacts
#
# Reuses cached CSVs under the combo run directory. Does not recompute NAV.
#
# Methods shown:
# - ``ldoi`` → percolation
# - ``ldoi+primer+cw`` → cw on reduced
# - ``cubewalkers`` → compute-time plot only
#
# Outputs:
# 1. ``method_compute_times.png``
# 2. ``nodes_pooled_ecdf_summary.png`` (ECDF | box over node×combo)
# 3. ``nodes_ldoi_unfixed_ecdf_summary.png`` (ECDF | box over node×combo)
# 4. ``combos_pooled_ecdf_summary.png`` (ECDF | box of per-combo mean |Δ|)
# 5. ``combos_ldoi_unfixed_ecdf_summary.png`` (same, unfixed means)

# %%
from __future__ import annotations

from pathlib import Path

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
SRC_RUN = (
    Path(__file__).resolve().parent
    / "logs"
    / "investigate_ldoi_ra_243_source_combos"
    / "243"
    / "combo-seed-0"
)
OUT_DIR = (
    Path(__file__).resolve().parent
    / "logs"
    / "plot_ldoi_ra_243_focus"
    / "243"
    / "combo-seed-0"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_LDOI = "ldoi"
METHOD_PCW = "ldoi+primer+cw"
METHOD_CW = "cubewalkers"

DISPLAY = {
    METHOD_LDOI: "percolation",
    METHOD_PCW: "cw on reduced",
    METHOD_CW: "cubewalkers",
}
COLORS = {
    METHOD_LDOI: "#2c3e50",
    METHOD_PCW: "#ff7f0e",
    METHOD_CW: "#7f7f7f",
}
DELTA_METHODS = [METHOD_LDOI, METHOD_PCW]
TIMING_METHODS = [METHOD_LDOI, METHOD_PCW, METHOD_CW]

ABS_COL = {
    METHOD_LDOI: "abs_delta_ldoi",
    METHOD_PCW: "abs_delta_ldoi_pcw",
}
PER_COMBO_MEAN_COL = {
    METHOD_LDOI: "mean_abs_delta_ldoi",
    METHOD_PCW: "mean_abs_delta_ldoi_pcw",
}
PER_COMBO_MEAN_UNFIXED_COL = {
    METHOD_LDOI: "mean_abs_delta_ldoi_unfixed",
    METHOD_PCW: "mean_abs_delta_ldoi_pcw_unfixed",
}

# %% [markdown]
# ## Load artifacts

# %%
node_cmp = pd.read_csv(SRC_RUN / "node_cmp.csv")
per_combo = pd.read_csv(SRC_RUN / "per_combo_summary.csv")
timings_summary = pd.read_csv(SRC_RUN / "method_timings_summary.csv")

if "ldoi_unfixed" not in node_cmp.columns:
    raise SystemExit(
        f"missing ldoi_unfixed column in {SRC_RUN / 'node_cmp.csv'}; "
        "re-run investigate_ldoi_ra_243_source_combos.py first"
    )
for col in list(PER_COMBO_MEAN_COL.values()) + list(
    PER_COMBO_MEAN_UNFIXED_COL.values()
):
    if col not in per_combo.columns:
        raise SystemExit(
            f"missing {col} in {SRC_RUN / 'per_combo_summary.csv'}; "
            "re-run investigate_ldoi_ra_243_source_combos.py first"
        )

print(f"source run: {SRC_RUN}")
print(f"out dir   : {OUT_DIR}")
print(f"node×combo rows: {len(node_cmp)}")
print(f"ldoi_unfixed rows: {int(node_cmp['ldoi_unfixed'].sum())}")
print(f"combos: {len(per_combo)}")
print(
    f"combos with ≥1 LDOI-unfixed: "
    f"{int((per_combo['n_ldoi_unfixed'] > 0).sum())}"
)
print()

# %% [markdown]
# ## 1. Method compute times


# %%
def plot_method_compute_times(path: Path) -> Path:
    means = []
    for m in TIMING_METHODS:
        sub = timings_summary.loc[timings_summary["method"] == m, "mean_s"]
        if sub.empty:
            raise SystemExit(f"missing timing for method={m!r}")
        means.append(float(sub.iloc[0]))

    labels = [DISPLAY[m] for m in TIMING_METHODS]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(TIMING_METHODS))
    bars = ax.bar(
        x,
        means,
        color=[COLORS[m] for m in TIMING_METHODS],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.62,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Mean wall time per combo (s)")
    ax.set_title("Compute time by method (mean over combos)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    for bar, val in zip(bars, means):
        if val < 0.01:
            label = f"{val:.3f}s"
        elif val < 100:
            label = f"{val:.2f}s"
        else:
            label = f"{val:.1f}s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# %% [markdown]
# ## 2–5. ECDF + box plot (side by side)


# %%
def _series_from_node_cmp(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        m: df[ABS_COL[m]].to_numpy(dtype=np.float64) for m in DELTA_METHODS
    }


def _series_from_per_combo(
    df: pd.DataFrame,
    col_map: dict[str, str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for m in DELTA_METHODS:
        vals = df[col_map[m]].to_numpy(dtype=np.float64)
        out[m] = vals[np.isfinite(vals)]
    return out


def plot_ecdf_box(
    series: dict[str, np.ndarray],
    *,
    path: Path,
    title: str,
    value_label: str = "|Δ| NAV vs CubeWalkers (sync)",
    ecdf_item: str = "items",
) -> Path:
    fig, (ax_ecdf, ax_box) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.4),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    nonempty = [v for v in series.values() if len(v)]
    if not nonempty:
        print(f"  (skipped {path.name} — no values)")
        plt.close(fig)
        return path
    xmax = float(np.nanmax(np.concatenate(nonempty)))
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 0.5
    xmax = max(xmax, 0.5)

    for m in DELTA_METHODS:
        vals = np.sort(series[m])
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax_ecdf.step(
            vals,
            y,
            where="post",
            color=COLORS[m],
            linewidth=2.2,
            label=DISPLAY[m],
        )
    ax_ecdf.set_xlabel(value_label)
    ax_ecdf.set_ylabel(f"ECDF  (fraction of {ecdf_item} ≤ x)")
    ax_ecdf.set_title("Empirical CDF")
    ax_ecdf.legend(frameon=False, loc="lower right")
    ax_ecdf.set_xlim(0.0, xmax * 1.02)
    ax_ecdf.set_ylim(0.0, 1.02)
    ax_ecdf.grid(alpha=0.3)

    data = [series[m] for m in DELTA_METHODS]
    labels = [DISPLAY[m] for m in DELTA_METHODS]
    bp = ax_box.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        whis=(5, 95),
        widths=0.55,
        patch_artist=True,
    )
    for patch, m in zip(bp["boxes"], DELTA_METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.45)
        patch.set_edgecolor(COLORS[m])
    for m, vals, xpos in zip(DELTA_METHODS, data, range(1, len(DELTA_METHODS) + 1)):
        if len(vals) == 0:
            continue
        ax_box.scatter(
            xpos,
            np.mean(vals),
            color=COLORS[m],
            marker="D",
            s=42,
            zorder=4,
            edgecolors="white",
            linewidths=0.6,
            label="mean" if xpos == 1 else None,
        )
    ax_box.set_ylabel(value_label)
    ax_box.set_title("distribution")
    ax_box.set_ylim(0.0, xmax * 1.02)
    ax_box.tick_params(axis="x", rotation=12)
    ax_box.grid(axis="y", alpha=0.3)
    handles, labels_leg = ax_box.get_legend_handles_labels()
    if handles:
        ax_box.legend(handles[:1], labels_leg[:1], loc="upper right", frameon=False)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# %%
p_time = plot_method_compute_times(OUT_DIR / "method_compute_times.png")
print(f"wrote {p_time}")

p_pooled = plot_ecdf_box(
    _series_from_node_cmp(node_cmp),
    path=OUT_DIR / "nodes_pooled_ecdf_summary.png",
    title="All nodes × combos",
)
print(f"wrote {p_pooled}")

unfixed = node_cmp[node_cmp["ldoi_unfixed"].astype(bool)]
p_unfixed = plot_ecdf_box(
    _series_from_node_cmp(unfixed),
    path=OUT_DIR / "nodes_ldoi_unfixed_ecdf_summary.png",
    title="Unfixed nodes only (set varies by combo)",
)
print(f"wrote {p_unfixed}")

p_combo_pooled = plot_ecdf_box(
    _series_from_per_combo(per_combo, PER_COMBO_MEAN_COL),
    path=OUT_DIR / "combos_pooled_ecdf_summary.png",
    title="Per-combo mean |Δ| (all nodes)",
    value_label="mean |Δ| per combo (all nodes)",
    ecdf_item="combos",
)
print(f"wrote {p_combo_pooled}")

p_combo_unfixed = plot_ecdf_box(
    _series_from_per_combo(per_combo, PER_COMBO_MEAN_UNFIXED_COL),
    path=OUT_DIR / "combos_ldoi_unfixed_ecdf_summary.png",
    title="Per-combo mean |Δ| on unfixed nodes only",
    value_label="mean |Δ| per combo (unfixed nodes only)",
    ecdf_item="combos",
)
print(f"wrote {p_combo_unfixed}")

print()
print(f"outputs under: {OUT_DIR}")
