# %% [markdown]
# Plot GA evaluation summaries
#
# Loads a `*_evals.csv` from a boolmore run (default: `demo/demo_evals.csv`)
# and builds figures for score over iterations, survivors, skipped (LDOI upper-
# bound) evaluations, and related complexity metrics.
#
# Figures are saved next to the CSV under `<stem>_figures/` and shown inline.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib

# Interactive backends work in Cursor/VS Code `# %%` cells; Agg for plain scripts.
try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# %% [markdown]
# ## Config

# %%
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
EVALS_CSV = DEMO_DIR / "runs" / "test" / "seed_000" / "seed_000_evals.csv"

FIG_DIR = EVALS_CSV.parent / f"{EVALS_CSV.stem}_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual defaults
DPI = 150
SCORE_YLIM_PAD = 0.02  # fraction of max_score padded around data range

# %% [markdown]
# ## Load & prepare

# %%
raw = pd.read_csv(EVALS_CSV)
raw["is_upper_bound"] = raw["is_upper_bound"].astype(bool)
raw["satisfies_constraints"] = raw["satisfies_constraints"].astype(bool)

# CSV rows are already GA-ranked within each iteration (score desc, then order_by).
# Keep that order so "best" matches the run log, then drop keep-padded id copies.
df = raw.drop_duplicates(subset=["iteration", "id"], keep="first").reset_index(drop=True)

max_score = float(df["max_score"].iloc[0])

# First appearance of each id → newly generated (or start); later → survivor.
first_iter = df.groupby("id")["iteration"].transform("min")
df["is_new"] = df["iteration"] == first_iter
df["is_survivor"] = ~df["is_new"]

# First remaining row per iteration == population[0] in the GA.
best_idx = df.groupby("iteration", sort=True).head(1).index
df["is_best"] = False
df.loc[best_idx, "is_best"] = True

best = df.loc[df["is_best"]].sort_values("iteration")
print(f"Loaded {len(df)} unique (iteration, id) rows from {EVALS_CSV}")
print(f"Iterations: {df['iteration'].min()}–{df['iteration'].max()}")
print(f"Models: {df['id'].nunique()} unique ids")
print(f"Skipped (is_upper_bound): {df['is_upper_bound'].sum()}")
print(f"Best trajectory:\n{best[['iteration', 'id', 'generation', 'score']].to_string(index=False)}")

# %%
def _score_ylim(series: pd.Series, ceiling: float) -> tuple[float, float]:
    lo = float(series.min())
    hi = float(series.max())
    pad = max((hi - lo) * 0.08, ceiling * SCORE_YLIM_PAD)
    return max(0.0, lo - pad), min(ceiling * 1.02, hi + pad)


def _save_and_show(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved {path}")
    return path


# Colors / markers
C_EVAL = "#2c7bb6"
C_SKIP = "#d7191c"
C_BEST = "#fdae61"
C_LINE = "#636363"
C_BEST_LINE = "#1a1a1a"

# %% [markdown]
# ## Score over iterations
#
# Each point is a model in that iteration's population. Survivors (same `id`
# across iterations) are connected. Skipped full evaluations (`is_upper_bound`)
# use open markers. The best model each iteration is highlighted.

# %%
fig, ax = plt.subplots(figsize=(10, 5.5))

# Survivor / start lineages
for model_id, g in df.groupby("id"):
    g = g.sort_values("iteration")
    if len(g) < 2:
        continue
    ax.plot(
        g["iteration"],
        g["score"],
        color=C_LINE,
        alpha=0.35,
        linewidth=1.0,
        zorder=1,
        solid_capstyle="round",
    )

# Champion lineage on top
champ_id = int(best.iloc[-1]["id"])
champ = df.loc[df["id"] == champ_id].sort_values("iteration")
if len(champ) >= 2:
    ax.plot(
        champ["iteration"],
        champ["score"],
        color=C_BEST_LINE,
        alpha=0.85,
        linewidth=2.0,
        zorder=2,
        label=f"champion lineage (id {champ_id})",
    )

# Evaluated (full min_traps)
full = df.loc[~df["is_upper_bound"]]
ax.scatter(
    full["iteration"],
    full["score"],
    c=C_EVAL,
    s=36,
    alpha=0.75,
    edgecolors="white",
    linewidths=0.4,
    zorder=3,
    label="evaluated",
)

# Skipped (LDOI upper bound only)
skipped = df.loc[df["is_upper_bound"]]
if len(skipped):
    ax.scatter(
        skipped["iteration"],
        skipped["score"],
        facecolors="none",
        edgecolors=C_SKIP,
        s=42,
        linewidths=1.4,
        zorder=4,
        label="skipped (LDOI upper bound)",
    )

# Best of iteration
ax.scatter(
    best["iteration"],
    best["score"],
    marker="*",
    s=220,
    c=C_BEST,
    edgecolors=C_BEST_LINE,
    linewidths=0.6,
    zorder=5,
    label="best of iteration",
)
for _, row in best.iterrows():
    ax.annotate(
        f"id {int(row['id'])}",
        (row["iteration"], row["score"]),
        textcoords="offset points",
        xytext=(5, 6),
        fontsize=8,
        color=C_BEST_LINE,
        alpha=0.9,
    )

ax.axhline(max_score, color="#bbbbbb", linestyle="--", linewidth=1, zorder=0)
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Score (max {max_score:g})")
ax.set_title("Population scores over iterations")
ax.set_xticks(sorted(df["iteration"].unique()))
ax.set_ylim(*_score_ylim(df["score"], max_score))
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
_save_and_show(fig, "score_over_iterations.png")

# %% [markdown]
# ## Best score trajectory & prune rate
#
# Left: best score per iteration with generation annotated.
# Right: fraction of the population that was LDOI-pruned (skipped).

# %%
iter_stats = (
    df.groupby("iteration")
    .agg(
        best_score=("score", "max"),
        mean_score=("score", "mean"),
        n=("id", "count"),
        n_skipped=("is_upper_bound", "sum"),
        n_survivors=("is_survivor", "sum"),
    )
    .reset_index()
)
iter_stats["frac_skipped"] = iter_stats["n_skipped"] / iter_stats["n"]
iter_stats = iter_stats.merge(
    best[["iteration", "id", "generation"]].rename(
        columns={"id": "best_id", "generation": "best_gen"}
    ),
    on="iteration",
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.fill_between(
    iter_stats["iteration"],
    iter_stats["mean_score"],
    iter_stats["best_score"],
    color=C_EVAL,
    alpha=0.15,
    label="mean → best band",
)
ax.plot(
    iter_stats["iteration"],
    iter_stats["mean_score"],
    color=C_EVAL,
    alpha=0.5,
    linewidth=1.2,
    label="mean score",
)
ax.plot(
    iter_stats["iteration"],
    iter_stats["best_score"],
    color=C_BEST_LINE,
    linewidth=2.0,
    marker="o",
    markersize=5,
    label="best score",
)
for _, row in iter_stats.iterrows():
    ax.annotate(
        f"g{int(row['best_gen'])}",
        (row["iteration"], row["best_score"]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=8,
        color=C_LINE,
    )
ax.axhline(max_score, color="#bbbbbb", linestyle="--", linewidth=1)
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Score (max {max_score:g})")
ax.set_title("Best & mean score")
ax.set_xticks(iter_stats["iteration"])
ax.set_ylim(*_score_ylim(iter_stats[["mean_score", "best_score"]].stack(), max_score))
ax.legend(fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.bar(
    iter_stats["iteration"],
    iter_stats["frac_skipped"],
    color=C_SKIP,
    alpha=0.75,
    width=0.7,
    label="skipped",
)
ax.bar(
    iter_stats["iteration"],
    1.0 - iter_stats["frac_skipped"],
    bottom=iter_stats["frac_skipped"],
    color=C_EVAL,
    alpha=0.55,
    width=0.7,
    label="evaluated",
)
ax.set_xlabel("Iteration")
ax.set_ylabel("Fraction of population")
ax.set_title("LDOI prune / skip rate")
ax.set_xticks(iter_stats["iteration"])
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
_save_and_show(fig, "best_and_prune_rate.png")

# %% [markdown]
# ## Score vs model complexity
#
# How score relates to prime-implicant count and effective edges. Skipped
# evaluations and the per-iteration bests are marked the same way as above.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, xcol, xlabel in [
    (axes[0], "n_prime_implicants", "Prime implicants"),
    (axes[1], "n_eff_edges", "Effective edges"),
]:
    full = df.loc[~df["is_upper_bound"]]
    skipped = df.loc[df["is_upper_bound"]]
    ax.scatter(
        full[xcol],
        full["score"],
        c=C_EVAL,
        s=28,
        alpha=0.55,
        edgecolors="none",
        label="evaluated",
    )
    if len(skipped):
        ax.scatter(
            skipped[xcol],
            skipped["score"],
            facecolors="none",
            edgecolors=C_SKIP,
            s=36,
            linewidths=1.2,
            label="skipped",
        )
    ax.scatter(
        best[xcol],
        best["score"],
        marker="*",
        s=160,
        c=C_BEST,
        edgecolors=C_BEST_LINE,
        linewidths=0.5,
        zorder=5,
        label="best of iter",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Score (max {max_score:g})")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_title("Score vs prime implicants")
axes[1].set_title("Score vs effective edges")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
_save_and_show(fig, "score_vs_complexity.png")

# %% [markdown]
# ## Complexity of the best model over iterations

# %%
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(
    best["iteration"],
    best["n_prime_implicants"],
    marker="o",
    color="#7570b3",
    label="prime implicants",
)
ax.plot(
    best["iteration"],
    best["n_eff_edges"],
    marker="s",
    color="#1b9e77",
    label="effective edges",
)
ax.plot(
    best["iteration"],
    best["n_extra_edges"],
    marker="^",
    color="#e7298a",
    label="extra edges",
)
ax.set_xlabel("Iteration")
ax.set_ylabel("Count")
ax.set_title("Complexity of the best model per iteration")
ax.set_xticks(best["iteration"])
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
_save_and_show(fig, "best_complexity.png")

# %% [markdown]
# ## Survival spans
#
# Horizontal spans show how long each model id remains in the kept population.

# %%
spans = (
    df.groupby("id")
    .agg(
        first=("iteration", "min"),
        last=("iteration", "max"),
        peak_score=("score", "max"),
        ever_best=("is_best", "any"),
        ever_skipped=("is_upper_bound", "any"),
    )
    .reset_index()
)
spans["span"] = spans["last"] - spans["first"]
# Keep models that appear more than once, plus any that were best once.
plot_spans = spans.loc[(spans["span"] > 0) | spans["ever_best"]].sort_values(
    ["peak_score", "id"], ascending=[False, True]
)

if len(plot_spans):
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.28 * len(plot_spans) + 1.2)))
    y_pos = np.arange(len(plot_spans))
    for i, row in enumerate(plot_spans.itertuples()):
        color = C_BEST if row.ever_best else C_EVAL
        ax.hlines(
            y=i,
            xmin=row.first,
            xmax=max(row.last, row.first + 0.15),
            colors=color,
            linewidth=3.5 if row.ever_best else 2.0,
            alpha=0.85,
        )
        ax.plot(row.first, i, "o", color=color, markersize=5)
        ax.plot(row.last, i, "o", color=color, markersize=5)
        if row.ever_skipped:
            ax.plot(row.last, i, "o", mfc="none", mec=C_SKIP, markersize=9, mew=1.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"id {int(i)}" for i in plot_spans["id"]], fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_title("Model survival in the population")
    ax.set_xticks(sorted(df["iteration"].unique()))
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color=C_BEST, lw=3, label="was best at least once"),
        Line2D([0], [0], color=C_EVAL, lw=2, label="survivor / kept"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=C_SKIP,
            markersize=8,
            label="had skipped eval",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save_and_show(fig, "survival_spans.png")
else:
    print("No multi-iteration survivors to plot.")

# %% [markdown]
# ## Summary table

# %%
summary = iter_stats[
    [
        "iteration",
        "best_id",
        "best_gen",
        "best_score",
        "mean_score",
        "n",
        "n_skipped",
        "n_survivors",
        "frac_skipped",
    ]
].copy()
summary["best_score"] = summary["best_score"].round(3)
summary["mean_score"] = summary["mean_score"].round(3)
summary["frac_skipped"] = summary["frac_skipped"].round(2)
display_df = summary
try:
    from IPython.display import display

    display(display_df)
except ImportError:
    print(display_df.to_string(index=False))

print(f"\nFigures written to {FIG_DIR}/")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"  {p.name}")
