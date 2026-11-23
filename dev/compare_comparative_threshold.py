# %% [markdown]
# Comparative higher/lower threshold: fixed 0.5 vs level-dependent `1/(2k)`
#
# For a GA batch run folder, take each seed's **best** exported model,
# predict once, then score twice on the same predictions:
#
# - **before** — force `entity_max_level → 1` (Boolean `0.5` for all higher/lower)
# - **after** — current level-dependent threshold (`1/(2k)`)
#
# Also cross-checks the run's stored `*_score.csv` totals against the
# re-scored "before" totals (should match closely if the run used the old
# fixed threshold).
#
# Outputs under `dev/logs/compare_comparative_threshold/<run_id>/`.

# %%
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from boolmore.evaluation.agreement import (
    build_related_nodes,
    entity_max_level,
)
from boolmore.evaluation.score import (
    EvaluationItemScore,
    EvaluationResult,
    get_model_score,
)
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.io.export import export_nav_report
from boolmore.io.load import load_model
from boolmore.io.paths import project_root
from boolmore.outcome import ComparativeOutcome, format_outcome
from boolmore.run_setup import build_evaluator

# %% [markdown]
# ## Config

# %%
REPO = project_root()
RUN = REPO / "case_study/breast_cancer/runs/20261119T130225Z"
CONFIG = REPO / "case_study/breast_cancer/config.json"

# Optional: limit work while iterating. `None` = every seed in summary.csv.
MAX_SEEDS: int | None = None
SEED_FILTER: set[int] | None = None  # e.g. {0, 4, 6}

WRITE_PER_SEED_REPORTS = False  # after-method NAV score CSVs
SHOW_TOP_ITEM_DELTAS = 15

OUT_DIR = (
    Path(__file__).resolve().parent
    / "logs"
    / "compare_comparative_threshold"
    / RUN.name
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"run: {RUN}")
print(f"out: {OUT_DIR}")

# %% [markdown]
# ## Load run summary + evaluator

# %%
ensure_asp_backend()

summary = pd.read_csv(RUN / "summary.csv")
if SEED_FILTER is not None:
    summary = summary[summary["seed"].isin(SEED_FILTER)].copy()
if MAX_SEEDS is not None:
    summary = summary.head(MAX_SEEDS).copy()
summary = summary.sort_values("seed").reset_index(drop=True)
print(f"seeds to score: {len(summary)}")
display_cols = ["seed", "score", "max_score", "pct", "best_model"]
print(summary[display_cols].to_string(index=False))

evaluator = build_evaluator(CONFIG, verbose=True)
assert evaluator.run_type == "NAV"

# Precompute related-node map once (stable across candidates sharing primes
# topology for sibling discovery — use base primes; per-model primes may add
# nodes rarely, so recompute per model below if needed).
related_base = build_related_nodes(evaluator.base.primes, evaluator.experiments)
level_by_observed = {
    base: entity_max_level(base, sibs) for base, sibs in related_base.items()
}
print("\nobserved entity max levels (from base primes):")
for base, k in sorted(level_by_observed.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {base:<20} k={k}  threshold={1 / (2 * k):.4f}")

# %% [markdown]
# ## Helpers


# %%
def _is_higher_lower(item: EvaluationItemScore) -> bool:
    return (
        isinstance(item.outcome, ComparativeOutcome)
        and item.outcome.relation in ("higher", "lower")
    )


def _is_same(item: EvaluationItemScore) -> bool:
    return (
        isinstance(item.outcome, ComparativeOutcome)
        and item.outcome.relation == "same"
    )


def _sum_scores(items: list[EvaluationItemScore], pred) -> float:
    return sum(item.score for item in items if pred(item))


def _sum_agreements(items: list[EvaluationItemScore], pred) -> float:
    return sum(item.agreement for item in items if pred(item))


def score_before_after(model) -> tuple[list[EvaluationItemScore], list[EvaluationItemScore]]:
    """Predict once; score with fixed k=1 (before) and level-dependent (after)."""
    predictions = evaluator.prediction_fn(model.primes, evaluator.experiments)
    items_after = evaluator.score_fn(evaluator.experiments, predictions)
    with patch(
        "boolmore.evaluation.agreement.entity_max_level",
        return_value=1,
    ):
        items_before = evaluator.score_fn(evaluator.experiments, predictions)
    return items_before, items_after


def load_stored_score_csv(seed_row: pd.Series) -> Path:
    stem = Path(str(seed_row["best_model"])).stem  # seed_000_9918_gen116
    seed = int(seed_row["seed"])
    path = RUN / f"seed_{seed:03d}" / f"{stem}_score.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


# %% [markdown]
# ## Re-score each seed's best model

# %%
rows: list[dict] = []
item_deltas: list[dict] = []

for _, seed_row in summary.iterrows():
    seed = int(seed_row["seed"])
    best_rel = str(seed_row["best_model"])
    stem = Path(best_rel).stem
    seed_dir = RUN / f"seed_{seed:03d}"
    pkl_path = seed_dir / f"{stem}.pkl"
    if not pkl_path.exists():
        # fall back to .bnet
        pkl_path = seed_dir / f"{stem}.bnet"
    stored_csv = load_stored_score_csv(seed_row)
    stored_score = float(pd.read_csv(stored_csv)["score"].sum())
    summary_score = float(seed_row["score"])

    print(f"\n=== seed {seed:03d}  {stem} ===")
    # Evolved primes can include regulators outside the base/edge_pool
    # (e.g. remutated self-loops). Load without base so validation passes;
    # scoring only needs the primes.
    model = load_model(pkl_path)
    t0 = perf_counter()
    items_before, items_after = score_before_after(model)
    elapsed = perf_counter() - t0

    max_b, score_b = get_model_score(items_before)
    max_a, score_a = get_model_score(items_after)
    assert abs(max_b - max_a) < 1e-9

    hl_b = _sum_scores(items_before, _is_higher_lower)
    hl_a = _sum_scores(items_after, _is_higher_lower)
    same_b = _sum_scores(items_before, _is_same)
    same_a = _sum_scores(items_after, _is_same)
    abs_b = score_b - hl_b - same_b
    abs_a = score_a - hl_a - same_a

    row = {
        "seed": seed,
        "model": stem,
        "elapsed_s": round(elapsed, 2),
        "max_score": max_a,
        "stored_csv_score": stored_score,
        "summary_score": summary_score,
        "before_score": score_b,
        "after_score": score_a,
        "delta_score": score_a - score_b,
        "before_pct": 100.0 * score_b / max_a if max_a else 0.0,
        "after_pct": 100.0 * score_a / max_a if max_a else 0.0,
        "delta_pct_pts": 100.0 * (score_a - score_b) / max_a if max_a else 0.0,
        "before_higher_lower": hl_b,
        "after_higher_lower": hl_a,
        "delta_higher_lower": hl_a - hl_b,
        "before_same": same_b,
        "after_same": same_a,
        "delta_same": same_a - same_b,
        "before_absolute": abs_b,
        "after_absolute": abs_a,
        "delta_absolute": abs_a - abs_b,
        "stored_minus_before": stored_score - score_b,
    }
    rows.append(row)
    print(
        f"  before {score_b:.3f} → after {score_a:.3f}  "
        f"(Δ {score_a - score_b:+.3f}, {row['delta_pct_pts']:+.2f} ppt)  "
        f"[{elapsed:.1f}s]"
    )
    print(
        f"  higher/lower Δ {hl_a - hl_b:+.3f}  "
        f"same Δ {same_a - same_b:+.3f}  "
        f"absolute Δ {abs_a - abs_b:+.3f}"
    )
    print(
        f"  stored CSV {stored_score:.3f} vs re-scored before {score_b:.3f}  "
        f"(diff {stored_score - score_b:+.3g})"
    )

    by_id_before = {item.id: item for item in items_before}
    for item in items_after:
        prev = by_id_before[item.id]
        if abs(item.score - prev.score) < 1e-12 and abs(
            item.agreement - prev.agreement
        ) < 1e-12:
            continue
        k = level_by_observed.get(item.observed or "", 1)
        item_deltas.append(
            {
                "seed": seed,
                "id": item.id,
                "observed": item.observed,
                "max_level": k,
                "threshold_after": 1.0 / (2 * k),
                "outcome": format_outcome(item.outcome) if item.outcome is not None else "",
                "predict_value": item.predict_value,
                "agreement_before": prev.agreement,
                "agreement_after": item.agreement,
                "delta_agreement": item.agreement - prev.agreement,
                "score_before": prev.score,
                "score_after": item.score,
                "delta_score": item.score - prev.score,
                "weight": item.weight,
                "hierarchy": item.hierarchy,
            }
        )

    if WRITE_PER_SEED_REPORTS:
        related = build_related_nodes(model.primes, evaluator.experiments)
        result_after = EvaluationResult(
            model_id=int(seed_row["model_id"]),
            max_score=max_a,
            score=score_a,
            n_eff_edges=0,
            n_eff_self_edges=0,
            n_prime_implicants=0,
            n_extra_edges=0,
            satisfies_constraints=True,
            item_scores=items_after,
            kind="NAV",
        )
        export_nav_report(
            result_after,
            OUT_DIR / f"{stem}_after_score.csv",
            related_nodes=related,
        )

compare = pd.DataFrame(rows)
deltas = pd.DataFrame(item_deltas)
compare_path = OUT_DIR / "seed_score_compare.csv"
deltas_path = OUT_DIR / "item_score_deltas.csv"
compare.to_csv(compare_path, index=False)
deltas.to_csv(deltas_path, index=False)
print(f"\nwrote {compare_path}")
print(f"wrote {deltas_path}")

# %% [markdown]
# ## Summary tables

# %%
print("=== per-seed totals ===")
print(
    compare[
        [
            "seed",
            "before_score",
            "after_score",
            "delta_score",
            "before_pct",
            "after_pct",
            "delta_pct_pts",
            "delta_higher_lower",
            "stored_minus_before",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\n=== aggregate ===")
print(
    f"mean Δ score:     {compare['delta_score'].mean():+.4f}  "
    f"(median {compare['delta_score'].median():+.4f})"
)
print(
    f"mean Δ pct pts:   {compare['delta_pct_pts'].mean():+.3f}  "
    f"(median {compare['delta_pct_pts'].median():+.3f})"
)
print(
    f"mean Δ higher/lower score: {compare['delta_higher_lower'].mean():+.4f}"
)
print(
    f"mean |stored − before|: {compare['stored_minus_before'].abs().mean():.3g}"
)
print(
    f"max  |stored − before|: {compare['stored_minus_before'].abs().max():.3g}"
)

if not deltas.empty:
    print(f"\nitems with any score/agreement change: {len(deltas)}")
    print(
        f"  of which higher/lower outcomes: "
        f"{deltas['outcome'].str.contains('higher|lower', regex=True).sum()}"
    )
    by_obs = (
        deltas.groupby(["observed", "max_level"], dropna=False)
        .agg(
            n=("id", "count"),
            mean_d_agree=("delta_agreement", "mean"),
            mean_d_score=("delta_score", "mean"),
            sum_d_score=("delta_score", "sum"),
        )
        .reset_index()
        .sort_values("sum_d_score", ascending=False)
    )
    print("\n=== Δ by observed entity (changed items only, summed over seeds) ===")
    print(by_obs.round(4).to_string(index=False))
    by_obs.to_csv(OUT_DIR / "delta_by_observed.csv", index=False)

    top = deltas.reindex(deltas["delta_score"].abs().sort_values(ascending=False).index)
    print(f"\n=== top {SHOW_TOP_ITEM_DELTAS} |Δ score| items ===")
    print(
        top[
            [
                "seed",
                "id",
                "observed",
                "max_level",
                "outcome",
                "predict_value",
                "agreement_before",
                "agreement_after",
                "delta_score",
            ]
        ]
        .head(SHOW_TOP_ITEM_DELTAS)
        .round(4)
        .to_string(index=False)
    )

# %% [markdown]
# ## Plots

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

ax = axes[0]
ax.scatter(compare["before_pct"], compare["after_pct"], s=36, c="#1f4e79")
lims = [
    min(compare["before_pct"].min(), compare["after_pct"].min()) - 0.5,
    max(compare["before_pct"].max(), compare["after_pct"].max()) + 0.5,
]
ax.plot(lims, lims, color="0.6", lw=1, ls="--")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("before % (fixed 0.5)")
ax.set_ylabel("after % (1/(2k))")
ax.set_title("Score % before vs after")
ax.set_aspect("equal", adjustable="box")

ax = axes[1]
seeds = compare["seed"].to_numpy()
ax.bar(seeds, compare["delta_pct_pts"], color="#2a6f6f", width=0.7)
ax.axhline(0.0, color="0.4", lw=0.8)
ax.set_xlabel("seed")
ax.set_ylabel("Δ percentage points")
ax.set_title("After − before (pct pts)")
if len(seeds) <= 40:
    ax.set_xticks(seeds)

ax = axes[2]
parts = compare[["delta_absolute", "delta_same", "delta_higher_lower"]].mean()
ax.bar(
    ["absolute", "same", "higher/lower"],
    parts.values,
    color=["#6b6b6b", "#8a6d3b", "#1f4e79"],
)
ax.axhline(0.0, color="0.4", lw=0.8)
ax.set_ylabel("mean Δ score")
ax.set_title("Where the Δ comes from")

fig.suptitle(f"Comparative threshold change — {RUN.name}", fontsize=11)
fig.tight_layout()
fig_path = OUT_DIR / "compare_summary.png"
fig.savefig(fig_path, dpi=140)
plt.close(fig)
print(f"wrote {fig_path}")

if not deltas.empty:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for k, sub in deltas.groupby("max_level"):
        ax.scatter(
            sub["predict_value"],
            sub["delta_agreement"],
            s=18,
            alpha=0.7,
            label=f"k={int(k)} thr={1 / (2 * int(k)):.3f}",
        )
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("predict_value (oriented Δ)")
    ax.set_ylabel("Δ agreement (after − before)")
    ax.set_title("Agreement change vs predicted difference")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig_path2 = OUT_DIR / "agreement_delta_vs_predict.png"
    fig.savefig(fig_path2, dpi=140)
    plt.close(fig)
    print(f"wrote {fig_path2}")

# %% [markdown]
# ## Manifest

# %%
manifest = {
    "run": str(RUN),
    "config": str(CONFIG),
    "n_seeds": int(len(compare)),
    "seeds": compare["seed"].tolist(),
    "mean_delta_score": float(compare["delta_score"].mean()),
    "mean_delta_pct_pts": float(compare["delta_pct_pts"].mean()),
    "mean_abs_stored_minus_before": float(
        compare["stored_minus_before"].abs().mean()
    ),
    "level_by_observed": level_by_observed,
    "outputs": {
        "seed_score_compare": str(compare_path),
        "item_score_deltas": str(deltas_path),
        "compare_summary_png": str(OUT_DIR / "compare_summary.png"),
    },
}
manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"wrote {manifest_path}")
print("done.")
