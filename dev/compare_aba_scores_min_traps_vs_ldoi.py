# %% [markdown]
# ABA scores: `min_traps` vs `ldoi`
#
# Score every `.bnet` under `case_study/ABA` against the experiments from
# `ABA_B.json` (`data_B.tsv`, `default_sources`) using both NAV prediction
# methods from `boolmore.inference.prediction`:
#
# - **min_traps**: minimal trap spaces → average node values
# - **ldoi**: logical domain of influence → 0/1/0.5 approximation
#
# Models that fail (missing nodes, ASP errors, etc.) are logged and skipped.
# For successes, compare total scores: Pearson/Spearman correlation, pairwise
# order preservation, and rank agreement. Saves plots under `dev/logs/`
# (no interactive display).
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.

# %%
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from boolmore.evaluation.nav_score import get_NAV_scores
from boolmore.evaluation.score import get_model_score
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.prediction import get_NAV_prediction
from boolmore.io.load import load_json_config, load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.io.paths import project_root

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
ABA = REPO / "case_study" / "ABA"
CONFIG_REL = "case_study/ABA/data/ABA_B.json"
LOG_DIR = Path(__file__).resolve().parent / "logs"
HIERARCHY = True
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
LOG_PATH = LOG_DIR / f"compare_aba_scores_min_traps_vs_ldoi_{RUN_ID}.log"
PLOT_SCORE = LOG_DIR / f"compare_aba_scores_scatter_{RUN_ID}.png"
PLOT_RANK = LOG_DIR / f"compare_aba_ranks_scatter_{RUN_ID}.png"
PLOT_DIFF = LOG_DIR / f"compare_aba_score_diff_{RUN_ID}.png"
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print()
# %% [markdown]
# ## Load config, experiments, enumerate bnets

# %%
config = load_json_config((project_root() / CONFIG_REL).resolve())
data_path = (project_root() / config["data"]).resolve()
default_sources = {str(k): int(v) for k, v in config["default_sources"].items()}

experiments = import_NAV_exps(data_path)
bnet_paths = sorted(ABA.rglob("*.bnet"))

print(f"config: {CONFIG_REL}")
print(f"data:   {data_path.relative_to(REPO)}")
print(f"default_sources: {default_sources}")
print(f"hierarchy: {HIERARCHY}")
print(f"experiments: {len(experiments)}")
print(f"bnet files: {len(bnet_paths)}")
for p in bnet_paths:
    print(f"  {p.relative_to(ABA)}")
print()

ensure_asp_backend()

# %% [markdown]
# ## Helpers


# %%
@dataclass
class ModelScoreRow:
    rel: str
    score_min: float | None = None
    score_ldoi: float | None = None
    max_score: float | None = None
    t_min: float | None = None
    t_ldoi: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.score_min is not None
            and self.score_ldoi is not None
        )


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Competition-style average ranks for ties (1-based, higher score → better/lower rank)."""
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def pairwise_order_agreement(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """Fraction of pairs with the same relative order (ties must match).

    Returns (agreement, n_agree, n_pairs).
    """
    n_agree = 0
    n_pairs = 0
    for i, j in combinations(range(len(x)), 2):
        n_pairs += 1
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        if dx == 0 and dy == 0:
            n_agree += 1
        elif dx == 0 or dy == 0:
            continue  # one tied, other not → disagreement
        elif (dx > 0) == (dy > 0):
            n_agree += 1
    agreement = n_agree / n_pairs if n_pairs else float("nan")
    return agreement, n_agree, n_pairs


def top_k_overlap(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int) -> float:
    set_a = set(np.where(ranks_a <= k)[0])
    set_b = set(np.where(ranks_b <= k)[0])
    if not set_a and not set_b:
        return float("nan")
    return len(set_a & set_b) / k

# %% [markdown]
# ## Score all models

# %%
rows: list[ModelScoreRow] = []

for idx, bnet in enumerate(bnet_paths, start=1):
    rel = str(bnet.relative_to(ABA))
    print(f"[{idx}/{len(bnet_paths)}] {rel}")
    row = ModelScoreRow(rel=rel)

    try:
        primes = load_primes(bnet)
    except Exception as exc:
        row.error = f"load_primes: {type(exc).__name__}: {exc}"
        print(f"  FAIL  {row.error}")
        rows.append(row)
        continue

    try:
        t0 = perf_counter()
        preds_min = get_NAV_prediction(primes, experiments, method="min_traps")
        items_min = get_NAV_scores(
            experiments,
            preds_min,
            default_sources=default_sources,
            hierarchy=HIERARCHY,
        )
        max_min, score_min = get_model_score(items_min)
        row.t_min = perf_counter() - t0
        row.score_min = score_min
        row.max_score = max_min
    except Exception as exc:
        row.error = f"min_traps: {type(exc).__name__}: {exc}"
        print(f"  FAIL  {row.error}")
        rows.append(row)
        continue

    try:
        t0 = perf_counter()
        preds_ldoi = get_NAV_prediction(primes, experiments, method="ldoi")
        items_ldoi = get_NAV_scores(
            experiments,
            preds_ldoi,
            default_sources=default_sources,
            hierarchy=HIERARCHY,
        )
        max_ldoi, score_ldoi = get_model_score(items_ldoi)
        row.t_ldoi = perf_counter() - t0
        row.score_ldoi = score_ldoi
        if row.max_score != max_ldoi:
            print(
                f"  WARN  max_score mismatch "
                f"min_traps={row.max_score} ldoi={max_ldoi}"
            )
    except Exception as exc:
        row.error = f"ldoi: {type(exc).__name__}: {exc}"
        print(f"  FAIL  {row.error}")
        rows.append(row)
        continue

    print(
        f"  OK    min_traps={score_min:.2f}/{max_min} ({row.t_min:.2f}s)  "
        f"ldoi={score_ldoi:.2f}/{max_ldoi} ({row.t_ldoi:.2f}s)  "
        f"diff={score_ldoi - score_min:+.2f}"
    )
    rows.append(row)

print()

# %% [markdown]
# ## Failures

# %%
failed = [r for r in rows if not r.ok]
print(f"failed: {len(failed)} / {len(rows)}")
for r in failed:
    print(f"  {r.rel}: {r.error}")
print()

# %% [markdown]
# ## Comparison (both methods succeeded)

# %%
ok_rows = [r for r in rows if r.ok]
n = len(ok_rows)

print(f"succeeded (both methods): {n} / {len(rows)}")
print()

if n == 0:
    print("No successful models to compare.")
else:
    scores_min = np.array([r.score_min for r in ok_rows], dtype=float)
    scores_ldoi = np.array([r.score_ldoi for r in ok_rows], dtype=float)
    ranks_min = average_ranks(scores_min)
    ranks_ldoi = average_ranks(scores_ldoi)
    diffs = scores_ldoi - scores_min

    # table sorted by min_traps score descending
    order = np.argsort(-scores_min, kind="mergesort")
    print(
        f"{'model':<55} {'min_traps':>10} {'ldoi':>10} {'diff':>8} "
        f"{'rank_m':>7} {'rank_l':>7}"
    )
    print("-" * 100)
    for i in order:
        r = ok_rows[i]
        print(
            f"{r.rel:<55} "
            f"{scores_min[i]:10.2f} "
            f"{scores_ldoi[i]:10.2f} "
            f"{diffs[i]:+8.2f} "
            f"{ranks_min[i]:7.1f} "
            f"{ranks_ldoi[i]:7.1f}"
        )
    print("-" * 100)
    print()

    r_pearson = pearson(scores_min, scores_ldoi)
    r_spearman = spearman(scores_min, scores_ldoi)
    order_agree, n_agree, n_pairs = pairwise_order_agreement(scores_min, scores_ldoi)
    mean_abs_rank_diff = float(np.mean(np.abs(ranks_min - ranks_ldoi)))

    print("Correlation / rank agreement")
    print(f"  Pearson r:              {r_pearson:.4f}")
    print(f"  Spearman rho:           {r_spearman:.4f}")
    print(
        f"  Pairwise order agree:   {order_agree:.4f} "
        f"({n_agree}/{n_pairs} pairs)"
    )
    print(f"  Mean |rank_min - rank_ldoi|: {mean_abs_rank_diff:.3f}")
    print(
        f"  Score diff (ldoi - min): "
        f"mean={diffs.mean():+.3f}  "
        f"std={diffs.std(ddof=1) if n > 1 else 0.0:.3f}  "
        f"min={diffs.min():+.3f}  max={diffs.max():+.3f}"
    )

    for k in (1, 3, 5):
        if n >= k:
            ov = top_k_overlap(ranks_min, ranks_ldoi, k)
            print(f"  Top-{k} set overlap:     {ov:.3f}")
    print()

    print("Scatter pairs (min_traps, ldoi)")
    for i in order:
        print(f"  ({scores_min[i]:.2f}, {scores_ldoi[i]:.2f})  {ok_rows[i].rel}")
    print()

# %% [markdown]
# ## Plots
#
# Score scatter, rank scatter, and score-difference bars. Saved as PNGs
# next to the log (no interactive display — avoids blocking on plt.show).

# %%
saved_plots: list[Path] = []

if n == 0:
    print("No plots (no successful models).")
else:
    short_labels = [Path(r.rel).stem for r in ok_rows]

    # --- score scatter ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(scores_min, scores_ldoi, s=40, zorder=3)
    lo = float(min(scores_min.min(), scores_ldoi.min()))
    hi = float(max(scores_min.max(), scores_ldoi.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="0.5", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("score (min_traps)")
    ax.set_ylabel("score (ldoi)")
    ax.set_title(
        f"ABA NAV scores  (n={n}, Pearson r={r_pearson:.3f}, "
        f"Spearman ρ={r_spearman:.3f})"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_SCORE, dpi=150)
    plt.close(fig)
    saved_plots.append(PLOT_SCORE)
    print(f"saved: {PLOT_SCORE}")

    # --- rank scatter ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ranks_min, ranks_ldoi, s=40, zorder=3)
    rlim = (0.5, n + 0.5)
    ax.plot(rlim, rlim, color="0.5", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(rlim)
    ax.set_ylim(rlim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("rank (min_traps, 1 = best)")
    ax.set_ylabel("rank (ldoi, 1 = best)")
    ax.set_title(
        f"ABA NAV ranks  (pairwise order={order_agree:.3f}, "
        f"mean |Δrank|={mean_abs_rank_diff:.2f})"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_RANK, dpi=150)
    plt.close(fig)
    saved_plots.append(PLOT_RANK)
    print(f"saved: {PLOT_RANK}")

    # --- score difference by model ---
    order_diff = np.argsort(diffs)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * n)))
    y = np.arange(n)
    ax.barh(y, diffs[order_diff], color="C0", alpha=0.85)
    ax.axvline(0.0, color="0.4", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([short_labels[i] for i in order_diff], fontsize=8)
    ax.set_xlabel("score(ldoi) − score(min_traps)")
    ax.set_title("Per-model score difference")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIFF, dpi=150)
    plt.close(fig)
    saved_plots.append(PLOT_DIFF)
    print(f"saved: {PLOT_DIFF}")
    print()

# %% [markdown]
# ## Summary

# %%
print("Summary")
print(f"  bnets scanned:     {len(rows)}")
print(f"  both methods OK:   {len(ok_rows)}")
print(f"  failed:            {len(failed)}")
if ok_rows:
    print(f"  Pearson r:         {r_pearson:.4f}")
    print(f"  Spearman rho:      {r_spearman:.4f}")
    print(f"  pairwise order:    {order_agree:.4f}")
    print(f"  mean |Δrank|:      {mean_abs_rank_diff:.3f}")
print(f"  Log written to:    {LOG_PATH}")
for p in saved_plots:
    print(f"  Plot written to:   {p}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
for p in saved_plots:
    print(f"plot saved: {p}")
