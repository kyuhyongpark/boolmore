# %% [markdown]
# ABA scores: `min_traps` vs LDOI family
#
# Score every `.bnet` under `case_study/ABA` against the experiments from
# `ABA_{A|B}.json` (`data_{A|B}`, `default_sources`) using NAV prediction
# methods from `boolmore.inference.prediction`:
#
# - **min_traps**: minimal trap spaces → average node values (reference)
# - **ldoi**: plain logical domain of influence
# - **ldoi_enhanced** with `cutoff=1`
# - **ldoi_enhanced** with `cutoff=2`
# - **ldoi_enhanced** with `cutoff=2` and `joint_self_loops=True`
#
# Switch dataset with `DATA_SET` below (`"A"` or `"B"`). Models that fail
# (missing nodes, ASP errors, etc.) are logged and skipped. For successes,
# compare each LDOI variant to `min_traps`: Pearson/Spearman, pairwise order
# preservation, and rank agreement. Also records per-method wall times.
#
# `min_traps` scores + wall times are cached under the log dir (JSON) so
# re-runs can skip the expensive trap-space step; set
# `FORCE_RECOMPUTE_MIN_TRAPS=True` to refresh.
#
# Outputs are overwritten under
# `dev/logs/compare_aba_scores_min_traps_vs_ldoi/{A|B}/` (stable names, no
# timestamps): log, score/rank/diff/timing plots, method_timings.csv, README.md.
# No interactive display.

# %%
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
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
#
# Set `DATA_SET` to `"A"` or `"B"` (or env `ABA_DATA_SET`). Everything else
# (config path + log folder) follows from that.

# %%
# --- session knobs ---
# DATA_SET = "B"  # "A" or "B"
DATA_SET = "A"
# Or: ABA_DATA_SET=A python ...
USE_MIN_TRAPS_CACHE = True
FORCE_RECOMPUTE_MIN_TRAPS = False  # True → ignore cache and rewrite it
# ---------------------

_env_set = os.environ.get("ABA_DATA_SET", "").strip().upper()
if _env_set:
    DATA_SET = _env_set
DATA_SET = DATA_SET.strip().upper()
if DATA_SET not in {"A", "B"}:
    raise ValueError(f'DATA_SET must be "A" or "B", got {DATA_SET!r}')

REPO = Path(__file__).resolve().parents[1]
ABA = REPO / "case_study" / "ABA"
CONFIG_REL = f"case_study/ABA/data/ABA_{DATA_SET}.json"
LOG_DIR = (
    Path(__file__).resolve().parent
    / "logs"
    / "compare_aba_scores_min_traps_vs_ldoi"
    / DATA_SET
)
HIERARCHY = True
STARTED_UTC = datetime.now(timezone.utc)

# LDOI-family variants compared against min_traps (key, plot/table label, kwargs).
LDOI_VARIANTS: list[tuple[str, str, dict]] = [
    ("ldoi", "ldoi", {"method": "ldoi"}),
    (
        "enh_c1",
        "enhanced(c=1)",
        {"method": "ldoi_enhanced", "cutoff": 1, "joint_self_loops": False},
    ),
    (
        "enh_c2",
        "enhanced(c=2)",
        {"method": "ldoi_enhanced", "cutoff": 2, "joint_self_loops": False},
    ),
    (
        "enh_c2_j",
        "enhanced+j(c=2)",
        {"method": "ldoi_enhanced", "cutoff": 2, "joint_self_loops": True},
    ),
]
LDOI_KEYS = [k for k, _, _ in LDOI_VARIANTS]
LDOI_LABEL = {k: lab for k, lab, _ in LDOI_VARIANTS}
METHOD_COLORS = {
    "min_traps": "#7f7f7f",
    "ldoi": "#2c3e50",
    "enh_c1": "#1f77b4",
    "enh_c2": "#2ca02c",
    "enh_c2_j": "#bcbd22",
}
ALL_METHOD_KEYS = ["min_traps", *LDOI_KEYS]
ALL_METHOD_LABELS = {
    "min_traps": "min_traps",
    **LDOI_LABEL,
}

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
LOG_PATH = LOG_DIR / "compare_aba_scores_min_traps_vs_ldoi.log"
PLOT_SCORE = LOG_DIR / "compare_aba_scores_scatter.png"
PLOT_RANK = LOG_DIR / "compare_aba_ranks_scatter.png"
PLOT_DIFF = LOG_DIR / "compare_aba_score_diff.png"
PLOT_TIMING = LOG_DIR / "compare_aba_method_timings.png"
TIMINGS_CSV = LOG_DIR / "method_timings.csv"
README_PATH = LOG_DIR / "README.md"
MIN_TRAPS_CACHE_PATH = LOG_DIR / "min_traps_cache.json"
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"DATA_SET: {DATA_SET}")
print(f"log dir:  {LOG_DIR}")
print(f"log file: {LOG_PATH}")
print(f"started (UTC): {STARTED_UTC.isoformat()}")
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
print(f"LDOI variants: {[LDOI_LABEL[k] for k in LDOI_KEYS]}")
print(f"min_traps cache: {MIN_TRAPS_CACHE_PATH}")
print(f"  use_cache={USE_MIN_TRAPS_CACHE}  force_recompute={FORCE_RECOMPUTE_MIN_TRAPS}")
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
    max_score: float | None = None
    t_min: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    times: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    min_from_cache: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.score_min is not None
            and all(k in self.scores for k in LDOI_KEYS)
        )


def _bnet_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def _cache_fingerprint() -> dict:
    """Inputs that must match for a cache entry to be reused."""
    return {
        "data_set": DATA_SET,
        "config_rel": CONFIG_REL,
        "data": str(data_path.relative_to(REPO)),
        "hierarchy": HIERARCHY,
        "default_sources": dict(sorted(default_sources.items())),
        "n_experiments": len(experiments),
    }


def load_min_traps_cache(path: Path) -> dict:
    """Load cache JSON; returns empty shell if missing/incompatible."""
    empty = {"fingerprint": _cache_fingerprint(), "models": {}}
    if not path.is_file():
        return empty
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"min_traps cache unreadable ({exc}); starting fresh")
        return empty
    if not isinstance(raw, dict) or "models" not in raw:
        print("min_traps cache malformed; starting fresh")
        return empty
    fp = raw.get("fingerprint")
    expected = _cache_fingerprint()
    if fp != expected:
        print("min_traps cache fingerprint mismatch; starting fresh")
        print(f"  cached:   {fp}")
        print(f"  expected: {expected}")
        return empty
    models = raw.get("models") or {}
    if not isinstance(models, dict):
        print("min_traps cache models malformed; starting fresh")
        return empty
    return {"fingerprint": expected, "models": dict(models)}


def save_min_traps_cache(path: Path, cache: dict) -> None:
    cache = {
        "fingerprint": _cache_fingerprint(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "models": cache.get("models", {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def lookup_min_traps_cache(
    cache: dict, rel: str, bnet: Path
) -> dict | None:
    """Return cached entry if present and bnet mtime matches."""
    entry = cache["models"].get(rel)
    if not isinstance(entry, dict):
        return None
    try:
        mtime = _bnet_mtime_ns(bnet)
    except OSError:
        return None
    if entry.get("bnet_mtime_ns") != mtime:
        return None
    if entry.get("error") is None and "score_min" in entry and "t_min" in entry:
        return entry
    if entry.get("error"):
        return entry
    return None


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


def _score_with_method(primes, method_kwargs: dict) -> tuple[float, float, float]:
    """Return (max_score, score, wall_s) for one NAV method."""
    t0 = perf_counter()
    preds = get_NAV_prediction(primes, experiments, **method_kwargs)
    items = get_NAV_scores(
        experiments,
        preds,
        default_sources=default_sources,
        hierarchy=HIERARCHY,
    )
    max_s, score = get_model_score(items)
    return max_s, score, perf_counter() - t0


@dataclass
class VariantMetrics:
    key: str
    label: str
    scores: np.ndarray
    ranks: np.ndarray
    diffs: np.ndarray
    pearson: float
    spearman: float
    order_agree: float
    n_agree: int
    n_pairs: int
    mean_abs_rank_diff: float
    top_k_overlaps: dict[int, float]


# %% [markdown]
# ## Score all models
#
# `min_traps` results (score + wall time) are read from / written to
# `min_traps_cache.json` when caching is enabled. LDOI variants always run fresh.

# %%
min_traps_cache = (
    load_min_traps_cache(MIN_TRAPS_CACHE_PATH)
    if USE_MIN_TRAPS_CACHE
    else {"fingerprint": _cache_fingerprint(), "models": {}}
)
cache_hits = 0
cache_misses = 0
cache_dirty = False

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

    cached = None
    if USE_MIN_TRAPS_CACHE and not FORCE_RECOMPUTE_MIN_TRAPS:
        cached = lookup_min_traps_cache(min_traps_cache, rel, bnet)

    if cached is not None:
        cache_hits += 1
        row.min_from_cache = True
        if cached.get("error"):
            row.error = str(cached["error"])
            print(f"  FAIL  {row.error}  [min_traps cache]")
            rows.append(row)
            continue
        row.score_min = float(cached["score_min"])
        row.max_score = float(cached["max_score"])
        row.t_min = float(cached["t_min"])
        print(
            f"  CACHE min_traps={row.score_min:.2f}/{row.max_score} "
            f"({row.t_min:.2f}s cached)"
        )
    else:
        cache_misses += 1
        try:
            max_min, score_min, t_min = _score_with_method(
                primes, {"method": "min_traps"}
            )
            row.t_min = t_min
            row.score_min = score_min
            row.max_score = max_min
            if USE_MIN_TRAPS_CACHE:
                min_traps_cache["models"][rel] = {
                    "score_min": float(score_min),
                    "max_score": float(max_min),
                    "t_min": float(t_min),
                    "bnet_mtime_ns": _bnet_mtime_ns(bnet),
                    "error": None,
                }
                cache_dirty = True
            print(f"  OK    min_traps={score_min:.2f}/{max_min} ({t_min:.2f}s)")
        except Exception as exc:
            row.error = f"min_traps: {type(exc).__name__}: {exc}"
            print(f"  FAIL  {row.error}")
            if USE_MIN_TRAPS_CACHE:
                try:
                    mtime = _bnet_mtime_ns(bnet)
                except OSError:
                    mtime = None
                min_traps_cache["models"][rel] = {
                    "error": row.error,
                    "bnet_mtime_ns": mtime,
                }
                cache_dirty = True
            rows.append(row)
            continue

    ldoi_errors: list[str] = []
    for key, label, kwargs in LDOI_VARIANTS:
        try:
            max_s, score, t_s = _score_with_method(primes, kwargs)
            row.scores[key] = float(score)
            row.times[key] = float(t_s)
            if row.max_score is not None and row.max_score != max_s:
                print(
                    f"  WARN  max_score mismatch "
                    f"min_traps={row.max_score} {label}={max_s}"
                )
            print(f"  OK    {label}={score:.2f}/{max_s} ({t_s:.2f}s)")
        except Exception as exc:
            msg = f"{key}: {type(exc).__name__}: {exc}"
            ldoi_errors.append(msg)
            print(f"  FAIL  {msg}")

    if ldoi_errors:
        row.error = "; ".join(ldoi_errors)
        rows.append(row)
        continue

    parts = [
        f"{LDOI_LABEL[k]}={row.scores[k]:.2f} "
        f"(Δ{row.scores[k] - row.score_min:+.2f})"
        for k in LDOI_KEYS
    ]
    print(f"  DONE  " + "  ".join(parts))
    rows.append(row)

if USE_MIN_TRAPS_CACHE and cache_dirty:
    save_min_traps_cache(MIN_TRAPS_CACHE_PATH, min_traps_cache)
    print(f"min_traps cache written: {MIN_TRAPS_CACHE_PATH}")
elif USE_MIN_TRAPS_CACHE:
    print(f"min_traps cache unchanged: {MIN_TRAPS_CACHE_PATH}")

print(
    f"min_traps cache: hits={cache_hits}  misses/recomputes={cache_misses}  "
    f"(n_models={len(bnet_paths)})"
)
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
# ## Comparison (min_traps + all LDOI variants succeeded)

# %%
ok_rows = [r for r in rows if r.ok]
n = len(ok_rows)

scores_min = np.array([], dtype=float)
ranks_min = np.array([], dtype=float)
variant_metrics: dict[str, VariantMetrics] = {}

print(f"succeeded (all methods): {n} / {len(rows)}")
print()

if n == 0:
    print("No successful models to compare.")
else:
    scores_min = np.array([r.score_min for r in ok_rows], dtype=float)
    ranks_min = average_ranks(scores_min)
    order = np.argsort(-scores_min, kind="mergesort")

    # Wide score table
    hdr = f"{'model':<45} {'min_traps':>10}"
    for k in LDOI_KEYS:
        hdr += f" {LDOI_LABEL[k]:>16}"
    print(hdr)
    print("-" * len(hdr))
    for i in order:
        r = ok_rows[i]
        line = f"{r.rel:<45} {scores_min[i]:10.2f}"
        for k in LDOI_KEYS:
            line += f" {r.scores[k]:16.2f}"
        print(line)
    print("-" * len(hdr))
    print()

    for key, label, _ in LDOI_VARIANTS:
        scores_v = np.array([r.scores[key] for r in ok_rows], dtype=float)
        ranks_v = average_ranks(scores_v)
        diffs = scores_v - scores_min
        r_p = pearson(scores_min, scores_v)
        r_s = spearman(scores_min, scores_v)
        o_agree, n_agree, n_pairs = pairwise_order_agreement(scores_min, scores_v)
        mean_abs = float(np.mean(np.abs(ranks_min - ranks_v)))
        top_ks: dict[int, float] = {}
        for kk in (1, 3, 5):
            if n >= kk:
                top_ks[kk] = top_k_overlap(ranks_min, ranks_v, kk)
        variant_metrics[key] = VariantMetrics(
            key=key,
            label=label,
            scores=scores_v,
            ranks=ranks_v,
            diffs=diffs,
            pearson=r_p,
            spearman=r_s,
            order_agree=o_agree,
            n_agree=n_agree,
            n_pairs=n_pairs,
            mean_abs_rank_diff=mean_abs,
            top_k_overlaps=top_ks,
        )

        print(f"Agreement vs min_traps — {label}")
        print(f"  Pearson r:              {r_p:.4f}")
        print(f"  Spearman rho:           {r_s:.4f}")
        print(
            f"  Pairwise order agree:   {o_agree:.4f} "
            f"({n_agree}/{n_pairs} pairs)"
        )
        print(f"  Mean |Δrank|:           {mean_abs:.3f}")
        print(
            f"  Score diff (method − min): "
            f"mean={diffs.mean():+.3f}  "
            f"std={diffs.std(ddof=1) if n > 1 else 0.0:.3f}  "
            f"min={diffs.min():+.3f}  max={diffs.max():+.3f}"
        )
        for kk, ov in top_ks.items():
            print(f"  Top-{kk} set overlap:     {ov:.3f}")
        print()

# %% [markdown]
# ## Method timings
#
# Wall time for prediction + NAV scoring + model score, per method, on models
# where that method completed.

# %%
def _timing_stats(vals: np.ndarray) -> dict[str, float]:
    if len(vals) == 0:
        return {
            "n": 0.0,
            "total_s": float("nan"),
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
        }
    return {
        "n": float(len(vals)),
        "total_s": float(vals.sum()),
        "mean_s": float(vals.mean()),
        "median_s": float(np.median(vals)),
        "min_s": float(vals.min()),
        "max_s": float(vals.max()),
    }


timing_by_method: dict[str, dict[str, float]] = {
    "min_traps": _timing_stats(
        np.array([r.t_min for r in rows if r.t_min is not None], dtype=float)
    ),
}
for key in LDOI_KEYS:
    timing_by_method[key] = _timing_stats(
        np.array(
            [r.times[key] for r in rows if key in r.times],
            dtype=float,
        )
    )

print("Method timings (prediction + NAV score + model score)")
for key in ALL_METHOD_KEYS:
    stats = timing_by_method[key]
    label = ALL_METHOD_LABELS[key]
    if stats["n"] == 0:
        print(f"  {label}: (no successful runs)")
        continue
    print(
        f"  {label}: n={int(stats['n'])}  "
        f"total={stats['total_s']:.2f}s  "
        f"mean={stats['mean_s']:.2f}s  "
        f"median={stats['median_s']:.2f}s  "
        f"min={stats['min_s']:.2f}s  "
        f"max={stats['max_s']:.2f}s"
    )
print()

with TIMINGS_CSV.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "method",
            "label",
            "n",
            "total_s",
            "mean_s",
            "median_s",
            "min_s",
            "max_s",
        ],
    )
    writer.writeheader()
    for key in ALL_METHOD_KEYS:
        writer.writerow(
            {
                "method": key,
                "label": ALL_METHOD_LABELS[key],
                **timing_by_method[key],
            }
        )
print(f"saved: {TIMINGS_CSV}")

per_model_csv = LOG_DIR / "method_timings_per_model.csv"
if n > 0:
    fieldnames = ["model", "t_min_traps_s"] + [f"t_{k}_s" for k in LDOI_KEYS]
    with per_model_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in ok_rows:
            row_out = {"model": r.rel, "t_min_traps_s": r.t_min}
            for k in LDOI_KEYS:
                row_out[f"t_{k}_s"] = r.times[k]
            writer.writerow(row_out)
    print(f"saved: {per_model_csv}")
print()

# %% [markdown]
# ## Plots
#
# Side-by-side panels per LDOI variant for score/rank/diff; method timings
# show all methods together.

# %%
saved_plots: list[Path] = []
n_panels = len(LDOI_KEYS)


def _panel_grid(n_cols: int) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.4))
    if n_cols == 1:
        axes = np.array([axes])
    return fig, axes


if n == 0:
    print("No score/rank plots (no successful models).")
else:
    short_labels = [Path(r.rel).stem for r in ok_rows]

    # --- score scatter (side by side) ---
    fig, axes = _panel_grid(n_panels)
    for ax, key in zip(axes, LDOI_KEYS):
        vm = variant_metrics[key]
        ax.scatter(
            scores_min,
            vm.scores,
            s=36,
            zorder=3,
            color=METHOD_COLORS[key],
            alpha=0.9,
        )
        lo = float(min(scores_min.min(), vm.scores.min()))
        hi = float(max(scores_min.max(), vm.scores.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("score (min_traps)")
        ax.set_ylabel(f"score ({vm.label})")
        ax.set_title(
            f"{vm.label}\nr={vm.pearson:.3f}, ρ={vm.spearman:.3f}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"ABA-{DATA_SET} NAV scores vs min_traps  (n={n})", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_SCORE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(PLOT_SCORE)
    print(f"saved: {PLOT_SCORE}")

    # --- rank scatter (side by side) ---
    fig, axes = _panel_grid(n_panels)
    rlim = (0.5, n + 0.5)
    for ax, key in zip(axes, LDOI_KEYS):
        vm = variant_metrics[key]
        ax.scatter(
            ranks_min,
            vm.ranks,
            s=36,
            zorder=3,
            color=METHOD_COLORS[key],
            alpha=0.9,
        )
        ax.plot(rlim, rlim, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlim(rlim)
        ax.set_ylim(rlim)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("rank (min_traps)")
        ax.set_ylabel(f"rank ({vm.label})")
        ax.set_title(
            f"{vm.label}\norder={vm.order_agree:.3f}, "
            f"|Δrank|={vm.mean_abs_rank_diff:.2f}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"ABA-{DATA_SET} NAV ranks vs min_traps  (n={n})", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_RANK, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(PLOT_RANK)
    print(f"saved: {PLOT_RANK}")

    # --- score difference by model (side by side; shared model order) ---
    # Order models by mean absolute diff across variants for readability.
    mean_abs_diff = np.mean(
        np.vstack([np.abs(variant_metrics[k].diffs) for k in LDOI_KEYS]),
        axis=0,
    )
    order_diff = np.argsort(mean_abs_diff)
    fig_h = max(3.5, 0.28 * n)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, fig_h), sharey=True)
    if n_panels == 1:
        axes = np.array([axes])
    y = np.arange(n)
    for ax, key in zip(axes, LDOI_KEYS):
        vm = variant_metrics[key]
        ax.barh(
            y,
            vm.diffs[order_diff],
            color=METHOD_COLORS[key],
            alpha=0.85,
        )
        ax.axvline(0.0, color="0.4", linewidth=1)
        ax.set_xlabel(f"{vm.label} − min_traps")
        ax.set_title(vm.label, fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short_labels[i] for i in order_diff], fontsize=7)
    fig.suptitle(f"ABA-{DATA_SET}: per-model score difference", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIFF, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(PLOT_DIFF)
    print(f"saved: {PLOT_DIFF}")

# --- method timing bars (all methods side by side) ---
methods_with_time = [
    k for k in ALL_METHOD_KEYS if timing_by_method[k]["n"] > 0
]
if methods_with_time:
    labels = [ALL_METHOD_LABELS[k] for k in methods_with_time]
    totals = [timing_by_method[k]["total_s"] for k in methods_with_time]
    means = [timing_by_method[k]["mean_s"] for k in methods_with_time]
    counts = [int(timing_by_method[k]["n"]) for k in methods_with_time]
    colors = [METHOD_COLORS[k] for k in methods_with_time]
    fig_w = max(9.0, 1.4 * len(methods_with_time) + 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.4))
    x = np.arange(len(methods_with_time))

    axes[0].bar(x, totals, color=colors, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Total wall time (s)")
    axes[0].set_title("Total time across models")
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, (val, cnt) in enumerate(zip(totals, counts)):
        axes[0].text(
            i, val, f"{val:.1f}s\n(n={cnt})", ha="center", va="bottom", fontsize=7
        )

    axes[1].bar(x, means, color=colors, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Mean wall time (s)")
    axes[1].set_title("Mean time per model")
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, val in enumerate(means):
        axes[1].text(i, val, f"{val:.2f}s", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"ABA-{DATA_SET} NAV scoring: method wall times", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_TIMING, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(PLOT_TIMING)
    print(f"saved: {PLOT_TIMING}")
else:
    print("No timing plot (no timed runs).")
print()

# %% [markdown]
# ## Summary + README

# %%
finished_utc = datetime.now(timezone.utc)
wall_s = (finished_utc - STARTED_UTC).total_seconds()

n_min_cached = sum(1 for r in rows if r.min_from_cache)

print("Summary")
print(f"  DATA_SET:          {DATA_SET}")
print(f"  bnets scanned:     {len(rows)}")
print(f"  all methods OK:    {len(ok_rows)}")
print(f"  failed:            {len(failed)}")
print(
    f"  min_traps cache:   hits={cache_hits}  "
    f"misses={cache_misses}  used_on_rows={n_min_cached}"
)
for key in LDOI_KEYS:
    if key not in variant_metrics:
        continue
    vm = variant_metrics[key]
    print(
        f"  {vm.label:<18} r={vm.pearson:.4f}  ρ={vm.spearman:.4f}  "
        f"order={vm.order_agree:.4f}"
    )
for key in ALL_METHOD_KEYS:
    stats = timing_by_method[key]
    label = ALL_METHOD_LABELS[key]
    if stats["n"] == 0:
        print(f"  {label} time: (none)")
    else:
        note = ""
        if key == "min_traps" and n_min_cached:
            note = f"  ({n_min_cached} from cache)"
        print(
            f"  {label} time: "
            f"total={stats['total_s']:.2f}s  mean={stats['mean_s']:.2f}s"
            f"{note}"
        )
print(f"  wall clock:        {wall_s:.1f}s")
print(f"  Log written to:    {LOG_PATH}")
for p in saved_plots:
    print(f"  Plot written to:   {p}")
print(f"finished (UTC): {finished_utc.isoformat()}")


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


readme_lines = [
    f"# ABA-{DATA_SET} scores: `min_traps` vs LDOI family",
    "",
    "Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,",
    f"using experiments from `{CONFIG_REL}` "
    f"(`{data_path.name}`, `default_sources`).",
    "",
    "Reference method: `min_traps`. Compared LDOI variants:",
    "",
]
for key in LDOI_KEYS:
    readme_lines.append(f"- `{LDOI_LABEL[key]}`")
readme_lines.extend(
    [
        "",
        f"- **Data set:** `{DATA_SET}`",
        f"- **Started (UTC):** {STARTED_UTC.isoformat()}",
        f"- **Finished (UTC):** {finished_utc.isoformat()}",
        f"- **Wall clock:** {wall_s:.1f}s",
        f"- **Hierarchy:** {HIERARCHY}",
        f"- **Experiments:** {len(experiments)}",
        f"- **Bnets scanned:** {len(rows)}",
        f"- **All methods OK:** {len(ok_rows)}",
        f"- **Failed:** {len(failed)}",
        f"- **min_traps cache:** "
        f"`use={USE_MIN_TRAPS_CACHE}`, "
        f"`force_recompute={FORCE_RECOMPUTE_MIN_TRAPS}`, "
        f"hits={cache_hits}, misses={cache_misses}",
        "",
        "## Score agreement vs min_traps",
        "",
    ]
)
if ok_rows:
    readme_lines.append(
        "| method | Pearson r | Spearman ρ | pairwise order | "
        "mean |Δrank| | mean Δscore |"
    )
    readme_lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for key in LDOI_KEYS:
        vm = variant_metrics[key]
        readme_lines.append(
            f"| `{vm.label}` | `{_fmt(vm.pearson)}` | `{_fmt(vm.spearman)}` | "
            f"`{_fmt(vm.order_agree)}` ({vm.n_agree}/{vm.n_pairs}) | "
            f"`{_fmt(vm.mean_abs_rank_diff, 3)}` | "
            f"`{_fmt(float(vm.diffs.mean()), 3)}` |"
        )
    for key in LDOI_KEYS:
        vm = variant_metrics[key]
        if not vm.top_k_overlaps:
            continue
        tops = ", ".join(
            f"top-{kk}={_fmt(ov, 3)}" for kk, ov in vm.top_k_overlaps.items()
        )
        readme_lines.append(f"- `{vm.label}` overlaps: {tops}")
else:
    readme_lines.append("- No successful models to compare.")

readme_lines.extend(["", "## Method timings", ""])
readme_lines.append(
    "Wall time for prediction + NAV scoring + model score "
    "(models where that method completed). "
    "`min_traps` times reuse the originally measured values when loaded "
    "from cache."
)
readme_lines.append("")
readme_lines.append(
    "| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |"
)
readme_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for key in ALL_METHOD_KEYS:
    stats = timing_by_method[key]
    label = ALL_METHOD_LABELS[key]
    if stats["n"] == 0:
        readme_lines.append(f"| `{label}` | 0 | — | — | — | — | — |")
    else:
        readme_lines.append(
            f"| `{label}` | {int(stats['n'])} | "
            f"{stats['total_s']:.2f} | {stats['mean_s']:.2f} | "
            f"{stats['median_s']:.2f} | {stats['min_s']:.2f} | "
            f"{stats['max_s']:.2f} |"
        )

if failed:
    readme_lines.extend(["", "## Failures", ""])
    for r in failed:
        readme_lines.append(f"- `{r.rel}`: {r.error}")

readme_lines.extend(
    [
        "",
        "## Artifacts",
        "",
        "Stable filenames (overwritten on each run):",
        "",
        f"- [`{LOG_PATH.name}`]({LOG_PATH.name}) — full run log",
        f"- [`{MIN_TRAPS_CACHE_PATH.name}`]({MIN_TRAPS_CACHE_PATH.name}) — "
        "`min_traps` score + timing cache",
        f"- [`{TIMINGS_CSV.name}`]({TIMINGS_CSV.name}) — timing summary CSV",
    ]
)
if n > 0:
    readme_lines.append(
        f"- [`{per_model_csv.name}`]({per_model_csv.name}) — per-model wall times"
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
print(f"timings csv: {TIMINGS_CSV}")
