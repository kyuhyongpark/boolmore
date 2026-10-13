# %% [markdown]
# Speed: `min_traps` vs `ldoi` NAV prediction
#
# Benchmarks `get_NAV_prediction` from
# `boolmore.inference.prediction` with
# `method="min_traps"` vs `method="ldoi"` on the ABA case study
# (`ABA_GA_base_A.bnet` + `data_A.tsv`).
#
# - **min_traps**: fix interventions → minimal trap spaces (ASP) → average node values
# - **ldoi**: logical domain of influence (percolation) → 0/1/0.5 approximation
#
# Reports wall time for full prediction calls and per unique intervention.

# %%
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from boolmore.boolean_functions import assignment_to_dict
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.inference.prediction import (
    _nav_from_ldoi,
    _nav_from_min_traps,
    get_NAV_prediction,
)
from boolmore.io.load import load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.types import Assignment

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
ABA = REPO / "case_study" / "ABA"
BNET = ABA / "baseline_models" / "ABA_GA_base_A.bnet"
DATA = ABA / "data" / "data_A.tsv"

N_WARMUP = 1
N_REPEATS = 3

# %% [markdown]
# ## Load model and experiments

# %%
primes = load_primes(BNET)
experiments = import_NAV_exps(DATA)
interventions = sorted({exp.fixes for exp in experiments})

print(f"model: {BNET.name} ({len(primes)} nodes)")
print(f"experiments: {len(experiments)}")
print(f"unique interventions: {len(interventions)}")

# ASP probe once up front so min_traps warmup is not the first failure point
ensure_asp_backend()

# %% [markdown]
# ## Helpers


# %%
def format_fixes(fixes: Assignment) -> str:
    if not fixes:
        return "(none)"
    return "{" + ", ".join(f"{n}={v}" for n, v in fixes) + "}"


def timed_call(fn, *args, n_warmup: int = N_WARMUP, n_repeats: int = N_REPEATS, **kwargs):
    """Return (last_result, list of wall times in seconds)."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    times: list[float] = []
    result = None
    for _ in range(n_repeats):
        t0 = perf_counter()
        result = fn(*args, **kwargs)
        times.append(perf_counter() - t0)
    return result, times


def summarize(times: list[float]) -> str:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return (
        f"mean={mean:.4f}s  stdev={stdev:.4f}s  "
        f"min={min(times):.4f}s  max={max(times):.4f}s  "
        f"(n={len(times)})"
    )

# %% [markdown]
# ## Full `get_NAV_prediction` timing

# %%
preds_min, times_min = timed_call(
    get_NAV_prediction, primes, experiments, method="min_traps"
)
preds_ldoi, times_ldoi = timed_call(
    get_NAV_prediction, primes, experiments, method="ldoi"
)

mean_min = statistics.mean(times_min)
mean_ldoi = statistics.mean(times_ldoi)

print("get_NAV_prediction (all unique interventions)")
print(f"  min_traps: {summarize(times_min)}")
print(f"  ldoi:      {summarize(times_ldoi)}")
if mean_ldoi < mean_min:
    print(f"  ldoi faster by {mean_min / mean_ldoi:.1f}x")
else:
    print(f"  min_traps faster by {mean_ldoi / mean_min:.1f}x")
print(f"  predictions: min_traps={len(preds_min)}, ldoi={len(preds_ldoi)}")

# %% [markdown]
# ## Per-intervention timing
#
# Times `_nav_from_min_traps` and `_nav_from_ldoi` once per unique `fixes`
# (single shot each; no per-call warmup beyond the full-call warmup above).

# %%
rows: list[dict] = []

for fixes in interventions:
    perturbation = assignment_to_dict(fixes)

    t0 = perf_counter()
    avg_min = _nav_from_min_traps(primes, perturbation)
    t_min = perf_counter() - t0

    t0 = perf_counter()
    avg_ldoi = _nav_from_ldoi(primes, perturbation)
    t_ldoi = perf_counter() - t0

    rows.append(
        {
            "fixes": fixes,
            "t_min": t_min,
            "t_ldoi": t_ldoi,
            "n_forced_ldoi": sum(
                1 for v in avg_ldoi.values() if v in (0.0, 1.0)
            ),
            "n_nodes": len(avg_min),
        }
    )

# group experiment ids by fixes for labels
ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    ids_by_fixes[exp.fixes].append(exp.id)

print(
    f"{'fixes':<55} {'min_traps':>10} {'ldoi':>10} "
    f"{'min/ldoi':>9}  exp ids"
)
print("-" * 110)
for row in sorted(rows, key=lambda r: r["t_min"], reverse=True):
    fixes = row["fixes"]
    ratio = row["t_min"] / row["t_ldoi"] if row["t_ldoi"] > 0 else float("inf")
    print(
        f"{format_fixes(fixes):<55} "
        f"{row['t_min']:10.4f} "
        f"{row['t_ldoi']:10.4f} "
        f"{ratio:9.2f}x  "
        f"{ids_by_fixes[fixes]}"
    )

total_min = sum(r["t_min"] for r in rows)
total_ldoi = sum(r["t_ldoi"] for r in rows)
print("-" * 110)
print(
    f"{'TOTAL (sum of singles)':<55} "
    f"{total_min:10.4f} "
    f"{total_ldoi:10.4f} "
    f"{total_min / total_ldoi if total_ldoi else float('inf'):9.2f}x"
)

# %% [markdown]
# ## Summary

# %%
print("Summary")
print(
    f"  Full get_NAV_prediction mean: "
    f"min_traps={statistics.mean(times_min):.4f}s, "
    f"ldoi={statistics.mean(times_ldoi):.4f}s"
)
print(
    f"  Per-intervention sum: "
    f"min_traps={total_min:.4f}s, ldoi={total_ldoi:.4f}s"
)
print(
    f"  Slowest min_traps intervention: "
    f"{format_fixes(max(rows, key=lambda r: r['t_min'])['fixes'])} "
    f"({max(r['t_min'] for r in rows):.4f}s)"
)
print(
    f"  Slowest ldoi intervention: "
    f"{format_fixes(max(rows, key=lambda r: r['t_ldoi'])['fixes'])} "
    f"({max(r['t_ldoi'] for r in rows):.4f}s)"
)
