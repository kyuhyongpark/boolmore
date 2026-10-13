# %% [markdown]
# Boolmore LDOI vs pystablemotifs LDOI (demo)
#
# Compare `boolmore.inference.ldoi.logical_domain_of_influence` with
# `pystablemotifs.drivers.logical_domain_of_influence` on the CAD demo.
#
# pystablemotifs returns `(implied, contradicted)`. A thin wrapper maps that
# to boolmore's `dict[str, int]` shape as `seed ∪ implied` (documented
# equivalence when there is no contradiction).
#
# Checks:
# 1. Same output for every unique experiment intervention
# 2. Wall-time difference (warmup + repeats)
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.
#
# Note: `pystablemotifs` is not a declared boolmore dependency; use an env
# that has it installed (e.g. the project `.venv314`).

# %%
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pystablemotifs.drivers import (
    logical_domain_of_influence as psm_logical_domain_of_influence,
)

from boolmore.inference.ldoi import (
    logical_domain_of_influence as boolmore_logical_domain_of_influence,
)
from boolmore.io.load import load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.types import Assignment

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
BNET = REPO / "demo" / "CAD.bnet"
DATA = REPO / "demo" / "CAD_data.csv"
LOG_DIR = Path(__file__).resolve().parent / "logs"

N_WARMUP = 1
N_REPEATS = 10

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
LOG_PATH = LOG_DIR / (
    f"bench_ldoi_boolmore_vs_pystablemotifs_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print()

# %% [markdown]
# ## Helpers


# %%
def fixes_to_dict(fixes: Assignment) -> dict[str, int]:
    return {node: int(value) for node, value in fixes}


def format_state(state: dict[str, int]) -> str:
    if not state:
        return "(none)"
    return "{" + ", ".join(f"{n}={state[n]}" for n in sorted(state)) + "}"


def ldoi_boolmore(primes: dict, seed: dict[str, int]) -> dict[str, int]:
    """Boolmore LDOI: Yang/pystablemotifs prime-implicant closure → seed ∪ LDOI."""
    return boolmore_logical_domain_of_influence(primes, seed)


def ldoi_pystablemotifs_wrapped(
    primes: dict, seed: dict[str, int]
) -> dict[str, int]:
    """pystablemotifs LDOI wrapped to boolmore's dict output.

    Returns ``seed ∪ implied``. Contradiction nodes (if any) are not folded
    into the dict; callers can inspect them via ``psm_logical_domain_of_influence``
    directly when diagnosing mismatches.
    """
    implied, _contradicted = psm_logical_domain_of_influence(seed, primes)
    return {**seed, **implied}


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
        f"mean={mean:.6f}s  stdev={stdev:.6f}s  "
        f"min={min(times):.6f}s  max={max(times):.6f}s  "
        f"(n={len(times)})"
    )

# %% [markdown]
# ## Load model and experiments

# %%
primes = load_primes(BNET)
experiments = import_NAV_exps(DATA)

ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    ids_by_fixes[exp.fixes].append(exp.id)

interventions = sorted(ids_by_fixes)

print(f"model: {BNET.name} ({len(primes)} nodes)")
print(f"data:  {DATA.name}")
print(f"experiments: {len(experiments)}")
print(f"unique interventions: {len(interventions)}")
print()

# %% [markdown]
# ## Correctness: same output?

# %%
mismatches: list[dict] = []
n_with_contra = 0

print(f"{'fixes':<40} {'match':>5} {'contra':>6}  exp ids")
print("-" * 90)

for fixes in interventions:
    seed = fixes_to_dict(fixes)
    bm = ldoi_boolmore(primes, seed)
    implied, contradicted = psm_logical_domain_of_influence(seed, primes)
    wrapped = {**seed, **implied}
    match = bm == wrapped
    if contradicted:
        n_with_contra += 1
    if not match:
        mismatches.append(
            {
                "fixes": fixes,
                "boolmore": bm,
                "wrapped": wrapped,
                "implied": implied,
                "contradicted": contradicted,
            }
        )

    print(
        f"{format_state(seed):<40} "
        f"{str(match):>5} "
        f"{str(bool(contradicted)):>6}  "
        f"{ids_by_fixes[fixes]}"
    )

print("-" * 90)
print(
    f"matches: {len(interventions) - len(mismatches)}/{len(interventions)}; "
    f"mismatches: {len(mismatches)}; "
    f"with nonempty psm contradiction: {n_with_contra}"
)

if mismatches:
    print()
    print("Mismatch details")
    for row in mismatches:
        print(f"  fixes:        {format_state(fixes_to_dict(row['fixes']))}")
        print(f"  boolmore:     {format_state(row['boolmore'])}")
        print(f"  psm wrapped:  {format_state(row['wrapped'])}")
        print(f"  psm implied:  {format_state(row['implied'])}")
        print(f"  psm contra:   {format_state(row['contradicted'])}")
else:
    print("All unique interventions: boolmore LDOI == seed ∪ psm implied.")

print()

# %% [markdown]
# ## Speed: all unique interventions in one pass

# %%
def run_all_boolmore() -> dict[Assignment, dict[str, int]]:
    return {
        fixes: ldoi_boolmore(primes, fixes_to_dict(fixes))
        for fixes in interventions
    }


def run_all_psm_wrapped() -> dict[Assignment, dict[str, int]]:
    return {
        fixes: ldoi_pystablemotifs_wrapped(primes, fixes_to_dict(fixes))
        for fixes in interventions
    }


_, times_bm = timed_call(run_all_boolmore)
_, times_psm = timed_call(run_all_psm_wrapped)

mean_bm = statistics.mean(times_bm)
mean_psm = statistics.mean(times_psm)

print("Full pass over all unique interventions")
print(f"  boolmore LDOI:           {summarize(times_bm)}")
print(f"  pystablemotifs wrapped:  {summarize(times_psm)}")
if mean_psm < mean_bm:
    print(f"  pystablemotifs faster by {mean_bm / mean_psm:.1f}x")
else:
    print(f"  boolmore faster by {mean_psm / mean_bm:.1f}x")
print()

# %% [markdown]
# ## Speed: per intervention

# %%
rows: list[dict] = []

# one extra warmup each so first timed row is not cold
_ = ldoi_boolmore(primes, {})
_ = ldoi_pystablemotifs_wrapped(primes, {})

for fixes in interventions:
    seed = fixes_to_dict(fixes)

    t0 = perf_counter()
    for _ in range(N_REPEATS):
        ldoi_boolmore(primes, seed)
    t_bm = (perf_counter() - t0) / N_REPEATS

    t0 = perf_counter()
    for _ in range(N_REPEATS):
        ldoi_pystablemotifs_wrapped(primes, seed)
    t_psm = (perf_counter() - t0) / N_REPEATS

    rows.append({"fixes": fixes, "t_bm": t_bm, "t_psm": t_psm})

print(
    f"{'fixes':<40} {'boolmore':>10} {'psm_wrap':>10} "
    f"{'bm/psm':>8}  exp ids"
)
print("-" * 95)
for row in sorted(rows, key=lambda r: r["t_bm"], reverse=True):
    fixes = row["fixes"]
    ratio = row["t_bm"] / row["t_psm"] if row["t_psm"] > 0 else float("inf")
    print(
        f"{format_state(fixes_to_dict(fixes)):<40} "
        f"{row['t_bm']:10.6f} "
        f"{row['t_psm']:10.6f} "
        f"{ratio:8.2f}x  "
        f"{ids_by_fixes[fixes]}"
    )

total_bm = sum(r["t_bm"] for r in rows)
total_psm = sum(r["t_psm"] for r in rows)
print("-" * 95)
print(
    f"{'TOTAL (sum of means)':<40} "
    f"{total_bm:10.6f} "
    f"{total_psm:10.6f} "
    f"{total_bm / total_psm if total_psm else float('inf'):8.2f}x"
)
print()

# %% [markdown]
# ## Summary

# %%
print("Summary")
print(
    f"  Correctness: "
    f"{len(interventions) - len(mismatches)}/{len(interventions)} match "
    f"(boolmore == seed ∪ psm implied); "
    f"psm contradiction nonempty on {n_with_contra} seeds"
)
print(
    f"  Full-pass mean: boolmore={mean_bm:.6f}s, "
    f"pystablemotifs={mean_psm:.6f}s"
)
print(
    f"  Per-intervention sum of means: "
    f"boolmore={total_bm:.6f}s, pystablemotifs={total_psm:.6f}s"
)
print(f"  Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# restore stdout and close log
sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
