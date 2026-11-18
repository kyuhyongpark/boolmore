# %% [markdown]
# LDOI closure micro-bench: scan vs fast
#
# Compare the legacy whole-network scan closure with the compiled
# event-driven closure on breast-cancer interventions (falls back to CAD).
#
# Checks:
# 1. Identical LDOI dicts for every unique intervention
# 2. Wall-time difference (warmup + repeats), with compile cache warm
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.

# %%
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from boolmore.inference.ldoi import (
    ldoi_closure_engine,
    logical_domain_of_influence,
)
from boolmore.io.load import load_json_config, load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.io.paths import project_root
from boolmore.types import Assignment

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs"

BC_CONFIG = REPO / "case_study" / "breast_cancer" / "config.json"
CAD_BNET = REPO / "demo" / "CAD.bnet"
CAD_DATA = REPO / "demo" / "CAD_data.csv"

N_WARMUP = 1
N_REPEATS = 5
# Cap interventions so the micro-bench stays quick on large datasets.
MAX_SEEDS = 200

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
    f"bench_ldoi_scan_vs_fast_"
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


def timed_pass(engine: str, seeds: list[dict[str, int]], primes: dict) -> list[float]:
    """Time one full pass over all seeds under ``engine``; return per-repeat totals."""
    times: list[float] = []
    with ldoi_closure_engine(engine):  # type: ignore[arg-type]
        for _ in range(N_WARMUP):
            for seed in seeds:
                logical_domain_of_influence(primes, seed)
        for _ in range(N_REPEATS):
            t0 = perf_counter()
            for seed in seeds:
                logical_domain_of_influence(primes, seed)
            times.append(perf_counter() - t0)
    return times


def summarize(times: list[float]) -> str:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return (
        f"mean={mean:.6f}s  stdev={stdev:.6f}s  "
        f"min={min(times):.6f}s  max={max(times):.6f}s  "
        f"(n={len(times)})"
    )


def load_breast_cancer():
    config = load_json_config(BC_CONFIG)
    root = project_root()
    primes = load_primes((root / config["base"]).resolve())
    nav_cols = config.get("nav_columns", {})
    experiments = import_NAV_exps(
        (root / config["data"]).resolve(),
        id_col=nav_cols.get("id", "id"),
        weight_col=nav_cols.get("weight", "Weight"),
        observed_col=nav_cols.get("observed", "Observed node"),
        outcome_col=nav_cols.get("outcome", "Value"),
        context_cols=nav_cols.get("context"),
        intervention_cols=nav_cols.get("intervention"),
    )
    return "breast_cancer", primes, experiments


def load_cad():
    primes = load_primes(CAD_BNET)
    experiments = import_NAV_exps(CAD_DATA)
    return "CAD", primes, experiments

# %% [markdown]
# ## Load model and interventions

# %%
if BC_CONFIG.is_file():
    label, primes, experiments = load_breast_cancer()
else:
    label, primes, experiments = load_cad()

ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    ids_by_fixes[exp.fixes].append(exp.id)

interventions = sorted(ids_by_fixes)[:MAX_SEEDS]
seeds = [fixes_to_dict(fixes) for fixes in interventions]

print(f"dataset: {label}")
print(f"model nodes: {len(primes)}")
print(f"experiments: {len(experiments)}")
print(f"unique interventions used: {len(seeds)} (cap={MAX_SEEDS})")
print(f"warmup={N_WARMUP}, repeats={N_REPEATS}")
print()

# %% [markdown]
# ## Correctness: scan == fast?

# %%
mismatches = 0
for seed in seeds:
    with ldoi_closure_engine("scan"):
        scan = logical_domain_of_influence(primes, seed)
    with ldoi_closure_engine("fast"):
        fast = logical_domain_of_influence(primes, seed)
    if scan != fast:
        mismatches += 1
        label_seed = ",".join(f"{n}={v}" for n, v in sorted(seed.items()))
        print(f"MISMATCH: {label_seed}")
        print(f"  scan={scan}")
        print(f"  fast={fast}")

print(f"mismatches: {mismatches} / {len(seeds)}")
if mismatches:
    raise SystemExit("scan and fast LDOI disagree; aborting timing")
print("all interventions match")
print()

# %% [markdown]
# ## Timing

# %%
scan_times = timed_pass("scan", seeds, primes)
fast_times = timed_pass("fast", seeds, primes)

print("Full-pass timing (all selected interventions per repeat)")
print(f"  scan: {summarize(scan_times)}")
print(f"  fast: {summarize(fast_times)}")

scan_mean = statistics.mean(scan_times)
fast_mean = statistics.mean(fast_times)
if fast_mean > 0:
    print(f"  fast speedup: {scan_mean / fast_mean:.2f}x")
print()
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
