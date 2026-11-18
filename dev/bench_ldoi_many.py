# %% [markdown]
# LDOI batch micro-bench: scan vs Python fast vs Cython batch
#
# Times a full pass over breast-cancer interventions (falls back to CAD).

# %%
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from boolmore.inference.ldoi import (
    _LDOI_CYTHON,
    _get_compiled,
    _ldoi_many_cython,
    _ldoi_many_python,
    ldoi_closure_engine,
    logical_domain_of_influence,
)
from boolmore.io.load import load_json_config, load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.io.paths import project_root
from boolmore.types import Assignment

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs"
BC_CONFIG = REPO / "case_study" / "breast_cancer" / "config.json"
CAD_BNET = REPO / "demo" / "CAD.bnet"
CAD_DATA = REPO / "demo" / "CAD_data.csv"

N_WARMUP = 1
N_REPEATS = 5
MAX_SEEDS = 200


class Tee:
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


def fixes_to_dict(fixes: Assignment) -> dict[str, int]:
    return {node: int(value) for node, value in fixes}


def summarize(times: list[float]) -> str:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return (
        f"mean={mean:.6f}s  stdev={stdev:.6f}s  "
        f"min={min(times):.6f}s  max={max(times):.6f}s  "
        f"(n={len(times)})"
    )


def timed(fn) -> list[float]:
    for _ in range(N_WARMUP):
        fn()
    times: list[float] = []
    for _ in range(N_REPEATS):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return times


LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / (
    f"bench_ldoi_many_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"ldoi_cython available: {_LDOI_CYTHON}")
print()

if BC_CONFIG.is_file():
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
    label = "breast_cancer"
else:
    primes = load_primes(CAD_BNET)
    experiments = import_NAV_exps(CAD_DATA)
    label = "CAD"

ids_by_fixes: dict[Assignment, list[int]] = defaultdict(list)
for exp in experiments:
    ids_by_fixes[exp.fixes].append(exp.id)
interventions = sorted(ids_by_fixes)[:MAX_SEEDS]
seeds = [fixes_to_dict(fixes) for fixes in interventions]
compiled = _get_compiled(primes)

print(f"dataset: {label}")
print(f"model nodes: {len(primes)}")
print(f"unique interventions: {len(seeds)} (cap={MAX_SEEDS})")
print()

with ldoi_closure_engine("scan"):
    scan_ref = [logical_domain_of_influence(primes, seed) for seed in seeds]
py_ref = _ldoi_many_python(compiled, seeds)
assert py_ref == scan_ref
if _LDOI_CYTHON:
    cy_ref = _ldoi_many_cython(compiled, seeds)
    assert cy_ref == scan_ref
print("correctness: scan == python fast" + (" == cython" if _LDOI_CYTHON else ""))
print()

with ldoi_closure_engine("scan"):
    scan_times = timed(
        lambda: [
            logical_domain_of_influence(primes, seed) for seed in seeds
        ]
    )
py_times = timed(lambda: _ldoi_many_python(compiled, seeds))
if _LDOI_CYTHON:
    cy_times = timed(lambda: _ldoi_many_cython(compiled, seeds))

print("Full-pass timing")
print(f"  scan:        {summarize(scan_times)}")
print(f"  python fast: {summarize(py_times)}")
if _LDOI_CYTHON:
    print(f"  cython batch:{summarize(cy_times)}")
    scan_mean = statistics.mean(scan_times)
    py_mean = statistics.mean(py_times)
    cy_mean = statistics.mean(cy_times)
    print(f"  cython vs scan: {scan_mean / cy_mean:.2f}x")
    print(f"  cython vs python fast: {py_mean / cy_mean:.2f}x")

print()
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")
