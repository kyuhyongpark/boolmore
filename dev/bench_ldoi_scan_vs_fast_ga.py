# %% [markdown]
# Speed: LDOI scan vs fast on whole-process breast-cancer GA
#
# Benchmarks genetic-algorithm wall time on the breast-cancer case study when
# NAV prediction uses either:
#
# 1. **scan**: legacy whole-network prime-implicant LDOI closure
# 2. **fast**: compiled event-driven LDOI closure (default)
#
# Same seed, core, and GA params for both arms. Uses `CORE=1` so trajectories
# are comparable; start and final scores must match if closures are equivalent.
#
# Related-node caching from `build_evaluator` is on for both arms.
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.
# Does not call `run_ga` (avoids exporting `.bnet` / run logs into CWD).

# %%
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from boolmore.evaluation.score import format_score
from boolmore.genetic.algorithm import ga_main
from boolmore.genetic.population import Candidate
from boolmore.genetic.selection import Selector
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.inference.ldoi import (
    get_ldoi_closure_engine,
    ldoi_closure_engine,
    set_ldoi_closure_engine,
)
from boolmore.io.load import load_json_config, load_model
from boolmore.io.paths import project_root
from boolmore.run_setup import build_evaluator, build_ga_config

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
CONFIG_REL = "case_study/breast_cancer/config.json"
LOG_DIR = Path(__file__).resolve().parent / "logs"

SEED = 0
# Single process so scan vs fast share the same RNG trajectory.
CORE = 1

# Short overrides so A/B finishes in minutes.
PARAM_OVERRIDES = {
    "total_iter": 3,
    "per_iter": 100,
    "keep": 20,
    "mix": 20,
    "stop_if_max": False,
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
LOG_PATH = LOG_DIR / (
    f"bench_ldoi_scan_vs_fast_ga_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print()

# %% [markdown]
# ## Load breast-cancer config

# %%
base_config = load_json_config((project_root() / CONFIG_REL).resolve())
base_config = copy.deepcopy(base_config)
base_config["parameters"].update(PARAM_OVERRIDES)
# LDOI path (no ASP); matches the case-study default.
base_config["nav_method"] = "ldoi"
params = base_config["parameters"]

print(f"config: {CONFIG_REL}")
print(f"data:   {base_config['data']}")
print(f"base:   {base_config['base']}")
print(f"start:  {base_config['start']}")
print(f"run_type: {base_config['run_type']}")
print(f"nav_method: {base_config.get('nav_method')}")
print(f"hierarchy: {base_config['hierarchy']}")
print(
    f"GA params: total_iter={params['total_iter']}, "
    f"per_iter={params['per_iter']}, keep={params['keep']}, "
    f"mix={params['mix']}, prob={params['prob']}"
)
print(f"seed: {SEED}, core: {CORE}")
print("related_nodes: cached via build_evaluator (both arms)")
print()

# %% [markdown]
# ## Helpers


# %%
@dataclass
class RunResult:
    engine: str
    wall_seconds: float
    n_iterations: int
    n_generated: int
    start_score: float
    start_max: float
    final_score: float
    final_max: float


def run_ga_bench(*, engine: str) -> RunResult:
    """Run breast-cancer GA once under the given LDOI closure engine."""
    config = copy.deepcopy(base_config)

    print("=" * 72)
    print(
        f"Run: ldoi_engine={engine}, nav_method={config['nav_method']}, "
        f"core={CORE}"
    )
    print("=" * 72)

    with ldoi_closure_engine(engine):  # type: ignore[arg-type]
        assert get_ldoi_closure_engine() == engine

        evaluator = build_evaluator(config, verbose=True)
        ga = build_ga_config(config, seed=SEED, core=CORE)

        start_path = (project_root() / config["start"]).resolve()
        base_path = (project_root() / config["base"]).resolve()
        if start_path == base_path:
            start_model = evaluator.base
        else:
            print(f"Loading starting model from {start_path}")
            start_model = load_model(start_path, base=evaluator.base)
            print("Starting model loaded.")

        start_result = evaluator.evaluate(start_model, model_id=0, detail=False)
        start = Candidate(start_model, 0, ga.starting_gen, start_result)
        print(start.model.info())
        print(format_score(start.eval_result))
        print()

        selector = Selector(keep=ga.keep, order_by=ga.order_by)
        reproducer = Reproducer(
            selector=selector, constraints=evaluator.constraints
        )

        t0 = perf_counter()
        outcome = ga_main(start, evaluator, selector, reproducer, ga)
        wall = perf_counter() - t0

        final = outcome.final.population[0]
        assert start.eval_result is not None
        assert final.eval_result is not None

        result = RunResult(
            engine=engine,
            wall_seconds=wall,
            n_iterations=len(outcome.history),
            n_generated=outcome.final.generated,
            start_score=start.eval_result.score,
            start_max=start.eval_result.max_score,
            final_score=final.eval_result.score,
            final_max=final.eval_result.max_score,
        )

        print()
        print(f"  GA wall time: {wall:.4f}s")
        print(
            f"  iterations={result.n_iterations}, "
            f"generated={result.n_generated}"
        )
        print(
            f"  start score: {result.start_score:.2f} / {result.start_max}"
        )
        print(
            f"  final score: {result.final_score:.2f} / {result.final_max}"
        )
        print()
        return result

# %% [markdown]
# ## Run GA: scan vs fast
#
# Same breast-cancer config, seed, and core. Only the LDOI closure engine
# differs.

# %%
# Ensure a clean default after the script (also restored by context managers).
set_ldoi_closure_engine("fast")

results: dict[str, RunResult] = {}
results["scan"] = run_ga_bench(engine="scan")
results["fast"] = run_ga_bench(engine="fast")

# %% [markdown]
# ## Summary

# %%
scan = results["scan"]
fast = results["fast"]


def _pct(score: float, max_score: float) -> float:
    return 100.0 * score / max_score if max_score else 0.0


print("Summary: breast-cancer NAV GA, LDOI scan vs fast")
print("-" * 72)
print(f"{'metric':<28} {'scan':>14} {'fast':>14}")
print("-" * 72)
print(
    f"{'GA wall time (s)':<28} "
    f"{scan.wall_seconds:14.4f} {fast.wall_seconds:14.4f}"
)
print(
    f"{'iterations':<28} "
    f"{scan.n_iterations:14d} {fast.n_iterations:14d}"
)
print(
    f"{'models generated':<28} "
    f"{scan.n_generated:14d} {fast.n_generated:14d}"
)
print(
    f"{'start score':<28} "
    f"{scan.start_score:8.2f}/{scan.start_max:<5.0f} "
    f"{fast.start_score:8.2f}/{fast.start_max:<5.0f}"
)
print(
    f"{'final score':<28} "
    f"{scan.final_score:8.2f}/{scan.final_max:<5.0f} "
    f"{fast.final_score:8.2f}/{fast.final_max:<5.0f}"
)
print(
    f"{'final %':<28} "
    f"{_pct(scan.final_score, scan.final_max):13.1f}% "
    f"{_pct(fast.final_score, fast.final_max):13.1f}%"
)
print("-" * 72)

if scan.wall_seconds > 0 and fast.wall_seconds > 0:
    if fast.wall_seconds < scan.wall_seconds:
        print(
            f"fast faster by {scan.wall_seconds / fast.wall_seconds:.2f}x "
            f"(saved {scan.wall_seconds - fast.wall_seconds:.4f}s)"
        )
    elif scan.wall_seconds < fast.wall_seconds:
        print(
            f"scan faster by {fast.wall_seconds / scan.wall_seconds:.2f}x "
            f"(delta {fast.wall_seconds - scan.wall_seconds:.4f}s)"
        )
    else:
        print("wall times equal")

scores_match = (
    scan.start_score == fast.start_score
    and scan.start_max == fast.start_max
    and scan.final_score == fast.final_score
    and scan.final_max == fast.final_max
    and scan.n_generated == fast.n_generated
    and scan.n_iterations == fast.n_iterations
)
print()
if scores_match:
    print(
        "PASS: start/final scores, iterations, and generated counts match "
        "(CORE=1, identical seed)."
    )
else:
    print(
        "FAIL: GA trajectories diverged between scan and fast "
        "(scores or counts differ)."
    )

print()
print(
    "Note: CORE=1 keeps RNG trajectories aligned so equivalent LDOI must "
    "yield identical scores."
)
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
set_ldoi_closure_engine("fast")
print(f"log saved: {LOG_PATH}")

if not scores_match:
    raise SystemExit(1)
