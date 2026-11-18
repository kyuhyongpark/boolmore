# %% [markdown]
# Speed: pure-Python vs Cython `get_uni_rr` on breast-cancer GA
#
# Benchmarks genetic-algorithm wall time on the breast-cancer case study when
# canonical rr closure uses either:
#
# 1. **python**: force `_get_uni_rr_py` (Cython dispatch off)
# 2. **cython**: normal dispatch through the compiled kernel (requires the
#    `boolmore._get_uni_rr` extension)
#
# Same seed, core, and GA params for both arms. Uses `CORE=1` so RNG
# trajectories match; start/final scores must agree (same algorithm).
#
# Related-node caching from `build_evaluator` is on for both arms.
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.
# Does not call `run_ga` (avoids exporting `.bnet` / run logs into CWD).

# %%
from __future__ import annotations

import copy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import boolmore.boolean_functions as bf
from boolmore.evaluation.score import format_score
from boolmore.genetic.algorithm import ga_main
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.population import Candidate
from boolmore.genetic.selection import Selector
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
# Single process so python vs cython share the same RNG trajectory.
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
    f"bench_uni_rr_py_vs_cython_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print()

if not bf._UNI_RR_CYTHON or bf._get_uni_rr_cy is None:
    print("ERROR: Cython get_uni_rr extension is not available.")
    print("Rebuild with a C compiler, e.g. `pip install -e .` from the package root.")
    sys.stdout = _stdout
    _log_file.close()
    raise SystemExit(1)

print(
    f"Cython extension loaded; dispatch min k={bf._UNI_RR_CYTHON_MIN_K} "
    f"(n >= {1 << bf._UNI_RR_CYTHON_MIN_K})"
)
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
    backend: str
    wall_seconds: float
    n_iterations: int
    n_generated: int
    start_score: float
    start_max: float
    final_score: float
    final_max: float


@contextmanager
def uni_rr_backend(*, use_cython: bool) -> Iterator[None]:
    """Toggle Cython dispatch for ``get_uni_rr`` (call-time module flag)."""
    previous = bf._UNI_RR_CYTHON
    bf._UNI_RR_CYTHON = bool(use_cython) and previous
    try:
        yield
    finally:
        bf._UNI_RR_CYTHON = previous


def run_ga_bench(*, use_cython: bool) -> RunResult:
    """Run breast-cancer GA once with Cython get_uni_rr on or off."""
    backend = "cython" if use_cython else "python"
    config = copy.deepcopy(base_config)

    print("=" * 72)
    print(
        f"Run: uni_rr={backend}, nav_method={config['nav_method']}, "
        f"core={CORE}"
    )
    print("=" * 72)

    with uni_rr_backend(use_cython=use_cython):
        assert bf._UNI_RR_CYTHON is use_cython

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
            backend=backend,
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
# ## Run GA: python vs cython
#
# Same breast-cancer config, seed, and core. Only ``get_uni_rr`` dispatch
# differs.

# %%
results: dict[str, RunResult] = {}
results["python"] = run_ga_bench(use_cython=False)
results["cython"] = run_ga_bench(use_cython=True)

# %% [markdown]
# ## Summary

# %%
python = results["python"]
cython = results["cython"]


def _pct(score: float, max_score: float) -> float:
    return 100.0 * score / max_score if max_score else 0.0


print("Summary: breast-cancer NAV GA, get_uni_rr python vs cython")
print("-" * 72)
print(f"{'metric':<28} {'python':>14} {'cython':>14}")
print("-" * 72)
print(
    f"{'GA wall time (s)':<28} "
    f"{python.wall_seconds:14.4f} {cython.wall_seconds:14.4f}"
)
print(
    f"{'iterations':<28} "
    f"{python.n_iterations:14d} {cython.n_iterations:14d}"
)
print(
    f"{'models generated':<28} "
    f"{python.n_generated:14d} {cython.n_generated:14d}"
)
print(
    f"{'start score':<28} "
    f"{python.start_score:8.2f}/{python.start_max:<5.0f} "
    f"{cython.start_score:8.2f}/{cython.start_max:<5.0f}"
)
print(
    f"{'final score':<28} "
    f"{python.final_score:8.2f}/{python.final_max:<5.0f} "
    f"{cython.final_score:8.2f}/{cython.final_max:<5.0f}"
)
print(
    f"{'final %':<28} "
    f"{_pct(python.final_score, python.final_max):13.1f}% "
    f"{_pct(cython.final_score, cython.final_max):13.1f}%"
)
print("-" * 72)

if python.wall_seconds > 0 and cython.wall_seconds > 0:
    if cython.wall_seconds < python.wall_seconds:
        print(
            f"cython faster by {python.wall_seconds / cython.wall_seconds:.2f}x "
            f"(saved {python.wall_seconds - cython.wall_seconds:.4f}s)"
        )
    elif python.wall_seconds < cython.wall_seconds:
        print(
            f"python faster by {cython.wall_seconds / python.wall_seconds:.2f}x "
            f"(delta {cython.wall_seconds - python.wall_seconds:.4f}s)"
        )
    else:
        print("wall times equal")

scores_match = (
    python.start_score == cython.start_score
    and python.start_max == cython.start_max
    and python.final_score == cython.final_score
    and python.final_max == cython.final_max
    and python.n_generated == cython.n_generated
    and python.n_iterations == cython.n_iterations
)
print()
if scores_match:
    print(
        "PASS: start/final scores, iterations, and generated counts match "
        "(CORE=1, identical seed)."
    )
else:
    print(
        "FAIL: GA trajectories diverged between python and cython "
        "(scores or counts differ)."
    )

print()
print(
    "Note: CORE=1 keeps RNG trajectories aligned so equivalent get_uni_rr "
    "must yield identical scores."
)
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")

if not scores_match:
    raise SystemExit(1)
