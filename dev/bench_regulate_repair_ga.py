# %% [markdown]
# Regulate-repair modes on breast-cancer GA
#
# Benchmarks genetic-algorithm wall time when regulate failures use:
#
# 1. **remutate**: no ``try_add_regulator``; full remutation only
# 2. **oneshot**: at most one ``try_add_regulator``, then remutate if still invalid
#
# Same seed, core, and GA params for all arms. Uses ``CORE=1``. Trajectories
# diverge once repair consumes RNG, so scores need not match.
#
# Related-node caching from ``build_evaluator`` is on for all arms.
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
from boolmore.genetic.operators import mutation as mut
from boolmore.genetic.operators.mutation import (
    get_repair_stats,
    regulate_repair_mode,
    reset_repair_stats,
)
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
CORE = 1

PARAM_OVERRIDES = {
    "total_iter": 3,
    "per_iter": 100,
    "keep": 20,
    "mix": 20,
    "stop_if_max": False,
}

MODES = ("remutate", "oneshot")

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
    f"bench_regulate_repair_ga_"
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
print(f"modes: {MODES}")
print("related_nodes: cached via build_evaluator (all arms)")
print()

# %% [markdown]
# ## Helpers


# %%
@dataclass
class RunResult:
    mode: str
    wall_seconds: float
    n_iterations: int
    n_generated: int
    start_score: float
    start_max: float
    final_score: float
    final_max: float
    trials: int
    retries: int
    repair_attempts: int
    repair_accepted: int
    repair_forward: int
    repair_irr_fallback: int
    repair_irr_fallback_fixed: int
    repair_noop: int


def run_ga_bench(*, mode: str) -> RunResult:
    """Run breast-cancer GA once with the given regulate-repair mode."""
    config = copy.deepcopy(base_config)

    print("=" * 72)
    print(
        f"Run: regulate_repair_mode={mode}, nav_method={config['nav_method']}, "
        f"core={CORE}"
    )
    print("=" * 72)

    reset_repair_stats()
    with regulate_repair_mode(mode):
        assert mut.REGULATE_REPAIR_MODE == mode

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
        stats = get_repair_stats()

        result = RunResult(
            mode=mode,
            wall_seconds=wall,
            n_iterations=len(outcome.history),
            n_generated=outcome.final.generated,
            start_score=start.eval_result.score,
            start_max=start.eval_result.max_score,
            final_score=final.eval_result.score,
            final_max=final.eval_result.max_score,
            trials=stats["trials"],
            retries=stats["retries"],
            repair_attempts=stats["repair_attempts"],
            repair_accepted=stats["repair_accepted"],
            repair_forward=stats["repair_forward"],
            repair_irr_fallback=stats["repair_irr_fallback"],
            repair_irr_fallback_fixed=stats["repair_irr_fallback_fixed"],
            repair_noop=stats["repair_noop"],
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
        print(
            f"  mutation trials={result.trials}, retries={result.retries}, "
            f"repair_attempts={result.repair_attempts}, "
            f"repair_accepted={result.repair_accepted}"
        )
        print(
            f"  forward={result.repair_forward}, "
            f"irr_fallback={result.repair_irr_fallback} "
            f"(fixed={result.repair_irr_fallback_fixed}), "
            f"noop={result.repair_noop}"
        )
        print()
        return result


# %% [markdown]
# ## Run GA: remutate vs oneshot

# %%
results: dict[str, RunResult] = {}
for mode in MODES:
    results[mode] = run_ga_bench(mode=mode)

# %% [markdown]
# ## Summary

# %%


def _pct(score: float, max_score: float) -> float:
    return 100.0 * score / max_score if max_score else 0.0


print("Summary: breast-cancer NAV GA, regulate repair remutate / oneshot")
print("-" * 88)
header = f"{'metric':<28}" + "".join(f"{m:>20}" for m in MODES)
print(header)
print("-" * 88)

rows = [
    ("GA wall time (s)", "wall_seconds", ".4f"),
    ("iterations", "n_iterations", "d"),
    ("models generated", "n_generated", "d"),
    ("mutation trials", "trials", "d"),
    ("mutation retries", "retries", "d"),
    ("repair_attempts", "repair_attempts", "d"),
    ("repair_accepted", "repair_accepted", "d"),
    ("repair_forward", "repair_forward", "d"),
    ("repair_irr_fallback", "repair_irr_fallback", "d"),
    ("irr_fallback_fixed", "repair_irr_fallback_fixed", "d"),
    ("repair_noop", "repair_noop", "d"),
]

for label, attr, fmt in rows:
    cells = []
    for m in MODES:
        val = getattr(results[m], attr)
        cells.append(f"{val:{'>'}{20}{fmt}}")
    print(f"{label:<28}" + "".join(cells))

print(
    f"{'start score':<28}"
    + "".join(
        f"{results[m].start_score:12.2f}/{results[m].start_max:<7.0f}"
        for m in MODES
    )
)
print(
    f"{'final score':<28}"
    + "".join(
        f"{results[m].final_score:12.2f}/{results[m].final_max:<7.0f}"
        for m in MODES
    )
)
print(
    f"{'final %':<28}"
    + "".join(
        f"{_pct(results[m].final_score, results[m].final_max):19.1f}%"
        for m in MODES
    )
)
print("-" * 88)

walls = {m: results[m].wall_seconds for m in MODES}
best = min(walls, key=walls.get)
print(f"fastest wall: {best} ({walls[best]:.4f}s)")
for m in MODES:
    if m == best or walls[best] <= 0:
        continue
    if walls[m] >= walls[best]:
        print(
            f"  {best} faster than {m} by {walls[m] / walls[best]:.2f}x "
            f"(saved {walls[m] - walls[best]:.4f}s)"
        )

print()
print(
    "Note: CORE=1 but repair modes change RNG use, so scores/generated "
    "need not match across arms."
)
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
