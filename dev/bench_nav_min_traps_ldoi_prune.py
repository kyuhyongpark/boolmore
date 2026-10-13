# %% [markdown]
# Speed: `min_traps` NAV scoring with vs without LDOI prune
#
# Benchmarks genetic-algorithm wall time on the CAD demo when
# `nav_method="min_traps"` and `ldoi_prune` is on vs off.
#
# LDOI prune is only meaningful inside the GA: survivors set
# `score_threshold = min(survivor_scores)`, and offspring whose optimistic
# LDOI upper bound is already below that threshold skip full ASP
# `min_traps` evaluation (`EvaluationResult.is_upper_bound=True`).
#
# Standalone `evaluate(model)` (no threshold) never prunes.
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.
# Does not call `run_ga` (avoids exporting `.bnet` / run logs into CWD).

# %%
from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from boolmore.evaluation.score import format_score
from boolmore.genetic.algorithm import GeneticAlgorithm, ga_main
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.population import Candidate
from boolmore.genetic.selection import Selector
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.io.load import load_json_config, load_model
from boolmore.io.paths import project_root
from boolmore.run_setup import build_evaluator, build_ga_config

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
CONFIG_REL = "demo/CAD_config.json"
LOG_DIR = Path(__file__).resolve().parent / "logs"

SEED = 0
CORE = 2

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
    f"bench_nav_min_traps_ldoi_prune_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print()

# %% [markdown]
# ## Load CAD config

# %%
base_config = load_json_config((project_root() / CONFIG_REL).resolve())
params = base_config["parameters"]

print(f"config: {CONFIG_REL}")
print(f"data:   {base_config['data']}")
print(f"base:   {base_config['base']}")
print(f"start:  {base_config['start']}")
print(f"run_type: {base_config['run_type']}")
print(f"nav_method: {base_config.get('nav_method', 'min_traps')}")
print(f"hierarchy: {base_config['hierarchy']}")
print(
    f"GA params: total_iter={params['total_iter']}, "
    f"per_iter={params['per_iter']}, keep={params['keep']}, "
    f"mix={params['mix']}, prob={params['prob']}"
)
print(f"seed: {SEED}, core: {CORE}")
print()

ensure_asp_backend()

# %% [markdown]
# ## Helpers


# %%
@dataclass
class EvalStats:
    n_eval: int = 0
    n_pruned: int = 0
    n_with_threshold: int = 0
    eval_seconds: float = 0.0


@dataclass
class RunResult:
    ldoi_prune: bool
    wall_seconds: float
    n_iterations: int
    n_generated: int
    start_score: float
    start_max: float
    final_score: float
    final_max: float
    stats: EvalStats = field(default_factory=EvalStats)


def instrument_new_candidates() -> tuple[EvalStats, Callable[[], None]]:
    """Patch ``GeneticAlgorithm.new_candidates`` to tally eval stats in-process.

    Avoids wrapping ``evaluator.evaluate``, which breaks joblib pickling when
    ``core > 1``. Stats are collected in the parent after Parallel returns.
    """
    stats = EvalStats()
    original = GeneticAlgorithm.new_candidates

    def wrapped(
        self,
        offsprings: list[Candidate],
        evaluator,
        score_threshold: float | None = None,
    ) -> list[Candidate]:
        if score_threshold is not None:
            stats.n_with_threshold += len(offsprings)
        t0 = perf_counter()
        candidates = original(self, offsprings, evaluator, score_threshold)
        stats.eval_seconds += perf_counter() - t0
        stats.n_eval += len(candidates)
        stats.n_pruned += sum(
            1
            for c in candidates
            if c.eval_result is not None and c.eval_result.is_upper_bound
        )
        return candidates

    GeneticAlgorithm.new_candidates = wrapped  # type: ignore[method-assign]

    def restore() -> None:
        GeneticAlgorithm.new_candidates = original  # type: ignore[method-assign]

    return stats, restore


def run_ga_bench(ldoi_prune: bool) -> RunResult:
    """Run CAD GA once with the given ``ldoi_prune`` flag (no model export)."""
    config = copy.deepcopy(base_config)
    config["nav_method"] = "min_traps"
    config["ldoi_prune"] = ldoi_prune

    print("=" * 72)
    print(f"Run: nav_method=min_traps, ldoi_prune={ldoi_prune}")
    print("=" * 72)

    evaluator = build_evaluator(config, verbose=True)
    ga = build_ga_config(config, seed=SEED, core=CORE)
    stats, restore = instrument_new_candidates()

    try:
        start_path = (project_root() / config["start"]).resolve()
        base_path = (project_root() / config["base"]).resolve()
        if start_path == base_path:
            start_model = evaluator.base
        else:
            print(f"Loading starting model from {start_path}")
            start_model = load_model(start_path, base=evaluator.base)
            print("Starting model loaded.")

        # Start evaluation has no threshold → never pruned (even if ldoi_prune).
        t0 = perf_counter()
        start_result = evaluator.evaluate(start_model, model_id=0)
        stats.eval_seconds += perf_counter() - t0
        stats.n_eval += 1
        start = Candidate(start_model, 0, ga.starting_gen, start_result)
        print(start.model.info())
        print(format_score(start.eval_result))
        print()

        selector = Selector(keep=ga.keep, order_by=ga.order_by)
        reproducer = Reproducer(selector=selector, constraints=evaluator.constraints)

        t0 = perf_counter()
        states = ga_main(start, evaluator, selector, reproducer, ga)
        wall = perf_counter() - t0

        final = states[-1].population[0]
        assert start.eval_result is not None
        assert final.eval_result is not None

        result = RunResult(
            ldoi_prune=ldoi_prune,
            wall_seconds=wall,
            n_iterations=len(states) - 1,
            n_generated=states[-1].generated,
            start_score=start.eval_result.score,
            start_max=start.eval_result.max_score,
            final_score=final.eval_result.score,
            final_max=final.eval_result.max_score,
            stats=stats,
        )

        prune_pct = (
            100.0 * stats.n_pruned / stats.n_eval if stats.n_eval else 0.0
        )
        print()
        print(f"  GA wall time: {wall:.4f}s")
        print(
            f"  evals={stats.n_eval}, pruned={stats.n_pruned} "
            f"({prune_pct:.1f}%), with_threshold={stats.n_with_threshold}"
        )
        print(f"  eval time sum: {stats.eval_seconds:.4f}s")
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
    finally:
        restore()

# %% [markdown]
# ## Run GA: `ldoi_prune=True` vs `False`
#
# Same CAD config and seed. Prune only affects offspring evaluation after
# survivors exist (`score_threshold` set by the GA).

# %%
results: dict[bool, RunResult] = {}
for prune_flag in (True, False):
    results[prune_flag] = run_ga_bench(prune_flag)

# %% [markdown]
# ## Summary

# %%
on = results[True]
off = results[False]


def _pct(score: float, max_score: float) -> float:
    return 100.0 * score / max_score if max_score else 0.0


print("Summary: min_traps NAV GA, LDOI prune on vs off (CAD demo)")
print("-" * 72)
print(f"{'metric':<28} {'prune=ON':>14} {'prune=OFF':>14}")
print("-" * 72)
print(
    f"{'GA wall time (s)':<28} "
    f"{on.wall_seconds:14.4f} {off.wall_seconds:14.4f}"
)
print(
    f"{'eval time sum (s)':<28} "
    f"{on.stats.eval_seconds:14.4f} {off.stats.eval_seconds:14.4f}"
)
print(
    f"{'#evaluate calls':<28} "
    f"{on.stats.n_eval:14d} {off.stats.n_eval:14d}"
)
print(
    f"{'#pruned (is_upper_bound)':<28} "
    f"{on.stats.n_pruned:14d} {off.stats.n_pruned:14d}"
)
print(
    f"{'#calls with threshold':<28} "
    f"{on.stats.n_with_threshold:14d} {off.stats.n_with_threshold:14d}"
)
on_pct = (
    100.0 * on.stats.n_pruned / on.stats.n_eval if on.stats.n_eval else 0.0
)
off_pct = (
    100.0 * off.stats.n_pruned / off.stats.n_eval if off.stats.n_eval else 0.0
)
print(f"{'prune rate (%)':<28} {on_pct:14.1f} {off_pct:14.1f}")
print(
    f"{'iterations':<28} "
    f"{on.n_iterations:14d} {off.n_iterations:14d}"
)
print(
    f"{'models generated':<28} "
    f"{on.n_generated:14d} {off.n_generated:14d}"
)
print(
    f"{'start score':<28} "
    f"{on.start_score:8.2f}/{on.start_max:<5.0f} "
    f"{off.start_score:8.2f}/{off.start_max:<5.0f}"
)
print(
    f"{'final score':<28} "
    f"{on.final_score:8.2f}/{on.final_max:<5.0f} "
    f"{off.final_score:8.2f}/{off.final_max:<5.0f}"
)
print(
    f"{'final %':<28} "
    f"{_pct(on.final_score, on.final_max):13.1f}% "
    f"{_pct(off.final_score, off.final_max):13.1f}%"
)
print("-" * 72)

if on.wall_seconds > 0 and off.wall_seconds > 0:
    if on.wall_seconds < off.wall_seconds:
        print(
            f"prune=ON faster by {off.wall_seconds / on.wall_seconds:.2f}x "
            f"(saved {off.wall_seconds - on.wall_seconds:.4f}s)"
        )
    elif off.wall_seconds < on.wall_seconds:
        print(
            f"prune=OFF faster by {on.wall_seconds / off.wall_seconds:.2f}x "
            f"(delta {on.wall_seconds - off.wall_seconds:.4f}s)"
        )
    else:
        print("wall times equal")

print()
print(
    "Note: prune=OFF should report 0 pruned; prune=ON may still prune 0 "
    "models if no offspring UB falls below the survivor threshold."
)
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# restore stdout and close log
sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
