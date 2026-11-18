# %% [markdown]
# Speed: legacy two-phase vs fused reproduce+evaluate
#
# Benchmarks genetic-algorithm wall time on the breast-cancer case study when
# each generation either:
#
# 1. **Legacy**: serial `Reproducer.sexual` / `asexual`, then parallel
#    `new_candidates` (evaluate only).
# 2. **Fused**: select parents on the main process, then Parallel
#    (reproduce → evaluate) with ``batch_size=1``.
#
# Related-node caching from `build_evaluator` is on for both arms.
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.
# Does not call `run_ga` (avoids exporting `.bnet` / run logs into CWD).

# %%
from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from boolmore.evaluation.score import format_score
from boolmore.genetic.algorithm import GAState, GeneticAlgorithm, ga_main
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.population import Candidate, sort_population
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
CORE = 4

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
    f"bench_fused_reproduce_eval_"
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
    mode: str
    wall_seconds: float
    n_iterations: int
    n_generated: int
    start_score: float
    start_max: float
    final_score: float
    final_max: float


def _legacy_next_state(self: GeneticAlgorithm, state: GAState) -> GAState:
    """Pre-rework path: serial reproduce, then parallel evaluate only."""
    iteration = state.iteration + 1
    population = self.selector.select_survivors(state.population)
    generated = state.generated
    pruned_ids: list[int] = []
    n_evaluated = 0

    score_threshold = None
    if (
        self.evaluator.ldoi_prune
        and self.evaluator.nav_method == "min_traps"
        and population
    ):
        survivor_scores = [
            c.eval_result.score
            for c in population
            if c.eval_result is not None
        ]
        if survivor_scores:
            score_threshold = min(survivor_scores)

    if iteration > 1:
        mixed_offsprings = self.reproducer.sexual(population, self.config.mix)
        candidates = self.new_candidates(
            mixed_offsprings, self.evaluator, score_threshold
        )
        population.extend(candidates)
        generated += len(candidates)
        n_evaluated += len(candidates)
        pruned_ids.extend(
            c.id
            for c in candidates
            if c.eval_result is not None and c.eval_result.is_upper_bound
        )

    offsprings = self.reproducer.asexual(
        population=population,
        prob=self.config.prob_list[iteration - 1],
        edge_prob=self.config.edge_prob,
        n=self.config.keep + self.config.per_iter - len(population),
    )
    candidates = self.new_candidates(
        offsprings, self.evaluator, score_threshold
    )
    population.extend(candidates)
    generated += len(candidates)
    n_evaluated += len(candidates)
    pruned_ids.extend(
        c.id
        for c in candidates
        if c.eval_result is not None and c.eval_result.is_upper_bound
    )

    population = sort_population(population, self.config.order_by)
    return GAState(
        iteration=iteration,
        population=population,
        generated=generated,
        score_threshold=score_threshold,
        pruned_ids=pruned_ids,
        n_evaluated=n_evaluated,
    )


def _patch_legacy_next_state() -> Callable[[], None]:
    original = GeneticAlgorithm.next_state
    GeneticAlgorithm.next_state = _legacy_next_state  # type: ignore[method-assign]

    def restore() -> None:
        GeneticAlgorithm.next_state = original  # type: ignore[method-assign]

    return restore


def run_ga_bench(*, fused: bool) -> RunResult:
    """Run breast-cancer GA once in legacy or fused mode (no model export)."""
    mode = "fused" if fused else "legacy"
    config = copy.deepcopy(base_config)

    print("=" * 72)
    print(f"Run: mode={mode}, nav_method={config['nav_method']}, core={CORE}")
    print("=" * 72)

    evaluator = build_evaluator(config, verbose=True)
    ga = build_ga_config(config, seed=SEED, core=CORE)

    restore: Callable[[], None] | None = None
    if not fused:
        restore = _patch_legacy_next_state()

    try:
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
        reproducer = Reproducer(selector=selector, constraints=evaluator.constraints)

        t0 = perf_counter()
        outcome = ga_main(start, evaluator, selector, reproducer, ga)
        wall = perf_counter() - t0

        final = outcome.final.population[0]
        assert start.eval_result is not None
        assert final.eval_result is not None

        result = RunResult(
            mode=mode,
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
    finally:
        if restore is not None:
            restore()

# %% [markdown]
# ## Run GA: legacy vs fused
#
# Same breast-cancer config, seed, and core. Legacy patches `next_state` to
# the old two-phase loop; fused uses the current implementation.

# %%
results: dict[str, RunResult] = {}
results["legacy"] = run_ga_bench(fused=False)
results["fused"] = run_ga_bench(fused=True)

# %% [markdown]
# ## Summary

# %%
legacy = results["legacy"]
fused = results["fused"]


def _pct(score: float, max_score: float) -> float:
    return 100.0 * score / max_score if max_score else 0.0


print("Summary: breast-cancer NAV GA, legacy two-phase vs fused reproduce+evaluate")
print("-" * 72)
print(f"{'metric':<28} {'legacy':>14} {'fused':>14}")
print("-" * 72)
print(
    f"{'GA wall time (s)':<28} "
    f"{legacy.wall_seconds:14.4f} {fused.wall_seconds:14.4f}"
)
print(
    f"{'iterations':<28} "
    f"{legacy.n_iterations:14d} {fused.n_iterations:14d}"
)
print(
    f"{'models generated':<28} "
    f"{legacy.n_generated:14d} {fused.n_generated:14d}"
)
print(
    f"{'start score':<28} "
    f"{legacy.start_score:8.2f}/{legacy.start_max:<5.0f} "
    f"{fused.start_score:8.2f}/{fused.start_max:<5.0f}"
)
print(
    f"{'final score':<28} "
    f"{legacy.final_score:8.2f}/{legacy.final_max:<5.0f} "
    f"{fused.final_score:8.2f}/{fused.final_max:<5.0f}"
)
print(
    f"{'final %':<28} "
    f"{_pct(legacy.final_score, legacy.final_max):13.1f}% "
    f"{_pct(fused.final_score, fused.final_max):13.1f}%"
)
print("-" * 72)

if legacy.wall_seconds > 0 and fused.wall_seconds > 0:
    if fused.wall_seconds < legacy.wall_seconds:
        print(
            f"fused faster by {legacy.wall_seconds / fused.wall_seconds:.2f}x "
            f"(saved {legacy.wall_seconds - fused.wall_seconds:.4f}s)"
        )
    elif legacy.wall_seconds < fused.wall_seconds:
        print(
            f"legacy faster by {fused.wall_seconds / legacy.wall_seconds:.2f}x "
            f"(delta {fused.wall_seconds - legacy.wall_seconds:.4f}s)"
        )
    else:
        print("wall times equal")

print()
print(
    "Note: RNG trajectories differ under core>1 (per-job seeds in fused), "
    "so final scores need not match. Related-node caching is on for both."
)
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# restore stdout and close log
sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
