# %% [markdown]
# Regulate-repair heuristic A/B on breast-cancer mutation
#
# Compares ``mutate_model`` with:
#
# 1. **baseline**: ``REGULATE_REPAIR=False`` (full remutation on regulate fail)
# 2. **repair**: ``REGULATE_REPAIR=True`` (one ``try_add_regulator`` attempt)
#
# Reports ``get_uni_rr`` call counts, mutation trial/retry counts, repair
# attempt / success / fail breakdown, and wall time. Mutation-only stress
# (no LDOI scoring) so call reduction is attributable to the heuristic.
#
# Prints to stdout and writes a timestamped log under `dev/logs/`.

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
from boolmore.genetic.operators import mutation as mut
from boolmore.genetic.operators.mutation import (
    get_repair_stats,
    mutate_model,
    regulate_repair_enabled,
    reset_repair_stats,
)
from boolmore.io.load import load_json_config, load_model
from boolmore.io.paths import project_root

# %% [markdown]
# ## Config

# %%
REPO = Path(__file__).resolve().parents[1]
CONFIG_REL = "case_study/breast_cancer/config.json"
LOG_DIR = Path(__file__).resolve().parent / "logs"

SEED = 0
N_MUTATIONS = 300  # mutate_model calls per arm
# Config ``prob`` is often a schedule dict; use a fixed stress rate.
PROB = 0.5
EDGE_PROB = 0.0

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
    f"bench_regulate_repair_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print()

# %% [markdown]
# ## Load breast-cancer model + constraints

# %%
base_config = load_json_config((project_root() / CONFIG_REL).resolve())
base_config = copy.deepcopy(base_config)
params = base_config["parameters"]
constraints = base_config["constraints"]
edge_pool = base_config["edge_pool"]

base_path = (project_root() / base_config["base"]).resolve()
start_path = (project_root() / base_config["start"]).resolve()

print(f"config: {CONFIG_REL}")
print(f"base:   {base_path}")
print(f"start:  {start_path}")
print(
    f"config prob={params['prob']}, edge_prob={params['edge_prob']} "
    f"(bench uses PROB={PROB}, EDGE_PROB={EDGE_PROB})"
)
print(f"N_MUTATIONS={N_MUTATIONS}, seed={SEED}")
print()

print(f"Loading base model from {base_path}")
base_model = load_model(base_path, edge_pool=edge_pool)
print("Base model loaded.")

if start_path == base_path:
    start_model = base_model
else:
    print(f"Loading starting model from {start_path}")
    start_model = load_model(start_path, base=base_model)
    print("Starting model loaded.")
print()

# %% [markdown]
# ## Helpers


# %%
@dataclass
class ArmResult:
    name: str
    wall_seconds: float
    get_uni_rr_calls: int
    trials: int
    retries: int
    repair_attempts: int
    repair_noop: int
    repair_forward: int
    repair_irr_fallback: int
    repair_irr_fallback_fixed: int
    repair_regulate_fixed: int
    repair_regulate_still_false: int
    repair_undone_by_necessary: int
    repair_accepted: int
    repair_regulate_fixed_but_other_fail: int


@contextmanager
def count_get_uni_rr() -> Iterator[list[int]]:
    """Temporarily wrap ``bf.get_uni_rr`` and yield a one-element call counter."""
    counter = [0]
    original = bf.get_uni_rr

    def wrapped(rr: str, max: bool = True) -> str:
        counter[0] += 1
        return original(rr, max=max)

    bf.get_uni_rr = wrapped  # type: ignore[assignment]
    try:
        yield counter
    finally:
        bf.get_uni_rr = original  # type: ignore[assignment]


def run_arm(*, repair: bool) -> ArmResult:
    name = "repair" if repair else "baseline"
    print("=" * 72)
    print(f"Run: {name} (REGULATE_REPAIR={repair})")
    print("=" * 72)

    reset_repair_stats()
    with regulate_repair_enabled(repair), count_get_uni_rr() as counter:
        assert mut.REGULATE_REPAIR is repair
        model = start_model
        t0 = perf_counter()
        for i in range(N_MUTATIONS):
            model = mutate_model(
                model,
                probability=PROB,
                edge_prob=EDGE_PROB,
                constraints=constraints,
                bias=0.5,
                seed=SEED + i,
            )
        wall = perf_counter() - t0

    stats = get_repair_stats()
    result = ArmResult(
        name=name,
        wall_seconds=wall,
        get_uni_rr_calls=counter[0],
        trials=stats["trials"],
        retries=stats["retries"],
        repair_attempts=stats["repair_attempts"],
        repair_noop=stats["repair_noop"],
        repair_forward=stats["repair_forward"],
        repair_irr_fallback=stats["repair_irr_fallback"],
        repair_irr_fallback_fixed=stats["repair_irr_fallback_fixed"],
        repair_regulate_fixed=stats["repair_regulate_fixed"],
        repair_regulate_still_false=stats["repair_regulate_still_false"],
        repair_undone_by_necessary=stats["repair_undone_by_necessary"],
        repair_accepted=stats["repair_accepted"],
        repair_regulate_fixed_but_other_fail=stats[
            "repair_regulate_fixed_but_other_fail"
        ],
    )

    print(f"  wall:              {result.wall_seconds:.4f}s")
    print(f"  get_uni_rr calls:  {result.get_uni_rr_calls}")
    print(f"  trials:            {result.trials}")
    print(f"  retries:           {result.retries}")
    print(f"  repair_attempts:   {result.repair_attempts}")
    print(f"  repair_noop:       {result.repair_noop}")
    print(f"  repair_forward:    {result.repair_forward}")
    print(f"  repair_irr_fallback: {result.repair_irr_fallback}")
    print(
        f"  irr_fallback_fixed_regulate: "
        f"{result.repair_irr_fallback_fixed}"
    )
    print(
        f"  regulate_fixed (after flip):    {result.repair_regulate_fixed}"
    )
    print(
        f"  regulate_still_false (after flip): "
        f"{result.repair_regulate_still_false}"
    )
    print(
        f"  undone_by_necessary: {result.repair_undone_by_necessary}"
    )
    print(f"  accepted_after_repair: {result.repair_accepted}")
    print(
        f"  regulate_fixed_but_other_fail: "
        f"{result.repair_regulate_fixed_but_other_fail}"
    )
    print()
    return result


# %% [markdown]
# ## Run baseline vs repair

# %%
results: dict[str, ArmResult] = {}
results["baseline"] = run_arm(repair=False)
results["repair"] = run_arm(repair=True)

# %% [markdown]
# ## Summary

# %%
base = results["baseline"]
rep = results["repair"]

print("Summary: breast-cancer mutate_model, regulate repair A/B")
print("-" * 72)
print(f"{'metric':<36} {'baseline':>14} {'repair':>14}")
print("-" * 72)
print(
    f"{'wall time (s)':<36} "
    f"{base.wall_seconds:14.4f} {rep.wall_seconds:14.4f}"
)
print(
    f"{'get_uni_rr calls':<36} "
    f"{base.get_uni_rr_calls:14d} {rep.get_uni_rr_calls:14d}"
)
print(f"{'trials':<36} {base.trials:14d} {rep.trials:14d}")
print(f"{'retries':<36} {base.retries:14d} {rep.retries:14d}")
print(
    f"{'repair_attempts':<36} "
    f"{base.repair_attempts:14d} {rep.repair_attempts:14d}"
)
print(
    f"{'repair_noop':<36} "
    f"{base.repair_noop:14d} {rep.repair_noop:14d}"
)
print(
    f"{'repair_forward':<36} "
    f"{base.repair_forward:14d} {rep.repair_forward:14d}"
)
print(
    f"{'repair_irr_fallback':<36} "
    f"{base.repair_irr_fallback:14d} {rep.repair_irr_fallback:14d}"
)
print(
    f"{'irr_fallback_fixed_regulate':<36} "
    f"{base.repair_irr_fallback_fixed:14d} "
    f"{rep.repair_irr_fallback_fixed:14d}"
)
print(
    f"{'regulate_fixed (after flip)':<36} "
    f"{base.repair_regulate_fixed:14d} {rep.repair_regulate_fixed:14d}"
)
print(
    f"{'regulate_still_false (after flip)':<36} "
    f"{base.repair_regulate_still_false:14d} "
    f"{rep.repair_regulate_still_false:14d}"
)
print(
    f"{'undone_by_necessary':<36} "
    f"{base.repair_undone_by_necessary:14d} "
    f"{rep.repair_undone_by_necessary:14d}"
)
print(
    f"{'accepted_after_repair':<36} "
    f"{base.repair_accepted:14d} {rep.repair_accepted:14d}"
)
print(
    f"{'regulate_fixed_but_other_fail':<36} "
    f"{base.repair_regulate_fixed_but_other_fail:14d} "
    f"{rep.repair_regulate_fixed_but_other_fail:14d}"
)
print("-" * 72)

if base.get_uni_rr_calls > 0:
    delta_calls = base.get_uni_rr_calls - rep.get_uni_rr_calls
    pct_calls = 100.0 * abs(delta_calls) / base.get_uni_rr_calls
    if delta_calls > 0:
        print(
            f"get_uni_rr calls: repair fewer by {delta_calls} "
            f"({pct_calls:.2f}% below baseline)"
        )
    elif delta_calls < 0:
        print(
            f"get_uni_rr calls: repair more by {-delta_calls} "
            f"({pct_calls:.2f}% above baseline)"
        )
    else:
        print("get_uni_rr calls: equal")

if base.wall_seconds > 0 and rep.wall_seconds > 0:
    if rep.wall_seconds < base.wall_seconds:
        print(
            f"wall: repair faster by "
            f"{base.wall_seconds / rep.wall_seconds:.2f}x "
            f"(saved {base.wall_seconds - rep.wall_seconds:.4f}s)"
        )
    elif base.wall_seconds < rep.wall_seconds:
        print(
            f"wall: baseline faster by "
            f"{rep.wall_seconds / base.wall_seconds:.2f}x "
            f"(delta {rep.wall_seconds - base.wall_seconds:.4f}s)"
        )
    else:
        print("wall times equal")

if rep.repair_attempts > 0:
    print()
    print(
        "Counts: regulate_fixed / still_false are checked right after "
        "try_add_regulator; undone_by_necessary is flip-fixed then lost "
        "after impose_necessary."
    )
    print(
        f"forward path / attempts: "
        f"{rep.repair_forward}/{rep.repair_attempts}"
    )
    print(
        f"irr fallback (forward had no target) / attempts: "
        f"{rep.repair_irr_fallback}/{rep.repair_attempts}"
    )
    print(
        f"irr fallback fixed regulate: "
        f"{rep.repair_irr_fallback_fixed} "
        f"(of {rep.repair_irr_fallback} irr fallbacks)"
    )
    if rep.repair_irr_fallback_fixed > 0:
        print(
            "YES: there are cases where forward finds no target and "
            "irr still repairs regulate."
        )
    else:
        print(
            "NO cases where irr fallback fixed regulate "
            "(irr never helped beyond forward, or never reached)."
        )
    print(
        f"flip hit rate (regulate_fixed / attempts): "
        f"{rep.repair_regulate_fixed}/{rep.repair_attempts} = "
        f"{100.0 * rep.repair_regulate_fixed / rep.repair_attempts:.1f}%"
    )
    if rep.repair_regulate_fixed > 0:
        print(
            f"undone by necessary / flip-fixed: "
            f"{rep.repair_undone_by_necessary}/{rep.repair_regulate_fixed} = "
            f"{100.0 * rep.repair_undone_by_necessary / rep.repair_regulate_fixed:.1f}%"
        )
    print(
        f"repair accept rate (accepted / attempts): "
        f"{rep.repair_accepted}/{rep.repair_attempts} = "
        f"{100.0 * rep.repair_accepted / rep.repair_attempts:.1f}%"
    )

print()
print(
    "Note: mutate-only stress (no scoring). Scores need not match between "
    "arms; RNG trajectories diverge once repair consumes random draws."
)
print(f"Log written to: {LOG_PATH}")
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

sys.stdout = _stdout
_log_file.close()
print(f"log saved: {LOG_PATH}")
