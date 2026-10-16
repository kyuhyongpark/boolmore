"""Batch GA runner: multi-seed runs under ``{RUNS_ROOT}/{batch_name}/``.

Edit the knobs below, then::

    python boolmore/scripts/run.py

``RUN_NAME`` set → batch folder is that name (recreated / overwritten).
``RUN_NAME`` None → UTC stamp folder (e.g. ``20261016T121200Z``).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from boolmore.runner import run_ga
from boolmore.io.paths import project_root

# --- edit these to run a different model ---
CONFIG = "demo/CAD_config.json"
# CONFIG = "case_study/breast_cancer/BreastCancer_config.json"
# CONFIG = "case_study/ABA/data/ABA_B.json"

RUN_NAME = "demo"
# RUN_NAME = None          # None → UTC timestamp folder under RUNS_ROOT
# RUN_NAME = "aba_B"       # named folder; overwritten on re-run

RUNS_ROOT = "runs"
# RUNS_ROOT = "case_study/ABA/runs"

SEEDS = list(range(3))
# SEEDS = [11]
CORE = 4

# Applied on top of the JSON for this script (empty = no change).
CONFIG_OVERRIDES: dict = {}
# CONFIG_OVERRIDES = {"nav_method": "ldoi"}

# Applied on top of the JSON ``parameters`` block (empty = no change).
PARAMETER_OVERRIDES: dict = {}
# PARAMETER_OVERRIDES = {
#     "total_iter": 10,
#     "per_iter": 10,
#     "keep": 2,
#     "mix": 2,
# }

# NAV CSV column names (same knobs as evaluate_model.py).
# Set any of these to None to use import_NAV_exps defaults.
ID_COL = None
WEIGHT_COL = None
OBSERVED_COL = None
OUTCOME_COL = None
FIXES_COLS = None
# For breast cancer, use e.g.:
# ID_COL = "id"
# WEIGHT_COL = "Weight"
# OBSERVED_COL = "Observed node"
# OUTCOME_COL = "Value"
# FIXES_COLS = (
#     "Signals",
#     "Drug treatment/intervention",
#     "Cell line characteristic profile",
#     "Additional features of the cell population",
# )
# ------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _last_iter_line(seed_log: str) -> str | None:
    """Return the last GA progress line (``iter N/M | …``) from a seed log."""
    last = None
    with open(seed_log, encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("iter "):
                last = line.rstrip("\n")
    return last


def main() -> None:
    source_config = project_root() / CONFIG
    runs_root = project_root() / RUNS_ROOT
    runs_root.mkdir(parents=True, exist_ok=True)

    named = RUN_NAME is not None
    batch_name = RUN_NAME if named else _utc_stamp()
    batch_dir = runs_root / batch_name

    if named:
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True)
    else:
        batch_dir.mkdir(parents=True, exist_ok=False)

    with open(source_config) as fp:
        config = json.load(fp)
    config.update(CONFIG_OVERRIDES)
    if PARAMETER_OVERRIDES:
        config.setdefault("parameters", {}).update(PARAMETER_OVERRIDES)

    config_path = batch_dir / "config.json"
    with open(config_path, "w") as fp:
        json.dump(config, fp, indent=2)
        fp.write("\n")

    batch_log = batch_dir / "batch.log"
    summaries: list[tuple[int, str, str]] = []  # seed, elapsed, last iter line

    with open(batch_log, "w") as log:
        log.write(f"batch_dir={batch_dir}\n")
        log.write(f"source_config={source_config}\n")
        log.write(f"config_overrides={CONFIG_OVERRIDES}\n")
        log.write(f"parameter_overrides={PARAMETER_OVERRIDES}\n")
        log.write(f"seeds={list(SEEDS)}\n")
        log.write(f"core={CORE}\n\n")
        log.flush()

        for seed in SEEDS:
            seed_dir = batch_dir / f"seed_{seed:03d}"
            seed_dir.mkdir()
            run_prefix = seed_dir / f"seed_{seed:03d}"
            seed_log_path = f"{run_prefix}_log.txt"

            started = _utc_now()
            log.write(f"[{started.isoformat()}] start seed={seed} dir={seed_dir}\n")
            log.flush()
            print(f"\n=== seed={seed} → {seed_dir} ===\n")

            run_ga(
                json_file=str(config_path),
                run_name=str(run_prefix),
                seed=seed,
                core=CORE,
                id_col=ID_COL,
                weight_col=WEIGHT_COL,
                observed_col=OBSERVED_COL,
                outcome_col=OUTCOME_COL,
                fixes_cols=FIXES_COLS,
            )

            finished = _utc_now()
            elapsed = finished - started
            iter_line = _last_iter_line(seed_log_path) or "(no iter line found)"
            summaries.append((seed, str(elapsed), iter_line))

            log.write(
                f"[{finished.isoformat()}] done  seed={seed} "
                f"elapsed={elapsed}\n"
            )
            log.write(f"  {iter_line}\n")
            log.flush()

        log.write("\n--- summary ---\n")
        for seed, elapsed, iter_line in summaries:
            log.write(f"seed={seed}  elapsed={elapsed}\n")
            log.write(f"  {iter_line}\n")
        log.flush()

    print(f"\nAll runs finished. Outputs in {batch_dir}")


if __name__ == "__main__":
    main()
