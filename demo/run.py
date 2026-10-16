"""Run the CAD demo batch.

Edit the knobs below, then::

    python boolmore/demo/run.py
"""

from __future__ import annotations

from boolmore.runner import run_batch

# --- session knobs ---
CONFIG = "demo/CAD_config.json"
RUN_NAME = "test"  # None → UTC timestamp folder under RUNS_ROOT
RUNS_ROOT = "demo/runs"
SEEDS = list(range(3))
CORE = 4

CONFIG_OVERRIDES: dict = {}
PARAMETER_OVERRIDES: dict = {}
# ---------------------


if __name__ == "__main__":
    run_batch(
        CONFIG,
        seeds=SEEDS,
        core=CORE,
        run_name=RUN_NAME,
        runs_root=RUNS_ROOT,
        config_overrides=CONFIG_OVERRIDES or None,
        parameter_overrides=PARAMETER_OVERRIDES or None,
    )
