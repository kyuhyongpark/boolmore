"""Run breast-cancer model refinement.

Edit the knobs below, then::

    python boolmore/case_study/breast_cancer/run.py

NAV CSV columns come from ``nav_columns`` in the config JSON.
"""

from __future__ import annotations

from boolmore.runner import run_batch

# --- session knobs ---
CONFIG = "case_study/breast_cancer/BreastCancer_config.json"
RUN_NAME = None  # None → UTC timestamp folder under RUNS_ROOT
RUNS_ROOT = "case_study/breast_cancer/runs"
SEEDS = [0]
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
