"""Run ABA stomatal-closure refinement.

Edit the knobs below, then::

    python boolmore/case_study/ABA/run.py
"""

from __future__ import annotations

from boolmore.runner import run_batch

# --- session knobs ---
CONFIG = "case_study/ABA/data/ABA_B.json"
# CONFIG = "case_study/ABA/data/ABA_A.json"
RUN_NAME = "aba_B"  # None → UTC timestamp folder under RUNS_ROOT
RUNS_ROOT = "case_study/ABA/runs"
SEEDS = [11]
CORE = 6

CONFIG_OVERRIDES: dict = {}
# CONFIG_OVERRIDES = {"nav_method": "ldoi"}
PARAMETER_OVERRIDES: dict = {}
# PARAMETER_OVERRIDES = {
#     "total_iter": 10,
#     "per_iter": 10,
#     "keep": 2,
#     "mix": 2,
# }
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
