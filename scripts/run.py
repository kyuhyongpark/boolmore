from boolmore.runner import run_ga
from boolmore.io.paths import project_root

# --- edit these to run a different model ---
CONFIG = "demo/CAD_config.json"
# CONFIG = "case_study/breast_cancer/BreastCancer_config.json"
RUN_NAME = "demo"
# RUN_NAME = None    # None → auto {start_stem}_{timestamp}
SEED = 0
CORE = 4

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

config_path = project_root() / CONFIG

base, start, states = run_ga(
    json_file=str(config_path),
    run_name=RUN_NAME,
    seed=SEED,
    core=CORE,
    id_col=ID_COL,
    weight_col=WEIGHT_COL,
    observed_col=OBSERVED_COL,
    outcome_col=OUTCOME_COL,
    fixes_cols=FIXES_COLS,
)
