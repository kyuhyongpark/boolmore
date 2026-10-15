from pathlib import Path
from time import perf_counter

from boolmore.evaluation.score import format_score
from boolmore.io.export import export_nav_report
from boolmore.io.load import load_model
from boolmore.io.paths import project_root
from boolmore.run_setup import build_evaluator

# --- edit these to evaluate a different model ---
CONFIG = "case_study/breast_cancer/BreastCancer_config.json"
MODEL = "case_study/breast_cancer/BreastCancer_primes_parsimonious.pkl"  # e.g. demo/CAD.bnet
REPORT = True  # set True to write a per-experiment NAV score CSV

# NAV CSV column names (same knobs as check_nav_import.py).
# Set any of these to None to use import_NAV_exps defaults.
ID_COL = "id"
WEIGHT_COL = "Weight"  # typo in the breast-cancer source CSV header
OBSERVED_COL = "Observed node"
OUTCOME_COL = "Value"
FIXES_COLS = (
    "Signals",
    "Drug treatment/intervention",
    "Cell line characteristic profile",
    "Additional features of the cell population",
)
# ------------------------------------------------

root = project_root()
config_path = root / CONFIG
model_path = root / MODEL

evaluator = build_evaluator(
    config_path,
    verbose=True,
    id_col=ID_COL,
    weight_col=WEIGHT_COL,
    observed_col=OBSERVED_COL,
    outcome_col=OUTCOME_COL,
    fixes_cols=FIXES_COLS,
)

base_result = evaluator.evaluate(evaluator.base)
print(evaluator.base.info())
print(f"base {format_score(base_result)}\n")

print(f"Loading model from {model_path}")
model = load_model(model_path, base=evaluator.base)
print("Model loaded.\n")

t0 = perf_counter()
result = evaluator.evaluate(model)
elapsed = perf_counter() - t0

print(model.info())
print(format_score(result))
print(f"Evaluation took {elapsed:.3f} s")

if REPORT:
    report_path = export_nav_report(
        result, Path.cwd() / f"{model_path.stem}_score.csv"
    )
    print(f"Wrote detailed report to {report_path}")
