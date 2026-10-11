from pathlib import Path

from boolmore.evaluation.score import format_score
from boolmore.run_setup import load_run_setup

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

# change these to evaluate a different model
CONFIG = DEMO_DIR / "CAD_config.json"
MODEL = DEMO_DIR / "CAD_start.bnet"
REPORT = False  # set True to write a per-experiment NAV score TSV

setup = load_run_setup(CONFIG, verbose=True)

_, base_result = setup.evaluate(setup.base_model)
print(setup.base_model.info())
print(f"base {format_score(base_result)}\n")

print(f"Loading model from {MODEL}")
model, result = setup.evaluate(MODEL)
print("Model loaded.\n")
print(model.info())
print(format_score(result))

if REPORT:
    report_path = setup.write_nav_report(
        result, Path.cwd() / f"{Path(MODEL).stem}_score.tsv"
    )
    print(f"Wrote detailed report to {report_path}")
