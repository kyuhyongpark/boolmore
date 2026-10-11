from pathlib import Path

from boolmore.evaluation.score import format_score, write_nav_report
from boolmore.io.load import load_model
from boolmore.run_setup import build_evaluator

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

# change these to evaluate a different model
CONFIG = DEMO_DIR / "CAD_config.json"
MODEL = DEMO_DIR / "CAD_start.bnet"
REPORT = True  # set True to write a per-experiment NAV score TSV

evaluator = build_evaluator(CONFIG, verbose=True)

base_result = evaluator.evaluate(evaluator.base)
print(evaluator.base.info())
print(f"base {format_score(base_result)}\n")

print(f"Loading model from {MODEL}")
model = load_model(MODEL, base=evaluator.base)
print("Model loaded.\n")
result = evaluator.evaluate(model)
print(model.info())
print(format_score(result))

if REPORT:
    report_path = write_nav_report(
        result, evaluator, Path.cwd() / f"{Path(MODEL).stem}_score.tsv"
    )
    print(f"Wrote detailed report to {report_path}")
