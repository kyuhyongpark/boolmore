# %% [markdown]
# Probe: `A_avg` + comparative NAV scoring
#
# Builds a tiny cascade model and one NAV `data.csv` with absolute `Some`
# plus reciprocal higher/lower and same-as pairs (including co-located
# absolute + comparative rows on the same fixes). Evaluates like
# `scripts/evaluate_model.py` and checks expected agreements:
#
# | id | fixes | outcome | expect A_avg / Δ | agreement |
# |----|-------|---------|------------------|-----------|
# | 1 | none | Some | 0.5 | 1.0 |
# | 2 | Inhib1=1 | lower than 3 | Δ = 1/6 | 1/6 |
# | 3 | Inhib2=1 | higher than 2 | Δ = 1/6 | 1/6 |
# | 4 | none | same as 5 | \|Δ\| = 1/6 | 5/6 |
# | 5 | Inhib2=1 | same as 4 | \|Δ\| = 1/6 | 5/6 |

# %%
from __future__ import annotations

import json
from pathlib import Path

from boolmore.evaluation.score import format_score
from boolmore.inference.asp_backend import ensure_asp_backend
from boolmore.io.export import export_nav_report
from boolmore.io.paths import project_root
from boolmore.run_setup import build_evaluator

# %% [markdown]
# ## Write probe artifacts

# %%
REPO = project_root()
PROBE_DIR = Path(__file__).resolve().parent / "scoring_probe"
PROBE_DIR.mkdir(parents=True, exist_ok=True)

BNET = PROBE_DIR / "cascade.bnet"
DATA = PROBE_DIR / "data.csv"
CONFIG = PROBE_DIR / "config.json"
REPORT = PROBE_DIR / "cascade_score.csv"

BNET.write_text(
    "\n".join(
        [
            "A, A",
            "A_2, A & !Inhib1",
            "A_3, A_2 & !Inhib2",
            "Inhib1, 0",
            "Inhib2, 0",
            "",
        ]
    ),
    encoding="utf-8",
)

DATA.write_text(
    "\n".join(
        [
            "ID,Score,Source,Perturbation,Observed node,Categorization",
            "1,1.0,,,A_avg,Some",
            "2,1.0,,Inhib1=1,A_avg,lower than 3",
            "3,1.0,,Inhib2=1,A_avg,higher than 2",
            "4,1.0,,,A_avg,same as 5",
            "5,1.0,,Inhib2=1,A_avg,same as 4",
            "",
        ]
    ),
    encoding="utf-8",
)

rel = lambda p: str(p.relative_to(REPO)).replace("\\", "/")
config = {
    "run_type": "NAV",
    "data": rel(DATA),
    "base": rel(BNET),
    "start": rel(BNET),
    "hierarchy": False,
    "nav_method": "min_traps",
    "default_sources": {},
    "generate_defaults": False,
    "constraints": {
        "fixed": [],
        "regulate": {},
        "necessary": {},
        "group": {},
        "possible_constant": [],
    },
    "edge_pool": [],
    "parameters": {
        "stop_if_max": True,
        "starting_gen": 0,
        "total_iter": 1,
        "per_iter": 1,
        "keep": 1,
        "mix": 0,
        "prob": 0.0,
        "edge_prob": 0.0,
        "order_by": ["n_extra_edges", "n_prime_implicants"],
    },
}
CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

print(f"wrote {BNET}")
print(f"wrote {DATA}")
print(f"wrote {CONFIG}")

# %% [markdown]
# ## Evaluate (same path as `evaluate_model.py`)

# %%
ensure_asp_backend()

evaluator = build_evaluator(CONFIG, verbose=True)
result = evaluator.evaluate(evaluator.base)
print(evaluator.base.info())
print(format_score(result))

report_path = export_nav_report(result, REPORT)
print(f"Wrote detailed report to {report_path}")

# %% [markdown]
# ## Assert expected scores

# %%
EXPECTED = {
    1: {"predict": 0.5, "agreement": 1.0},
    2: {"predict": 1 / 6, "agreement": 1 / 6},
    3: {"predict": 1 / 6, "agreement": 1 / 6},
    4: {"predict": 1 / 6, "agreement": 5 / 6},
    5: {"predict": 1 / 6, "agreement": 5 / 6},
}

by_id = {item.id: item for item in result.item_scores}
assert set(by_id) == set(EXPECTED), f"unexpected ids: {sorted(by_id)}"

ok = True
for exp_id, expect in EXPECTED.items():
    item = by_id[exp_id]
    predict_ok = abs(item.predict_value - expect["predict"]) < 1e-9
    agree_ok = abs(item.agreement - expect["agreement"]) < 1e-9
    status = "PASS" if predict_ok and agree_ok else "FAIL"
    if status == "FAIL":
        ok = False
    print(
        f"id={exp_id} {status}: "
        f"predict={item.predict_value!r} (expect {expect['predict']!r}), "
        f"agreement={item.agreement!r} (expect {expect['agreement']!r}), "
        f"outcome={item.outcome!r}, fixes={item.fixes!r}"
    )

if not ok:
    raise AssertionError("scoring probe expectations failed")

print("\nAll probe expectations passed.")
