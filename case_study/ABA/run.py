"""Run the ABA case-study GA with dated, self-contained output directories.

Legacy outputs under ``generated_models/`` and ``analysis_result/`` are left
alone. New runs write under ``runs/<timestamp>/``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from boolmore.genetic.runner import run_ga
from boolmore.io.paths import project_root

HERE = Path(__file__).resolve().parent
CONFIG = project_root() / "case_study" / "ABA" / "data" / "ABA_B.json"
RUNS_ROOT = HERE / "runs"

# Top-level config keys applied on top of the JSON for this script.
CONFIG_OVERRIDES = {
    "nav_method": "ldoi",
}

# Applied on top of the JSON ``parameters`` block for this script.
PARAMETER_OVERRIDES = {
    "total_iter": 10,
    "per_iter": 10,
    "keep": 2,
    "mix": 2,
}

SEEDS = range(3, 30)
CORE = 6


def main() -> None:
    batch_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = RUNS_ROOT / batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)

    with open(CONFIG) as fp:
        config = json.load(fp)
    config.update(CONFIG_OVERRIDES)
    config["parameters"].update(PARAMETER_OVERRIDES)

    config_path = batch_dir / "config.json"
    with open(config_path, "w") as fp:
        json.dump(config, fp, indent=2)
        fp.write("\n")

    batch_log = batch_dir / "batch.log"
    with open(batch_log, "w") as log:
        log.write(f"batch_dir={batch_dir}\n")
        log.write(f"source_config={CONFIG}\n")
        log.write(f"config_overrides={CONFIG_OVERRIDES}\n")
        log.write(f"parameter_overrides={PARAMETER_OVERRIDES}\n")
        log.write(f"seeds={list(SEEDS)}\n")
        log.write(f"core={CORE}\n\n")
        log.flush()

        for seed in SEEDS:
            seed_dir = batch_dir / f"seed_{seed:03d}"
            seed_dir.mkdir()
            run_prefix = seed_dir / f"seed_{seed:03d}"

            started = datetime.datetime.now()
            log.write(f"[{started.isoformat()}] start seed={seed} dir={seed_dir}\n")
            log.flush()
            print(f"\n=== seed={seed} → {seed_dir} ===\n")

            run_ga(
                json_file=str(config_path),
                run_name=str(run_prefix),
                core=CORE,
                seed=seed,
            )

            finished = datetime.datetime.now()
            log.write(
                f"[{finished.isoformat()}] done  seed={seed} "
                f"elapsed={finished - started}\n"
            )
            log.flush()

    print(f"\nAll runs finished. Outputs in {batch_dir}")


if __name__ == "__main__":
    main()
