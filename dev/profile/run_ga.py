#!/usr/bin/env python3
"""Profile breast-cancer GA wall time for the current commit.

Writes under ``dev/profile/runs/<stamp>_<sha>_ga_breast_cancer/``:

* ``manifest.json`` — task knobs, git/env fingerprint, wall times, scores
* ``console.log`` — full stdout tee

Re-run after each commit of interest, then compare with ``compare_runs.py``.

Does **not** call ``run_batch`` / ``run_ga`` (no case_study run folders).

Examples
--------
::

    python boolmore/dev/profile/run_ga.py
    python boolmore/dev/profile/run_ga.py --core 4 --total-iter 5
    python boolmore/dev/profile/run_ga.py --note "after fingerprint fix"
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

# Allow ``python boolmore/dev/profile/run_ga.py`` without installing the helper
# as a package; boolmore itself must still be importable (editable install).
_PROFILE_DIR = Path(__file__).resolve().parent
if str(_PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROFILE_DIR))

from common import (  # noqa: E402
    SCHEMA_VERSION,
    TASK_VERSION,
    Tee,
    collect_environment,
    collect_git_info,
    make_run_dir,
    sha256_file,
    task_fingerprint,
    utc_now,
    write_manifest,
)
from boolmore.evaluation.score import format_score  # noqa: E402
from boolmore.genetic.algorithm import ga_main  # noqa: E402
from boolmore.genetic.operators.reproduction import Reproducer  # noqa: E402
from boolmore.genetic.population import Candidate  # noqa: E402
from boolmore.genetic.selection import Selector  # noqa: E402
from boolmore.io.load import load_json_config, load_model  # noqa: E402
from boolmore.io.paths import project_root  # noqa: E402
from boolmore.run_setup import build_evaluator, build_ga_config  # noqa: E402

TASK_NAME = "ga_breast_cancer"
DEFAULT_CONFIG_REL = "case_study/breast_cancer/config.json"
DEFAULT_PARAM_OVERRIDES = {
    "total_iter": 3,
    "per_iter": 100,
    "keep": 20,
    "mix": 20,
    "stop_if_max": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a fixed breast-cancer GA profile for the current git commit "
            "and write a comparable manifest under dev/profile/runs/."
        )
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_REL,
        help=f"Config path relative to project root (default: {DEFAULT_CONFIG_REL})",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--core", type=int, default=1)
    p.add_argument("--total-iter", type=int, default=DEFAULT_PARAM_OVERRIDES["total_iter"])
    p.add_argument("--per-iter", type=int, default=DEFAULT_PARAM_OVERRIDES["per_iter"])
    p.add_argument("--keep", type=int, default=DEFAULT_PARAM_OVERRIDES["keep"])
    p.add_argument("--mix", type=int, default=DEFAULT_PARAM_OVERRIDES["mix"])
    p.add_argument(
        "--stop-if-max",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Honor stop_if_max (default: off for stable iteration counts).",
    )
    p.add_argument(
        "--nav-method",
        default=None,
        help="Override config nav_method (default: keep config value).",
    )
    p.add_argument(
        "--note",
        default="",
        help="Free-text note stored in the manifest (not part of task fingerprint).",
    )
    p.add_argument(
        "--quiet-build",
        action="store_true",
        help="Pass verbose=False to build_evaluator.",
    )
    return p.parse_args(argv)


def build_task_spec(
    *,
    args: argparse.Namespace,
    config: dict,
    config_path: Path,
) -> dict:
    params = dict(config["parameters"])
    return {
        "name": TASK_NAME,
        "task_version": TASK_VERSION,
        "config_rel": args.config,
        "config_sha256": sha256_file(config_path),
        "run_type": config.get("run_type"),
        "nav_method": config.get("nav_method"),
        "hierarchy": config.get("hierarchy"),
        "ldoi_prune": config.get("ldoi_prune", True),
        "data_rel": config.get("data"),
        "base_rel": config.get("base"),
        "start_rel": config.get("start"),
        "seed": args.seed,
        "core": args.core,
        "parameters": {
            "total_iter": params["total_iter"],
            "per_iter": params["per_iter"],
            "keep": params["keep"],
            "mix": params["mix"],
            "prob": params.get("prob"),
            "edge_prob": params.get("edge_prob"),
            "order_by": params.get("order_by"),
            "stop_if_max": params.get("stop_if_max"),
            "starting_gen": params.get("starting_gen"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root()
    config_path = (root / args.config).resolve()
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2

    git = collect_git_info(root)
    run_dir = make_run_dir(task_name=TASK_NAME, commit_short=git.get("commit_short"))
    log_path = run_dir / "console.log"
    log_file = log_path.open("w", encoding="utf-8")
    stdout = sys.stdout
    sys.stdout = Tee(stdout, log_file)

    started = utc_now()
    print(f"run_dir: {run_dir}")
    print(f"started (UTC): {started.isoformat()}")
    print(f"git: {git.get('describe')} dirty={git.get('dirty')}")
    print()

    try:
        config = copy.deepcopy(load_json_config(config_path))
        if args.nav_method is not None:
            config["nav_method"] = args.nav_method
        config["parameters"].update(
            {
                "total_iter": args.total_iter,
                "per_iter": args.per_iter,
                "keep": args.keep,
                "mix": args.mix,
                "stop_if_max": args.stop_if_max,
            }
        )

        env = collect_environment()
        task = build_task_spec(args=args, config=config, config_path=config_path)
        # Algorithmic toggles belong in the task fingerprint so a Cython-off
        # run is not treated as identical work to a Cython-on run.
        task["ldoi_closure_engine"] = env["ldoi_closure_engine"]
        task["ldoi_cython"] = env["ldoi_cython"]
        task["regulate_repair_mode"] = env["regulate_repair_mode"]
        task["uni_rr_cython"] = env["uni_rr_cython"]
        task["uni_rr_cython_min_k"] = env["uni_rr_cython_min_k"]
        fingerprint = task_fingerprint(task)

        print("=== task ===")
        for key in (
            "name",
            "task_version",
            "config_rel",
            "config_sha256",
            "nav_method",
            "hierarchy",
            "seed",
            "core",
            "ldoi_closure_engine",
            "ldoi_cython",
            "regulate_repair_mode",
            "uni_rr_cython",
        ):
            print(f"  {key}: {task[key]}")
        print(f"  parameters: {task['parameters']}")
        print(f"  task_fingerprint: {fingerprint}")
        print()
        print("=== environment ===")
        for key, value in env.items():
            print(f"  {key}: {value}")
        print()

        evaluator = build_evaluator(config, verbose=not args.quiet_build)
        ga = build_ga_config(config, seed=args.seed, core=args.core)

        start_path = (root / config["start"]).resolve()
        base_path = (root / config["base"]).resolve()
        if start_path == base_path:
            start_model = evaluator.base
        else:
            print(f"Loading starting model from {start_path}")
            start_model = load_model(start_path, base=evaluator.base)
            print("Starting model loaded.")

        start_result = evaluator.evaluate(start_model, model_id=0, detail=False)
        start = Candidate(start_model, 0, ga.starting_gen, start_result)
        print(format_score(start.eval_result))
        print()

        selector = Selector(keep=ga.keep, order_by=ga.order_by)
        reproducer = Reproducer(
            selector=selector, constraints=evaluator.constraints
        )

        t0 = perf_counter()
        outcome = ga_main(start, evaluator, selector, reproducer, ga)
        wall = perf_counter() - t0

        final = outcome.final.population[0]
        assert start.eval_result is not None
        assert final.eval_result is not None

        history = [
            {
                "iteration": h.iteration,
                "generated": h.generated,
                "n_evaluated": h.n_evaluated,
                "n_pruned": h.n_pruned,
                "best_id": h.best_id,
                "best_generation": h.best_generation,
                "best_score": h.best_score,
                "best_max_score": h.best_max_score,
            }
            for h in outcome.history
        ]

        print()
        print(f"GA wall time: {wall:.4f}s")
        print(
            f"iterations={len(outcome.history)}, "
            f"generated={outcome.final.generated}"
        )
        print(
            f"start score: {start.eval_result.score:.2f} / "
            f"{start.eval_result.max_score}"
        )
        print(
            f"final score: {final.eval_result.score:.2f} / "
            f"{final.eval_result.max_score}"
        )

        finished = utc_now()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "note": args.note,
            "task": task,
            "task_fingerprint": fingerprint,
            "git": git,
            "environment": env,
            "results": {
                "wall_seconds": wall,
                "n_iterations": len(outcome.history),
                "n_generated": outcome.final.generated,
                "start_score": start.eval_result.score,
                "start_max_score": start.eval_result.max_score,
                "final_score": final.eval_result.score,
                "final_max_score": final.eval_result.max_score,
                "best_id": final.id,
                "best_generation": final.generation,
                "history": history,
            },
        }
        manifest_path = write_manifest(run_dir, manifest)
        print()
        print(f"wrote {manifest_path}")
        print(f"wrote {log_path}")
        return 0
    finally:
        sys.stdout = stdout
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
