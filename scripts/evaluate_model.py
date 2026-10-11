#!/usr/bin/env python3
"""Evaluate a Boolean model with the same fitness pipeline as the genetic algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

from boolmore.evaluation.score import format_score
from boolmore.genetic.setup import load_run_setup
from boolmore.io.paths import find_project_root, resolve_existing_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Boolean model with the same scoring pipeline "
            "used by the genetic algorithm."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo",
        action="store_true",
        help="Use the CAD demo config under <project_root>/demo/ (default).",
    )
    source.add_argument(
        "--config",
        type=Path,
        help="Path to a GA config JSON file.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "Model .bnet to evaluate. Defaults to the config start model. "
            "With --demo, bare filenames are also looked up under demo/."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a per-experiment NAV score TSV in the current directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = find_project_root()
    demo = project_root / "demo"

    use_demo = args.config is None
    if args.config is not None:
        config_path = args.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
    else:
        config_path = demo / "CAD_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Demo config not found: {config_path}")

    print(f"Project root: {project_root}")
    print(f"Config: {config_path}")

    setup = load_run_setup(config_path, verbose=True)

    if args.model is None:
        model_path = setup.start_path
    else:
        search = [setup.config_dir]
        if use_demo:
            search.insert(0, demo)
        model_path = resolve_existing_path(args.model, *search)

    print(f"Model: {model_path}\n")

    _, base_result = setup.evaluate(setup.base_model)
    print(setup.base_model.info())
    print(f"base {format_score(base_result)}")
    if not base_result.following_constraints:
        print("WARNING: base model does not follow constraints")
    print()

    print(f"Loading model from {model_path}")
    model, result = setup.evaluate(model_path)
    print("Model loaded.\n")
    print(model.info())
    print(format_score(result))
    if not result.following_constraints:
        print("WARNING: model does not follow constraints")

    if args.report:
        try:
            report_path = setup.write_nav_report(
                result, Path.cwd() / f"{model_path.stem}_score.tsv"
            )
        except ValueError as exc:
            print(f"Skipping detailed report: {exc}")
        else:
            print(f"Wrote detailed report to {report_path}")


if __name__ == "__main__":
    main()
