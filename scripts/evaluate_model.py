#!/usr/bin/env python3
"""Evaluate a Boolean model with the same fitness pipeline as the genetic algorithm."""

from __future__ import annotations

import argparse
import json
import pickle
from functools import partial
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.evaluation.score import (
    Evaluator,
    EvalResult,
    get_NAV_scores,
    get_phenotype_scores,
)
from boolmore.inference.prediction import get_NAV_prediction, get_phenotype_prediction
from boolmore.io.load import import_NAV_exps, import_phenotypes
from boolmore.model import Model


def find_project_root(start: Path | None = None) -> Path:
    """Walk parents until a directory containing pyproject.toml is found."""
    starts = []
    if start is not None:
        starts.append(start.resolve())
    else:
        starts.append(Path(__file__).resolve().parent)
        starts.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for origin in starts:
        current = origin if origin.is_dir() else origin.parent
        while current not in seen:
            seen.add(current)
            if (current / "pyproject.toml").is_file():
                return current
            if current.parent == current:
                break
            current = current.parent

    raise FileNotFoundError(
        "Could not find project root (directory containing pyproject.toml). "
        f"Searched from: {', '.join(str(s) for s in starts)}"
    )


def resolve_under(base_dir: Path, path: str | Path) -> Path:
    """Resolve path; relative paths are taken relative to base_dir."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def resolve_model_path(
    model_arg: str | None,
    *,
    config_dir: Path,
    start_from_config: str,
    demo_dir: Path | None,
    use_demo: bool,
) -> Path:
    """Resolve the model to evaluate; default is the config start model."""
    if model_arg is None:
        return resolve_under(config_dir, start_from_config)

    candidate = Path(model_arg)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"Model not found: {candidate}")
        return candidate.resolve()

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.is_file():
        return cwd_path

    if use_demo and demo_dir is not None:
        demo_path = (demo_dir / candidate).resolve()
        if demo_path.is_file():
            return demo_path

    config_path = (config_dir / candidate).resolve()
    if config_path.is_file():
        return config_path

    raise FileNotFoundError(
        f"Model not found: {model_arg} "
        f"(tried cwd, config dir{', demo dir' if use_demo else ''})"
    )


def load_primes(path: Path) -> dict:
    if path.suffix == ".bnet":
        return bnet_file2primes(str(path))
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported model file format: {path}")


def build_evaluator(config: dict, config_dir: Path) -> tuple[Evaluator, dict]:
    """Build the same Evaluator wiring used by run_ga."""
    run_type = config["run_type"]
    data_path = resolve_under(config_dir, config["data"])
    hierarchy = config["hierarchy"]
    default_sources = dict(config["default_sources"])
    generate_defaults = config["generate_defaults"]
    constraints = config["constraints"]
    edge_pool = config["edge_pool"]

    base_path = resolve_under(config_dir, config["base"])
    print(f"Loading base model from {base_path}")
    base_primes = load_primes(base_path)
    base_model = Model.from_primes(base_primes, edge_pool=edge_pool)
    print("Base model loaded.")

    print(f"Loading experimental data from {data_path}")
    if run_type == "NAV":
        if generate_defaults:
            for node in base_primes:
                if base_primes[node] == [[{node: 0}], [{node: 1}]]:
                    default_sources[node] = 0

        exps = import_NAV_exps(str(data_path))
        prediction_fn = get_NAV_prediction
        score_fn = partial(
            get_NAV_scores,
            default_sources=default_sources,
            hierarchy=hierarchy,
        )
    elif run_type == "Phenotype":
        exps = import_phenotypes(str(data_path))
        prediction_fn = get_phenotype_prediction
        score_fn = get_phenotype_scores
    else:
        raise ValueError(f"Unsupported run type: {run_type}")

    evaluator = Evaluator(
        exps=exps,
        prediction_fn=prediction_fn,
        score_fn=score_fn,
        constraints=constraints,
    )
    print("Experimental data loaded.\n")

    meta = {
        "run_type": run_type,
        "base_model": base_model,
        "base_primes": base_primes,
        "default_sources": default_sources,
        "hierarchy": hierarchy,
    }
    return evaluator, meta


def format_score(result: EvalResult) -> str:
    pct = result.score / result.max_score * 100 if result.max_score else 0.0
    return (
        f"score: {round(result.score, 2)} / {result.max_score} "
        f"({round(pct, 1)}%)"
    )


def maybe_write_report(
    evaluator: Evaluator,
    result: EvalResult,
    meta: dict,
    model_path: Path,
) -> Path | None:
    if meta["run_type"] != "NAV":
        print("Skipping detailed report: only supported for NAV run_type.")
        return None

    report_path = Path.cwd() / f"{model_path.stem}_score.tsv"
    predictions = result.details[0]
    get_NAV_scores(
        evaluator.exps,
        predictions,
        default_sources=meta["default_sources"],
        hierarchy=meta["hierarchy"],
        report=True,
        file=str(report_path),
    )
    print(f"Wrote detailed report to {report_path.resolve()}")
    return report_path


def evaluate_model(
    config_path: Path,
    model_path: Path,
    *,
    report: bool = False,
) -> EvalResult:
    config_dir = config_path.parent
    with open(config_path) as f:
        config = json.load(f)

    evaluator, meta = build_evaluator(config, config_dir)
    base_model: Model = meta["base_model"]

    base_result = evaluator.evaluate(base_model)
    print(base_model.info())
    print(f"base {format_score(base_result)}")
    if not base_result.following_constraints:
        print("WARNING: base model does not follow constraints")
    print()

    print(f"Loading model from {model_path}")
    model_primes = load_primes(model_path)
    model = Model.from_primes(model_primes, base=base_model)
    print("Model loaded.\n")

    result = evaluator.evaluate(model)
    print(model.info())
    print(format_score(result))
    if not result.following_constraints:
        print("WARNING: model does not follow constraints")

    if report:
        maybe_write_report(evaluator, result, meta, model_path)

    return result


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
        help="Write a per-experiment NAV score TSV next to the current directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = find_project_root()
    demo_dir = project_root / "demo"

    use_demo = args.config is None
    if args.config is not None:
        config_path = args.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
    else:
        # --demo or default
        config_path = demo_dir / "CAD_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Demo config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    model_path = resolve_model_path(
        args.model,
        config_dir=config_path.parent,
        start_from_config=config["start"],
        demo_dir=demo_dir,
        use_demo=use_demo,
    )

    print(f"Project root: {project_root}")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}\n")

    evaluate_model(config_path, model_path, report=args.report)


if __name__ == "__main__":
    main()
