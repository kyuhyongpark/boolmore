"""Module CLI: ``python -m boolmore run CONFIG [options]``."""

from __future__ import annotations

import argparse
import sys

from boolmore.runner import run_batch, run_ga


def _parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config",
        help="Config path relative to the project root (or absolute).",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds (default: 0). Single seed = single-run batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed (overrides --seeds).",
    )
    parser.add_argument("--core", type=int, default=1, help="Parallel cores.")
    parser.add_argument(
        "--name",
        default=None,
        help="Batch folder name (default: UTC stamp). Overwrites if set.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Parent directory for batch folders (default: runs).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=(
            "Call run_ga once without batch folder layout "
            "(uses --seed or first of --seeds; --name is the run prefix)."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="python -m boolmore",
        description="boolmore Boolean model refinement",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser(
        "run",
        help="Run GA from a study config (batch layout by default).",
    )
    _add_run_arguments(run_parser)

    # Backward-compatible: ``python -m boolmore CONFIG`` → run batch
    if argv and not argv[0].startswith("-") and argv[0] != "run":
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        raise SystemExit(2)

    seeds = [args.seed] if args.seed is not None else _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("At least one seed is required.")

    if args.single:
        run_ga(
            json_file=args.config,
            seed=seeds[0],
            core=args.core,
            run_name=args.name,
        )
        return

    run_batch(
        args.config,
        seeds=seeds,
        core=args.core,
        run_name=args.name,
        runs_root=args.runs_root,
    )


if __name__ == "__main__":
    main()
