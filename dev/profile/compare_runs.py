#!/usr/bin/env python3
"""Compare wall times of identical profile tasks across commits.

Loads ``manifest.json`` files under ``dev/profile/runs/``. Two runs are
comparable only when ``task_fingerprint`` matches (same config hash, GA
knobs, seed, core, etc.). Environment differences are reported as warnings
but do not block comparison.

Examples
--------
::

    # Table of all runs, grouped by task fingerprint
    python boolmore/dev/profile/compare_runs.py

    # Compare two specific manifests (must share task fingerprint)
    python boolmore/dev/profile/compare_runs.py \\
        dev/profile/runs/<run_a>/manifest.json \\
        dev/profile/runs/<run_b>/manifest.json

    # Restrict to one task name
    python boolmore/dev/profile/compare_runs.py --task ga_breast_cancer
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_PROFILE_DIR = Path(__file__).resolve().parent
if str(_PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROFILE_DIR))

from common import (  # noqa: E402
    RUNS_DIR,
    env_warning_report,
    iter_manifests,
    load_manifest,
    task_mismatch_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare profile run manifests. Refuses to mix runs whose task "
            "fingerprints differ (apples-to-apples)."
        )
    )
    p.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        help="Optional explicit manifest.json paths (2+ for pairwise table).",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help=f"Directory of run folders (default: {RUNS_DIR})",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Only include runs whose task.name matches.",
    )
    p.add_argument(
        "--fingerprint",
        default=None,
        help="Only include runs with this exact task_fingerprint.",
    )
    p.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Print wall times even if task fingerprints differ (not recommended).",
    )
    return p.parse_args(argv)


def _resolve_manifests(args: argparse.Namespace) -> list[dict]:
    paths: list[Path]
    if args.manifests:
        paths = []
        for raw in args.manifests:
            path = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
            if path.is_dir():
                path = path / "manifest.json"
            if not path.is_file():
                raise SystemExit(f"manifest not found: {path}")
            paths.append(path)
    else:
        paths = iter_manifests(args.runs_dir)

    runs: list[dict] = []
    for path in paths:
        manifest = load_manifest(path)
        manifest["_path"] = str(path)
        if args.task is not None and (manifest.get("task") or {}).get("name") != args.task:
            continue
        if (
            args.fingerprint is not None
            and manifest.get("task_fingerprint") != args.fingerprint
        ):
            continue
        runs.append(manifest)
    return runs


def _fmt_wall(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.4f}"


def _run_label(m: dict) -> str:
    git = m.get("git") or {}
    short = git.get("commit_short") or "?"
    dirty = "*" if git.get("dirty") else ""
    return f"{short}{dirty}"


def print_group_table(runs: list[dict]) -> None:
    """Print one comparison table for runs that share a fingerprint."""
    runs = sorted(
        runs,
        key=lambda m: (
            (m.get("git") or {}).get("commit") or "",
            m.get("started_utc") or "",
        ),
    )
    fp = runs[0].get("task_fingerprint", "?")
    task = runs[0].get("task") or {}
    print("=" * 88)
    print(f"task: {task.get('name')}  task_version={task.get('task_version')}")
    print(f"task_fingerprint: {fp}")
    print(
        f"knobs: seed={task.get('seed')} core={task.get('core')} "
        f"nav_method={task.get('nav_method')} params={task.get('parameters')}"
    )
    print(f"config: {task.get('config_rel')} sha256={task.get('config_sha256')}")
    print("-" * 88)
    header = (
        f"{'commit':<12} {'dirty':<5} {'wall_s':>10} {'delta%':>8} "
        f"{'gen':>5} {'final':>10} {'started_utc':<22} {'run_id'}"
    )
    print(header)
    print("-" * 88)

    baseline = (runs[0].get("results") or {}).get("wall_seconds")
    for m in runs:
        git = m.get("git") or {}
        res = m.get("results") or {}
        wall = res.get("wall_seconds")
        if baseline and wall is not None and baseline > 0:
            delta = 100.0 * (wall - baseline) / baseline
            delta_s = f"{delta:+.1f}%"
        else:
            delta_s = "—"
        final = res.get("final_score")
        final_s = f"{final:.2f}" if isinstance(final, (int, float)) else "n/a"
        print(
            f"{_run_label(m):<12} "
            f"{'yes' if git.get('dirty') else 'no':<5} "
            f"{_fmt_wall(wall):>10} "
            f"{delta_s:>8} "
            f"{res.get('n_generated', 'n/a'):>5} "
            f"{final_s:>10} "
            f"{(m.get('started_utc') or '')[:22]:<22} "
            f"{m.get('run_id', Path(m['_path']).parent.name)}"
        )

    if len(runs) >= 2 and baseline:
        latest = (runs[-1].get("results") or {}).get("wall_seconds")
        if latest is not None:
            speedup = baseline / latest if latest > 0 else float("inf")
            print("-" * 88)
            print(
                f"vs first in group: {_fmt_wall(baseline)}s → {_fmt_wall(latest)}s "
                f"({speedup:.3f}x; {(100.0 * (latest - baseline) / baseline):+.1f}%)"
            )

    # pairwise env warnings vs first
    warn = []
    for m in runs[1:]:
        warn.extend(env_warning_report(runs[0], m))
    # unique preserve order
    seen: set[str] = set()
    uniq = []
    for w in warn:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    if uniq:
        print()
        print("environment warnings (may skew wall times):")
        for w in uniq:
            print(f"  - {w}")
    notes = [m.get("note") for m in runs if m.get("note")]
    if notes:
        print()
        print("notes:")
        for m in runs:
            if m.get("note"):
                print(f"  - {_run_label(m)}: {m['note']}")
    print()


def compare_explicit(runs: list[dict], *, allow_mismatch: bool) -> int:
    fps = {m.get("task_fingerprint") for m in runs}
    if len(fps) > 1 and not allow_mismatch:
        print("ERROR: task fingerprints differ — not apples-to-apples.", file=sys.stderr)
        print(file=sys.stderr)
        for i, m in enumerate(runs):
            print(
                f"  [{i}] {_run_label(m)}  fp={m.get('task_fingerprint')}",
                file=sys.stderr,
            )
        print(file=sys.stderr)
        print("Diffs vs first:", file=sys.stderr)
        for m in runs[1:]:
            for line in task_mismatch_report(runs[0], m):
                print(f"  {line}", file=sys.stderr)
        print(
            "\nRe-run with matching knobs, or pass --allow-mismatch to force.",
            file=sys.stderr,
        )
        return 2
    if len(fps) > 1:
        print("WARNING: comparing mismatched tasks (--allow-mismatch)\n")
        for m in runs[1:]:
            for line in task_mismatch_report(runs[0], m):
                print(f"  {line}")
        print()
    print_group_table(runs)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runs = _resolve_manifests(args)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if not runs:
        print(f"No profile runs found under {args.runs_dir}")
        print("Create one with: python boolmore/dev/profile/run_ga.py")
        return 1

    if args.manifests and len(args.manifests) >= 2:
        return compare_explicit(runs, allow_mismatch=args.allow_mismatch)

    by_fp: dict[str, list[dict]] = defaultdict(list)
    for m in runs:
        by_fp[m.get("task_fingerprint") or "unknown"].append(m)

    print(f"runs_dir: {args.runs_dir}")
    print(f"loaded: {len(runs)} run(s) in {len(by_fp)} task group(s)\n")
    for fp in sorted(by_fp, key=lambda k: min(m.get("started_utc") or "" for m in by_fp[k])):
        print_group_table(by_fp[fp])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
