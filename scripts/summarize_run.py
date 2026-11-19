#!/usr/bin/env python3
"""Print a human-readable summary of a boolmore batch run folder.

``summary.csv`` is written by ``run_batch`` itself. This script only reports
(and can optionally regenerate the CSV for older batches).

Edit ``RUN_DIR`` below, then::

    python boolmore/scripts/summarize_run.py

Or pass a folder on the CLI (overrides ``RUN_DIR``)::

    python boolmore/scripts/summarize_run.py \\
        case_study/breast_cancer/runs/20261118T231453Z
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

from boolmore.batch_summary import (
    SeedSummary,
    elapsed_to_seconds,
    summarize_batch,
    write_summary_csv,
)
from boolmore.io.paths import project_root

# --- edit these to summarize a different batch ---
RUN_DIR = "case_study/breast_cancer/runs/20261118T231453Z"
TOP = 0  # 0 = print all seeds; N = top-N by score
# -------------------------------------------------

_ENV_KEYS = (
    "ldoi_closure_engine",
    "ldoi_cython",
    "uni_rr_cython",
    "uni_rr_cython_min_k",
    "regulate_repair_mode",
    "hostname",
    "platform",
    "python",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Print a summary of a boolmore multi-seed batch run. "
            "summary.csv is normally produced by run_batch; use --write-csv "
            "only to backfill older batches."
        )
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Batch run folder (contains seed_NNN/ and usually config.json). "
            f"Default: RUN_DIR knob ({RUN_DIR!r})."
        ),
    )
    p.add_argument(
        "--write-csv",
        action="store_true",
        help="Regenerate summary.csv in the run directory (backfill).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional explicit CSV path (implies --write-csv).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help="Print only the top-N seeds by score (default: TOP knob).",
    )
    return p.parse_args(argv)


def _resolve_run_dir(raw: Path | str) -> Path:
    """Resolve under project root unless absolute; also accept cwd-relative."""
    p = Path(raw)
    if p.is_absolute():
        path = p.resolve()
    else:
        under_root = (project_root() / p).resolve()
        under_cwd = (Path.cwd() / p).resolve()
        if under_root.is_dir():
            path = under_root
        elif under_cwd.is_dir():
            path = under_cwd
        else:
            path = under_root
    if not path.is_dir():
        raise SystemExit(f"run directory not found: {path}")
    return path


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else float("nan")


def _fmt(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "unknown"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d} ({seconds:.1f}s)"
    return f"{m}:{s:02d} ({seconds:.1f}s)"


def _print_environment(run_meta: dict | None) -> None:
    print("--- environment ---")
    if not run_meta:
        print("run_meta.json: unknown (older batch; re-run to capture provenance)")
        print("git: unknown")
        for key in _ENV_KEYS:
            print(f"  {key}: unknown")
        return

    git = run_meta.get("git") or {}
    env = run_meta.get("environment") or {}
    describe = git.get("describe") or git.get("commit_short") or "unknown"
    dirty = git.get("dirty")
    dirty_s = f" dirty={dirty}" if dirty is not None else ""
    print(f"git: {describe}{dirty_s}")
    if run_meta.get("core") is not None:
        print(f"core: {run_meta.get('core')}")
    if run_meta.get("started_utc") or run_meta.get("finished_utc"):
        print(
            f"started_utc: {run_meta.get('started_utc') or 'unknown'}  "
            f"finished_utc: {run_meta.get('finished_utc') or 'unknown'}"
        )
    for key in _ENV_KEYS:
        val = env.get(key, "unknown") if env else "unknown"
        if val is None:
            val = "unknown"
        if key == "python" and isinstance(val, str) and len(val) > 80:
            val = val[:77] + "..."
        print(f"  {key}: {val}")


def print_report(
    summaries: list[SeedSummary],
    meta: dict,
    *,
    top: int = 0,
) -> None:
    run_dir = Path(meta["run_dir"])
    config = meta.get("config") or {}
    params = config.get("parameters") or {}
    run_meta = meta.get("run_meta")

    print(f"run: {run_dir}")
    if config:
        print(
            f"type={config.get('run_type')}  "
            f"nav_method={config.get('nav_method')}  "
            f"hierarchy={config.get('hierarchy')}"
        )
        print(
            f"params: total_iter={params.get('total_iter')}  "
            f"per_iter={params.get('per_iter')}  "
            f"keep={params.get('keep')}  mix={params.get('mix')}  "
            f"edge_prob={params.get('edge_prob')}"
        )
    start = meta.get("start_score")
    if start is not None:
        s, m = start
        print(f"start/base score: {s:.2f}/{m:.1f} ({100.0 * s / m:.1f}%)")
    print(f"seeds with evals: {len(summaries)}")
    print()
    _print_environment(run_meta)
    print()

    rows = sorted(summaries, key=lambda s: s.score, reverse=True)
    if top > 0:
        rows = rows[:top]

    header = (
        f"{'seed':>4} {'score':>8} {'pct':>6} {'gen':>4} {'id':>6} "
        f"{'extra':>5} {'self':>4} {'edges':>5} {'pi':>5} "
        f"{'elapsed':>14}  model"
    )
    print(header)
    print("-" * len(header))
    for s in rows:
        print(
            f"{s.seed:>4} "
            f"{s.score:>8.2f} "
            f"{s.pct:>5.1f}% "
            f"{s.generation:>4} "
            f"{s.model_id:>6} "
            f"{s.n_extra_edges:>5} "
            f"{s.n_eff_self_edges:>4} "
            f"{s.n_eff_edges:>5} "
            f"{s.n_prime_implicants:>5} "
            f"{(s.elapsed or 'n/a'):>14}  "
            f"{s.best_model or '—'}"
        )

    scores = [s.score for s in summaries]
    pcts = [s.pct for s in summaries]
    max_score = summaries[0].max_score if summaries else float("nan")
    best = max(summaries, key=lambda s: s.score)
    worst = min(summaries, key=lambda s: s.score)
    elapsed_secs = [
        sec
        for s in summaries
        if (sec := elapsed_to_seconds(s.elapsed)) is not None
    ]

    print()
    print("--- aggregate ---")
    print(
        f"score: mean={_fmt(_mean(scores), 2)}  "
        f"median={_fmt(statistics.median(scores), 2)}  "
        f"stdev={_fmt(_stdev(scores), 2)}  "
        f"min={worst.score:.2f} (seed {worst.seed})  "
        f"max={best.score:.2f} (seed {best.seed})"
    )
    print(
        f"pct of {max_score:g}: mean={_fmt(_mean(pcts))}%  "
        f"median={_fmt(statistics.median(pcts))}%  "
        f"range={worst.pct:.1f}–{best.pct:.1f}%"
    )
    if start is not None:
        s0, _ = start
        print(
            f"vs start ({s0:.2f}): "
            f"mean Δ={_fmt(_mean(scores) - s0, 2)}  "
            f"best Δ={best.score - s0:.2f}"
        )
    if elapsed_secs:
        print(
            f"elapsed/seed: mean={_fmt_duration(_mean(elapsed_secs))}  "
            f"median={_fmt_duration(statistics.median(elapsed_secs))}  "
            f"min={_fmt_duration(min(elapsed_secs))}  "
            f"max={_fmt_duration(max(elapsed_secs))}  "
            f"({len(elapsed_secs)}/{len(summaries)} seeds)"
        )
    else:
        print("elapsed/seed: unknown")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = _resolve_run_dir(args.run_dir if args.run_dir is not None else RUN_DIR)
    top = TOP if args.top is None else args.top

    summaries, meta = summarize_batch(run_dir)
    if not summaries:
        print("No seed summaries could be built.", file=sys.stderr)
        return 1
    print_report(summaries, meta, top=top)

    if args.write_csv or args.out is not None:
        out = args.out if args.out is not None else run_dir / "summary.csv"
        if not out.is_absolute():
            out = (Path.cwd() / out).resolve()
        write_summary_csv(summaries, out)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
