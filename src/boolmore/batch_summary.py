"""Aggregate per-seed results for a boolmore batch run folder.

Used by ``run_batch`` (writes ``summary.csv`` by default) and by
``scripts/summarize_run.py`` (human-readable report).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
_ELAPSED_RE = re.compile(
    r"done\s+seed=(?P<seed>\d+)\s+elapsed=(?P<elapsed>[^\s]+)"
)
_SCORE_HEADER_RE = re.compile(
    r"#\s*score:\s*(?P<score>[0-9.]+)\s*/\s*(?P<max>[0-9.]+)"
)
_TIMEDELTA_RE = re.compile(
    r"^(?:(?P<days>\d+)\s+days?,\s+)?"
    r"(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)$"
)

SUMMARY_CSV_NAME = "summary.csv"


@dataclass
class SeedSummary:
    seed: int
    score: float
    max_score: float
    pct: float
    generation: int
    model_id: int
    n_extra_edges: int
    n_eff_self_edges: int
    n_eff_edges: int
    n_prime_implicants: int
    iterations: int
    n_evaluated: int
    elapsed: str | None
    best_model: str | None
    evals_csv: str


def seed_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _SEED_DIR_RE.match(child.name)
        if m:
            found.append((int(m.group(1)), child))
    return found


def parse_elapsed(batch_log: Path) -> dict[int, str]:
    if not batch_log.is_file():
        return {}
    out: dict[int, str] = {}
    with open(batch_log, encoding="utf-8") as fp:
        for line in fp:
            m = _ELAPSED_RE.search(line)
            if m:
                out[int(m.group("seed"))] = m.group("elapsed")
    return out


def elapsed_to_seconds(text: str | None) -> float | None:
    """Parse ``batch.log`` elapsed strings (``datetime.timedelta`` repr)."""
    if not text:
        return None
    m = _TIMEDELTA_RE.match(text.strip())
    if not m:
        return None
    days = int(m.group("days") or 0)
    hours = int(m.group("hours"))
    minutes = int(m.group("minutes"))
    seconds = float(m.group("seconds"))
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def load_run_meta(run_dir: Path) -> dict | None:
    path = run_dir / "run_meta.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def best_exported_model(seed_dir: Path, seed: int) -> Path | None:
    preferred = list(seed_dir.glob(f"seed_{seed:03d}_*_gen*.bnet"))
    if preferred:
        preferred.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return preferred[0]
    return None


def best_from_evals(evals_csv: Path) -> dict:
    """Return the best model row: first unique id of the last iteration."""
    with open(evals_csv, encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"empty evals csv: {evals_csv}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"no rows in {evals_csv}")

    seen: set[tuple[int, int]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (int(row["iteration"]), int(row["id"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    last_iter = max(int(r["iteration"]) for r in unique)
    last_rows = [r for r in unique if int(r["iteration"]) == last_iter]
    best = last_rows[0]
    return {
        "score": float(best["score"]),
        "max_score": float(best["max_score"]),
        "generation": int(best["generation"]),
        "model_id": int(best["id"]),
        "n_extra_edges": int(best["n_extra_edges"]),
        "n_eff_self_edges": int(best["n_eff_self_edges"]),
        "n_eff_edges": int(best["n_eff_edges"]),
        "n_prime_implicants": int(best["n_prime_implicants"]),
        "iterations": last_iter,
        "n_evaluated": len({int(r["id"]) for r in unique}),
    }


def start_score_from_log(seed_log: Path) -> tuple[float, float] | None:
    if not seed_log.is_file():
        return None
    with open(seed_log, encoding="utf-8") as fp:
        for line in fp:
            m = _SCORE_HEADER_RE.search(line)
            if m:
                return float(m.group("score")), float(m.group("max"))
    return None


def collect_seed_summaries(
    run_dir: Path,
    *,
    warn: bool = True,
) -> list[SeedSummary]:
    """Build one ``SeedSummary`` per ``seed_NNN`` folder that has evals."""
    dirs = seed_dirs(run_dir)
    if not dirs:
        return []

    elapsed_by_seed = parse_elapsed(run_dir / "batch.log")
    summaries: list[SeedSummary] = []
    for seed, seed_dir in dirs:
        evals = seed_dir / f"seed_{seed:03d}_evals.csv"
        if not evals.is_file():
            matches = sorted(seed_dir.glob("*_evals.csv"))
            if not matches:
                if warn:
                    print(
                        f"warning: missing evals for seed {seed}",
                        file=sys.stderr,
                    )
                continue
            evals = matches[0]
        best = best_from_evals(evals)
        model = best_exported_model(seed_dir, seed)
        max_score = best["max_score"]
        summaries.append(
            SeedSummary(
                seed=seed,
                score=best["score"],
                max_score=max_score,
                pct=100.0 * best["score"] / max_score if max_score else float("nan"),
                generation=best["generation"],
                model_id=best["model_id"],
                n_extra_edges=best["n_extra_edges"],
                n_eff_self_edges=best["n_eff_self_edges"],
                n_eff_edges=best["n_eff_edges"],
                n_prime_implicants=best["n_prime_implicants"],
                iterations=best["iterations"],
                n_evaluated=best["n_evaluated"],
                elapsed=elapsed_by_seed.get(seed),
                best_model=model.name if model else None,
                evals_csv=str(evals.relative_to(run_dir)),
            )
        )
    return summaries


def summarize_batch(run_dir: Path) -> tuple[list[SeedSummary], dict]:
    """Return seed summaries plus metadata for reporting."""
    run_dir = Path(run_dir)
    summaries = collect_seed_summaries(run_dir)
    config: dict = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as fp:
            config = json.load(fp)

    start_score = None
    for seed, seed_dir in seed_dirs(run_dir):
        parsed = start_score_from_log(seed_dir / f"seed_{seed:03d}_log.txt")
        if parsed is not None:
            start_score = parsed
            break

    meta = {
        "run_dir": str(run_dir),
        "config": config,
        "run_meta": load_run_meta(run_dir),
        "start_score": start_score,
        "n_seeds": len(summaries),
    }
    return summaries, meta


def write_summary_csv(
    summaries: list[SeedSummary],
    path: Path | str,
) -> Path:
    """Write ``summary.csv`` (seed-sorted) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(asdict(summaries[0]).keys())
        if summaries
        else ["seed", "score", "max_score", "pct"]
    )
    with open(out, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(summaries, key=lambda x: x.seed):
            writer.writerow(asdict(s))
    return out


def write_batch_summary_csv(run_dir: Path | str) -> Path | None:
    """Collect seed results under ``run_dir`` and write ``summary.csv``.

    Returns the written path, or ``None`` if no seed evals were found.
    """
    run_dir = Path(run_dir)
    summaries = collect_seed_summaries(run_dir, warn=False)
    if not summaries:
        return None
    return write_summary_csv(summaries, run_dir / SUMMARY_CSV_NAME)
