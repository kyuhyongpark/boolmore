"""Shared helpers for commit-comparable GA profile runs.

Run artifacts live under ``dev/profile/runs/`` (gitignored). Each run writes
``manifest.json`` (machine-readable) and ``console.log`` (tee of stdout).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boolmore.runtime_info import (  # noqa: F401 — re-export for profile scripts
    collect_environment,
    collect_git_info,
)

PROFILE_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROFILE_DIR / "runs"
SCHEMA_VERSION = 1

# Bump when the default task definition changes in a way that makes old runs
# non-comparable even if CLI knobs match.
TASK_VERSION = 1


class Tee:
    """Write to stdout and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.strip()


def git_commit_order(
    commits: list[str],
    *,
    repo_root: Path | None = None,
) -> dict[str, int]:
    """Map full commit SHAs to oldest-first history rank.

    Uses ``git rev-list --date-order --reverse`` over the union of the given
    commits, then keeps only those commits in that order. Unknown / missing
    commits are omitted (caller should fall back).
    """
    uniq = [c for c in dict.fromkeys(commits) if c]
    if not uniq:
        return {}
    root = repo_root or PROFILE_DIR.parents[1]
    out = _git("rev-list", "--date-order", "--reverse", *uniq, cwd=root)
    if not out:
        return {}
    wanted = set(uniq)
    order: dict[str, int] = {}
    for line in out.splitlines():
        sha = line.strip()
        if sha in wanted and sha not in order:
            order[sha] = len(order)
    return order


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def task_fingerprint(task: dict[str, Any]) -> str:
    """Stable hash of the task knobs that define 'identical work'."""
    return hashlib.sha256(canonical_json(task).encode("utf-8")).hexdigest()


def make_run_dir(*, task_name: str, commit_short: str | None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    short = commit_short or "nogit"
    run_id = f"{stamp}_{short}_{task_name}"
    path = RUNS_DIR / run_id
    path.mkdir(parents=False, exist_ok=False)
    return path


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_manifests(runs_dir: Path | None = None) -> list[Path]:
    root = runs_dir or RUNS_DIR
    if not root.is_dir():
        return []
    return sorted(root.glob("*/manifest.json"))


def task_mismatch_report(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Human-readable list of task-field differences (empty if identical)."""
    ta = a.get("task") or {}
    tb = b.get("task") or {}
    keys = sorted(set(ta) | set(tb))
    diffs: list[str] = []
    for key in keys:
        if ta.get(key) != tb.get(key):
            diffs.append(f"task.{key}: {ta.get(key)!r} vs {tb.get(key)!r}")
    return diffs


def env_warning_report(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Non-fatal env differences that can still skew wall times."""
    ea = a.get("environment") or {}
    eb = b.get("environment") or {}
    watch = (
        "uni_rr_cython",
        "uni_rr_cython_min_k",
        "ldoi_closure_engine",
        "regulate_repair_mode",
        "python",
        "platform",
        "hostname",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    warnings: list[str] = []
    for key in watch:
        if ea.get(key) != eb.get(key):
            warnings.append(f"environment.{key}: {ea.get(key)!r} vs {eb.get(key)!r}")
    ga = a.get("git") or {}
    gb = b.get("git") or {}
    if ga.get("dirty") or gb.get("dirty"):
        warnings.append(
            f"git.dirty: {ga.get('dirty')} ({ga.get('commit_short')}) vs "
            f"{gb.get('dirty')} ({gb.get('commit_short')})"
        )
    return warnings
