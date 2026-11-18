"""Shared helpers for commit-comparable GA profile runs.

Run artifacts live under ``dev/profile/runs/`` (gitignored). Each run writes
``manifest.json`` (machine-readable) and ``console.log`` (tee of stdout).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def collect_git_info(repo_root: Path) -> dict[str, Any]:
    """Snapshot of the git tree the profile was executed against."""
    commit = _git("rev-parse", "HEAD", cwd=repo_root)
    short = _git("rev-parse", "--short", "HEAD", cwd=repo_root)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
    dirty_out = _git("status", "--porcelain", cwd=repo_root)
    describe = _git("describe", "--always", "--dirty", "--tags", cwd=repo_root)
    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": bool(dirty_out),
        "dirty_paths": dirty_out.splitlines() if dirty_out else [],
        "describe": describe,
    }


def collect_environment() -> dict[str, Any]:
    """Runtime environment that can affect wall time."""
    import boolmore.boolean_functions as bf
    from boolmore.genetic.operators import mutation as mut
    from boolmore.inference import ldoi as ldoi_mod
    from boolmore.inference.ldoi import get_ldoi_closure_engine

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "uni_rr_cython": bool(bf._UNI_RR_CYTHON and bf._get_uni_rr_cy is not None),
        "uni_rr_cython_min_k": bf._UNI_RR_CYTHON_MIN_K,
        "ldoi_cython": bool(ldoi_mod._LDOI_CYTHON and ldoi_mod._batch_closure_cy is not None),
        "ldoi_closure_engine": get_ldoi_closure_engine(),
        "regulate_repair_mode": mut.REGULATE_REPAIR_MODE,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


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
