"""Runtime provenance helpers for GA runs (git + environment snapshot)."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


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
    """Snapshot of the git tree the run was executed against."""
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
    """Runtime environment that can affect wall time / scoring paths."""
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
        "ldoi_cython": bool(
            ldoi_mod._LDOI_CYTHON and ldoi_mod._batch_closure_cy is not None
        ),
        "ldoi_closure_engine": get_ldoi_closure_engine(),
        "regulate_repair_mode": mut.REGULATE_REPAIR_MODE,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def write_json(path: Path | str, obj: dict[str, Any]) -> Path:
    """Write ``obj`` as pretty JSON (sorted keys) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return out


def format_meta_oneline(*, git: dict[str, Any], environment: dict[str, Any]) -> str:
    """Short human-readable provenance line for batch/seed logs."""
    # ``git describe --dirty`` already encodes dirty trees; fall back otherwise.
    describe = git.get("describe") or git.get("commit_short") or "nogit"
    if not git.get("describe") and git.get("dirty"):
        describe = f"{describe}-dirty"
    return (
        f"git={describe}  "
        f"ldoi_closure_engine={environment.get('ldoi_closure_engine')!r}  "
        f"ldoi_cython={environment.get('ldoi_cython')}  "
        f"uni_rr_cython={environment.get('uni_rr_cython')}  "
        f"regulate_repair_mode={environment.get('regulate_repair_mode')!r}"
    )
