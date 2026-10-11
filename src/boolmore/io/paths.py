"""Path helpers that do not depend on the current working directory."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk parents until a directory containing ``pyproject.toml`` is found.

    If ``start`` is omitted, search from this module's location, then from
    ``Path.cwd()``.
    """
    starts: list[Path] = []
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


def resolve_under(base_dir: Path | str, path: str | Path) -> Path:
    """Resolve ``path``; relative paths are taken relative to ``base_dir``."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (Path(base_dir) / p).resolve()


def resolve_existing_path(
    path: str | Path,
    *search_dirs: Path,
) -> Path:
    """Resolve an existing file: absolute, cwd-relative, then each search dir."""
    candidate = Path(path)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"File not found: {candidate}")
        return candidate.resolve()

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.is_file():
        return cwd_path

    for directory in search_dirs:
        found = (Path(directory) / candidate).resolve()
        if found.is_file():
            return found

    tried = "cwd" + "".join(f", {d}" for d in search_dirs)
    raise FileNotFoundError(f"File not found: {path} (tried {tried})")
