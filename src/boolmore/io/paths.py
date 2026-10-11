"""Path helpers that do not depend on the current working directory."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (directory containing ``pyproject.toml``).

    Walks upward from this package file so the result does not depend on the
    process working directory.
    """
    here = Path(__file__).resolve()

    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml above "
        f"{here})."
    )
