"""Load primes, config JSON, Model objects, and experiment data from files."""

import json
import pickle
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.io.nav import import_NAV_exps
from boolmore.io.phenotype import (
    CSVParseException,
    ParseError,
    check_phenotypes,
    import_phenotypes,
)
from boolmore.model import Model
from boolmore.types import PrimeType

__all__ = [
    "load_primes",
    "load_json_config",
    "load_model",
    "import_NAV_exps",
    "ParseError",
    "CSVParseException",
    "import_phenotypes",
    "check_phenotypes",
]


def load_primes(path: str | Path) -> dict[str, PrimeType]:
    """Load PyBoolNet primes from a ``.bnet`` or ``.pkl`` file."""
    path = Path(path)
    if path.suffix == ".bnet":
        return bnet_file2primes(str(path))
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported model file format: {path}")


def load_json_config(path: str | Path) -> dict:
    """Load a GA / evaluation JSON config file.

    Coerces ``edge_pool`` entries from JSON lists to ``(regulator, target, sign)``
    tuples expected by ``Model``.
    """
    with open(path) as f:
        config = json.load(f)

    if "edge_pool" in config:
        config["edge_pool"] = [
            tuple(edge) if not isinstance(edge, tuple) else edge
            for edge in config["edge_pool"]
        ]

    return config


def load_model(
    path: str | Path,
    *,
    base: Model | None = None,
    edge_pool: list | None = None,
) -> Model:
    """Load a ``.bnet`` / ``.pkl`` model, optionally relative to a base model."""
    primes = load_primes(path)
    return Model.from_primes(primes, base=base, edge_pool=edge_pool)
