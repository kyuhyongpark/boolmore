"""Load primes, config JSON, and Model objects from files."""

import json
import pickle
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_file2primes

from boolmore.model import Model
from boolmore.types import PrimeType


def load_primes(path: str | Path) -> dict[str, PrimeType]:
    """Load PyBoolNet primes from a ``.bnet`` or ``.pkl`` file."""
    path = Path(path)
    if path.suffix == ".bnet":
        return bnet_file2primes(str(path))
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported model file format: {path}")


def load_config(path: str | Path) -> dict:
    """Load a GA / evaluation JSON config file."""
    with open(path) as f:
        return json.load(f)


def load_model(
    path: str | Path,
    *,
    base: Model | None = None,
    edge_pool: list | None = None,
) -> Model:
    """Load a ``.bnet`` / ``.pkl`` model, optionally relative to a base model."""
    primes = load_primes(path)
    return Model.from_primes(primes, base=base, edge_pool=edge_pool)
