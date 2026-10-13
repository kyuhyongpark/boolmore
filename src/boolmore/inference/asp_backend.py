"""Probe and cache availability of PyBoolNet's ASP trap-space backend."""

from __future__ import annotations

from typing import Optional

from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.trap_spaces import compute_trap_spaces

_CHECKED: Optional[bool] = None
_FAILURE: Optional[RuntimeError] = None

_ERROR_MESSAGE = (
    "PyBoolNet ASP backend (gringo/clasp) is not working: trap-space "
    "computation returned an unexpected result for a known probe network. "
    "Empty results from a broken backend look like valid 'no trap space' "
    "answers. See the Install / Troubleshooting section of the boolmore "
    "README (gringo + clasp binaries; not `pip install clingo`)."
)

_EXPECTED_MIN_TRAPS = frozenset(
    (
        frozenset((("A", 0),)),
        frozenset((("A", 1),)),
    )
)


def reset_asp_backend_check() -> None:
    """Clear the cached probe result (for tests)."""
    global _CHECKED, _FAILURE
    _CHECKED = None
    _FAILURE = None


def _probe_ok() -> bool:
    primes = bnet_text2primes("A, A\n")
    traps = compute_trap_spaces(primes, "min")
    if not isinstance(traps, list):
        return False
    normalized = frozenset(
        frozenset(trap.items()) for trap in traps if isinstance(trap, dict)
    )
    return normalized == _EXPECTED_MIN_TRAPS


def ensure_asp_backend() -> None:
    """Ensure PyBoolNet gringo/clasp can compute trap spaces.

    Runs a one-time probe on the network ``A, A``, which must yield the
    minimal trap spaces ``{A: 0}`` and ``{A: 1}``. Success and failure are
    both cached so a broken environment does not re-run the probe.

    Raises
    ------
    RuntimeError
        If the ASP binaries are missing, crash silently, or otherwise fail
        the probe.
    """
    global _CHECKED, _FAILURE

    if _CHECKED is True:
        return
    if _CHECKED is False:
        assert _FAILURE is not None
        raise _FAILURE

    try:
        ok = _probe_ok()
    except Exception as exc:
        _CHECKED = False
        _FAILURE = RuntimeError(_ERROR_MESSAGE)
        _FAILURE.__cause__ = exc
        raise _FAILURE from exc

    if not ok:
        _CHECKED = False
        _FAILURE = RuntimeError(_ERROR_MESSAGE)
        raise _FAILURE

    _CHECKED = True
