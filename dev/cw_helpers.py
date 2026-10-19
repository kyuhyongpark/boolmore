"""Reusable cubewalkers helpers for ``dev/`` notebooks and percent-scripts.

Not part of the installed ``boolmore`` package. Import from sibling scripts via::

    from cw_helpers import require_cubewalkers, simulate_mean_nav, bootstrap_se, ...

or, if the kernel cwd is not ``dev/``::

    import sys
    from pathlib import Path
    _DEV = Path(__file__).resolve().parent
    if str(_DEV) not in sys.path:
        sys.path.insert(0, str(_DEV))
    from cw_helpers import ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

_CUDA_HELP = """
Cubewalkers requires CUDA + CuPy inside WSL2.

Setup checklist:
  1. Windows: NVIDIA driver with WSL2 GPU support (`nvidia-smi` in PowerShell)
  2. WSL: CUDA toolkit matching the driver major version
  3. pip install cupy-cuda12x   # or cupy-cuda11x
  4. pip install git+https://github.com/jcrozum/cubewalkers
  5. Re-run nvidia-smi / cupy device count / tiny cubewalkers smoke test
""".strip()

DEFAULT_THREADS_PER_BLOCK: tuple[int, int] = (16, 16)
DEFAULT_SEED = 0
DEFAULT_N_BOOT = 1000
DEFAULT_BOOT_CHUNK = 50


def require_cubewalkers():
    """Import cubewalkers and verify a CUDA device is available."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit(
            f"CuPy is not installed.\n\n{_CUDA_HELP}\n\nOriginal error: {exc}"
        ) from exc

    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"CUDA runtime failed.\n\n{_CUDA_HELP}\n\nOriginal error: {exc}"
        ) from exc

    if n_dev < 1:
        raise SystemExit(f"No CUDA devices visible (count={n_dev}).\n\n{_CUDA_HELP}")

    try:
        import cubewalkers as cw
    except ImportError as exc:
        raise SystemExit(
            "cubewalkers is not installed.\n"
            "  pip install git+https://github.com/jcrozum/cubewalkers\n\n"
            f"{_CUDA_HELP}\n\nOriginal error: {exc}"
        ) from exc

    try:
        probe = cw.Model("A, !A\nB, A", n_time_steps=2, n_walkers=4)
        probe.simulate_ensemble(
            T_window=1,
            averages_only=True,
            maskfunction=cw.update_schemes.asynchronous,
        )
        cp.cuda.Device().synchronize()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"cubewalkers smoke simulation failed.\n\n{_CUDA_HELP}\n\n"
            f"Original error: {exc}"
        ) from exc

    return cw, cp


def bnet_rules_for_cubewalkers(path: Path) -> str:
    """Load a BoolNet ``.bnet``, dropping the ``targets, factors`` header row."""
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().replace("\t", " ").replace(" ", "") in {
            "targets,factors",
            "targets,factors,",
        }:
            continue
        left = stripped.split(",", 1)[0].strip().lower()
        if left == "targets":
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def make_model_namer() -> Callable[[str], str]:
    """Return a callable that yields unique CuPy-safe ``RawKernel`` names."""
    counter = 0

    def next_name(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter}"

    return next_name


def clamps_to_experiment(
    clamps: Sequence[tuple[str, int]] | Iterable[tuple[str, int]] | None,
) -> Any:
    """Build a cubewalkers ``Experiment`` that permanently clamps each fix.

    ``clamps`` is an iterable of ``(node, value)`` pairs (e.g. a boolmore
    ``Assignment``). Empty / ``None`` → no experiment (``None``).
    """
    if not clamps:
        return None
    import cubewalkers as cw

    lines = [f"{node},0,inf,{int(value)}" for node, value in clamps]
    return cw.Experiment("\n".join(lines))


def _build_model(
    rules: str,
    *,
    n_time_steps: int,
    n_walkers: int,
    experiment: Any = None,
    model_name: str | None = None,
):
    """Construct a ``cw.Model`` with optional permanent clamps."""
    import cubewalkers as cw

    model_kwargs: dict = {
        "n_time_steps": n_time_steps,
        "n_walkers": n_walkers,
    }
    if model_name is not None:
        model_kwargs["model_name"] = model_name
    return cw.Model(rules, experiment=experiment, **model_kwargs)


def simulate_walker_nav(
    rules: str,
    *,
    n_time_steps: int,
    t_window: int,
    n_walkers: int,
    clamps: Sequence[tuple[str, int]] | Iterable[tuple[str, int]] | None = None,
    threads_per_block: tuple[int, int] = DEFAULT_THREADS_PER_BLOCK,
    model_name: str | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Run an ensemble; return per-walker NAV and split timings.

    Parameters
    ----------
    clamps
        Optional permanent interventions ``(node, value)``. ``None`` / empty
        → unclamped ensemble.

    Returns
    -------
    walker_nav : ndarray, shape (n_nodes, n_walkers)
        Time-window mean for each node and walker.
    varnames : list[str]
    timings : dict with t_setup, t_simulate, t_reduce, t_total

    Notes
    -----
    ``cw.Model`` builds a ``RawKernel`` during setup, but NVRTC compile is often
    deferred until the first kernel launch inside ``simulate_ensemble``.
    Call :func:`require_cubewalkers` first for a clearer fail-fast message.
    """
    import cubewalkers as cw
    import cupy as cp

    if t_window > n_time_steps or t_window < 1:
        raise ValueError(f"T_WINDOW={t_window} must be in [1, T={n_time_steps}]")

    experiment = clamps_to_experiment(clamps)

    t0 = perf_counter()
    model = _build_model(
        rules,
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
        experiment=experiment,
        model_name=model_name,
    )
    cp.cuda.Device().synchronize()
    t_setup = perf_counter() - t0

    t0 = perf_counter()
    model.simulate_ensemble(
        T_window=t_window,
        averages_only=False,
        maskfunction=cw.update_schemes.asynchronous,
        threads_per_block=threads_per_block,
    )
    cp.cuda.Device().synchronize()
    t_simulate = perf_counter() - t0

    t0 = perf_counter()
    traj = model.trajectories  # (T_window, N, W)
    walker_nav_gpu = traj.mean(axis=0)  # (N, W)
    walker_nav = cp.asnumpy(walker_nav_gpu).astype(np.float64, copy=False)
    varnames = list(model.varnames)
    del traj, walker_nav_gpu, model
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
    t_reduce = perf_counter() - t0

    timings = {
        "t_setup": float(t_setup),
        "t_simulate": float(t_simulate),
        "t_reduce": float(t_reduce),
        "t_total": float(t_setup + t_simulate + t_reduce),
    }
    return walker_nav, varnames, timings


def simulate_mean_nav(
    rules: str,
    *,
    n_time_steps: int,
    t_window: int,
    n_walkers: int,
    clamps: Sequence[tuple[str, int]] | Iterable[tuple[str, int]] | None = None,
    threads_per_block: tuple[int, int] = DEFAULT_THREADS_PER_BLOCK,
    model_name: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run an ensemble with ``averages_only``; return mean NAV dict + timings.

    Same clamping / horizon conventions as :func:`simulate_walker_nav`, but
    keeps only the walker-averaged trajectories (cheap enough for many
    intervention loops).

    Returns
    -------
    nav : dict[str, float]
        Time-window mean activity per node (cubewalkers varname order).
    timings : dict with t_setup, t_simulate, t_reduce, t_total
    """
    import cubewalkers as cw
    import cupy as cp

    if t_window > n_time_steps or t_window < 1:
        raise ValueError(f"T_WINDOW={t_window} must be in [1, T={n_time_steps}]")

    experiment = clamps_to_experiment(clamps)

    t0 = perf_counter()
    model = _build_model(
        rules,
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
        experiment=experiment,
        model_name=model_name,
    )
    cp.cuda.Device().synchronize()
    t_setup = perf_counter() - t0

    t0 = perf_counter()
    model.simulate_ensemble(
        T_window=t_window,
        averages_only=True,
        maskfunction=cw.update_schemes.asynchronous,
        threads_per_block=threads_per_block,
    )
    cp.cuda.Device().synchronize()
    t_simulate = perf_counter() - t0

    t0 = perf_counter()
    traj = model.trajectories  # (T_window, N)
    means_gpu = traj.mean(axis=0)
    means = cp.asnumpy(means_gpu).astype(np.float64, copy=False)
    nav = {name: float(means[i]) for i, name in enumerate(model.varnames)}
    del traj, means_gpu, model
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
    t_reduce = perf_counter() - t0

    timings = {
        "t_setup": float(t_setup),
        "t_simulate": float(t_simulate),
        "t_reduce": float(t_reduce),
        "t_total": float(t_setup + t_simulate + t_reduce),
    }
    return nav, timings


def bootstrap_se(
    walker_nav: np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    chunk: int = DEFAULT_BOOT_CHUNK,
    seed: int = DEFAULT_SEED,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bootstrap SE of the walker-mean NAV for each node.

    Parameters
    ----------
    walker_nav : (n_nodes, n_walkers)

    Returns
    -------
    means : (n_nodes,)
    se : (n_nodes,)
    boot_means : (n_nodes, n_boot)
        Walker-mean replicates; reuse for independent B−A diffs.
    t_boot : float

    Notes
    -----
    Resamples are processed in chunks so we never materialize a full
    ``(n_nodes, n_boot, n_walkers)`` array (can be multi-GB at paper scale).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_nodes, n_walkers = walker_nav.shape
    t0 = perf_counter()
    means = walker_nav.mean(axis=1)
    boot_means = np.empty((n_nodes, n_boot), dtype=np.float64)
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        k = end - start
        idx = rng.integers(0, n_walkers, size=(k, n_walkers))
        # walker_nav[:, idx] → (n_nodes, k, n_walkers)
        boot_means[:, start:end] = walker_nav[:, idx].mean(axis=2)
    se = boot_means.std(axis=1, ddof=1)
    t_boot = perf_counter() - t0
    return means, se, boot_means, float(t_boot)


def diff_from_boot(
    means_a: np.ndarray,
    boot_a: np.ndarray,
    means_b: np.ndarray,
    boot_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node Δ = mean_B − mean_A and bootstrap SE from stored replicates.

    Pairing independent setting-A and setting-B replicates is the same
    two-sample bootstrap as resampling both walker sets again.
    """
    if boot_a.shape != boot_b.shape:
        raise ValueError(f"boot shape mismatch: {boot_a.shape} vs {boot_b.shape}")
    diff = means_b - means_a
    se = (boot_b - boot_a).std(axis=1, ddof=1)
    return diff, se


def diff_summary(diff: np.ndarray, se: np.ndarray) -> dict[str, float]:
    """Summarize per-node |Δ| and SE(Δ) across nodes."""
    abs_diff = np.abs(diff)
    return {
        "vs_prev_median_abs": float(np.median(abs_diff)),
        "vs_prev_mean_abs": float(abs_diff.mean()),
        "vs_prev_max_abs": float(abs_diff.max()),
        "vs_prev_median_se": float(np.median(se)),
        "vs_prev_mean_se": float(se.mean()),
        "vs_prev_max_se": float(se.max()),
    }


def format_timings(timings: dict[str, float]) -> str:
    """Format split wall-time dict from simulate_* helpers."""
    return (
        f"setup={timings['t_setup']:.3f}s  "
        f"simulate={timings['t_simulate']:.3f}s  "
        f"reduce={timings['t_reduce']:.3f}s  "
        f"total={timings['t_total']:.3f}s"
    )
