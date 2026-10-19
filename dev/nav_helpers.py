"""Reusable NAV-comparison helpers for ``dev/`` notebooks and percent-scripts.

Shared by source-sampling / accuracy notebooks (not cubewalkers — see
``cw_helpers``). Not part of the installed ``boolmore`` package. Import via::

    from nav_helpers import analyze_fixes, nav_from_sampled_sources, ...

or, if the kernel cwd is not ``dev/``::

    import sys
    from pathlib import Path
    _DEV = Path(__file__).resolve().parent
    if str(_DEV) not in sys.path:
        sys.path.insert(0, str(_DEV))
    from nav_helpers import ...
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np

from boolmore.boolean_functions import assignment_to_dict
from boolmore.evaluation.agreement import (
    _AVG_SUFFIX,
    _average_related_values,
    _related_node_names,
)
from boolmore.inference.prediction import (
    NAVPrediction,
    _nav_from_ldoi,
    _nav_from_min_traps,
)
from boolmore.types import Assignment


class Tee:
    """Write to stdout and a log file (or any open text streams)."""

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


def source_nodes_from_primes(primes: Mapping[str, Any]) -> list[str]:
    """Nodes whose only rules are self-constant 0 / 1 (input sources)."""
    return sorted(
        node for node in primes if primes[node] == [[{node: 0}], [{node: 1}]]
    )


def n_unfixed_sources(fixes: Assignment, source_set: set[str]) -> int:
    """Count source nodes not present in ``fixes``."""
    return len(source_set - {node for node, _ in fixes})


def analyze_fixes(
    fixes: Assignment,
    source_set: set[str],
    *,
    max_unfixed_m1: int | None = None,
) -> dict:
    """Per-intervention source coverage and ``2**x`` min-trap lower bound.

    If ``max_unfixed_m1`` is set, also include ``run_method1``
    (``n_unfixed_sources <= max_unfixed_m1``).
    """
    fixed_nodes = {node for node, _ in fixes}
    fixed_sources = sorted(fixed_nodes & source_set)
    unfixed_sources = sorted(source_set - fixed_nodes)
    x = len(unfixed_sources)
    out: dict = {
        "fixes": fixes,
        "n_fixes": len(fixes),
        "n_fixed_sources": len(fixed_sources),
        "n_unfixed_sources": x,
        "min_trap_lower_bound": 2**x,
        "fixed_sources": fixed_sources,
        "unfixed_sources": unfixed_sources,
    }
    if max_unfixed_m1 is not None:
        out["run_method1"] = x <= max_unfixed_m1
    return out


def format_fixes(fixes: Assignment, max_len: int = 60) -> str:
    """Short printable form of an intervention assignment."""
    if not fixes:
        s = "(none)"
    else:
        s = "{" + ", ".join(f"{n}={v}" for n, v in fixes) + "}"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def decode_source_index(
    index: int,
    unfixed: Sequence[str],
    fixed_sources: Mapping[str, int],
) -> dict[str, int]:
    """Map an integer in ``[0, 2**x)`` to a full source assignment."""
    assignment = dict(fixed_sources)
    x = len(unfixed)
    for j, name in enumerate(unfixed):
        bit = (index >> (x - 1 - j)) & 1
        assignment[name] = bit
    return assignment


def sample_source_indices(x: int, n_samples: int, rng: random.Random) -> list[int]:
    """Without-replacement sample of source-combination indices.

    Always shuffled (including full enumeration). Sequential ``0 .. 2**x-1``
    order would keep high-order source bits constant on nested prefixes, so
    Δprev would hit 0.5 artifactually on every power-of-two doubling.
    """
    total = 2**x
    n = min(n_samples, total)
    return rng.sample(range(total), n)


def sample_size_grid(requested: Sequence[int], x: int) -> list[int]:
    """Stability sample sizes for a fix with ``x`` free sources.

    Keeps requested sizes in ``[1, 2**x]``. If the grid's maximum is at least
    ``2**x``, also include ``2**x`` so full enumeration is the last grid point
    (no separate full-enum parameter).
    """
    if not requested:
        return []
    total = 2**x
    sizes = {s for s in requested if 1 <= s <= total}
    if total <= max(requested):
        sizes.add(total)
    return sorted(sizes)


def merge_perturbation(
    fixes: Assignment,
    source_assignment: Mapping[str, int],
) -> dict[str, int]:
    """Intervention dict plus full source assignment (sources override)."""
    pert = assignment_to_dict(fixes)
    pert.update(source_assignment)
    return pert


def average_nav_dicts(navs: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Equal-weight average of per-node NAV dicts."""
    if not navs:
        raise ValueError("no NAV dicts to average")
    nodes = navs[0].keys()
    n = len(navs)
    return {node: sum(d[node] for d in navs) / n for node in nodes}


def _nav_fn_for_method(method: str) -> Callable[..., dict[str, float]]:
    if method == "min_traps":
        return _nav_from_min_traps
    if method == "ldoi":
        return _nav_from_ldoi
    raise ValueError(f"unknown method {method!r}; expected 'min_traps' or 'ldoi'")


def nav_samples_from_sources(
    primes: Mapping[str, Any],
    fixes: Assignment,
    unfixed: Sequence[str],
    fixed_sources: Mapping[str, int],
    indices: Sequence[int],
    method: str,
    node_names: Sequence[str],
) -> np.ndarray:
    """Per-sample NAV matrix ``(n_nodes, n_samples)`` aligned to ``node_names``."""
    nav_fn = _nav_fn_for_method(method)
    n_nodes = len(node_names)
    n_samples = len(indices)
    out = np.empty((n_nodes, n_samples), dtype=np.float64)
    for j, idx in enumerate(indices):
        sources = decode_source_index(idx, unfixed, fixed_sources)
        pert = merge_perturbation(fixes, sources)
        nav = nav_fn(primes, pert)
        out[:, j] = [float(nav[n]) for n in node_names]
    return out


def nav_from_sampled_sources(
    primes: Mapping[str, Any],
    fixes: Assignment,
    unfixed: Sequence[str],
    fixed_sources: Mapping[str, int],
    indices: Sequence[int],
    method: str,
) -> dict[str, float]:
    """Average NAV over source combinations (``method``: ``min_traps`` | ``ldoi``)."""
    nav_fn = _nav_fn_for_method(method)
    navs: list[dict[str, float]] = []
    for idx in indices:
        sources = decode_source_index(idx, unfixed, fixed_sources)
        pert = merge_perturbation(fixes, sources)
        navs.append(nav_fn(primes, pert))
    return average_nav_dicts(navs)


def bootstrap_sample_mean(
    sample_nav: np.ndarray,
    *,
    n_boot: int = 1000,
    chunk: int = 50,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bootstrap SE of the equal-weight mean over the sample axis.

    Parameters
    ----------
    sample_nav : (n_nodes, n_samples)

    Returns
    -------
    means, se, boot_means ``(n_nodes,)`` / ``(n_nodes, n_boot)``, t_boot
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_nodes, n_samples = sample_nav.shape
    if n_samples < 1:
        raise ValueError("sample_nav must have at least one sample")
    t0 = perf_counter()
    means = sample_nav.mean(axis=1)
    boot_means = np.empty((n_nodes, n_boot), dtype=np.float64)
    if n_samples == 1:
        boot_means[:] = means[:, None]
        se = np.zeros(n_nodes, dtype=np.float64)
        return means, se, boot_means, float(perf_counter() - t0)
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        k = end - start
        idx = rng.integers(0, n_samples, size=(k, n_samples))
        boot_means[:, start:end] = sample_nav[:, idx].mean(axis=2)
    se = boot_means.std(axis=1, ddof=1)
    return means, se, boot_means, float(perf_counter() - t0)


def nav_vector_metrics(
    a: Mapping[str, float],
    b: Mapping[str, float],
    nodes: Sequence[str] | None = None,
) -> dict[str, float]:
    """MAE / RMSE / max|Δ| between two per-node NAV dicts."""
    if nodes is None:
        nodes = sorted(a.keys())
    diffs = [abs(a[n] - b[n]) for n in nodes]
    if not diffs:
        return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0}
    arr = np.asarray(diffs, dtype=float)
    return {
        "mae": float(arr.mean()),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "max_abs": float(arr.max()),
    }


def timed(fn: Callable[..., Any], *args, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, wall_seconds)``."""
    t0 = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - t0


def timing_summary_line(label: str, vals: Sequence[float]) -> str:
    """One-line total / mean / median / max timing summary."""
    if not vals:
        return f"  {label}: skipped / empty"
    return (
        f"  {label}: n={len(vals)}  total={sum(vals):.3f}s  "
        f"mean={np.mean(vals):.4f}s  median={np.median(vals):.4f}s  "
        f"max={max(vals):.3f}s"
    )


def to_nav_predictions(
    preds: Mapping[Assignment, Mapping[str, float]],
) -> dict[Assignment, NAVPrediction]:
    """Wrap raw NAV dicts as ``NAVPrediction`` objects for scoring."""
    return {
        fixes: NAVPrediction(fixes=fixes, average_values=dict(avg))
        for fixes, avg in preds.items()
    }


def discover_entity_bases(
    node_names: Sequence[str],
    experiments: Iterable[Any],
) -> list[str]:
    """Bases with ≥2 siblings, plus bases behind observed ``*_avg`` names."""
    names = list(node_names)
    bases: set[str] = set()
    for exp in experiments:
        obs = exp.observed
        if obs.endswith(_AVG_SUFFIX):
            base = obs[: -len(_AVG_SUFFIX)]
            if base and _related_node_names(base, names):
                bases.add(base)

    suffix_re = re.compile(r"^(.+?)_(?:\d+|[X-Z])$")
    candidates: set[str] = set(names)
    for name in names:
        m = suffix_re.match(name)
        if m:
            candidates.add(m.group(1))
    for base in candidates:
        related = _related_node_names(base, names)
        if len(related) >= 2:
            bases.add(base)

    return sorted(bases)


def entity_average_nav(
    nav: Mapping[str, float],
    bases: Sequence[str],
) -> dict[str, float]:
    """Map each entity base → mean of sibling node NAVs."""
    out: dict[str, float] = {}
    for base in bases:
        avg_name = f"{base}{_AVG_SUFFIX}"
        val = _average_related_values(nav, avg_name)
        if val is None:
            raise ValueError(f"entity average for {base!r} hit a None sibling NAV")
        out[base] = float(val)
    return out


def discover_full_entities(
    node_names: Sequence[str],
    experiments: Iterable[Any],
) -> tuple[list[str], list[str]]:
    """Merged multi-sibling bases plus singleton nodes as their own entities.

    Returns
    -------
    merged_bases
        Bases that average ≥2 related nodes (same as ``discover_entity_bases``).
    entity_keys
        ``merged_bases`` followed by sorted singletons (nodes not covered by
        any merged base's related set).
    """
    names = list(node_names)
    merged_bases = discover_entity_bases(names, experiments)
    covered: set[str] = set()
    for base in merged_bases:
        covered.update(_related_node_names(base, names))
    singletons = sorted(n for n in names if n not in covered)
    return merged_bases, list(merged_bases) + singletons


def entity_nav_dict(
    nav: Mapping[str, float],
    entity_keys: Sequence[str],
    merged_bases: Sequence[str] | set[str],
) -> dict[str, float]:
    """Entity NAV: sibling mean for merged bases, identity for singletons."""
    merged = set(merged_bases)
    out: dict[str, float] = {}
    for key in entity_keys:
        if key in merged:
            val = _average_related_values(nav, f"{key}{_AVG_SUFFIX}")
            if val is None:
                raise ValueError(f"entity average for {key!r} hit a None sibling NAV")
            out[key] = float(val)
        else:
            out[key] = float(nav[key])
    return out


def entity_from_node_array(
    node_arr: np.ndarray,
    node_names: Sequence[str],
    entity_keys: Sequence[str],
    merged_bases: Sequence[str] | set[str],
) -> np.ndarray:
    """Map node-aligned array to entities (sibling mean / identity).

    ``node_arr`` is ``(n_nodes,)`` or ``(n_nodes, ...)``; output matches the
    trailing dimensions with leading axis ``len(entity_keys)``.
    """
    if node_arr.shape[0] != len(node_names):
        raise ValueError(
            f"node_arr leading dim {node_arr.shape[0]} != len(node_names) "
            f"{len(node_names)}"
        )
    name_to_i = {n: i for i, n in enumerate(node_names)}
    merged = set(merged_bases)
    trailing = node_arr.shape[1:]
    out = np.empty((len(entity_keys),) + trailing, dtype=np.float64)
    for e, key in enumerate(entity_keys):
        if key in merged:
            related = _related_node_names(key, node_names)
            if not related:
                raise ValueError(f"no related nodes for merged entity {key!r}")
            idxs = [name_to_i[n] for n in related]
            out[e] = node_arr[idxs].mean(axis=0)
        else:
            out[e] = node_arr[name_to_i[key]]
    return out


def pooled_delta_entity_stats(cloud: np.ndarray) -> dict[str, np.ndarray]:
    """Per-entity stats over a pooled signed-Δ cloud ``(n_entities, n_points)``."""
    if cloud.ndim != 2:
        raise ValueError(f"cloud must be 2d, got shape {cloud.shape}")
    abs_c = np.abs(cloud)
    return {
        "median_abs": np.median(abs_c, axis=1),
        "mean_abs": abs_c.mean(axis=1),
        "max_abs": abs_c.max(axis=1),
        "rmse": np.sqrt(np.mean(cloud**2, axis=1)),
        "mean_signed": cloud.mean(axis=1),
    }


def summarize_across_entities(per_entity: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Mean / median / max of each per-entity statistic across entities."""
    out: dict[str, float] = {}
    for name, arr in per_entity.items():
        a = np.asarray(arr, dtype=float)
        out[f"{name}_mean"] = float(a.mean()) if a.size else float("nan")
        out[f"{name}_median"] = float(np.median(a)) if a.size else float("nan")
        out[f"{name}_max"] = float(a.max()) if a.size else float("nan")
    return out
