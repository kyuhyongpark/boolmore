#!/usr/bin/env python3
"""Batch consensus view of per-group agreement Δ across seeds.

For each seed's best ``*_score.csv`` in a batch run folder, computes group Δs
(observed node, cell line, intervened node) against a shared batch baseline:

    Δ = mean(group agreement) − batch mean agreement

where batch mean agreement is the average of each seed's overall mean
agreement (equal weight per seed). Unlike ``analyze_score.py``, deltas are
not centered on each model's own mean.

Stacks those vectors into a models × features signature, prints consensus
tables (median Δ, spread, sign agreement), and writes:

* strip overview — Δ across seeds for observed / cell / intervened
* heatmap overview — seed × group Δ for the same three facets
* optional long CSV of every (seed, facet, group) Δ

Higher ``n`` is used to rank which groups matter and to weight signature
distance; group ``n`` is typically identical across seeds (shared data.csv).

Edit the knobs below, then::

    python boolmore/scripts/analyze_score_batch.py

Or pass a batch folder on the CLI::

    python boolmore/scripts/analyze_score_batch.py \\
        case_study/breast_cancer/runs/20261119T130225Z
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from boolmore.io.paths import project_root

# --- edit these to analyze a different batch ---
# RUN_DIR = "case_study/breast_cancer/runs/20261118T231453Z"
# RUN_DIR = "case_study/breast_cancer/runs/20261119T130225Z"
RUN_DIR = "case_study/breast_cancer/runs/20261123T163416Z"
DATA = None  # None → batch data.csv / analyze_score default
MIN_N = 10
TOP = 0  # 0 = all groups (pass min_n); N = cap tables / strip / heatmap columns
PLOT = True
# ------------------------------------------------

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
_EPS = 1e-12

# Shared diverging Δ palette (ColorBrewer RdBu): red = below batch mean / worse,
# blue = above batch mean / better. Strips and heatmaps both use this cmap.
DELTA_CMAP = "RdBu"


def _delta_norm(values: Sequence[float]) -> TwoSlopeNorm:
    finite = [v for v in values if not math.isnan(v)]
    vmax = max((abs(v) for v in finite), default=0.05)
    vmax = max(vmax, 0.05)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _delta_scatter_kwargs(
    deltas: Sequence[float],
    *,
    norm: TwoSlopeNorm | None = None,
) -> dict:
    """Face colors from the shared RdBu scale (red − / blue +).

    Near-zero points are pale; a thin dark edge keeps them visible.
    """
    nrm = norm if norm is not None else _delta_norm(deltas)
    return {
        "c": list(deltas),
        "cmap": DELTA_CMAP,
        "norm": nrm,
        "edgecolors": "#333333",
        "linewidths": 0.55,
    }


def _load_analyze_score():
    """Import sibling ``analyze_score.py`` without requiring package install."""
    path = Path(__file__).resolve().parent / "analyze_score.py"
    spec = importlib.util.spec_from_file_location("analyze_score", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load analyze_score from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / annotations resolve cleanly.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ascore = _load_analyze_score()


@dataclass(frozen=True)
class SeedRef:
    seed: int
    score_path: Path
    label: str  # e.g. seed_000


@dataclass
class FacetEntry:
    seed: int
    seed_label: str
    facet: str
    group: str
    n: int
    mean_agreement: float
    delta_agreement: float  # mean_agreement − batch mean agreement
    overall_mean: float  # batch mean agreement (shared across seeds)


@dataclass
class ConsensusRow:
    facet: str
    group: str
    n: int  # rows in data (same across seeds when data is shared)
    n_seeds: int
    median_delta: float
    mean_delta: float
    stdev_delta: float
    frac_pos: float
    frac_neg: float
    frac_same_sign: float  # max(frac_pos, frac_neg); ties → 0.5
    significance: float  # |median| · log10(n+1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare per-group agreement Δ across seeds in a batch run folder."
        )
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help=f"Batch run folder (default: RUN_DIR knob ({RUN_DIR!r})).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data.csv (default: batch data.csv if present).",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=None,
        help=f"Drop groups with n < this (default: {MIN_N}).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help=(
            "Cap groups in tables / strips / heatmaps "
            f"(0 = all; default: {TOP})."
        ),
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing figures.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional long-form CSV of every (seed, facet, group) Δ.",
    )
    return p.parse_args(argv)


def _resolve_run_dir(raw: Path | str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        path = p.resolve()
    else:
        under_root = (project_root() / p).resolve()
        under_cwd = (Path.cwd() / p).resolve()
        if under_root.is_dir():
            path = under_root
        elif under_cwd.is_dir():
            path = under_cwd
        else:
            path = under_root
    if not path.is_dir():
        raise SystemExit(f"run directory not found: {path}")
    return path


def _resolve_data(run_dir: Path, cli: Path | None) -> Path:
    if cli is not None:
        return ascore._resolve_path(cli)
    batch = run_dir / "data.csv"
    if batch.is_file():
        return batch.resolve()
    if DATA:
        return ascore._resolve_path(DATA)
    return ascore._resolve_path("case_study/breast_cancer/data.csv")


def find_seed_scores(run_dir: Path) -> list[SeedRef]:
    """Pick one ``*_score.csv`` per ``seed_NNN/`` (lexicographically last if many)."""
    found: list[SeedRef] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _SEED_DIR_RE.match(child.name)
        if not m:
            continue
        scores = sorted(child.glob("*_score.csv"))
        if not scores:
            print(f"warning: no *_score.csv in {child.name}")
            continue
        if len(scores) > 1:
            print(
                f"warning: {len(scores)} score files in {child.name}; "
                f"using {scores[-1].name}"
            )
        found.append(
            SeedRef(seed=int(m.group(1)), score_path=scores[-1], label=child.name)
        )
    found.sort(key=lambda s: s.seed)
    return found


def _overall_means(rows: Sequence) -> tuple[float, float]:
    agrees = [r.agreement for r in rows]
    hiers = [
        r.hierarchy_agreement
        for r in rows
        if not math.isnan(r.hierarchy_agreement)
    ]
    return ascore._mean(agrees), ascore._mean(hiers)


def _facet_stats(rows: Sequence, mean_a: float, mean_h: float, min_n: int):
    """Return dict facet → list[GroupStats] (min_n applied)."""
    return {
        "observed": ascore._stats_for_key(
            rows,
            lambda r: r.observed or "no_obs",
            mean_a,
            mean_h,
            min_n=min_n,
        ),
        "cell_line": ascore._stats_for_key(
            rows, ascore._cell_group_label, mean_a, mean_h, min_n=min_n
        ),
        "intervened": ascore.intervention_node_stats(
            rows, mean_a, mean_h, min_n
        ),
    }


def collect_entries(
    seeds: Sequence[SeedRef],
    data_by_id: dict,
    min_n: int,
) -> tuple[list[FacetEntry], dict[int, float], float]:
    """Load every seed score; return long entries, per-seed means, batch mean.

    Deltas use the batch mean agreement (average of per-seed overall means),
    not each seed's own overall mean.
    """
    loaded: list[tuple[SeedRef, Sequence, float]] = []
    overall_by_seed: dict[int, float] = {}
    for ref in seeds:
        rows = ascore.load_score_rows(ref.score_path, data_by_id)
        if not rows:
            print(f"warning: empty score file {ref.score_path}")
            continue
        mean_a, mean_h = _overall_means(rows)
        overall_by_seed[ref.seed] = mean_a
        loaded.append((ref, rows, mean_h))
        print(
            f"  {ref.label}: overall agree={ascore._fmt(mean_a)}  "
            f"rows={len(rows)}  file={ref.score_path.name}"
        )

    if not overall_by_seed:
        return [], {}, float("nan")

    batch_mean = statistics.fmean(overall_by_seed.values())
    print(
        f"  batch mean agreement (Δ baseline): {ascore._fmt(batch_mean)}  "
        f"(mean of {len(overall_by_seed)} seed overalls)"
    )

    entries: list[FacetEntry] = []
    for ref, rows, mean_h in loaded:
        # Agreement Δ vs batch mean; hierarchy baseline stays per-seed (unused).
        for facet, stats in _facet_stats(rows, batch_mean, mean_h, min_n).items():
            for s in stats:
                entries.append(
                    FacetEntry(
                        seed=ref.seed,
                        seed_label=ref.label,
                        facet=facet,
                        group=s.label,
                        n=s.n,
                        mean_agreement=s.mean_agreement,
                        delta_agreement=s.delta_agreement,
                        overall_mean=batch_mean,
                    )
                )
    return entries, overall_by_seed, batch_mean


def _sign_fracs(deltas: Sequence[float]) -> tuple[float, float, float]:
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    n_pos = sum(1 for d in deltas if d > _EPS)
    n_neg = sum(1 for d in deltas if d < -_EPS)
    frac_pos = n_pos / n
    frac_neg = n_neg / n
    frac_same = max(frac_pos, frac_neg) if (n_pos + n_neg) else 0.5
    return frac_pos, frac_neg, frac_same


def consensus_table(entries: Sequence[FacetEntry], facet: str) -> list[ConsensusRow]:
    by_group: dict[str, list[FacetEntry]] = defaultdict(list)
    for e in entries:
        if e.facet == facet:
            by_group[e.group].append(e)

    rows: list[ConsensusRow] = []
    for group, members in by_group.items():
        deltas = [m.delta_agreement for m in members]
        ns = {m.n for m in members}
        n_rep = members[0].n
        if len(ns) > 1:
            # Rare: different score row subsets; take median n.
            n_rep = int(statistics.median(m.n for m in members))
        frac_pos, frac_neg, frac_same = _sign_fracs(deltas)
        med = statistics.median(deltas)
        rows.append(
            ConsensusRow(
                facet=facet,
                group=group,
                n=n_rep,
                n_seeds=len(members),
                median_delta=med,
                mean_delta=statistics.fmean(deltas),
                stdev_delta=(
                    statistics.stdev(deltas) if len(deltas) >= 2 else float("nan")
                ),
                frac_pos=frac_pos,
                frac_neg=frac_neg,
                frac_same_sign=frac_same,
                significance=abs(med) * math.log10(n_rep + 1.0),
            )
        )
    rows.sort(key=lambda r: (-r.significance, -abs(r.median_delta), r.group))
    return rows


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{100.0 * x:.0f}%"


def _take_groups(rows: Sequence[ConsensusRow], top: int) -> list[ConsensusRow]:
    """Return all rows when ``top <= 0``, else the first ``top``."""
    if top <= 0:
        return list(rows)
    return list(rows[:top])


def print_consensus(title: str, rows: Sequence[ConsensusRow], top: int) -> None:
    print(f"\n=== {title} (ordered by |median Δ|·log10(n+1)) ===")
    print(
        "− / + = share of seeds with Δ<0 / Δ>0; "
        "σ = stdev of Δ across seeds"
    )
    width = 28
    header = (
        f"{'group':<{width}} {'n':>5} {'seeds':>5} {'medΔ':>7} {'meanΔ':>7} "
        f"{'σΔ':>7} {'−':>4} {'+':>4} {'sig':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in _take_groups(rows, top):
        label = r.group if len(r.group) <= width else r.group[: width - 3] + "..."
        print(
            f"{label:<{width}} {r.n:5d} {r.n_seeds:5d} "
            f"{ascore._fmt(r.median_delta):>7} {ascore._fmt(r.mean_delta):>7} "
            f"{ascore._fmt(r.stdev_delta):>7} "
            f"{_fmt_pct(r.frac_neg):>4} {_fmt_pct(r.frac_pos):>4} "
            f"{ascore._fmt(r.significance):>6}"
        )


def print_seed_similarity(
    entries: Sequence[FacetEntry],
    facet: str,
    seeds: Sequence[SeedRef],
) -> None:
    """n-weighted pairwise correlation of Δ vectors; print mean off-diagonal r."""
    groups = sorted({e.group for e in entries if e.facet == facet})
    if len(groups) < 2 or len(seeds) < 2:
        return
    # Representative n per group (should be constant).
    n_by_group: dict[str, int] = {}
    for e in entries:
        if e.facet == facet and e.group not in n_by_group:
            n_by_group[e.group] = e.n
    weights = [math.log10(n_by_group[g] + 1.0) for g in groups]

    vectors: dict[int, list[float]] = {}
    lookup = {
        (e.seed, e.group): e.delta_agreement
        for e in entries
        if e.facet == facet
    }
    for ref in seeds:
        vec = [lookup.get((ref.seed, g), float("nan")) for g in groups]
        if any(math.isnan(x) for x in vec):
            continue
        vectors[ref.seed] = vec

    seed_ids = sorted(vectors)
    if len(seed_ids) < 2:
        return

    def wcorr(a: list[float], b: list[float]) -> float:
        # Weighted Pearson with weights ∝ log10(n+1).
        wsum = sum(weights)
        if wsum <= 0:
            return float("nan")
        ma = sum(w * x for w, x in zip(weights, a)) / wsum
        mb = sum(w * x for w, x in zip(weights, b)) / wsum
        cov = sum(w * (x - ma) * (y - mb) for w, x, y in zip(weights, a, b))
        va = sum(w * (x - ma) ** 2 for w, x in zip(weights, a))
        vb = sum(w * (y - mb) ** 2 for w, y in zip(weights, b))
        if va <= _EPS or vb <= _EPS:
            return float("nan")
        return cov / math.sqrt(va * vb)

    rs: list[float] = []
    for i, si in enumerate(seed_ids):
        for sj in seed_ids[i + 1 :]:
            r = wcorr(vectors[si], vectors[sj])
            if not math.isnan(r):
                rs.append(r)
    if not rs:
        return
    print(
        f"\n=== {facet}: n-weighted Δ signature correlation across seeds ==="
    )
    print(
        f"pairwise r: median={ascore._fmt(statistics.median(rs))}  "
        f"mean={ascore._fmt(statistics.fmean(rs))}  "
        f"min={ascore._fmt(min(rs))}  max={ascore._fmt(max(rs))}  "
        f"(n_pairs={len(rs)}, dims={len(groups)})"
    )


def write_long_csv(path: Path, entries: Sequence[FacetEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "seed_label",
        "facet",
        "group",
        "n",
        "mean_agreement",
        "delta_agreement",
        "overall_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            w.writerow(
                {
                    "seed": e.seed,
                    "seed_label": e.seed_label,
                    "facet": e.facet,
                    "group": e.group,
                    "n": e.n,
                    "mean_agreement": e.mean_agreement,
                    "delta_agreement": e.delta_agreement,
                    "overall_mean": e.overall_mean,
                }
            )
    print(f"\nwrote long CSV: {path}")


def _short(text: str, max_len: int = 24) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _marker_sizes(ns: Sequence[int], *, lo: float = 18.0, hi: float = 90.0) -> list[float]:
    if not ns:
        return []
    n_min, n_max = min(ns), max(ns)
    if n_max <= n_min:
        return [0.5 * (lo + hi)] * len(ns)
    out = []
    for n in ns:
        t = math.log10(n + 1.0) - math.log10(n_min + 1.0)
        t /= math.log10(n_max + 1.0) - math.log10(n_min + 1.0)
        out.append(lo + t * (hi - lo))
    return out


def _cluster_leaf_order(rows: Sequence[Sequence[float]]) -> list[int]:
    """Leaf order that groups similar vectors.

    Uses average-linkage hierarchical clustering on Euclidean distance when
    SciPy is available; otherwise sorts by the first principal component.
    ``rows[i]`` is the feature vector for item i (NaNs treated as 0).
    """
    n = len(rows)
    if n <= 1:
        return list(range(n))

    try:
        import numpy as np
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
    except ImportError:
        np = None  # type: ignore[assignment]
        linkage = None  # type: ignore[assignment]

    if linkage is not None:
        arr = np.asarray(rows, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        if np.allclose(arr, arr[0:1]):
            return list(range(n))
        Z = linkage(pdist(arr, metric="euclidean"), method="average")
        return [int(i) for i in leaves_list(Z)]

    # Fallback: order by projection onto the first singular vector.
    try:
        import numpy as np
    except ImportError:
        return list(range(n))
    arr = np.asarray(rows, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(arr, full_matrices=False)
        scores = arr @ vt[0]
    except np.linalg.LinAlgError:
        return list(range(n))
    return [int(i) for i in np.argsort(scores)]


def _build_clustered_heatmaps(
    entries: Sequence[FacetEntry],
    by_facet: dict[str, list[ConsensusRow]],
    seeds: Sequence[SeedRef],
    *,
    top: int,
    facets: Sequence[tuple[str, str]],
) -> tuple[list[str], list[tuple[str, list[str], list[list[float]]] | None]]:
    """Cluster seeds jointly; cluster groups per facet; return labels + panels.

    Each panel is ``(title, groups, matrix)`` with rows in the shared seed
    order and columns in that facet's group-cluster leaf order, or ``None``
    if that facet has no groups.
    """
    # Features: (facet, group, n) — membership for joint seed clustering.
    features: list[tuple[str, str, int]] = []
    for facet, _title in facets:
        for r in _take_groups(by_facet[facet], top):
            features.append((facet, r.group, r.n))

    seed_ids = [s.seed for s in seeds]
    seed_labels_all = [s.label for s in seeds]
    lookup = {(e.facet, e.seed, e.group): e.delta_agreement for e in entries}

    # Keep seeds that have at least one finite Δ somewhere.
    keep: list[int] = []
    weighted_rows: list[list[float]] = []
    for i, sid in enumerate(seed_ids):
        raw = [lookup.get((f, sid, g), float("nan")) for f, g, _n in features]
        if not any(not math.isnan(v) for v in raw):
            continue
        keep.append(i)
        # Weight by log10(n+1) so high-n groups drive the clustering more.
        weighted_rows.append(
            [
                (0.0 if math.isnan(v) else v) * math.log10(n + 1.0)
                for v, (_f, _g, n) in zip(raw, features)
            ]
        )

    if not keep:
        return [], [None] * len(facets)

    order_local = _cluster_leaf_order(weighted_rows)
    ordered_idx = [keep[j] for j in order_local]
    seed_labels = [seed_labels_all[i] for i in ordered_idx]
    ordered_seeds = [seed_ids[i] for i in ordered_idx]

    panels: list[tuple[str, list[str], list[list[float]]] | None] = []
    for facet, title in facets:
        groups = [r.group for r in _take_groups(by_facet[facet], top)]
        if not groups:
            panels.append(None)
            continue
        # Cluster groups on their across-seed Δ vectors (per facet).
        group_vectors = [
            [lookup.get((facet, sid, g), float("nan")) for sid in ordered_seeds]
            for g in groups
        ]
        group_order = _cluster_leaf_order(group_vectors)
        groups = [groups[j] for j in group_order]
        matrix = [
            [lookup.get((facet, sid, g), float("nan")) for g in groups]
            for sid in ordered_seeds
        ]
        panels.append((title, groups, matrix))
    return seed_labels, panels


def write_overview_heatmap(
    path: Path,
    entries: Sequence[FacetEntry],
    by_facet: dict[str, list[ConsensusRow]],
    seeds: Sequence[SeedRef],
    *,
    top: int,
) -> Path:
    """Three facets as heatmap panels (observed / cell / intervened)."""
    facets = (
        ("observed", "Observed node"),
        ("cell_line", "Cell line"),
        ("intervened", "Intervened node"),
    )
    seed_labels, panels = _build_clustered_heatmaps(
        entries, by_facet, seeds, top=top, facets=facets
    )
    all_vals = [
        v
        for payload in panels
        if payload is not None
        for row in payload[2]
        for v in row
        if not math.isnan(v)
    ]

    fig_h = max(5.5, 0.18 * max(len(seed_labels), 1) + 2.2)
    fig, axes = plt.subplots(1, 3, figsize=(16, fig_h), constrained_layout=True)
    norm = _delta_norm(all_vals)
    last_im = None

    for ax, payload in zip(axes, panels):
        if payload is None:
            ax.set_title("(no groups)")
            ax.text(0.5, 0.5, "(no groups)", ha="center", va="center", transform=ax.transAxes)
            continue
        title, groups, matrix = payload
        # RdBu: red = negative Δ (worse), blue = positive Δ (better).
        last_im = ax.imshow(matrix, aspect="auto", cmap=DELTA_CMAP, norm=norm)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            [_short(g, 14) for g in groups], rotation=55, ha="right", fontsize=6.5
        )
        ax.set_yticks(range(len(seed_labels)))
        ax.set_title(title)

    # Seed labels only on the leftmost drawn panel (shared row order).
    labeled = False
    for ax, payload in zip(axes, panels):
        if payload is None:
            ax.set_yticklabels([])
            continue
        if not labeled:
            ax.set_yticklabels(seed_labels, fontsize=6.5)
            ax.set_ylabel("seed (clustered)")
            labeled = True
        else:
            ax.set_yticklabels([])

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Δ agreement  (red − / blue +)")

    fig.suptitle(
        "Batch signature heatmap: seed × group Δ\n"
        "red = below batch mean (worse), blue = above (better); "
        "rows = seeds clustered on joint n-weighted Δ signature (shared); "
        "columns = groups clustered within each facet",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_overview_strip(
    path: Path,
    entries: Sequence[FacetEntry],
    by_facet: dict[str, list[ConsensusRow]],
    *,
    top: int,
) -> Path:
    """2×2-ish: three facets as strip panels (observed / cell / intervened)."""
    panels = (
        ("observed", "Observed node"),
        ("cell_line", "Cell line"),
        ("intervened", "Intervened node"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), constrained_layout=True)

    # Shared color scale across the three panels.
    overview_deltas: list[float] = []
    panel_data: list[tuple] = []
    for facet, title in panels:
        consensus = _take_groups(by_facet[facet], top)
        groups = [r.group for r in consensus]
        by_group: dict[str, list[FacetEntry]] = defaultdict(list)
        for e in entries:
            if e.facet == facet and e.group in {r.group for r in consensus}:
                by_group[e.group].append(e)
                overview_deltas.append(e.delta_agreement)
        panel_data.append((facet, title, consensus, groups, by_group))
    overview_norm = _delta_norm(overview_deltas)

    for ax, (_facet, title, consensus, groups, by_group) in zip(axes, panel_data):
        if not groups:
            ax.set_title(title)
            ax.text(0.5, 0.5, "(no groups)", ha="center", va="center", transform=ax.transAxes)
            continue
        y_pos = {g: len(groups) - 1 - i for i, g in enumerate(groups)}
        rng = random.Random(title)
        for g in groups:
            members = by_group[g]
            xs = [m.delta_agreement for m in members]
            ys = [y_pos[g] + rng.uniform(-0.15, 0.15) for _ in members]
            sizes = _marker_sizes([m.n for m in members])
            ax.scatter(
                xs,
                ys,
                s=sizes,
                alpha=0.85,
                zorder=2,
                **_delta_scatter_kwargs(xs, norm=overview_norm),
            )
            med = next(r.median_delta for r in consensus if r.group == g)
            ax.plot(
                [med],
                [y_pos[g]],
                marker="|",
                markersize=12,
                markeredgewidth=1.8,
                color="#111",
                zorder=3,
            )

        ax.axvline(0.0, color="#888", lw=0.8, ls="--", zorder=0)
        ax.set_yticks([y_pos[g] for g in groups])
        n_by_group = {r.group: r.n for r in consensus}
        ax.set_yticklabels(
            [f"{_short(g, 18)} (n={n_by_group[g]})" for g in groups],
            fontsize=7,
        )
        ax.set_xlabel("Δ agreement")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)

    # Shared x range so panel widths are comparable.
    if overview_deltas:
        half = max(abs(min(overview_deltas)), abs(max(overview_deltas)))
        pad = max(0.02, 0.06 * half)
        xlim = (-half - pad, half + pad)
        for ax in axes:
            ax.set_xlim(xlim)

    fig.suptitle(
        "Batch signature: Δ across seeds\n"
        "red = below batch mean (worse), blue = above (better); "
        "black tick = median; size ∝ log(n); "
        "y-label shows group n; ordered by |median Δ|·log10(n+1)",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = _resolve_run_dir(args.run_dir if args.run_dir is not None else RUN_DIR)
    data_path = _resolve_data(run_dir, args.data)
    min_n = MIN_N if args.min_n is None else args.min_n
    top = TOP if args.top is None else args.top
    do_plot = PLOT and not args.no_plot

    print(f"run_dir: {run_dir}")
    print(f"data:    {data_path}")
    print(f"min_n={min_n}  top={top if top > 0 else 'all'}")

    seeds = find_seed_scores(run_dir)
    if not seeds:
        raise SystemExit(f"no seed_*/ *_score.csv under {run_dir}")
    print(f"seeds with scores: {len(seeds)}")

    data_by_id = ascore.load_data_by_id(data_path)
    print("loading per-seed group Δs …")
    entries, overall_by_seed, batch_mean = collect_entries(
        seeds, data_by_id, min_n
    )
    if not entries:
        raise SystemExit("no group entries collected")

    overalls = list(overall_by_seed.values())
    print(
        f"\nper-seed overall agreement: "
        f"median={ascore._fmt(statistics.median(overalls))}  "
        f"mean={ascore._fmt(batch_mean)}  "
        f"σ={ascore._fmt(statistics.stdev(overalls) if len(overalls) >= 2 else float('nan'))}"
    )
    print(f"Δ baseline = batch mean agreement: {ascore._fmt(batch_mean)}")

    facets = ("observed", "cell_line", "intervened")
    titles = {
        "observed": "by observed node",
        "cell_line": "by Cell line",
        "intervened": "by intervened node",
    }
    by_facet = {f: consensus_table(entries, f) for f in facets}
    for f in facets:
        print_consensus(titles[f], by_facet[f], top)
        print_seed_similarity(entries, f, seeds)

    out_dir = run_dir / "score_batch_analysis"
    if args.out_csv is not None:
        out_csv = args.out_csv
        if not out_csv.is_absolute():
            out_csv = (Path.cwd() / out_csv).resolve()
    else:
        out_csv = out_dir / "delta_long.csv"
    write_long_csv(out_csv, entries)

    # Consensus CSV (all facets).
    cons_path = out_dir / "consensus.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with cons_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "facet",
            "group",
            "n",
            "n_seeds",
            "median_delta",
            "mean_delta",
            "stdev_delta",
            "frac_pos",
            "frac_neg",
            "frac_same_sign",
            "significance",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for facet in facets:
            for r in by_facet[facet]:
                w.writerow({k: getattr(r, k) for k in fields})
    print(f"wrote consensus CSV: {cons_path}")

    if do_plot:
        overview = write_overview_strip(
            out_dir / "delta_strips.png",
            entries,
            by_facet,
            top=top,
        )
        print(f"wrote strip overview: {overview}")
        heat = write_overview_heatmap(
            out_dir / "delta_heatmaps.png",
            entries,
            by_facet,
            seeds,
            top=top,
        )
        print(f"wrote heatmap overview: {heat}")
        print(f"\nfigures in {out_dir}")


if __name__ == "__main__":
    main()
