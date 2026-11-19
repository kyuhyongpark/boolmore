#!/usr/bin/env python3
"""Analyze a NAV ``_score.csv`` joined with experiment metadata from ``data.csv``.

Derives ``hierarchy_agreement = score / weight`` so raw agreement can be
distinguished from hierarchy-amplified agreement. Prints grouped reports,
intervention ladders, mismatch lists, and absolute/comparative statement
distributions. Writes a volcano-style summary figure next to the score CSV.
Optionally writes an annotated CSV.

Edit the knobs below, then::

    python boolmore/scripts/analyze_score.py

Or pass a score file on the CLI (overrides ``SCORE``)::

    python boolmore/scripts/analyze_score.py \\
        case_study/breast_cancer/runs/20261118T231453Z/seed_017/seed_017_9818_gen108_score.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from boolmore.io.paths import project_root

# --- edit these to analyze a different score file ---
SCORE = (
    "case_study/breast_cancer/runs/20261118T231453Z/"
    "seed_017/seed_017_9818_gen108_score.csv"
)
DATA = "case_study/breast_cancer/data.csv"  # None → auto-detect / default
MIN_N = 10
MIN_DELTA = 0.05
TOP = 30
PLOT = True
PLOT_LABEL_N = 8  # label this many most-significant points per subplot
PLOT_LABEL_MIN_N = 10  # only label groups with at least this many rows
# ----------------------------------------------------

_META_COLS = (
    "Cell line",
    "Culture",
    "Culture notes",
    "ER",
    "HER2",
    "Model",
)

_COMPARATIVE_RE = re.compile(
    r"(?P<dir>higher|lower)\s+than\s+(?P<other>\d+)",
    re.IGNORECASE,
)
_ASSIGN_RE = re.compile(r"([^=,\s]+)\s*=\s*([01])")

_EPS = 1e-12
_BINS: tuple[tuple[str, Callable[[float], bool]], ...] = (
    ("{0}", lambda x: x <= _EPS),
    ("(0,0.25]", lambda x: _EPS < x <= 0.25),
    ("(0.25,0.5]", lambda x: 0.25 < x <= 0.5),
    ("(0.5,0.75]", lambda x: 0.5 < x <= 0.75),
    ("(0.75,1)", lambda x: 0.75 < x < 1.0 - _EPS),
    ("{1}", lambda x: x >= 1.0 - _EPS),
)


@dataclass
class Row:
    """One score row joined with optional metadata."""

    id: int
    weight: float
    hierarchy: int
    context: str
    intervention: str
    fixes: str
    observed: str
    outcome: str
    outcome_range: str
    predict_value: float | None
    agreement: float
    score: float
    hierarchy_agreement: float
    cell_line: str = ""
    culture: str = ""
    culture_notes: str = ""
    er: str = ""
    her2: str = ""
    model: str = ""
    is_comparative: bool = False
    other_id: int | None = None
    intervention_map: dict[str, int] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze a NAV _score.csv joined with data.csv metadata "
            "(Cell line, Culture, …)."
        )
    )
    p.add_argument(
        "score",
        nargs="?",
        type=Path,
        default=None,
        help=f"Path to *_score.csv (default: SCORE knob ({SCORE!r})).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Path to data.csv. Default: batch runs/.../data.csv if present, "
            f"else DATA knob ({DATA!r})."
        ),
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=None,
        help=f"Minimum rows for combo / per-drug tables (default: {MIN_N}).",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help=(
            "Minimum |mean − overall| to print a combo "
            f"(default: {MIN_DELTA})."
        ),
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help=f"Cap for mismatch / ladder listings (default: {TOP}).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional annotated CSV path (score + metadata + hierarchy_agreement).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the volcano figure next to the score CSV.",
    )
    p.add_argument(
        "--plot-labels",
        type=int,
        default=None,
        help=f"How many points to label per subplot (default: {PLOT_LABEL_N}).",
    )
    return p.parse_args(argv)


def _resolve_path(raw: Path | str, *, must_exist: bool = True) -> Path:
    p = Path(raw)
    if p.is_absolute():
        path = p.resolve()
    else:
        under_root = (project_root() / p).resolve()
        under_cwd = (Path.cwd() / p).resolve()
        if under_root.exists():
            path = under_root
        elif under_cwd.exists():
            path = under_cwd
        else:
            path = under_root
    if must_exist and not path.is_file():
        raise SystemExit(f"file not found: {path}")
    return path


def _find_batch_data(score_path: Path) -> Path | None:
    """Walk parents of ``score_path`` for a batch ``data.csv`` under ``runs/``."""
    for parent in score_path.parents:
        candidate = parent / "data.csv"
        if candidate.is_file() and "runs" in parent.parts:
            return candidate
    return None


def _parse_assignment(text: str) -> dict[str, int]:
    if not text or not text.strip():
        return {}
    return {m.group(1): int(m.group(2)) for m in _ASSIGN_RE.finditer(text)}


def _parse_comparative(outcome: str) -> tuple[bool, int | None]:
    m = _COMPARATIVE_RE.search(outcome or "")
    if not m:
        return False, None
    return True, int(m.group("other"))


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _three_way(xs: Sequence[float]) -> tuple[int, int, int]:
    zeros = partial = ones = 0
    for x in xs:
        if x <= _EPS:
            zeros += 1
        elif x >= 1.0 - _EPS:
            ones += 1
        else:
            partial += 1
    return zeros, partial, ones


def _bin_counts(xs: Sequence[float]) -> list[tuple[str, int]]:
    counts = [0] * len(_BINS)
    for x in xs:
        for i, (_, pred) in enumerate(_BINS):
            if pred(x):
                counts[i] += 1
                break
    return [(label, counts[i]) for i, (label, _) in enumerate(_BINS)]


def load_data_by_id(path: Path) -> dict[int, dict[str, str]]:
    by_id: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise SystemExit(f"data.csv missing 'id' column: {path}")
        for raw in reader:
            try:
                exp_id = int(raw["id"])
            except (KeyError, TypeError, ValueError):
                continue
            by_id[exp_id] = raw
    return by_id


def load_score_rows(score_path: Path, data_by_id: dict[int, dict[str, str]]) -> list[Row]:
    rows: list[Row] = []
    unmatched = 0
    with score_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "id",
            "weight",
            "hierarchy",
            "agreement",
            "score",
            "observed",
            "outcome",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(
                f"score CSV missing columns {sorted(required - set(reader.fieldnames or []))}"
            )
        for raw in reader:
            exp_id = int(raw["id"])
            weight = float(raw["weight"])
            agreement = float(raw["agreement"])
            score = float(raw["score"])
            hier_agree = (score / weight) if weight > 0 else float("nan")
            is_comp, other_id = _parse_comparative(raw.get("outcome") or "")
            meta = data_by_id.get(exp_id)
            if meta is None:
                unmatched += 1
                meta = {}
            pv_raw = (raw.get("predict_value") or "").strip()
            try:
                predict_value = float(pv_raw) if pv_raw else None
            except ValueError:
                predict_value = None
            intervention = raw.get("intervention") or ""
            rows.append(
                Row(
                    id=exp_id,
                    weight=weight,
                    hierarchy=int(float(raw.get("hierarchy") or 0)),
                    context=raw.get("context") or "",
                    intervention=intervention,
                    fixes=raw.get("fixes") or "",
                    observed=raw.get("observed") or "",
                    outcome=raw.get("outcome") or "",
                    outcome_range=raw.get("outcome_range") or "",
                    predict_value=predict_value,
                    agreement=agreement,
                    score=score,
                    hierarchy_agreement=hier_agree,
                    cell_line=(meta.get("Cell line") or "").strip(),
                    culture=(meta.get("Culture") or "").strip(),
                    culture_notes=(meta.get("Culture notes") or "").strip(),
                    er=(meta.get("ER") or "").strip(),
                    her2=(meta.get("HER2") or "").strip(),
                    model=(meta.get("Model") or "").strip(),
                    is_comparative=is_comp,
                    other_id=other_id,
                    intervention_map=_parse_assignment(intervention),
                )
            )
    if unmatched:
        print(f"warning: {unmatched} score row(s) had no matching data.csv id")
    joined = [r for r in rows if r.cell_line or r.culture or r.model]
    if joined:
        for col in ("Cell line", "Culture"):
            attr = _meta_attr(col)
            if not any(getattr(r, attr) for r in joined):
                print(f"warning: metadata column {col!r} is empty for all joined rows")
    return rows


def _meta_attr(col: str) -> str:
    return {
        "Cell line": "cell_line",
        "Culture": "culture",
        "Culture notes": "culture_notes",
        "ER": "er",
        "HER2": "her2",
        "Model": "model",
    }[col]


@dataclass
class GroupStats:
    label: str
    n: int
    sum_weight: float
    mean_agreement: float
    mean_hierarchy_agreement: float
    frac_perfect: float
    frac_hier_hurt: float
    delta_agreement: float = 0.0
    delta_hierarchy: float = 0.0
    member_ids: frozenset[int] = field(default_factory=frozenset)
    n_factors: int = 1


def _group_stats(
    label: str,
    rows: Sequence[Row],
    overall_mean_agree: float,
    overall_mean_hier: float,
    *,
    n_factors: int = 1,
) -> GroupStats:
    n = len(rows)
    agrees = [r.agreement for r in rows]
    hiers = [r.hierarchy_agreement for r in rows if not math.isnan(r.hierarchy_agreement)]
    mean_a = _mean(agrees)
    mean_h = _mean(hiers)
    return GroupStats(
        label=label,
        n=n,
        sum_weight=sum(r.weight for r in rows),
        mean_agreement=mean_a,
        mean_hierarchy_agreement=mean_h,
        frac_perfect=sum(1 for a in agrees if a >= 1.0 - _EPS) / n if n else float("nan"),
        frac_hier_hurt=(
            sum(1 for r in rows if r.hierarchy_agreement + _EPS < r.agreement) / n
            if n
            else float("nan")
        ),
        delta_agreement=mean_a - overall_mean_agree if n else float("nan"),
        delta_hierarchy=mean_h - overall_mean_hier if n else float("nan"),
        member_ids=frozenset(r.id for r in rows),
        n_factors=n_factors,
    )


def _print_group_table(
    title: str,
    stats: Sequence[GroupStats],
    overall_mean_agree: float,
    overall_mean_hier: float,
) -> None:
    print(f"\n=== {title} ===")
    print(
        f"(overall mean agreement={_fmt(overall_mean_agree)}, "
        f"hierarchy_agreement={_fmt(overall_mean_hier)}; "
        "sorted by |Δ agreement|)"
    )
    width = 56
    header = (
        f"{'group':<{width}} {'n':>5} {'Σw':>7} {'agree':>7} {'hier':>7} "
        f"{'Δagree':>7} {'Δhier':>7} {'perf%':>6} {'hurt%':>6}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        label = s.label if len(s.label) <= width else s.label[: width - 3] + "..."
        print(
            f"{label:<{width}} {s.n:5d} {_fmt(s.sum_weight, 1):>7} "
            f"{_fmt(s.mean_agreement):>7} {_fmt(s.mean_hierarchy_agreement):>7} "
            f"{_fmt(s.delta_agreement):>7} {_fmt(s.delta_hierarchy):>7} "
            f"{_fmt_pct(s.frac_perfect):>6} {_fmt_pct(s.frac_hier_hurt):>6}"
        )


def _stats_for_key(
    rows: Sequence[Row],
    key_fn: Callable[[Row], str],
    overall_mean_agree: float,
    overall_mean_hier: float,
    *,
    min_n: int = 1,
    empty_label: str = "(empty)",
) -> list[GroupStats]:
    buckets: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        key = key_fn(r) or empty_label
        buckets[key].append(r)
    stats = [
        _group_stats(label, group, overall_mean_agree, overall_mean_hier)
        for label, group in buckets.items()
        if len(group) >= min_n
    ]
    stats.sort(key=lambda s: (-abs(s.delta_agreement), -s.n, s.label))
    return stats


def print_overall(rows: Sequence[Row]) -> tuple[float, float]:
    agrees = [r.agreement for r in rows]
    hiers = [r.hierarchy_agreement for r in rows if not math.isnan(r.hierarchy_agreement)]
    mean_a = _mean(agrees)
    mean_h = _mean(hiers)
    z_a, p_a, o_a = _three_way(agrees)
    z_h, p_h, o_h = _three_way(hiers)
    n = len(rows)
    hurt = sum(1 for r in rows if r.hierarchy_agreement + _EPS < r.agreement)
    print("=== overall ===")
    print(f"n={n}  Σweight={_fmt(sum(r.weight for r in rows), 1)}  "
          f"Σscore={_fmt(sum(r.score for r in rows), 3)}")
    print(f"mean agreement={_fmt(mean_a)}  mean hierarchy_agreement={_fmt(mean_h)}")
    print(
        f"agreement:     zero={z_a} ({_fmt_pct(z_a / n)})  "
        f"partial={p_a} ({_fmt_pct(p_a / n)})  "
        f"perfect={o_a} ({_fmt_pct(o_a / n)})"
    )
    print(
        f"hier_agree:    zero={z_h} ({_fmt_pct(z_h / n)})  "
        f"partial={p_h} ({_fmt_pct(p_h / n)})  "
        f"perfect={o_h} ({_fmt_pct(o_h / n)})"
    )
    print(f"hierarchy-amplified hurts (hier < agreement): {hurt} ({_fmt_pct(hurt / n)})")
    return mean_a, mean_h


def intervention_node_stats(
    rows: Sequence[Row],
    overall_mean_agree: float,
    overall_mean_hier: float,
    min_n: int,
) -> list[GroupStats]:
    by_drug: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        for node in r.intervention_map:
            by_drug[node].append(r)
    stats = [
        _group_stats(name, group, overall_mean_agree, overall_mean_hier)
        for name, group in by_drug.items()
        if len(group) >= min_n
    ]
    stats.sort(key=lambda s: (-abs(s.delta_agreement), -s.n, s.label))
    return stats


def print_drug_reports(
    rows: Sequence[Row],
    overall_mean_agree: float,
    overall_mean_hier: float,
    min_n: int,
) -> list[GroupStats]:
    with_alp = [r for r in rows if r.intervention_map.get("Alpelisib") == 1]
    without_alp = [r for r in rows if r.intervention_map.get("Alpelisib") != 1]
    alp_stats = [
        _group_stats("Alpelisib present", with_alp, overall_mean_agree, overall_mean_hier),
        _group_stats("Alpelisib absent", without_alp, overall_mean_agree, overall_mean_hier),
    ]
    _print_group_table(
        "Alpelisib present vs absent",
        alp_stats,
        overall_mean_agree,
        overall_mean_hier,
    )

    drug_stats = intervention_node_stats(
        rows, overall_mean_agree, overall_mean_hier, min_n
    )
    _print_group_table(
        f"by intervention node (n>={min_n})",
        drug_stats,
        overall_mean_agree,
        overall_mean_hier,
    )
    return drug_stats


def _posneg_tag(prefix: str, value: str) -> str:
    """Map pos/neg-style annotations to ``PREFIX+`` / ``PREFIX-``."""
    v = (value or "").strip().lower()
    if v in {"pos", "+", "1", "positive"}:
        return f"{prefix}+"
    if v in {"neg", "-", "0", "negative"}:
        return f"{prefix}-"
    if not v:
        return f"{prefix}?"
    return f"{prefix}({value.strip()})"


def _cell_group_label(r: Row) -> str:
    """Cell line name, or ER/HER signature when the cell line is missing."""
    if r.cell_line:
        return r.cell_line
    return f"{_posneg_tag('ER', r.er)}{_posneg_tag('HER', r.her2)}"


def _culture_label(r: Row) -> str:
    """Culture condition; empty → ``general``."""
    return r.culture or "general"


def _culture_notes_label(r: Row) -> str:
    """Culture notes; empty → copy culture (including ``general``)."""
    return r.culture_notes or _culture_label(r)


# Multi-factor combination axes (single-factor views are other panels).
# Empty cell line → ER±HER±; empty culture → general; empty notes → culture.
_COMBO_FACTORS: tuple[Callable[[Row], str], ...] = (
    lambda r: r.observed or "no_obs",
    _cell_group_label,
    _culture_label,
    _culture_notes_label,
    lambda r: r.intervention or "no_iv",
)


def collect_combinations(
    rows: Sequence[Row],
    overall_mean_agree: float,
    overall_mean_hier: float,
    min_n: int,
    min_delta: float,
) -> list[GroupStats]:
    """Build all multi-factor combos among observed×cell×culture×notes×intervention.

    Every subset of those five factors with size ≥ 2 is considered. Groups with
    ``n < min_n`` or ``|Δ| < min_delta`` are dropped. Callers typically pass the
    result through :func:`filter_minimal_combos` so longer labels that cover the
    same experiments as a shorter combo are removed.
    """
    combos: list[GroupStats] = []
    factor_indices = range(len(_COMBO_FACTORS))

    for n_fac in range(2, len(_COMBO_FACTORS) + 1):
        for idxs in itertools.combinations(factor_indices, n_fac):
            getters = [_COMBO_FACTORS[i] for i in idxs]
            buckets: dict[tuple[str, ...], list[Row]] = defaultdict(list)
            for row in rows:
                key = tuple(g(row) for g in getters)
                buckets[key].append(row)
            for key, group in buckets.items():
                if len(group) < min_n:
                    continue
                s = _group_stats(
                    " × ".join(key),
                    group,
                    overall_mean_agree,
                    overall_mean_hier,
                    n_factors=len(idxs),
                )
                if max(abs(s.delta_agreement), abs(s.delta_hierarchy)) >= min_delta:
                    combos.append(s)

    combos.sort(
        key=lambda s: (
            -max(abs(s.delta_agreement), abs(s.delta_hierarchy)),
            -s.n,
            s.label,
        )
    )
    return combos


def filter_minimal_combos(combos: Sequence[GroupStats]) -> list[GroupStats]:
    """Keep the shortest factor list when several combos cover the same ids.

    E.g. ``AKT × BT474`` and ``AKT × BT474 × 2D culture`` with identical
    member sets → keep only ``AKT × BT474``.
    """
    by_ids: dict[frozenset[int], list[GroupStats]] = defaultdict(list)
    for s in combos:
        by_ids[s.member_ids].append(s)

    kept: list[GroupStats] = []
    for group in by_ids.values():
        group.sort(key=lambda s: (s.n_factors, len(s.label), s.label))
        kept.append(group[0])

    kept.sort(
        key=lambda s: (
            -max(abs(s.delta_agreement), abs(s.delta_hierarchy)),
            -s.n,
            s.label,
        )
    )
    return kept


def print_combinations(
    rows: Sequence[Row],
    overall_mean_agree: float,
    overall_mean_hier: float,
    min_n: int,
    min_delta: float,
) -> list[GroupStats]:
    combos = collect_combinations(
        rows, overall_mean_agree, overall_mean_hier, min_n, min_delta
    )
    minimal = filter_minimal_combos(combos)
    dropped = len(combos) - len(minimal)
    title = f"combinations (n>={min_n}, |Δ|>={min_delta}, minimal)"
    if dropped:
        title += f"; dropped {dropped} redundant longer labels"
    _print_group_table(
        title,
        minimal,
        overall_mean_agree,
        overall_mean_hier,
    )
    return minimal


def _significance(s: GroupStats) -> float:
    """Volcano-style rank: large |Δ| and large n."""
    return abs(s.delta_agreement) * math.log10(s.n + 1.0)


def _short_label(text: str, max_len: int = 42) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _draw_volcano_panel(
    ax: plt.Axes,
    stats: Sequence[GroupStats],
    title: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("Δ agreement (vs overall)")
    ax.set_ylabel("n (log)")
    ax.grid(True, which="both", alpha=0.25)
    if not stats:
        ax.text(0.5, 0.5, "(no groups)", ha="center", va="center", transform=ax.transAxes)
        return

    xs = [s.delta_agreement for s in stats]
    ys = [s.n for s in stats]
    colors = ["#c0392b" if x < 0 else "#2980b9" for x in xs]
    ax.axvline(0.0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
    ax.scatter(xs, ys, c=colors, s=36, alpha=0.75, edgecolors="none", zorder=2)


def _add_volcano_labels(
    ax: plt.Axes,
    stats: Sequence[GroupStats],
    *,
    label_n: int,
    label_min_n: int = 10,
    fontsize: float = 7.0,
    max_label_len: int = 42,
    ensure_inside: bool = True,
) -> None:
    """Place top labels near dots; avoid label–label overlap only.

    Labels may sit on top of dots. Prefers ``adjustText`` when installed;
    otherwise uses a lightweight offset-points repulsion pass. Only groups
    with ``n >= label_min_n`` are considered for labeling.
    """
    eligible = [s for s in stats if s.n >= label_min_n]
    ranked = sorted(eligible, key=_significance, reverse=True)[: max(0, label_n)]
    if not ranked:
        return

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None  # type: ignore[assignment]

    xs = [s.delta_agreement for s in ranked]
    ys = [float(s.n) for s in ranked]
    labels = [_short_label(s.label, max_label_len) for s in ranked]

    if adjust_text is not None:
        # Seed with a small spread so coincident labels can explode apart.
        x0, x1 = ax.get_xlim()
        x_jitter = 0.015 * (x1 - x0 if x1 > x0 else 1.0)
        texts = []
        for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
            y_seed = y * (1.0 + 0.03 * ((i % 3) - 1))
            texts.append(
                ax.text(
                    x + x_jitter * ((i % 5) - 2),
                    max(y_seed, 1.0),
                    lab,
                    fontsize=fontsize,
                    color="#222222",
                    zorder=3,
                )
            )
        # Repel labels from each other only (not from dots). Targets still
        # drive leader lines back to the points.
        adjust_text(
            texts,
            ax=ax,
            target_x=xs,
            target_y=ys,
            avoid_self=False,
            force_text=(1.5, 1.8),
            force_static=(0.0, 0.0),
            force_explode=(1.2, 1.6),
            force_pull=(0.02, 0.02),
            expand=(1.3, 1.6),
            max_move=(120, 160),
            ensure_inside_axes=ensure_inside,
            expand_axes=not ensure_inside,
            prevent_crossings=True,
            min_arrow_len=2,
            iter_lim=600,
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 3,
            },
        )
        return

    # Fallback: label–label AABB repulsion only (dots may be covered).
    fig = ax.figure
    pt_to_px = fig.dpi / 72.0
    glyph_w = 0.55 * fontsize * pt_to_px
    glyph_h = 1.15 * fontsize * pt_to_px
    placed: list[dict] = []
    for i, (s, lab) in enumerate(zip(ranked, labels)):
        side = 1.0 if s.delta_agreement >= 0 else -1.0
        placed.append(
            {
                "s": s,
                "label": lab,
                "dx": side * (4.0 + 2.0 * (i % 3)),
                "dy": 6.0 * ((i % 5) - 2),
                "w": max(18.0, glyph_w * len(lab)),
                "h": glyph_h,
            }
        )

    for _ in range(50):
        centers: list[tuple[float, float, float, float]] = []
        for p in placed:
            x_d, y_d = ax.transData.transform((p["s"].delta_agreement, p["s"].n))
            ox = p["dx"] * pt_to_px
            oy = p["dy"] * pt_to_px
            cx = x_d + ox + (0.5 * p["w"] if p["dx"] >= 0 else -0.5 * p["w"])
            cy = y_d + oy
            centers.append((cx, cy, p["w"], p["h"]))

        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                cxi, cyi, wi, hi = centers[i]
                cxj, cyj, wj, hj = centers[j]
                if abs(cxi - cxj) >= 0.5 * (wi + wj) + 4:
                    continue
                if abs(cyi - cyj) >= 0.5 * (hi + hj) + 4:
                    continue
                push = 3.0
                if cyi <= cyj:
                    placed[i]["dy"] -= push
                    placed[j]["dy"] += push
                else:
                    placed[i]["dy"] += push
                    placed[j]["dy"] -= push
                moved = True

        for p in placed:
            p["dx"] = max(-120.0, min(120.0, p["dx"]))
            p["dy"] = max(-100.0, min(100.0, p["dy"]))
        if not moved:
            break

    for p in placed:
        ax.annotate(
            p["label"],
            xy=(p["s"].delta_agreement, p["s"].n),
            xytext=(p["dx"], p["dy"]),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left" if p["dx"] >= 0 else "right",
            va="center",
            color="#222222",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 3,
            },
            zorder=3,
            annotation_clip=False,
        )


def write_volcano_plot(
    path: Path,
    *,
    observed: Sequence[GroupStats],
    cell_line: Sequence[GroupStats],
    intervened: Sequence[GroupStats],
    combinations: Sequence[GroupStats],
    label_n: int,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = (
        (axes[0, 0], observed, "Observed node"),
        (axes[0, 1], cell_line, "Cell line"),
        (axes[1, 0], intervened, "Intervened node"),
        (
            axes[1, 1],
            combinations,
            "Combinations (obs×cell×culture×notes×iv, minimal)",
        ),
    )
    for ax, stats, title in panels:
        _draw_volcano_panel(ax, stats, title)

    all_deltas = [
        s.delta_agreement
        for _, stats, _ in panels
        for s in stats
        if not math.isnan(s.delta_agreement)
    ]
    all_ns = [
        s.n
        for _, stats, _ in panels
        for s in stats
        if s.n > 0
    ]
    if all_deltas:
        half = max(abs(min(all_deltas)), abs(max(all_deltas)))
        pad = max(0.02, 0.05 * half)
        xlim = (-half - pad, half + pad)
    else:
        xlim = None
    if all_ns:
        n_lo = min(all_ns)
        n_hi = max(all_ns)
        ylim = (max(n_lo / 1.25, 0.8), n_hi * 1.25)
    else:
        ylim = None

    for ax, _, _ in panels:
        ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    for ax, stats, title in panels:
        is_combo = title.startswith("Combinations")
        _add_volcano_labels(
            ax,
            stats,
            label_n=min(label_n, 6) if is_combo else label_n,
            label_min_n=PLOT_LABEL_MIN_N,
            max_label_len=32 if is_combo else 42,
            # Allow combo labels to spill slightly so dense clusters can separate.
            ensure_inside=not is_combo,
        )

    fig.suptitle(
        "Agreement Δ vs n (volcano-style)\n"
        "x = mean agreement − overall; y = group size (log, shared); "
        "labels = top by |Δ|·log10(n+1)",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _is_subset(a: dict[str, int], b: dict[str, int]) -> bool:
    return all(b.get(k) == v for k, v in a.items())


def print_intervention_ladders(rows: Sequence[Row], top: int) -> None:
    abs_rows = [r for r in rows if not r.is_comparative]
    by_key: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in abs_rows:
        by_key[(r.observed, r.context)].append(r)

    collapses: list[tuple[float, str, list[Row]]] = []
    for (observed, context), group in by_key.items():
        if len(group) < 2:
            continue
        # Prefer unique interventions; keep best weight if duplicates.
        by_iv: dict[tuple[tuple[str, int], ...], Row] = {}
        for r in group:
            key = tuple(sorted(r.intervention_map.items()))
            prev = by_iv.get(key)
            if prev is None or r.weight > prev.weight:
                by_iv[key] = r
        members = list(by_iv.values())
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (r.hierarchy, len(r.intervention_map), r.id))

        # Collapse: some shallower row has higher agreement than a deeper
        # superseting intervention (on agreement or hierarchy_agreement).
        severity = 0.0
        for i, shallow in enumerate(members):
            for deep in members[i + 1 :]:
                if not _is_subset(shallow.intervention_map, deep.intervention_map):
                    continue
                drop_a = shallow.agreement - deep.agreement
                drop_h = shallow.hierarchy_agreement - deep.hierarchy_agreement
                if drop_a > _EPS or drop_h > _EPS:
                    severity = max(severity, drop_a, drop_h)
        if severity > _EPS:
            ctx_short = context if len(context) <= 60 else context[:57] + "..."
            label = f"{observed} | ctx={ctx_short}"
            collapses.append((severity, label, members))

    collapses.sort(key=lambda t: (-t[0], t[1]))
    print(f"\n=== intervention ladders that collapse (top {top}) ===")
    if not collapses:
        print("(none)")
        return
    for severity, label, members in collapses[:top]:
        print(f"\n[{_fmt(severity)} drop] {label}")
        for r in members:
            iv = r.intervention or "(none)"
            print(
                f"  h={r.hierarchy} id={r.id:<6} iv={iv:<40} "
                f"agree={_fmt(r.agreement)} hier={_fmt(r.hierarchy_agreement)} "
                f"pred={_fmt(r.predict_value) if r.predict_value is not None else 'n/a'} "
                f"out={r.outcome}"
            )


def _print_mismatch_rows(title: str, rows: Sequence[Row]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(none)")
        return
    header = (
        f"{'id':>6} {'cell':<10} {'culture':<16} {'notes':<18} "
        f"{'observed':<14} {'agree':>6} {'hier':>6} {'score':>6} intervention"
    )
    print(header)
    print("-" * min(120, len(header) + 40))
    for r in rows:
        notes = (r.culture_notes or "")[:18]
        cult = (r.culture or "")[:16]
        cell = (r.cell_line or "")[:10]
        obs = (r.observed or "")[:14]
        iv = r.intervention or "(none)"
        if len(iv) > 36:
            iv = iv[:33] + "..."
        print(
            f"{r.id:6d} {cell:<10} {cult:<16} {notes:<18} "
            f"{obs:<14} {_fmt(r.agreement):>6} {_fmt(r.hierarchy_agreement):>6} "
            f"{_fmt(r.score):>6} {iv}"
        )
        print(
            f"{'':6} outcome={r.outcome}  predict="
            f"{_fmt(r.predict_value) if r.predict_value is not None else 'n/a'}"
        )


def print_mismatches(rows: Sequence[Row], top: int) -> None:
    worst = sorted(rows, key=lambda r: (r.agreement, -r.weight, r.id))[:top]
    _print_mismatch_rows(f"worst by agreement (top {top})", worst)

    hurts = [
        r for r in rows if r.hierarchy_agreement + _EPS < r.agreement
    ]
    hurts.sort(
        key=lambda r: (-(r.agreement - r.hierarchy_agreement), -r.weight, r.id)
    )
    _print_mismatch_rows(
        f"hierarchy-amplified hurts (top {top})",
        hurts[:top],
    )


def _print_distribution(title: str, values: Sequence[float]) -> None:
    print(f"\n--- {title} (n={len(values)}) ---")
    if not values:
        print("(none)")
        return
    print(f"mean={_fmt(_mean(values))}  median={_fmt(_median(values))}")
    for label, count in _bin_counts(values):
        frac = count / len(values)
        print(f"  {label:<12} {count:5d}  {_fmt_pct(frac)}")


def print_statement_distributions(rows: Sequence[Row]) -> None:
    print("\n=== statement distributions ===")
    absolute = [r for r in rows if not r.is_comparative]
    _print_distribution("absolute: agreement", [r.agreement for r in absolute])
    _print_distribution(
        "absolute: hierarchy_agreement",
        [r.hierarchy_agreement for r in absolute],
    )

    by_id = {r.id: r for r in rows}
    comparative = [r for r in rows if r.is_comparative]
    pairs: dict[tuple[int, int], list[Row]] = defaultdict(list)
    unpaired: list[Row] = []
    for r in comparative:
        if r.other_id is None:
            unpaired.append(r)
            continue
        key = (min(r.id, r.other_id), max(r.id, r.other_id))
        pairs[key].append(r)

    pair_agreements: list[float] = []
    pair_scores: list[float] = []
    incomplete = 0
    for (a, b), members in pairs.items():
        # Deduplicate if both sides listed the pair.
        uniq: dict[int, Row] = {m.id: m for m in members}
        # Ensure partner exists in the score file even if not in members yet.
        for pid in (a, b):
            if pid not in uniq and pid in by_id and by_id[pid].is_comparative:
                uniq[pid] = by_id[pid]
        if len(uniq) != 2:
            incomplete += 1
            continue
        left, right = uniq[a], uniq[b]
        pair_agreements.append(0.5 * (left.agreement + right.agreement))
        pair_scores.append(left.score + right.score)

    if unpaired:
        print(f"warning: {len(unpaired)} comparative row(s) with unparsable partner id")
    if incomplete:
        print(f"warning: {incomplete} comparative pair key(s) missing a partner row")

    _print_distribution(
        "comparative pairs: avg agreement (≡ Σscore when weights=0.5)",
        pair_agreements,
    )
    # Also show score-sum distribution (should match when weights are 0.5).
    if pair_scores and any(
        abs(a - s) > 1e-6 for a, s in zip(pair_agreements, pair_scores)
    ):
        _print_distribution("comparative pairs: Σscore", pair_scores)
    print(
        f"comparative pairs complete={len(pair_agreements)}  "
        f"absolute rows={len(absolute)}  comparative rows={len(comparative)}"
    )


def write_annotated_csv(path: Path, rows: Sequence[Row]) -> None:
    fieldnames = [
        "id",
        "weight",
        "hierarchy",
        "context",
        "intervention",
        "fixes",
        "observed",
        "outcome",
        "outcome_range",
        "predict_value",
        "agreement",
        "hierarchy_agreement",
        "score",
        "Cell line",
        "Culture",
        "Culture notes",
        "ER",
        "HER2",
        "Model",
        "is_comparative",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "id": r.id,
                    "weight": r.weight,
                    "hierarchy": r.hierarchy,
                    "context": r.context,
                    "intervention": r.intervention,
                    "fixes": r.fixes,
                    "observed": r.observed,
                    "outcome": r.outcome,
                    "outcome_range": r.outcome_range,
                    "predict_value": (
                        "" if r.predict_value is None else r.predict_value
                    ),
                    "agreement": r.agreement,
                    "hierarchy_agreement": r.hierarchy_agreement,
                    "score": r.score,
                    "Cell line": r.cell_line,
                    "Culture": r.culture,
                    "Culture notes": r.culture_notes,
                    "ER": r.er,
                    "HER2": r.her2,
                    "Model": r.model,
                    "is_comparative": r.is_comparative,
                }
            )
    print(f"\nwrote annotated CSV: {path}")


def run_analysis(
    rows: Sequence[Row],
    *,
    min_n: int,
    min_delta: float,
    top: int,
) -> dict[str, list[GroupStats]]:
    mean_a, mean_h = print_overall(rows)

    observed = _stats_for_key(
        rows, lambda r: r.observed or "no_obs", mean_a, mean_h
    )
    cell_line = _stats_for_key(rows, _cell_group_label, mean_a, mean_h)
    _print_group_table("by observed node", observed, mean_a, mean_h)
    _print_group_table("by Cell line", cell_line, mean_a, mean_h)
    _print_group_table(
        "by Culture",
        _stats_for_key(rows, _culture_label, mean_a, mean_h),
        mean_a,
        mean_h,
    )
    _print_group_table(
        "by Culture + Culture notes",
        _stats_for_key(
            rows,
            lambda r: f"{_culture_label(r)} | {_culture_notes_label(r)}",
            mean_a,
            mean_h,
        ),
        mean_a,
        mean_h,
    )
    _print_group_table(
        "by hierarchy depth",
        _stats_for_key(rows, lambda r: str(r.hierarchy), mean_a, mean_h),
        mean_a,
        mean_h,
    )
    print_drug_reports(rows, mean_a, mean_h, min_n)
    print_combinations(rows, mean_a, mean_h, min_n, min_delta)
    # Volcano plots: no min_n / min_delta cutoff (those are print-only).
    plot_intervened = intervention_node_stats(rows, mean_a, mean_h, min_n=1)
    plot_combinations = filter_minimal_combos(
        collect_combinations(rows, mean_a, mean_h, min_n=1, min_delta=0.0)
    )
    print_intervention_ladders(rows, top)
    print_mismatches(rows, top)
    print_statement_distributions(rows)
    return {
        "observed": observed,
        "cell_line": cell_line,
        "intervened": plot_intervened,
        "combinations": plot_combinations,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    score_raw = args.score if args.score is not None else SCORE
    score_path = _resolve_path(score_raw)

    if args.data is not None:
        data_path = _resolve_path(args.data)
    else:
        batch_data = _find_batch_data(score_path)
        if batch_data is not None:
            data_path = batch_data
        elif DATA:
            data_path = _resolve_path(DATA)
        else:
            data_path = _resolve_path("case_study/breast_cancer/data.csv")

    min_n = MIN_N if args.min_n is None else args.min_n
    min_delta = MIN_DELTA if args.min_delta is None else args.min_delta
    top = TOP if args.top is None else args.top
    label_n = PLOT_LABEL_N if args.plot_labels is None else args.plot_labels
    do_plot = PLOT and not args.no_plot

    print(f"score: {score_path}")
    print(f"data:  {data_path}")
    print(f"min_n={min_n}  min_delta={min_delta}  top={top}")

    data_by_id = load_data_by_id(data_path)
    rows = load_score_rows(score_path, data_by_id)
    if not rows:
        raise SystemExit("no score rows loaded")

    groups = run_analysis(rows, min_n=min_n, min_delta=min_delta, top=top)

    if do_plot:
        plot_path = score_path.with_name(f"{score_path.stem}_volcano.png")
        write_volcano_plot(
            plot_path,
            observed=groups["observed"],
            cell_line=groups["cell_line"],
            intervened=groups["intervened"],
            combinations=groups["combinations"],
            label_n=label_n,
        )
        print(f"\nwrote volcano plot: {plot_path}")

    if args.out is not None:
        out_path = args.out
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        write_annotated_csv(out_path, rows)


if __name__ == "__main__":
    main()
