# %% [markdown]
# NAV import check
#
# Probe whether a NAV-style CSV/TSV can be loaded with
# `boolmore.io.nav.import_NAV_exps`.
#
# Edit the config cell below to point at another dataset. Column overrides
# (e.g. one fixes column winning over another on shared nodes) are applied
# here only — not in `nav.py`.

# %%
from __future__ import annotations

import csv
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from boolmore.io.nav import (
    _DEFAULT_FIXES_COLS,
    _DEFAULT_ID_COL,
    _DEFAULT_OBSERVED_COL,
    _DEFAULT_OUTCOME_COL,
    _DEFAULT_WEIGHT_COL,
    _merge_fixes,
    _normalize_header,
    _parse_fixes_cell,
    import_NAV_exps,
)

# %% [markdown]
# ## Config
#
# Change these for another dataset. `FIXES_OVERRIDES` entries are
# `(override_col, overridden_col)`: when both assign the same node, keep the
# override column and drop that node from the overridden column.

# %%
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "case_study" / "breast_cancer" / "breast_cancer_data.csv"

ID_COL = "id"
WEIGHT_COL = "Weight"  # typo in the breast-cancer source CSV header
OBSERVED_COL = "Observed node"
OUTCOME_COL = "Value"
FIXES_COLS = (
    "Signals",
    "Drug treatment/intervention",
    "Cell line characteristic profile",
    "Additional features of the cell population",
)
# Additional features wins over Cell line characteristic profile.
FIXES_OVERRIDES: tuple[tuple[str, str], ...] = (
    (
        "Additional features of the cell population",
        "Cell line characteristic profile",
    ),
)

# %%
@dataclass(frozen=True)
class OverrideEvent:
    """One node taken from an override column instead of the overridden column."""

    row_id: str
    node: str
    overridden_col: str
    overridden_value: int
    override_col: str
    override_value: int

    @property
    def kind(self) -> str:
        if self.overridden_value == self.override_value:
            return "duplicate"
        return "inconsistent"


def _format_fixes(pairs: Sequence[tuple[str, int]]) -> str:
    return ", ".join(f"{node}={value}" for node, value in pairs)


def apply_fixes_overrides(
    rows: list[dict[str, str]],
    overrides: Sequence[tuple[str, str]],
    *,
    id_col: str = "id",
) -> list[OverrideEvent]:
    """Apply column overrides in place; return where they took effect.

    For each ``(override_col, overridden_col)`` pair, any node present in both
    columns is kept only from ``override_col``. The node is removed from the
    overridden column cell, so later ``_merge_fixes`` / ``import_NAV_exps``
    neither see duplicates nor inconsistencies for that node.
    """
    events: list[OverrideEvent] = []
    for override_col, overridden_col in overrides:
        for row in rows:
            rid = (row.get(id_col) or "").strip()
            override_cell = (row.get(override_col) or "").strip()
            overridden_cell = (row.get(overridden_col) or "").strip()
            if not override_cell or not overridden_cell:
                continue
            try:
                override_map = dict(_parse_fixes_cell(override_cell))
                overridden_pairs = _parse_fixes_cell(overridden_cell)
            except ValueError:
                # Leave unparsable cells for the diagnostic pass.
                continue
            if not override_map:
                continue

            kept: list[tuple[str, int]] = []
            for node, value in overridden_pairs:
                if node in override_map:
                    events.append(
                        OverrideEvent(
                            row_id=rid,
                            node=node,
                            overridden_col=overridden_col,
                            overridden_value=value,
                            override_col=override_col,
                            override_value=override_map[node],
                        )
                    )
                else:
                    kept.append((node, value))
            row[overridden_col] = _format_fixes(kept)
    return events


def clean_fixes_cell(cell: str) -> str:
    cell = (cell or "").strip()
    if not cell or cell == "-":
        return ""
    # "PTEN_T_2=0 SGK1_T=0" -> "PTEN_T_2=0, SGK1_T=0"
    cell = re.sub(r"(?<=[01])\s+(?=[A-Za-z_])", ", ", cell)
    parts: list[str] = []
    for part in cell.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            parts.append(part)
        elif re.fullmatch(r"[A-Za-z0-9_]+", part):
            parts.append(f"{part}=1")
        else:
            parts.append(part)
    return ", ".join(parts)


def clean_weight(cell: str) -> str:
    cell = (cell or "").strip()
    if not cell or cell == "?":
        return "1"
    return cell


def _norm_msg(msg: str) -> str:
    return re.sub(r"\(row id=\d+\)", "(row id=...)", msg)


def _report_grouped(
    title: str,
    items: list[tuple[str, str]],
    *,
    limit_groups: int | None = None,
) -> None:
    """Print error counts grouped by message, with every row id listed."""
    print(f"\n{title}: {len(items)}")
    grouped: dict[str, list[str]] = {}
    for rid, msg in items:
        grouped.setdefault(_norm_msg(msg), []).append(rid)
    groups = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if limit_groups is not None:
        groups = groups[:limit_groups]
    for msg, ids in groups:
        print(f"  {len(ids):4d}  {msg}")
        print(f"        ids: {', '.join(ids)}")


def _report_overrides(events: Sequence[OverrideEvent]) -> None:
    print(f"\nfixes column overrides applied: {len(events)}")
    if not events:
        return
    by_kind = Counter(e.kind for e in events)
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind}: {n}")

    grouped: dict[str, list[str]] = {}
    for e in events:
        key = (
            f"{e.kind}  node={e.node!r}  "
            f"{e.overridden_col!r}={e.overridden_value} "
            f"-> kept {e.override_col!r}={e.override_value}"
        )
        grouped.setdefault(key, []).append(e.row_id)
    print("  where applied:")
    for key, ids in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"    {len(ids):4d}  {key}")
        print(f"          ids: {', '.join(ids)}")


# %% [markdown]
# ## 1. Inspect headers vs `import_NAV_exps` defaults

# %%
with DATA.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = list(reader.fieldnames or [])
    rows = list(reader)

print(f"file: {DATA}")
print(f"rows: {len(rows)}")
print("headers:")
for h in headers:
    print(f"  {h!r}  ->  {_normalize_header(h)!r}")

print("\ndefault roles:")
for role, aliases in [
    ("id", _DEFAULT_ID_COL),
    ("weight", _DEFAULT_WEIGHT_COL),
    ("observed", _DEFAULT_OBSERVED_COL),
    ("outcome", _DEFAULT_OUTCOME_COL),
    ("fixes", _DEFAULT_FIXES_COLS),
]:
    norms = {_normalize_header(a) for a in aliases}
    hits = [h for h in headers if _normalize_header(h) in norms]
    print(f"  {role:8} aliases={list(aliases)!r}  matches={hits!r}")

print("\nactive mapping:")
print(f"  id={ID_COL!r}")
print(f"  weight={WEIGHT_COL!r}")
print(f"  observed={OBSERVED_COL!r}")
print(f"  outcome={OUTCOME_COL!r}")
print(f"  fixes={list(FIXES_COLS)!r}")
if FIXES_OVERRIDES:
    for override_col, overridden_col in FIXES_OVERRIDES:
        print(f"  override: {override_col!r} overrides {overridden_col!r}")
else:
    print("  override: (none)")

# %% [markdown]
# ## 2. Default import (expected to fail on mismatched columns)

# %%
try:
    exps = import_NAV_exps(DATA)
    print(f"default import OK: {len(exps)} experiments")
except Exception as exc:
    print(f"default import FAILED: {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 3. Custom column mapping (raw file, no cleanup / override)

# %%
try:
    exps = import_NAV_exps(
        DATA,
        weight_col=WEIGHT_COL,
        observed_col=OBSERVED_COL,
        outcome_col=OUTCOME_COL,
        fixes_cols=FIXES_COLS,
    )
    print(f"custom-mapping import OK: {len(exps)} experiments")
    print("first:", exps[0])
except Exception as exc:
    print(f"custom-mapping import FAILED: {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 4. Light cleanup + column overrides
#
# Mechanical fixes (dataset-specific, not in `nav.py`):
# - treat `-` as empty intervention
# - insert missing commas between `=0|1` tokens
# - append `=1` to bare drug names (`Alpelisib`, `Ipatasertib`)
# - fill empty / `?` weights with `1.0`
#
# Then apply `FIXES_OVERRIDES` so shared nodes keep the override column.

# %%
cleaned_rows: list[dict[str, str]] = []
for r in rows:
    nr = dict(r)
    for col in FIXES_COLS:
        nr[col] = clean_fixes_cell(nr.get(col, ""))
    nr[WEIGHT_COL] = clean_weight(nr.get(WEIGHT_COL, ""))
    cleaned_rows.append(nr)

override_events = apply_fixes_overrides(
    cleaned_rows,
    FIXES_OVERRIDES,
    id_col=ID_COL,
)
_report_overrides(override_events)

# %% [markdown]
# ## 5. Row-level diagnostics (after cleanup / override)

# %%
weight_counts = Counter((r.get(WEIGHT_COL) or "").strip() for r in cleaned_rows)
print("Weight value counts:", weight_counts.most_common())

parse_errors: list[tuple[str, str, str, str]] = []
merge_errors: list[tuple[str, str]] = []
observed_in_fixes: list[tuple[str, str]] = []
weight_errors: list[tuple[str, str]] = []

for r in cleaned_rows:
    rid = (r.get(ID_COL) or "").strip()
    cells: list[str] = []
    row_parse_ok = True
    for col in FIXES_COLS:
        cell = (r.get(col) or "").strip()
        if not cell:
            continue
        try:
            _merge_fixes([cell], row_id=rid)
            cells.append(cell)
        except Exception as exc:
            parse_errors.append((rid, col, cell[:100], str(exc)))
            row_parse_ok = False
            break
    if not row_parse_ok:
        continue

    try:
        fixes = _merge_fixes(cells, row_id=rid or None)
    except Exception as exc:
        merge_errors.append((rid, str(exc)))
        continue

    observed = (r.get(OBSERVED_COL) or "").strip()
    if observed in {node for node, _ in fixes}:
        observed_in_fixes.append((rid, observed))

    w = (r.get(WEIGHT_COL) or "").strip()
    try:
        float(w)
    except Exception:
        weight_errors.append((rid, w))

_report_grouped(
    "fixes parse errors",
    [(rid, msg) for rid, _col, _cell, msg in parse_errors],
)

_report_grouped(
    "fixes merge errors (duplicates / inconsistent nodes)",
    merge_errors,
)

print(f"\nobserved node also in fixes: {len(observed_in_fixes)}")
for rid, observed in observed_in_fixes:
    print(f"  id={rid}  observed={observed!r}")

_report_grouped(
    f"non-numeric {WEIGHT_COL}",
    [(rid, f"value={w!r}") for rid, w in weight_errors],
)

# %% [markdown]
# ## 6. Example problematic cells

# %%
print("parse failures (one example cell per message):")
seen_msgs: set[str] = set()
for rid, col, cell, msg in parse_errors:
    key = _norm_msg(msg)
    if key in seen_msgs:
        continue
    seen_msgs.add(key)
    ids = [r for r, _c, _cell, m in parse_errors if _norm_msg(m) == key]
    print(f"  ids={', '.join(ids)}")
    print(f"    example id={rid}  col={col!r}")
    print(f"    cell={cell!r}")
    print(f"    err={msg}")

print("\nmerge failures (one example row per message):")
seen_merge: set[str] = set()
for rid, msg in merge_errors:
    key = _norm_msg(msg)
    if key in seen_merge:
        continue
    seen_merge.add(key)
    ids = [r for r, m in merge_errors if _norm_msg(m) == key]
    r = next(x for x in cleaned_rows if (x.get(ID_COL) or "").strip() == rid)
    print(f"  ids={', '.join(ids)}")
    print(f"    example id={rid}: {msg}")
    for col in FIXES_COLS:
        print(f"      {col}: {(r.get(col) or '')[:120]!r}")

# %% [markdown]
# ## 7. Import trial
#
# Rows that still violate NAV constraints are dropped for this trial.

# %%
kept: list[dict[str, str]] = []
dropped: list[tuple[str, str]] = []
for r in cleaned_rows:
    rid = (r.get(ID_COL) or "").strip()
    try:
        fixes = _merge_fixes([r[c] for c in FIXES_COLS], row_id=rid or None)
        observed = (r.get(OBSERVED_COL) or "").strip()
        if observed in {node for node, _ in fixes}:
            raise ValueError(
                f"Node '{observed}' appears in both fixes and observed"
            )
        float((r.get(WEIGHT_COL) or "").strip())
    except Exception as exc:
        dropped.append((rid, str(exc)))
        continue
    kept.append(r)

print(f"kept after cleanup/override: {len(kept)} / {len(rows)}")
_report_grouped("dropped", dropped)

with tempfile.TemporaryDirectory() as tmp:
    cleaned_path = Path(tmp) / f"{DATA.stem}_cleaned_trial{DATA.suffix or '.csv'}"
    with cleaned_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    try:
        exps = import_NAV_exps(
            cleaned_path,
            weight_col=WEIGHT_COL,
            observed_col=OBSERVED_COL,
            outcome_col=OUTCOME_COL,
            fixes_cols=FIXES_COLS,
        )
        print(f"\ncleaned trial import OK: {len(exps)} experiments")
        print("first:", exps[0])
        print(f"n fixes (first): {len(exps[0].fixes)}")
    except Exception as exc:
        print(f"\ncleaned trial import FAILED: {type(exc).__name__}: {exc}")
