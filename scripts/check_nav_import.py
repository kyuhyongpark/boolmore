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
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from boolmore.evaluation.agreement import check_co_located_outcomes
from boolmore.io.nav import (
    _COMPARATIVE_INVERSE,
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
from boolmore.outcome import ComparativeOutcome, OutcomeType, parse_outcome

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


@dataclass(frozen=True)
class ParsedRow:
    """One row that passed per-row fixes / outcome / weight checks."""

    exp_id: int
    row_id: str
    fixes: tuple[tuple[str, int], ...]
    observed: str
    outcome: OutcomeType


def _check_comparative_pairs(
    rows: Sequence[ParsedRow],
) -> list[tuple[str, str]]:
    """Return comparative-pair errors mirroring ``import_NAV_exps``."""
    by_id = {row.exp_id: row for row in rows}
    errors: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row.outcome, ComparativeOutcome):
            continue
        other_id = row.outcome.other_id
        other = by_id.get(other_id)
        if other is None:
            errors.append(
                (
                    row.row_id,
                    f"Comparative outcome for experiment {row.exp_id} "
                    f"references missing id {other_id}",
                )
            )
            continue
        if row.observed != other.observed:
            errors.append(
                (
                    row.row_id,
                    f"Comparative outcome for experiment {row.exp_id} "
                    f"references id {other_id} with different observed "
                    f"({row.observed!r} vs {other.observed!r})",
                )
            )
            continue
        expected_relation = _COMPARATIVE_INVERSE[row.outcome.relation]
        if not (
            isinstance(other.outcome, ComparativeOutcome)
            and other.outcome.relation == expected_relation
            and other.outcome.other_id == row.exp_id
        ):
            if expected_relation == "same":
                expected_phrase = f"same as {row.exp_id}"
            else:
                expected_phrase = f"{expected_relation} than {row.exp_id}"
            errors.append(
                (
                    row.row_id,
                    f"Comparative outcome for experiment {row.exp_id} "
                    f"({row.outcome}) expects reciprocal "
                    f"{expected_phrase!r} on experiment {other_id}",
                )
            )
    return errors


def _filter_co_located_and_comparative(
    rows: Sequence[ParsedRow],
) -> tuple[list[ParsedRow], list[tuple[str, str]], list[tuple[str, str]]]:
    """Keep rows that pass co-located then comparative checks.

    Co-located conflicts drop the later row (same order as import).
    Comparative failures are dropped and the check is repeated until stable,
    so leftover references to dropped ids are also removed.
    """
    accepted: list[ParsedRow] = []
    co_located_errors: list[tuple[str, str]] = []
    for row in rows:
        co_located = [
            (other.exp_id, other.outcome)
            for other in accepted
            if other.fixes == row.fixes and other.observed == row.observed
        ]
        try:
            check_co_located_outcomes(
                co_located,
                row.exp_id,
                row.outcome,
                row.observed,
                row.fixes,
            )
        except ValueError as exc:
            co_located_errors.append(
                (row.row_id, _format_co_located_error(str(exc)))
            )
            continue
        accepted.append(row)

    comparative_errors: list[tuple[str, str]] = []
    while True:
        batch = _check_comparative_pairs(accepted)
        if not batch:
            break
        comparative_errors.extend(batch)
        bad_ids = {rid for rid, _ in batch}
        accepted = [row for row in accepted if row.row_id not in bad_ids]
    return accepted, co_located_errors, comparative_errors


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


def clean_weight(cell: str) -> tuple[str, bool]:
    """Fill empty / ``?`` weights with ``1``. Return ``(cleaned, changed)``."""
    cell = (cell or "").strip()
    if not cell or cell == "?":
        return "1", True
    return cell, False


def _format_co_located_error(msg: str) -> str:
    """Rewrite co-located errors to lead with the duplicate ids."""
    match = re.fullmatch(
        r"Duplicate experimental entry: (\d+) and (\d+) share "
        r"observed=(?P<obs>.*?) fixes=.*? with (?P<reason>.*)",
        msg,
    )
    if match is None:
        return msg
    return (
        f"ids {match.group(1)} / {match.group(2)}  "
        f"observed={match.group('obs')}  {match.group('reason')}"
    )


def _norm_msg(msg: str) -> str:
    """Collapse row-/experiment-specific ids so similar errors group together."""
    text = re.sub(r"\(row id=\d+\)", "(row id=...)", msg)
    text = re.sub(r"\bexperiment \d+\b", "experiment ...", text)
    text = re.sub(r"\bmissing id \d+\b", "missing id ...", text)
    text = re.sub(r"\breferences id \d+\b", "references id ...", text)
    text = re.sub(r"\bon experiment \d+\b", "on experiment ...", text)
    text = re.sub(r"\bhigher than \d+\b", "higher than ...", text)
    text = re.sub(r"\blower than \d+\b", "lower than ...", text)
    text = re.sub(r"\bsame as \d+\b", "same as ...", text)
    text = re.sub(r"\bDuplicate id \d+\b", "Duplicate id ...", text)
    # Keep co-located "ids A / B" intact — those ids are the report.
    return text


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


def _report_weight_cleanups(n_rows: int) -> None:
    print(f"weight cleanups applied (empty/? -> 1): {n_rows}")


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
# - fill empty / `?` weights with `1`
#
# Then apply `FIXES_OVERRIDES` so shared nodes keep the override column.
# (`-` as empty intervention is already handled in `nav.py`.)

# %%
cleaned_rows: list[dict[str, str]] = []
n_weight_cleanups = 0
for r in rows:
    nr = dict(r)
    cleaned_w, changed = clean_weight(nr.get(WEIGHT_COL, ""))
    nr[WEIGHT_COL] = cleaned_w
    if changed:
        n_weight_cleanups += 1
    cleaned_rows.append(nr)

override_events = apply_fixes_overrides(
    cleaned_rows,
    FIXES_OVERRIDES,
    id_col=ID_COL,
)
_report_overrides(override_events)
_report_weight_cleanups(n_weight_cleanups)

# %% [markdown]
# ## 5. Row-level + comparative diagnostics (after cleanup / override)
#
# Mirrors ``import_NAV_exps`` checks: fixes parse/merge, observed-in-fixes,
# weights, outcome parse, co-located outcomes, and comparative pairs
# (missing other id, different observed, missing/wrong reciprocal).

# %%
weight_counts = Counter((r.get(WEIGHT_COL) or "").strip() for r in cleaned_rows)
print("Weight value counts:", weight_counts.most_common())

parse_errors: list[tuple[str, str, str, str]] = []
merge_errors: list[tuple[str, str]] = []
observed_in_fixes: list[tuple[str, str]] = []
weight_errors: list[tuple[str, str]] = []
id_errors: list[tuple[str, str]] = []
outcome_errors: list[tuple[str, str]] = []
duplicate_id_errors: list[tuple[str, str]] = []
parsed_rows: list[ParsedRow] = []
seen_exp_ids: set[int] = set()

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
        continue

    w = (r.get(WEIGHT_COL) or "").strip()
    try:
        float(w)
    except Exception:
        weight_errors.append((rid, w))
        continue

    try:
        exp_id = int(rid)
    except Exception:
        id_errors.append((rid, f"non-integer id={rid!r}"))
        continue
    if exp_id in seen_exp_ids:
        duplicate_id_errors.append((rid, f"Duplicate id {exp_id}"))
        continue

    raw_outcome = (r.get(OUTCOME_COL) or "").strip()
    try:
        outcome = parse_outcome(raw_outcome)
    except Exception as exc:
        outcome_errors.append((rid, str(exc)))
        continue

    seen_exp_ids.add(exp_id)
    parsed_rows.append(
        ParsedRow(
            exp_id=exp_id,
            row_id=rid,
            fixes=fixes,
            observed=observed,
            outcome=outcome,
        )
    )

(
    accepted_rows,
    co_located_errors,
    comparative_errors,
) = _filter_co_located_and_comparative(parsed_rows)

n_comparative = sum(
    1 for row in parsed_rows if isinstance(row.outcome, ComparativeOutcome)
)
print(f"\nrows passing per-row checks: {len(parsed_rows)}")
print(f"  of which comparative outcomes: {n_comparative}")
print(f"rows kept after co-located/comparative: {len(accepted_rows)}")

_report_grouped(
    "fixes parse errors",
    [(rid, msg) for rid, _col, _cell, msg in parse_errors],
)

_report_grouped(
    "fixes merge errors (duplicates / inconsistent nodes)",
    merge_errors,
)

_report_grouped(
    "observed node also in fixes",
    [(rid, f"observed={observed!r}") for rid, observed in observed_in_fixes],
)

_report_grouped(
    f"non-numeric {WEIGHT_COL}",
    [(rid, f"value={w!r}") for rid, w in weight_errors],
)

_report_grouped("non-integer ids", id_errors)
_report_grouped("duplicate ids", duplicate_id_errors)
_report_grouped("outcome parse errors", outcome_errors)
_report_grouped("co-located outcome errors", co_located_errors)
_report_grouped(
    "comparative pair errors (missing id / different observed / reciprocal)",
    comparative_errors,
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

print("\ncomparative failures (one example row per message):")
seen_comp: set[str] = set()
parsed_by_rid = {row.row_id: row for row in parsed_rows}
parsed_by_exp_id = {row.exp_id: row for row in parsed_rows}
for rid, msg in comparative_errors:
    key = _norm_msg(msg)
    if key in seen_comp:
        continue
    seen_comp.add(key)
    ids = [r for r, m in comparative_errors if _norm_msg(m) == key]
    row = parsed_by_rid.get(rid)
    print(f"  ids={', '.join(ids)}")
    print(f"    example id={rid}: {msg}")
    if row is not None:
        print(f"      observed={row.observed!r}  outcome={row.outcome}")
        if isinstance(row.outcome, ComparativeOutcome):
            other = parsed_by_exp_id.get(row.outcome.other_id)
            if other is not None:
                print(
                    f"      other id={other.row_id}  "
                    f"observed={other.observed!r}  outcome={other.outcome}"
                )

# %% [markdown]
# ## 7. Import trial
#
# Rows that still violate NAV constraints (including comparative pairs and
# co-located outcomes) are dropped for this trial.

# %%
accepted_ids = {row.row_id for row in accepted_rows}
kept: list[dict[str, str]] = []
dropped: list[tuple[str, str]] = []

per_row_drop_msgs: dict[str, str] = {}
for rid, msg in (
    [(r, msg) for r, _c, _cell, msg in parse_errors]
    + merge_errors
    + [(r, f"Node '{obs}' appears in both fixes and observed") for r, obs in observed_in_fixes]
    + [(r, f"non-numeric {WEIGHT_COL}={w!r}") for r, w in weight_errors]
    + id_errors
    + duplicate_id_errors
    + outcome_errors
    + co_located_errors
    + comparative_errors
):
    per_row_drop_msgs.setdefault(rid, msg)

for r in cleaned_rows:
    rid = (r.get(ID_COL) or "").strip()
    if rid in accepted_ids:
        kept.append(r)
    else:
        dropped.append((rid, per_row_drop_msgs.get(rid, "failed NAV checks")))

print(f"kept after cleanup/override/comparative: {len(kept)} / {len(rows)}")
_report_grouped("dropped", dropped)

trial_path = DATA.with_name(f"{DATA.stem}_trial{DATA.suffix or '.csv'}")
with trial_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(kept)
print(f"\nwrote cleaned trial CSV: {trial_path}  ({len(kept)} rows)")

try:
    exps = import_NAV_exps(
        trial_path,
        weight_col=WEIGHT_COL,
        observed_col=OBSERVED_COL,
        outcome_col=OUTCOME_COL,
        fixes_cols=FIXES_COLS,
    )
    print(f"\ncleaned trial import OK: {len(exps)} experiments")
    print("first:", exps[0])
    print(f"n fixes (first): {len(exps[0].fixes)}")
    n_comp = sum(
        1 for e in exps if isinstance(e.outcome, ComparativeOutcome)
    )
    print(f"comparative outcomes kept: {n_comp}")
except Exception as exc:
    print(f"\ncleaned trial import FAILED: {type(exc).__name__}: {exc}")
