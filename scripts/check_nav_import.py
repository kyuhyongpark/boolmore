# %% [markdown]
# NAV import check
#
# Probe whether a NAV-style CSV/TSV can be loaded with
# `boolmore.io.nav.import_NAV_exps`.
#
# Edit the config cell below to point at another dataset. Set `FIXES_COLS`
# when the file uses non-default fixes headers (e.g. breast cancer).

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
    _resolve_single_column,
    import_NAV_exps,
)
from boolmore.outcome import ComparativeOutcome, OutcomeType, parse_outcome

# %% [markdown]
# ## Config
#
# Change `DATA` / `FIXES_COLS` for another dataset. Leave `FIXES_COLS` as
# ``None`` to use ``import_NAV_exps`` defaults.

# %%
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "case_study" / "breast_cancer" / "breast_cancer_data.csv"

FIXES_COLS: Sequence[str] | None = (
    "Signals",
    "Drug treatment/intervention",
    "Cell line characteristic profile",
    "Additional features of the cell population",
)

# %%
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


# %% [markdown]
# ## 1. Inspect headers vs `import_NAV_exps` defaults

# %%
with DATA.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = list(reader.fieldnames or [])
    rows = list(reader)

ID_COL = _resolve_single_column(headers, _DEFAULT_ID_COL, "id")
WEIGHT_COL = _resolve_single_column(headers, _DEFAULT_WEIGHT_COL, "weight")
OBSERVED_COL = _resolve_single_column(headers, _DEFAULT_OBSERVED_COL, "observed")
OUTCOME_COL = _resolve_single_column(headers, _DEFAULT_OUTCOME_COL, "outcome")
active_fixes = list(FIXES_COLS) if FIXES_COLS is not None else list(_DEFAULT_FIXES_COLS)

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
print(f"  fixes={active_fixes!r}")

# %% [markdown]
# ## 2. Default import (expected to fail on mismatched fixes columns)

# %%
try:
    exps = import_NAV_exps(DATA)
    print(f"default import OK: {len(exps)} experiments")
except Exception as exc:
    print(f"default import FAILED: {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 3. Import with custom fixes columns (if configured)

# %%
nav_kwargs: dict = {}
if FIXES_COLS is not None:
    nav_kwargs["fixes_cols"] = FIXES_COLS

try:
    exps = import_NAV_exps(DATA, **nav_kwargs)
    print(f"import OK: {len(exps)} experiments")
    print("first:", exps[0])
except Exception as exc:
    print(f"import FAILED: {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 4. Row-level + comparative diagnostics
#
# Mirrors ``import_NAV_exps`` checks: fixes parse/merge, observed-in-fixes,
# weights, outcome parse, co-located outcomes, and comparative pairs
# (missing other id, different observed, missing/wrong reciprocal).

# %%
weight_counts = Counter((r.get(WEIGHT_COL) or "").strip() for r in rows)
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

for r in rows:
    rid = (r.get(ID_COL) or "").strip()
    cells: list[str] = []
    row_parse_ok = True
    for col in active_fixes:
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
# ## 5. Example problematic cells

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
    r = next(x for x in rows if (x.get(ID_COL) or "").strip() == rid)
    print(f"  ids={', '.join(ids)}")
    print(f"    example id={rid}: {msg}")
    for col in active_fixes:
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
# ## 6. Import trial
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

for r in rows:
    rid = (r.get(ID_COL) or "").strip()
    if rid in accepted_ids:
        kept.append(r)
    else:
        dropped.append((rid, per_row_drop_msgs.get(rid, "failed NAV checks")))

print(f"kept after diagnostics: {len(kept)} / {len(rows)}")
_report_grouped("dropped", dropped)

trial_path = DATA.with_name(f"{DATA.stem}_trial{DATA.suffix or '.csv'}")
with trial_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(kept)
print(f"\nwrote trial CSV: {trial_path}  ({len(kept)} rows)")

try:
    exps = import_NAV_exps(trial_path, **nav_kwargs)
    print(f"\ntrial import OK: {len(exps)} experiments")
    print("first:", exps[0])
    print(f"n fixes (first): {len(exps[0].fixes)}")
    n_comp = sum(
        1 for e in exps if isinstance(e.outcome, ComparativeOutcome)
    )
    print(f"comparative outcomes kept: {n_comp}")
except Exception as exc:
    print(f"\ntrial import FAILED: {type(exc).__name__}: {exc}")
