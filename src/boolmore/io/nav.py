"""Import NAV-style experiments from TSV or CSV files."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path

from boolmore.experiment import NAVExperiment
from boolmore.outcome import (
    format_outcome_range,
    parse_outcome_cell,
    resolve_outcome_range,
)

_DEFAULT_ID_COL: tuple[str, ...] = ("ID",)
_DEFAULT_WEIGHT_COL: tuple[str, ...] = ("Score", "Weight")
_DEFAULT_OBSERVED_COL: tuple[str, ...] = ("Observed node",)
_DEFAULT_OUTCOME_COL: tuple[str, ...] = ("Categorization", "Value")
_DEFAULT_FIXES_COLS: tuple[str, ...] = ("Source", "Perturbation", "Intervention")
_OUTCOME_RANGE_HEADER = "Outcome range"


def _normalize_header(name: str) -> str:
    """Normalize a column name for matching."""
    text = name.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _as_alias_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_single_column(
    headers: list[str],
    aliases: str | Sequence[str],
    role: str,
) -> str:
    """Return the unique original header matching one of ``aliases``.

    Raises
    ------
    ValueError
        If zero or more than one header matches.
    """
    alias_norms = {_normalize_header(a) for a in _as_alias_list(aliases)}
    matches = [h for h in headers if _normalize_header(h) in alias_norms]
    if not matches:
        raise ValueError(
            f"No column found for '{role}'. "
            f"Looked for aliases {_as_alias_list(aliases)!r} among headers {headers!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous columns for '{role}': {matches!r} "
            f"(aliases {_as_alias_list(aliases)!r})"
        )
    return matches[0]


def _resolve_fixes_columns(
    headers: list[str],
    fixes_cols: Sequence[str],
) -> list[str]:
    """Return original headers that match ``fixes_cols``, skipping absent names.

    Raises
    ------
    ValueError
        If a listed name matches more than one header.
    """
    header_by_norm: dict[str, list[str]] = {}
    for h in headers:
        header_by_norm.setdefault(_normalize_header(h), []).append(h)

    resolved: list[str] = []
    seen: set[str] = set()
    for name in fixes_cols:
        norm = _normalize_header(name)
        candidates = header_by_norm.get(norm, [])
        if not candidates:
            continue
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous columns for fixes name {name!r}: {candidates!r}"
            )
        header = candidates[0]
        if header not in seen:
            resolved.append(header)
            seen.add(header)
    return resolved


def _parse_fix_token(token: str) -> tuple[str, int]:
    """Parse one assignment token as ``Node=0|1`` or legacy ``Node KO|CA``."""
    token = token.strip()
    if not token:
        raise ValueError("Empty assignment token")

    if "=" in token:
        node, value_str = token.split("=", 1)
        node = node.strip()
        value_str = value_str.strip()
        if not node:
            raise ValueError(f"Empty node in assignment '{token}'")
        if value_str not in {"0", "1"}:
            raise ValueError(f"Invalid value '{value_str}' in assignment '{token}'")
        return node, int(value_str)

    parts = token.split()
    if len(parts) != 2:
        raise ValueError(
            f"Invalid assignment '{token}'. "
            "Expected 'Node=0|1' or legacy 'Node KO|CA'"
        )
    node, value_str = parts[0].strip(), parts[1].strip().upper()
    if not node:
        raise ValueError(f"Empty node in assignment '{token}'")
    if value_str == "KO":
        return node, 0
    if value_str == "CA":
        return node, 1
    raise ValueError(
        f"Invalid assignment '{token}'. "
        "Expected 'Node=0|1' or legacy 'Node KO|CA'"
    )


def _parse_fixes_cell(cell: str) -> list[tuple[str, int]]:
    """Parse a comma-separated fixes cell into ``(node, value)`` pairs.

    A blank cell, or a cell that is only ``-``, is treated as empty.
    """
    text = cell.strip() if cell else ""
    if not text or text == "-":
        return []
    items: list[tuple[str, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(_parse_fix_token(part))
    return items


def _merge_fixes(
    cells: Sequence[str],
    *,
    row_id: str | None = None,
) -> tuple[tuple[str, int], ...]:
    """Merge fixes cells; raise if any node name appears more than once."""
    merged: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    for cell in cells:
        for node, value in _parse_fixes_cell(cell):
            if node in seen:
                prev = seen[node]
                where = f" (row id={row_id})" if row_id is not None else ""
                if prev != value:
                    raise ValueError(
                        f"Inconsistent fixes for node '{node}'{where}: "
                        f"{prev} vs {value}"
                    )
                raise ValueError(f"Duplicate node '{node}' in fixes{where}")
            seen[node] = value
            merged.append((node, value))
    return tuple(sorted(merged, key=lambda x: x[0]))


def _reformat_fixes_cell(cell: str) -> str:
    """Rewrite a fixes cell to standard ``name=value`` tokens.

    A cell that is only ``-`` is rewritten as empty.
    """
    if not cell or not cell.strip():
        return cell
    if cell.strip() == "-":
        return ""
    tokens: list[str] = []
    for part in cell.split(","):
        part = part.strip()
        if not part:
            continue
        node, value = _parse_fix_token(part)
        tokens.append(f"{node}={value}")
    return ",".join(tokens)


def _fixes_cell_needs_rewrite(cell: str) -> bool:
    """Return True when a fixes cell is not already in canonical form."""
    return _reformat_fixes_cell(cell) != cell


def _outcome_range_header_index(headers: Sequence[str]) -> int | None:
    """Return the index of an existing Outcome range column, if any."""
    target = _normalize_header(_OUTCOME_RANGE_HEADER)
    for i, header in enumerate(headers):
        if _normalize_header(header) == target:
            return i
    return None


def _delimiter_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    raise ValueError(f"Unsupported NAV data format: {path} (expected .tsv or .csv)")


def _valid_lines(file):
    for line in file:
        if line.startswith("#") or line.isspace():
            continue
        yield line


def import_NAV_exps(
    location: str | Path,
    *,
    id_col: str | Sequence[str] = _DEFAULT_ID_COL,
    weight_col: str | Sequence[str] = _DEFAULT_WEIGHT_COL,
    observed_col: str | Sequence[str] = _DEFAULT_OBSERVED_COL,
    outcome_col: str | Sequence[str] = _DEFAULT_OUTCOME_COL,
    fixes_cols: Sequence[str] = _DEFAULT_FIXES_COLS,
) -> list[NAVExperiment]:
    """Import NAV experiments from a ``.tsv`` or ``.csv`` file.

    Columns are resolved by normalized header name (case-insensitive; spaces and
    hyphens become underscores), not by position.

    Parameters
    ----------
    location : str or Path
        Path to a ``.tsv`` or ``.csv`` file.
    id_col, weight_col, observed_col, outcome_col :
        Alias or aliases for each required single-column role. Exactly one
        header must match.
    fixes_cols :
        Columns whose cells are concatenated into ``NAVExperiment.fixes``.
        Absent names are skipped. Tokens may be ``Node=0|1`` or legacy
        ``Node KO|CA``. A cell that is only ``-`` is treated as empty.

    Notes
    -----
    Raises ``ValueError`` on duplicate experiment ids, invalid outcomes, an
    inconsistent or empty ``Outcome range`` column, or when the observed node
    also appears in that row's fixes.

    When an ``Outcome range`` column is present, each row is checked against
    the range resolved from the outcome column; matching ranges are treated as
    already audited (no rewrite of outcomes). When the column is absent and
    outcomes are not already ``lo,hi`` strings, or when fixes need normalizing,
    a ``{stem}_preprocessed.csv`` is written beside the input (always
    overwritten). No file is written when preprocessing is unnecessary, or
    when the input stem already ends with ``_preprocessed``.
    """
    path = Path(location)
    delimiter = _delimiter_for_path(path)

    with path.open("r", newline="") as file:
        reader = csv.reader(_valid_lines(file), delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty NAV data file: {path}") from exc

        headers = [h.strip() for h in headers]
        id_header = _resolve_single_column(headers, id_col, "id")
        weight_header = _resolve_single_column(headers, weight_col, "weight")
        observed_header = _resolve_single_column(headers, observed_col, "observed")
        outcome_header = _resolve_single_column(headers, outcome_col, "outcome")
        fixes_headers = _resolve_fixes_columns(headers, fixes_cols)

        header_index = {name: i for i, name in enumerate(headers)}
        existing_range_idx = _outcome_range_header_index(headers)

        raw_rows: list[list[str]] = []
        outcome_range_strings: list[str] = []
        exps: list[NAVExperiment] = []
        interventions: list[tuple] = []
        seen_ids: set[int] = set()
        needs_fixes_rewrite = False
        needs_outcome_ranges = False

        for row in reader:
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            raw_rows.append(list(row))

            def cell(name: str) -> str:
                return row[header_index[name]].strip()

            row_id_str = cell(id_header)
            exp_id = int(row_id_str)
            if exp_id in seen_ids:
                raise ValueError(f"Duplicate id {exp_id}")
            seen_ids.add(exp_id)

            fixes_cells = [cell(h) for h in fixes_headers]
            for header in fixes_headers:
                if _fixes_cell_needs_rewrite(row[header_index[header]]):
                    needs_fixes_rewrite = True
                    break

            fixes = _merge_fixes(
                fixes_cells,
                row_id=row_id_str or None,
            )
            observed = cell(observed_header)
            fixed_nodes = {node for node, _ in fixes}
            if observed in fixed_nodes:
                raise ValueError(
                    f"Node '{observed}' appears in both fixes and observed "
                    f"(row id={exp_id})"
                )

            raw_outcome = cell(outcome_header)
            outcome = parse_outcome_cell(raw_outcome)
            lo, hi = resolve_outcome_range(outcome)
            range_str = format_outcome_range(lo, hi)
            outcome_range_strings.append(range_str)

            if existing_range_idx is not None:
                raw_range = row[existing_range_idx].strip()
                if not raw_range:
                    raise ValueError(
                        f"Empty Outcome range (row id={exp_id})"
                    )
                try:
                    provided = parse_outcome_cell(raw_range)
                    provided_lo, provided_hi = resolve_outcome_range(provided)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid Outcome range {raw_range!r} (row id={exp_id})"
                    ) from exc
                provided_str = format_outcome_range(provided_lo, provided_hi)
                if provided_str != range_str:
                    raise ValueError(
                        f"Inconsistent Outcome range (row id={exp_id}): "
                        f"outcome {raw_outcome!r} resolves to {range_str!r}, "
                        f"but Outcome range is {provided_str!r}"
                    )
            elif raw_outcome != range_str:
                needs_outcome_ranges = True

            exp = NAVExperiment(
                id=exp_id,
                fixes=fixes,
                observed=observed,
                outcome=outcome,
                weight=float(cell(weight_header)),
            )

            if fixes not in interventions:
                interventions.append(fixes)
            else:
                for other_exp in exps:
                    if fixes == other_exp.fixes:
                        assert exp.observed != other_exp.observed, (
                            f"{other_exp.id} and {exp.id} are duplicates"
                        )

            exps.append(exp)

    skip_write = path.stem.endswith("_preprocessed")
    needs_write = (needs_fixes_rewrite or needs_outcome_ranges) and not skip_write
    if needs_write:
        out_headers = list(headers)
        if existing_range_idx is None:
            out_headers.append(_OUTCOME_RANGE_HEADER)

        fixes_indices = {header_index[h] for h in fixes_headers}
        rewritten_rows: list[list[str]] = []
        for row, range_str in zip(raw_rows, outcome_range_strings, strict=True):
            new_row = list(row[: len(headers)])
            if len(new_row) < len(headers):
                new_row = new_row + [""] * (len(headers) - len(new_row))
            for idx in fixes_indices:
                new_row[idx] = _reformat_fixes_cell(new_row[idx])
            if existing_range_idx is None:
                new_row.append(range_str)
            else:
                new_row[existing_range_idx] = range_str
            rewritten_rows.append(new_row)

        preprocessed_path = path.with_name(f"{path.stem}_preprocessed.csv")
        with preprocessed_path.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(out_headers)
            writer.writerows(rewritten_rows)

    return exps
