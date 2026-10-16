"""Export models and per-experiment score reports to disk."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

from boolmore.boolean_functions import assignment_to_dict
from boolmore.evaluation.score import EvaluationResult
from boolmore.genetic.population import Candidate
from boolmore.model import Model
from boolmore.outcome import format_outcome, format_outcome_range, resolve_outcome_range


def export_item_scores_report(result: EvaluationResult, path: str | Path) -> Path:
    """Write a generic per-experiment score CSV from an :class:`EvaluationResult`.

    Columns are ``id``, ``weight``, ``agreement``, and ``score``.

    Parameters
    ----------
    result : EvaluationResult
    path : str or Path
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    rows = [
        {
            "id": item.id,
            "weight": item.weight,
            "agreement": item.agreement,
            "score": item.score,
        }
        for item in result.item_scores
    ]
    output = Path(path)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.resolve()


def export_nav_report(result: EvaluationResult, path: str | Path) -> Path:
    """Write a per-experiment NAV score CSV from an :class:`EvaluationResult`.

    Parameters
    ----------
    result : EvaluationResult
        Must contain NAV ``item_scores`` (with ``fixes`` set).
    path : str or Path
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If any item score is missing NAV fields.
    """
    rows = []
    for item in result.item_scores:
        if item.fixes is None:
            raise ValueError("export_nav_report requires NAV item scores")
        if item.outcome is None:
            raise ValueError("export_nav_report requires NAV item scores with outcome")
        lo, hi = resolve_outcome_range(item.outcome)
        rows.append({
            "id": item.id,
            "hierarchy": item.hierarchy,
            "fixes": ",".join(f"{n}={v}" for n, v in item.fixes),
            "observed_node": item.observed,
            "experimental_outcome": format_outcome(item.outcome),
            "outcome_range": format_outcome_range(lo, hi),
            "predict_value": item.predict_value,
            "attractor_agreement": item.agreement,
            "score": item.score,
        })
    output = Path(path)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.resolve()


def export_phenotype_report(result: EvaluationResult, path: str | Path) -> Path:
    """Write a per-experiment phenotype score CSV from an :class:`EvaluationResult`.

    Parameters
    ----------
    result : EvaluationResult
        Must contain phenotype ``item_scores`` (with ``predicted_exists`` set).
    path : str or Path
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If any item score is missing phenotype fields.
    """
    rows = []
    for item in result.item_scores:
        if item.predicted_exists is None:
            raise ValueError("export_phenotype_report requires phenotype item scores")
        if (
            item.perturbation is None
            or item.sources is None
            or item.phenotype is None
            or item.expected_exists is None
            or item.maximal_trap_spaces is None
        ):
            raise ValueError("export_phenotype_report requires phenotype item scores")

        row = {
            "id": item.id,
            "perturbation": assignment_to_dict(item.perturbation),
            "sources": assignment_to_dict(item.sources),
            "phenotype": assignment_to_dict(item.phenotype),
            "expected_exists": item.expected_exists,
            "predicted_exists": item.predicted_exists,
            "agreement": item.agreement,
            "weight": item.weight,
            "score": item.score,
            "maximal_trap_spaces": item.maximal_trap_spaces,
        }
        if item.metadata:
            row.update(item.metadata)
        rows.append(row)

    output = Path(path)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.resolve()


def export_model(
    model: Model,
    file_name: str,
    *,
    header: str | None = None,
) -> Path:
    """Export a model as a ``.bnet`` file and pickle its primes.

    Parameters
    ----------
    model : Model
        Model to export.
    file_name : str
        Output path stem (without ``.bnet`` / ``.pkl`` extension).
    header : str or None, optional
        Comment header written before the ``targets, factors`` line.
        Defaults to ``model.info()``.

    Returns
    -------
    Path
        Absolute path of the written ``.bnet`` file.
    """
    stem = str(file_name)
    bnet_file_name = stem + ".bnet"
    pkl_file_name = stem + ".pkl"

    if header is None:
        header = model.info()

    with open(bnet_file_name, "w") as fp:
        fp.write(header + "\n")
        fp.write("targets,\tfactors\n")
        fp.write(model.bnet)

    with open(pkl_file_name, "wb") as f:
        pickle.dump(model.primes, f)

    print("Exported generated model to", os.path.abspath(bnet_file_name))
    print("Pickled primes to", os.path.abspath(pkl_file_name))
    return Path(bnet_file_name).resolve()


def export_candidate(candidate: Candidate, export_name: str) -> Path:
    """Export a GA candidate model and its evaluation score report.

    Writes:

    - ``{export_name}_{id}_gen{generation}.bnet`` with ``candidate.info()`` as header
    - ``{export_name}_{id}_gen{generation}.pkl`` with pickled primes
    - ``{export_name}_{id}_gen{generation}_score.csv`` with per-experiment scores
      (NAV or phenotype report from ``eval_result.kind``; otherwise a generic
      item-scores CSV for ``"Mixed"`` or untyped results)

    Parameters
    ----------
    candidate : Candidate
        Candidate with a non-``None`` ``eval_result``.
    export_name : str
        Prefix for output file names.

    Returns
    -------
    Path
        Absolute path of the output stem (without extension).

    Raises
    ------
    ValueError
        If ``candidate.eval_result`` is ``None``.
    """
    if candidate.eval_result is None:
        raise ValueError("export_candidate requires candidate.eval_result")

    stem = f"{export_name}_{candidate.id}_gen{candidate.generation}"
    export_model(candidate.model, stem, header=candidate.info())

    result = candidate.eval_result
    score_path = f"{stem}_score.csv"
    if result.kind == "NAV":
        export_nav_report(result, score_path)
    elif result.kind == "Phenotype":
        export_phenotype_report(result, score_path)
    else:
        export_item_scores_report(result, score_path)

    return Path(stem).resolve()
