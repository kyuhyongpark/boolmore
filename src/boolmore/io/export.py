"""Export models and per-experiment score reports to disk."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

from boolmore.boolean_functions import assignment_to_dict
from boolmore.evaluation.agreement import (
    _related_node_names,
    entity_max_level,
)
from boolmore.evaluation.score import EvaluationResult, Evaluator
from boolmore.genetic.population import Candidate
from boolmore.model import Model
from boolmore.outcome import (
    ComparativeOutcome,
    format_outcome,
    format_outcome_range,
    resolve_outcome_range,
)


def export_nav_report(
    result: EvaluationResult,
    path: str | Path,
    related_nodes: dict[str, list[str]] | None = None,
) -> Path:
    """Write a per-experiment NAV score CSV from an :class:`EvaluationResult`.

    Parameters
    ----------
    result : EvaluationResult
        Must contain NAV ``item_scores`` (with ``fixes`` set).
    path : str or Path
        Output CSV path.
    related_nodes : dict of str to list of str, optional
        Observed-base → sibling names used to resolve comparative
        higher/lower ``outcome_range`` (``1/(2k)``). When omitted, Boolean
        ``k=1`` (``0.5``) is assumed for those outcomes.

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
        max_level: int | None = None
        if (
            isinstance(item.outcome, ComparativeOutcome)
            and item.outcome.relation != "same"
            and item.observed is not None
            and related_nodes is not None
            and item.observed in related_nodes
        ):
            max_level = entity_max_level(
                item.observed, related_nodes[item.observed]
            )
        lo, hi = resolve_outcome_range(item.outcome, max_level=max_level)
        def _fmt(assignment) -> str:
            if not assignment:
                return ""
            return ",".join(f"{n}={v}" for n, v in assignment)

        rows.append({
            "id": item.id,
            "weight": item.weight,
            "hierarchy": item.hierarchy,
            "context": _fmt(item.context or ()),
            "intervention": _fmt(item.intervention or ()),
            "fixes": _fmt(item.fixes),
            "observed": item.observed,
            "outcome": format_outcome(item.outcome),
            "outcome_range": format_outcome_range(lo, hi),
            "predict_value": item.predict_value,
            "agreement": item.agreement,
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


def export_candidate(
    candidate: Candidate,
    export_name: str,
    evaluator: Evaluator | None = None,
) -> Path:
    """Export a GA candidate model and its evaluation score report.

    Writes:

    - ``{export_name}_{id}_gen{generation}.bnet`` with ``candidate.info()`` as header
    - ``{export_name}_{id}_gen{generation}.pkl`` with pickled primes
    - ``{export_name}_{id}_gen{generation}_score.csv`` with per-experiment scores
      (NAV or phenotype report from ``eval_result.kind``)

    Parameters
    ----------
    candidate : Candidate
        Candidate with a non-``None`` ``eval_result``.
    export_name : str
        Prefix for output file names.
    evaluator : Evaluator or None
        Required when ``eval_result.item_scores`` is empty (GA ran with
        ``detail=False``). Re-evaluates once with ``detail=True`` before
        writing the score CSV and stores the full result on the candidate.

    Returns
    -------
    Path
        Absolute path of the output stem (without extension).

    Raises
    ------
    ValueError
        If ``candidate.eval_result`` is ``None``, ``item_scores`` are empty
        and ``evaluator`` is missing, or ``kind`` is not ``"NAV"`` or
        ``"Phenotype"``.
    """
    if candidate.eval_result is None:
        raise ValueError("export_candidate requires candidate.eval_result")

    result = candidate.eval_result
    if not result.item_scores:
        if evaluator is None:
            raise ValueError(
                "export_candidate requires evaluator when item_scores are empty"
            )
        result = evaluator.evaluate(
            candidate.model, candidate.id, detail=True
        )
        candidate.eval_result = result

    stem = f"{export_name}_{candidate.id}_gen{candidate.generation}"
    export_model(candidate.model, stem, header=candidate.info())

    score_path = f"{stem}_score.csv"
    if result.kind == "NAV":
        observed = {
            item.observed
            for item in result.item_scores
            if item.observed is not None
        }
        related_nodes = {
            base: _related_node_names(base, candidate.model.primes)
            for base in observed
        }
        export_nav_report(result, score_path, related_nodes=related_nodes)
    elif result.kind == "Phenotype":
        export_phenotype_report(result, score_path)
    else:
        raise ValueError(
            f"export_candidate requires NAV or Phenotype result, got {result.kind!r}"
        )

    return Path(stem).resolve()
