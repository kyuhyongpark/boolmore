"""Export models and per-experiment score reports to disk."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

from boolmore.boolean_functions import assignment_to_dict
from boolmore.evaluation.score import EvaluationItemScore, EvaluationResult
from boolmore.experiment import PhenotypeExperiment
from boolmore.genetic.population import Candidate
from boolmore.inference.prediction import PhenotypePrediction
from boolmore.model import Model


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
        rows.append({
            "id": item.id,
            "hierarchy": item.hierarchy,
            "fixes": ",".join(f"{n}={v}" for n, v in item.fixes),
            "observed_node": item.observed,
            "experimental_outcome": item.outcome,
            "predict_value": item.predict_value,
            "attractor_agreement": item.agreement,
            "score": item.score,
        })
    output = Path(path)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.resolve()


def export_phenotype_report(
    experiments: list[PhenotypeExperiment],
    predictions: list[PhenotypePrediction],
    score_items: list[EvaluationItemScore],
    path: str | Path,
) -> Path:
    """Write a per-experiment phenotype score CSV.

    Experiments, predictions, and score items are matched by ``id``.

    Parameters
    ----------
    experiments : list of PhenotypeExperiment
    predictions : list of PhenotypePrediction
    score_items : list of EvaluationItemScore
    path : str or Path
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If a prediction or score item is missing for an experiment id.
    """
    pred_by_id = {pred.id: pred for pred in predictions}
    score_by_id = {score_item.id: score_item for score_item in score_items}

    rows = []
    for exp in experiments:
        pred = pred_by_id.get(exp.id)
        score_item = score_by_id.get(exp.id)
        if pred is None:
            raise ValueError(f"No Prediction found for Experiment id={exp.id}")
        if score_item is None:
            raise ValueError(f"No Score found for Experiment id={exp.id}")

        row = {
            "id": exp.id,
            "perturbation": assignment_to_dict(exp.perturbation),
            "sources": assignment_to_dict(exp.sources),
            "phenotype": assignment_to_dict(exp.phenotype),
            "expected_exists": exp.expected_exists,
            "predicted_exists": pred.predicted_exists,
            "agreement": score_item.agreement,
            "weight": exp.weight,
            "score": score_item.score,
            "maximal_trap_spaces": pred.maximal_trap_spaces,
        }
        row.update(exp.metadata)
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
      (NAV report when NAV fields are present; otherwise a generic item-scores CSV)

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
    if result.item_scores and any(item.fixes is not None for item in result.item_scores):
        export_nav_report(result, score_path)
    else:
        export_item_scores_report(result, score_path)

    return Path(stem).resolve()
