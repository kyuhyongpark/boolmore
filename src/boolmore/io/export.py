import csv
import os
import pickle

from boolmore import boolean_functions as bf
from boolmore.model import Model
from boolmore.experiment import PhenotypeExperiment
from boolmore.inference.prediction import PhenotypePrediction
from boolmore.evaluation.score import EvaluationItemScore
from boolmore.boolean_functions import assignment_to_dict


def export_phenotype_results(
    experiments: list[PhenotypeExperiment],
    predictions: list[PhenotypePrediction],
    score_items: list[EvaluationItemScore],
    filename: str,
) -> None:
    """
    Print a summary of each experiment/prediction pair and save it to a CSV.

    Experiments and Predictions are matched by their id.
    """
    pred_by_id = {pred.id: pred for pred in predictions}
    score_by_id = {score_item.id: score_item for score_item in score_items}

    headers = [
        "id",
        "perturbation",
        "sources",
        "phenotype",
        "expected_exists",
        "predicted_exists",
        "agreement",
        "weight",
        "score",
        "found_phenotypes",
    ]

    # Add metadata columns after the standard columns
    metadata_headers = sorted({
        key
        for exp in experiments
        for key in exp.metadata
    })

    headers.extend(metadata_headers)

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
            "found_phenotypes": pred.found_phenotypes,
        }

        row.update(exp.metadata)
        rows.append(row)

    # Print nicely
    widths = {
        h: max(len(h), *(len(str(r[h])) for r in rows))
        for h in headers
    }

    print(" | ".join(f"{h:<{widths[h]}}" for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))

    for row in rows:
        print(
            " | ".join(
                f"{str(row[h]):<{widths[h]}}"
                for h in headers
            )
        )

    # Save to CSV
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def export_model(model:Model, file_name: str, details: bool = True):
    """Export a model as a .bnet file and pickle its primes."""

    bnet_file_name = file_name + ".bnet"
    pkl_file_name = file_name + ".pkl"

    with open(bnet_file_name, "w") as fp:
        fp.write(model.info() + "\n")

        fp.write("targets,\tfactors\n")

        primes = {k: model.primes[k] for k in sorted(model.primes)}
        for k in primes:
            fp.write(bf.prime2bnet(k, primes[k]) + "\n")

    with open(pkl_file_name, "wb") as f:
        pickle.dump(model.primes, f)

    print("Exported generated model to", os.path.abspath(bnet_file_name))
    print("Pickled primes to", os.path.abspath(pkl_file_name))