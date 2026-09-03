import csv

from boolmore.core.experiment import PhenotypeExperiment
from boolmore.core.prediction import PhenotypePrediction
from boolmore.eval.score import EvaluationItemScore
from boolmore.core.conversions import assignment_to_dict


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