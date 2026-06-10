import csv

from boolmore.core.experiment import Experiment
from boolmore.core.prediction import PhenotypePrediction
from boolmore.core.conversions import assignment_to_dict


def export_phenotype_results(
    experiments: list[Experiment],
    predictions: list[PhenotypePrediction],
    filename: str,
) -> None:
    """
    Print a summary of each experiment/prediction pair and save it to a CSV.

    Experiments and Predictions are matched by their id.
    """
    pred_by_id = {pred.id: pred for pred in predictions}

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

    rows = []

    for exp in experiments:
        pred = pred_by_id.get(exp.id)
        if pred is None:
            raise ValueError(f"No Prediction found for Experiment id={exp.id}")

        row = {
            "id": exp.id,
            "perturbation": assignment_to_dict(exp.perturbation),
            "sources": assignment_to_dict(exp.sources),
            "phenotype": assignment_to_dict(exp.phenotype),
            "expected_exists": exp.expected_exists,
            "predicted_exists": pred.predicted_exists,
            "agreement": pred.agreement,
            "weight": exp.weight,
            "score": pred.score,
            "found_phenotypes": pred.found_phenotypes,
        }
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