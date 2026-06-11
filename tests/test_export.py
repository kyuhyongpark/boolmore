import csv

from boolmore.io.export import export_phenotype_results
from boolmore.core.experiment import PhenotypeExperiment
from boolmore.core.prediction import PhenotypePrediction
from boolmore.eval.score import EvaluationItemScore


def test_export_phenotype_results(tmp_path):
    exp = PhenotypeExperiment(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 1),),
        phenotype=(("P", 1),),
        expected_exists=True,
        weight=2.0,
    )

    pred = PhenotypePrediction(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 1),),
        phenotype=(("P", 1),),
        found_phenotypes=[],
        predicted_exists=True,
    )

    score_item = EvaluationItemScore(
        id=1,
        weight=2.0,
        agreement=1.0,
        score=2.0,
    )

    file_path = tmp_path / "results.csv"

    export_phenotype_results([exp], [pred], [score_item], str(file_path))

    assert file_path.exists()

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["expected_exists"] == "True"
    assert rows[0]["predicted_exists"] == "True"
    assert rows[0]["score"] == "2.0"