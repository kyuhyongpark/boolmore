import csv

from boolmore.io.export import export_phenotype_results
from boolmore.core.experiment import Experiment
from boolmore.core.prediction import Prediction


def test_export_phenotype_results(tmp_path):
    exp = Experiment(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 1),),
        phenotype=(("P", 1),),
        expected_exists=True,
        weight=2.0,
    )

    pred = Prediction(
        id=1,
        perturbation=(("A", 1),),
        sources=(("S", 1),),
        phenotype=(("P", 1),),
        found_phenotypes=[],
        predicted_exists=True,
        agreement=1.0,
        score=2.0,
    )

    file_path = tmp_path / "results.csv"

    export_phenotype_results([exp], [pred], str(file_path))

    assert file_path.exists()

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["expected_exists"] == "True"
    assert rows[0]["predicted_exists"] == "True"
    assert rows[0]["score"] == "2.0"