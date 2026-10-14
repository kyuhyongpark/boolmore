import json
import unittest
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.experiment import NAVExperiment, PhenotypeExperiment
from boolmore.inference.prediction import (
    _average_related_values,
    _fix_constants,
    _inject_avg_values,
    _related_node_names,
    get_NAV_prediction,
    get_NAV_upper_bound_prediction,
    get_phenotype_prediction,
)
from boolmore.io.load import load_primes
from boolmore.io.nav import import_NAV_exps

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "cad_nav_predictions.json"


class TestFixConstants(unittest.TestCase):

    def setUp(self):
        self.primes = bnet_text2primes("A, !A\nB, A\n")

    def test_fix_to_zero(self):
        result = _fix_constants(self.primes, {"A": 0})
        self.assertEqual(result["A"], [[{}], []])
        self.assertEqual(result["B"], self.primes["B"])

    def test_fix_to_one(self):
        result = _fix_constants(self.primes, {"A": 1})
        self.assertEqual(result["A"], [[], [{}]])
        self.assertEqual(result["B"], self.primes["B"])

    def test_missing_node_raises(self):
        with self.assertRaises(ValueError):
            _fix_constants(self.primes, {"Z": 1})


class TestGetPhenotypePrediction(unittest.TestCase):

    def setUp(self):
        self.bnet = """
        A, A
        B, B
        C, A & B
        D, C & D
        E, !E
        """

        self.primes = bnet_text2primes(self.bnet)

    def test_positive_and_negative_prediction(self):
        experiments = [
            PhenotypeExperiment(
                id=1,
                perturbation=(),
                sources=(("A", 1), ("B", 1)),
                phenotype=(("C", 1),),
                weight=1.0,
                expected_exists=True
            ),
            PhenotypeExperiment(
                id=2,
                perturbation=(),
                sources=(("A", 1), ("B", 0)),
                phenotype=(("C", 1),),
                weight=1.0,
                expected_exists=False
            ),
        ]

        results = get_phenotype_prediction(self.primes, experiments)

        self.assertEqual(len(results), 2)

        self.assertTrue(results[0].predicted_exists)
        self.assertEqual(results[0].maximal_trap_spaces, [{"A": 1, "B": 1, "C": 1}])

        self.assertFalse(results[1].predicted_exists)
        self.assertEqual(results[1].maximal_trap_spaces, [])

    def test_perturbation(self):
        experiments = [
            PhenotypeExperiment(
                id=1,
                perturbation=(("C", 1),),
                sources=(("A", 0), ("B", 0)),
                phenotype=(("D", 1),),
                weight=1.0,
                expected_exists=True
            ),
            PhenotypeExperiment(
                id=2,
                perturbation=(("D", 1),),
                sources=(("A", 1), ("B", 0)),
                phenotype=(("D", 1),),
                weight=1.0,
                expected_exists=True
            ),
        ]

        results = get_phenotype_prediction(self.primes, experiments)

        self.assertEqual(len(results), 2)

        self.assertTrue(results[0].predicted_exists)
        self.assertEqual(results[0].maximal_trap_spaces, [{"A": 0, "B": 0, "C": 1, "D": 1}])

        self.assertTrue(results[1].predicted_exists)
        self.assertEqual(results[1].maximal_trap_spaces, [{"A": 1, "B": 0, "D": 1}])

    def test_cache_reuse_same_perturbation(self):
        experiments = [
            PhenotypeExperiment(
                id=1,
                perturbation=(("C", 1),),
                sources=(("A", 0), ("B", 0)),
                phenotype=(("D", 1),),
                weight=1.0,
                expected_exists=True,
            ),
            # Same perturbation; sources/phenotype satisfied by the trap from exp 1.
            PhenotypeExperiment(
                id=2,
                perturbation=(("C", 1),),
                sources=(("A", 0),),
                phenotype=(("D", 1),),
                weight=1.0,
                expected_exists=True,
            ),
        ]

        results = get_phenotype_prediction(self.primes, experiments)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].predicted_exists)
        self.assertTrue(results[1].predicted_exists)
        self.assertEqual(
            results[1].maximal_trap_spaces,
            [{"A": 0, "B": 0, "C": 1, "D": 1}],
        )

    def test_missing_perturbation_node_raises(self):
        experiments = [
            PhenotypeExperiment(
                id=1,
                perturbation=(("Z", 1),),
                sources=(),
                phenotype=(("A", 1),),
                weight=1.0,
                expected_exists=True,
            ),
        ]
        with self.assertRaises(ValueError):
            get_phenotype_prediction(self.primes, experiments)

    def test_missing_subspace_node_raises(self):
        experiments = [
            PhenotypeExperiment(
                id=1,
                perturbation=(),
                sources=(("Z", 1),),
                phenotype=(("A", 1),),
                weight=1.0,
                expected_exists=True,
            ),
        ]
        with self.assertRaises(ValueError):
            get_phenotype_prediction(self.primes, experiments)


class TestGetNAVPrediction(unittest.TestCase):

    def setUp(self):
        self.primes = bnet_text2primes("A, A\nB, A\n")

    def test_unique_fixes_and_average_values(self):
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
            NAVExperiment(id=2, fixes=(("A", 1),), observed="B", outcome="ON", weight=1.0),
            NAVExperiment(id=3, fixes=(("A", 0),), observed="B", outcome="OFF", weight=1.0),
            # Duplicate intervention should collapse to one prediction entry.
            NAVExperiment(id=4, fixes=(("A", 1),), observed="A", outcome="ON", weight=0.5),
        ]

        preds = get_NAV_prediction(self.primes, experiments)

        self.assertEqual(len(preds), 3)
        self.assertEqual(
            set(preds.keys()),
            {(), (("A", 0),), (("A", 1),)},
        )

        self.assertEqual(
            preds[()].average_values,
            {"A": 0.5, "B": 0.5},
        )
        self.assertEqual(
            preds[(("A", 1),)].average_values,
            {"A": 1.0, "B": 1.0},
        )
        self.assertEqual(
            preds[(("A", 0),)].average_values,
            {"A": 0.0, "B": 0.0},
        )

    def test_missing_fix_node_raises(self):
        experiments = [
            NAVExperiment(id=1, fixes=(("Z", 1),), observed="A", outcome="ON", weight=1.0),
        ]
        with self.assertRaises(ValueError):
            get_NAV_prediction(self.primes, experiments)

    def test_ldoi_average_values(self):
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
            NAVExperiment(id=2, fixes=(("A", 1),), observed="B", outcome="ON", weight=1.0),
            NAVExperiment(id=3, fixes=(("A", 0),), observed="B", outcome="OFF", weight=1.0),
        ]

        preds = get_NAV_prediction(self.primes, experiments, method="ldoi")

        self.assertEqual(preds[()].average_values, {"A": 0.5, "B": 0.5})
        self.assertEqual(
            preds[(("A", 1),)].average_values,
            {"A": 1.0, "B": 1.0},
        )
        self.assertEqual(
            preds[(("A", 0),)].average_values,
            {"A": 0.0, "B": 0.0},
        )

    def test_ldoi_matches_min_traps_when_traps_agree(self):
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
            NAVExperiment(id=2, fixes=(("A", 1),), observed="B", outcome="ON", weight=1.0),
            NAVExperiment(id=3, fixes=(("A", 0),), observed="B", outcome="OFF", weight=1.0),
        ]

        traps = get_NAV_prediction(self.primes, experiments, method="min_traps")
        ldoi = get_NAV_prediction(self.primes, experiments, method="ldoi")

        for fixes in traps:
            self.assertEqual(traps[fixes].average_values, ldoi[fixes].average_values)

    def test_unknown_method_raises(self):
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
        ]
        with self.assertRaises(ValueError):
            get_NAV_prediction(self.primes, experiments, method="unknown")

    def test_avg_observed_mean_of_siblings(self):
        primes = bnet_text2primes("A, A\nA_2, A_2\nA_3, A_3\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A_avg", outcome="Some", weight=1.0),
        ]
        preds = get_NAV_prediction(primes, experiments, method="ldoi")
        values = preds[()].average_values
        self.assertEqual(values["A"], 0.5)
        self.assertEqual(values["A_2"], 0.5)
        self.assertEqual(values["A_3"], 0.5)
        self.assertEqual(values["A_avg"], 0.5)

    def test_avg_includes_alphabet_siblings_excludes_unrelated(self):
        values = {
            "A": 0.0,
            "A_X": 1.0,
            "A_Y": 0.5,
            "A_iT": 0.25,
            "A_2b": 0.75,
            "AB": 1.0,
        }
        self.assertEqual(
            sorted(_related_node_names("A", values)),
            ["A", "A_X", "A_Y"],
        )
        self.assertEqual(_average_related_values(values, "A_avg"), 0.5)

    def test_avg_none_when_any_sibling_none(self):
        primes = bnet_text2primes("A, !A\nA_2, A\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A_avg", outcome="Some", weight=1.0),
        ]
        preds = get_NAV_upper_bound_prediction(primes, experiments)
        values = preds[()].average_values
        self.assertIsNone(values["A"])
        self.assertIsNone(values["A_2"])
        self.assertIsNone(values["A_avg"])

    def test_avg_no_siblings_raises(self):
        primes = bnet_text2primes("B, B\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A_avg", outcome="Some", weight=1.0),
        ]
        with self.assertRaises(ValueError):
            get_NAV_prediction(primes, experiments, method="ldoi")

    def test_no_avg_observed_adds_no_synthetic_keys(self):
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
        ]
        preds = get_NAV_prediction(self.primes, experiments, method="ldoi")
        self.assertEqual(preds[()].average_values, {"A": 0.5, "B": 0.5})
        self.assertNotIn("A_avg", preds[()].average_values)

    def test_inject_avg_values_empty_is_noop(self):
        values = {"A": 1.0}
        self.assertIs(_inject_avg_values(values, set()), values)


def _serialize_nav_predictions(preds) -> list[dict]:
    out = []
    for fixes in sorted(preds.keys()):
        pred = preds[fixes]
        out.append(
            {
                "fixes": [list(pair) for pair in fixes],
                "average_values": {
                    k: pred.average_values[k] for k in sorted(pred.average_values)
                },
            }
        )
    return out


def test_cad_demo_nav_prediction_matches_golden():
    primes = load_primes(DEMO_DIR / "CAD.bnet")
    experiments = import_NAV_exps(DEMO_DIR / "CAD_data.csv")
    preds = get_NAV_prediction(primes, experiments)

    with GOLDEN_PATH.open(encoding="utf-8") as f:
        expected = json.load(f)

    actual = _serialize_nav_predictions(preds)

    assert len(actual) == len(expected)
    assert actual == expected


if __name__ == "__main__":
    unittest.main()
