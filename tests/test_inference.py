import unittest

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.core.experiment import PhenotypeExperiment
from boolmore.algo.inference import get_phenotype_prediction


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
        self.assertEqual(results[0].found_phenotypes, [{"A": 1, "B": 1, "C": 1}])

        self.assertFalse(results[1].predicted_exists)
        self.assertEqual(results[1].found_phenotypes, [])


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
        self.assertEqual(results[0].found_phenotypes, [{"A": 0, "B": 0, "C": 1, "D": 1}])

        self.assertTrue(results[1].predicted_exists)
        self.assertEqual(results[1].found_phenotypes, [{"A": 1, "B": 0, "D": 1}])


if __name__ == "__main__":
    unittest.main()