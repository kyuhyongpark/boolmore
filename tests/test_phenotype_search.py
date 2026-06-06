import unittest

from boolmore.phenotypes import find_matching_phenotype


class TestFindMatchingPhenotype(unittest.TestCase):

    def setUp(self):
        self.answer_sheet = {
            1: [
                {"A": 0, "B": 1, "C": 0, "D": 1},
                {"A": 0, "B": 1, "C": 0, "D": 0},
            ],
            3: [
                {"A": 1, "B": 1, "C": 1, "D": 1},
                {"A": 1, "B": 1, "C": 1, "D": 0},
            ],
        }

    def test_example_match(self):
        result = find_matching_phenotype(
            self.answer_sheet, {"C": 1, "D": 0}
        )
        self.assertEqual(
            result,
            (3, {"A": 1, "B": 1, "C": 1, "D": 0})
        )

    def test_no_match_returns_none(self):
        result = find_matching_phenotype(
            self.answer_sheet, {"C": 1, "D": 0, "A": 0}
        )
        self.assertIsNone(result)

    def test_partial_phenotype_match(self):
        # only C constraint should still match valid trap spaces
        result = find_matching_phenotype(
            self.answer_sheet, {"C": 1}
        )
        self.assertIsNotNone(result)
        mask, trap = result
        self.assertEqual(mask, 3)
        self.assertEqual(trap["C"], 1)

    def test_first_match_only(self):
        # ensure it stops at first match in traversal order
        answer_sheet = {
            5: [
                {"C": 1, "D": 0},  # would match but should NOT be reached if earlier match exists
            ],
            3: [
                {"C": 1, "D": 0},
            ],
        }

        result = find_matching_phenotype(answer_sheet, {"C": 1, "D": 0})
        self.assertEqual(result[0], 5)
        self.assertEqual(result[1], {"C": 1, "D": 0})


if __name__ == "__main__":
    unittest.main()