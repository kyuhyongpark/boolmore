import unittest

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.mask import (
    generate_source_masks,
    mask_to_sources,
    mask_to_str,
    )

from boolmore.phenotypes import (
    get_phenotypes_for_source_comb,
)


def _canon_phenotypes(phenotypes):
    """
    Convert unordered list of dicts into a stable comparable structure.
    """
    return sorted(
        [tuple(sorted(ph.items())) for ph in phenotypes]
    )


class TestPhenotypes(unittest.TestCase):
    def test_get_phenotypes_for_source_comb(self):
        bnet = """
        A, A
        B, B
        C, A & B
        D, D & B
        """

        primes = bnet_text2primes(bnet)

        source_comb = {"A": 0, "B": 1}

        phenotypes = get_phenotypes_for_source_comb(primes, source_comb)

        expected = [
            {'A': 0, 'B': 1, 'C': 0, 'D': 0},
            {'A': 0, 'B': 1, 'C': 0, 'D': 1},
        ]

        self.assertEqual(len(phenotypes), len(expected))

        for p in expected:
            self.assertIn(p, phenotypes)

        for p in phenotypes:
            self.assertIn(p, expected)

    def test_source_comb_order_independence(self):
        bnet = """
        A, A
        B, B
        C, A & B
        """

        primes = bnet_text2primes(bnet)

        comb1 = {"A": 0, "B": 1}
        comb2 = {"B": 1, "A": 0}

        p1 = get_phenotypes_for_source_comb(primes, comb1)
        p2 = get_phenotypes_for_source_comb(primes, comb2)

        self.assertEqual(p1, p2)

    def test_unknown_source_raises(self):
        bnet = """
        A, A
        B, B
        """

        primes = bnet_text2primes(bnet)

        source_comb = {"X": 1}

        with self.assertRaises(ValueError):
            get_phenotypes_for_source_comb(primes, source_comb)

    def test_determinism(self):
        bnet = """
        A, A
        B, B
        C, A & B
        """

        primes = bnet_text2primes(bnet)

        source_comb = {"A": 0, "B": 1}

        p1 = get_phenotypes_for_source_comb(primes, source_comb)
        p2 = get_phenotypes_for_source_comb(primes, source_comb)

        self.assertEqual(p1, p2)

    def test_primes_not_modified(self):
        bnet = """
        A, A
        B, B
        C, A & B
        """

        primes = bnet_text2primes(bnet)
        original = str(primes)  # cheap snapshot

        source_comb = {"A": 0, "B": 1}

        _ = get_phenotypes_for_source_comb(primes, source_comb)

        self.assertEqual(str(primes), original)

    def test_empty_source_comb(self):
        bnet = """
        A, A
        B, B
        """

        primes = bnet_text2primes(bnet)

        phenotypes = get_phenotypes_for_source_comb(primes, {})

        self.assertIn({'A': 0, 'B': 0}, phenotypes)
        self.assertIn({'A': 0, 'B': 1}, phenotypes)
        self.assertIn({'A': 1, 'B': 0}, phenotypes)
        self.assertIn({'A': 1, 'B': 1}, phenotypes)

    def test_empty_source_comb_with_DEBUG(self):
        bnet = """
        A, A
        B, B
        """

        primes = bnet_text2primes(bnet)

        with self.assertRaises(ValueError):
            get_phenotypes_for_source_comb(primes, {}, DEBUG=True)


class TestFullPipelineExactOutput(unittest.TestCase):

    def setUp(self):
        self.bnet = """
        A, A
        B, B
        C, A & B
        D, D & B
        E, !E
        """

        self.primes = bnet_text2primes(self.bnet)

        self.source_nodes = sorted([
            node for node in self.primes
            if self.primes[node] == [[{node: 0}], [{node: 1}]]
        ])

        self.sources_partial = {"B": 1}

        self.answer_sheet = {}
        for mask in generate_source_masks(self.source_nodes, self.sources_partial):
            source_comb = mask_to_sources(mask, self.source_nodes)
            phenotypes = get_phenotypes_for_source_comb(self.primes, source_comb)
            self.answer_sheet[mask_to_str(mask, len(self.source_nodes))] = phenotypes

    def test_exact_source_nodes(self):
        self.assertEqual(self.source_nodes, ["A", "B"])

    def test_exact_answer_sheet_unordered_safe(self):

        expected = {
            "01": [
                {"A": 0, "B": 1, "C": 0, "D": 1},
                {"A": 0, "B": 1, "C": 0, "D": 0},
            ],
            "11": [
                {"A": 1, "B": 1, "C": 1, "D": 1},
                {"A": 1, "B": 1, "C": 1, "D": 0},
            ],
        }

        # canonicalize both sides
        actual_norm = {
            k: _canon_phenotypes(v)
            for k, v in self.answer_sheet.items()
        }

        expected_norm = {
            k: _canon_phenotypes(v)
            for k, v in expected.items()
        }

        self.assertEqual(actual_norm, expected_norm)


if __name__ == "__main__":
    unittest.main()