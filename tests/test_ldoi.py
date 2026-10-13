import unittest

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.inference.ldoi import logical_domain_of_influence


class TestLogicalDomainOfInfluence(unittest.TestCase):

    def test_seed_always_included_even_if_not_self_forced(self):
        # A = A & B does not force A=1; seed is still in the returned dict.
        primes = bnet_text2primes("A, A & B\nB, B\n")
        result = logical_domain_of_influence(primes, {"A": 1})
        self.assertEqual(result.get("A"), 1)

    def test_empty_seed_forces_preexisting_constants(self):
        primes = bnet_text2primes("C, 1\nA, C\nB, A\n")
        result = logical_domain_of_influence(primes, {})
        self.assertEqual(result, {"A": 1, "B": 1, "C": 1})

    def test_missing_seed_node_raises(self):
        primes = bnet_text2primes("A, !A\nB, A\n")
        with self.assertRaises(ValueError):
            logical_domain_of_influence(primes, {"Z": 1})

    def test_contradicting_seed_keeps_seed_and_forces_dependents(self):
        # Rule A=0 contradicts seed A=1; seed wins, B still forced via A=1.
        primes = bnet_text2primes("A, 0\nB, A\n")
        result = logical_domain_of_influence(primes, {"A": 1})
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("A"), 1)
        self.assertEqual(result.get("B"), 1)


if __name__ == "__main__":
    unittest.main()
