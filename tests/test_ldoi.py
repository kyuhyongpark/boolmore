import unittest

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.inference.ldoi import LdoiCache, logical_domain_of_influence


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


class TestLdoiCache(unittest.TestCase):

    def test_empty_seed_cached_and_reused_as_warm_start(self):
        primes = bnet_text2primes("C, 1\nA, C\nB, A\n")
        cache = LdoiCache()
        empty = logical_domain_of_influence(primes, {}, cache=cache)
        self.assertEqual(empty, {"A": 1, "B": 1, "C": 1})
        self.assertEqual(len(cache), 1)

        with_seed = logical_domain_of_influence(
            primes, {"A": 1}, cache=cache
        )
        self.assertEqual(with_seed, {"A": 1, "B": 1, "C": 1})
        self.assertEqual(len(cache), 2)

    def test_warm_merge_equals_uncached(self):
        primes = bnet_text2primes(
            "A, A\nB, B\nX, A\nY, B\nZ, A & B\n"
        )
        uncached = logical_domain_of_influence(primes, {"A": 1, "B": 1})

        cache = LdoiCache()
        logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        logical_domain_of_influence(primes, {"B": 1}, cache=cache)
        cached = logical_domain_of_influence(
            primes, {"A": 1, "B": 1}, cache=cache
        )
        self.assertEqual(cached, uncached)
        self.assertEqual(cached.get("X"), 1)
        self.assertEqual(cached.get("Y"), 1)
        self.assertEqual(cached.get("Z"), 1)

    def test_conflicting_subset_ldoi_is_skipped(self):
        # LDOI({A:1}) forces B=0 via B = !A; seed {A:1, B:1} must not
        # inherit that opposite, and seed B=1 must win.
        primes = bnet_text2primes("A, A\nB, !A\nC, B\n")
        cache = LdoiCache()
        alone = logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        self.assertEqual(alone.get("B"), 0)

        merged = cache.compatible_merge({"A": 1, "B": 1})
        self.assertNotIn("B", merged)  # conflicting LDOI was skipped
        self.assertNotIn("C", merged)

        result = logical_domain_of_influence(
            primes, {"A": 1, "B": 1}, cache=cache
        )
        uncached = logical_domain_of_influence(primes, {"A": 1, "B": 1})
        self.assertEqual(result, uncached)
        self.assertEqual(result.get("B"), 1)
        self.assertEqual(result.get("C"), 1)

    def test_cache_hit_returns_copy(self):
        primes = bnet_text2primes("C, 1\nA, C\n")
        cache = LdoiCache()
        first = logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        second = logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)

        first["A"] = 0
        third = logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        self.assertEqual(third.get("A"), 1)
        self.assertEqual(len(cache), 1)

    def test_nested_probe_under_base(self):
        primes = bnet_text2primes("P, P\nA, A\nX, A\nY, P & A\n")
        cache = LdoiCache()
        base = {"P": 1}
        fixed = logical_domain_of_influence(primes, base, cache=cache)
        self.assertEqual(fixed.get("P"), 1)

        probe = logical_domain_of_influence(
            primes,
            {"A": 1},
            cache=cache,
            base=base,
            implied=fixed,
        )
        uncached = logical_domain_of_influence(
            primes, {"P": 1, "A": 1}
        )
        self.assertEqual(probe, uncached)
        self.assertEqual(probe.get("X"), 1)
        self.assertEqual(probe.get("Y"), 1)

        # Nested hit
        hit = cache.get({"A": 1}, base=base)
        self.assertEqual(hit, probe)

        # Top-level get(P ∪ SC) finds the probe via delta
        as_experiment = cache.get({"P": 1, "A": 1})
        self.assertEqual(as_experiment, probe)

        # One base LDOI + one probe
        self.assertEqual(len(cache), 2)

    def test_probe_reuse_as_experiment_under_empty_base(self):
        # Probe SC under empty intervention; later experiment equals SC.
        primes = bnet_text2primes("A, A\nX, A\n")
        cache = LdoiCache()
        empty_fixed = logical_domain_of_influence(primes, {}, cache=cache)
        probe = logical_domain_of_influence(
            primes,
            {"A": 1},
            cache=cache,
            base={},
            implied=empty_fixed,
        )
        # Experiment {A:1} should hit the probe stored under empty base.
        hit = logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        self.assertEqual(hit, probe)
        self.assertEqual(hit.get("X"), 1)

    def test_probe_under_other_base_not_confused_with_experiment(self):
        primes = bnet_text2primes("P, P\nA, A\nX, P & A\n")
        cache = LdoiCache()
        base = {"P": 1}
        fixed = logical_domain_of_influence(primes, base, cache=cache)
        probe = logical_domain_of_influence(
            primes,
            {"A": 1},
            cache=cache,
            base=base,
            implied=fixed,
        )
        self.assertEqual(probe.get("X"), 1)

        # Experiment {A:1} alone must NOT reuse P∪{A:1} (X not forced).
        alone = logical_domain_of_influence(primes, {"A": 1}, cache=cache)
        self.assertIsNone(alone.get("X"))
        self.assertNotEqual(alone, probe)


if __name__ == "__main__":
    unittest.main()
