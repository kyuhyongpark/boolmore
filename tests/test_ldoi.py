import random
import unittest
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.inference.ldoi import (
    LdoiCache,
    _ldoi_closure_fast,
    _ldoi_closure_scan,
    _get_compiled,
    get_ldoi_closure_engine,
    ldoi_closure_engine,
    logical_domain_of_influence,
    set_ldoi_closure_engine,
)
from boolmore.io.load import load_primes
from boolmore.io.nav import import_NAV_exps
from boolmore.types import Assignment


def _fixes_to_dict(fixes: Assignment) -> dict[str, int]:
    return {node: int(value) for node, value in fixes}


def _assert_fast_equals_scan(
    test: unittest.TestCase,
    primes: dict,
    seed: dict[str, int],
    *,
    base: dict[str, int] | None = None,
    implied: dict[str, int] | None = None,
) -> None:
    """Compare public API under both engines and raw closure helpers."""
    with ldoi_closure_engine("scan"):
        scan = logical_domain_of_influence(
            primes, seed, base=base, implied=implied
        )
    with ldoi_closure_engine("fast"):
        fast = logical_domain_of_influence(
            primes, seed, base=base, implied=implied
        )
    test.assertEqual(fast, scan)

    if base is not None:
        full = dict(base)
        full.update(seed)
    else:
        full = dict(seed)
    if implied:
        fixed = dict(implied)
        fixed.update(full)
    else:
        fixed = full
    test.assertEqual(
        _ldoi_closure_fast(_get_compiled(primes), fixed),
        _ldoi_closure_scan(primes, fixed),
    )


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


class TestLdoiClosureEquivalence(unittest.TestCase):

    def tearDown(self) -> None:
        set_ldoi_closure_engine("fast")

    def test_engine_switch_roundtrip(self):
        self.assertEqual(get_ldoi_closure_engine(), "fast")
        set_ldoi_closure_engine("scan")
        self.assertEqual(get_ldoi_closure_engine(), "scan")
        with ldoi_closure_engine("fast"):
            self.assertEqual(get_ldoi_closure_engine(), "fast")
        self.assertEqual(get_ldoi_closure_engine(), "scan")
        set_ldoi_closure_engine("fast")

    def test_hand_cases_fast_equals_scan(self):
        cases = [
            ("A, A & B\nB, B\n", {"A": 1}),
            ("C, 1\nA, C\nB, A\n", {}),
            ("A, 0\nB, A\n", {"A": 1}),
            ("A, A\nB, B\nX, A\nY, B\nZ, A & B\n", {"A": 1, "B": 1}),
            ("A, A\nB, !A\nC, B\n", {"A": 1, "B": 1}),
            ("A, !A\nB, A\n", {"A": 0}),
            ("A, A\nB, A | B\n", {"A": 1}),
            ("P, P\nA, A\nX, A\nY, P & A\n", {"P": 1, "A": 1}),
        ]
        for bnet, seed in cases:
            with self.subTest(bnet=bnet, seed=seed):
                primes = bnet_text2primes(bnet)
                _assert_fast_equals_scan(self, primes, seed)

    def test_implied_and_base_fast_equals_scan(self):
        primes = bnet_text2primes("P, P\nA, A\nX, A\nY, P & A\n")
        base = {"P": 1}
        with ldoi_closure_engine("scan"):
            implied = logical_domain_of_influence(primes, base)
        _assert_fast_equals_scan(
            self, primes, {"A": 1}, base=base, implied=implied
        )

    def test_random_networks_fast_equals_scan(self):
        rng = random.Random(0)
        formulas = [
            "0",
            "1",
            "{x}",
            "!{x}",
            "{x} & {y}",
            "{x} | {y}",
            "{x} & !{y}",
            "!{x} | {y}",
        ]
        for trial in range(40):
            n = rng.randint(3, 7)
            names = [f"N{i}" for i in range(n)]
            lines: list[str] = []
            for target in names:
                template = rng.choice(formulas)
                regs = rng.sample(names, k=min(2, n))
                fmt = {"x": regs[0], "y": regs[-1]}
                lines.append(f"{target}, {template.format(**fmt)}")
            primes = bnet_text2primes("\n".join(lines) + "\n")
            for _ in range(8):
                k = rng.randint(0, min(3, n))
                seed_nodes = rng.sample(names, k=k)
                seed = {node: rng.randint(0, 1) for node in seed_nodes}
                with self.subTest(trial=trial, seed=seed):
                    _assert_fast_equals_scan(self, primes, seed)

    def test_cad_interventions_fast_equals_scan(self):
        repo = Path(__file__).resolve().parents[1]
        bnet = repo / "demo" / "CAD.bnet"
        data = repo / "demo" / "CAD_data.csv"
        if not bnet.is_file() or not data.is_file():
            self.skipTest("CAD demo files not present")
        primes = load_primes(bnet)
        exps = import_NAV_exps(data)
        seen: set[frozenset[tuple[str, int]]] = set()
        for exp in exps:
            key = frozenset(_fixes_to_dict(exp.fixes).items())
            if key in seen:
                continue
            seen.add(key)
            seed = dict(key)
            with self.subTest(seed=seed):
                _assert_fast_equals_scan(self, primes, seed)

    def test_compile_cache_invalidates_on_node_replace(self):
        primes = bnet_text2primes("A, A\nB, A\n")
        compiled_a = _get_compiled(primes)
        primes["B"] = [[{}], []]  # B becomes constant 0
        compiled_b = _get_compiled(primes)
        self.assertIsNot(compiled_a, compiled_b)
        _assert_fast_equals_scan(self, primes, {})

    def test_compiled_reuse_matches_auto_compile(self):
        primes = bnet_text2primes("A, A\nB, A\nX, A & B\n")
        seed = {"A": 1}
        compiled = _get_compiled(primes)
        with ldoi_closure_engine("scan"):
            scan = logical_domain_of_influence(primes, seed)
        auto = logical_domain_of_influence(primes, seed)
        reused = logical_domain_of_influence(primes, seed, compiled=compiled)
        self.assertEqual(auto, scan)
        self.assertEqual(reused, scan)


if __name__ == "__main__":
    unittest.main()
