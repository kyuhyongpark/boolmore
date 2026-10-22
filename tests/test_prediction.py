import json
import unittest
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.experiment import NAVExperiment, PhenotypeExperiment
from boolmore.inference.prediction import (
    _enhanced_nav_from_forcers,
    _fix_constants,
    _nav_from_ldoi,
    _nav_from_ldoi_enhanced,
    _nav_from_min_traps,
    _prune_subsumed_forcers,
    _sc_subsets_by_size,
    get_NAV_prediction,
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

    def test_min_traps_averages_equally_over_source_combinations(self):
        # S=0 leaves A free (4 traps with C); S=1 forces A=0 (2 traps).
        # Flat trap average would give B=1/3; source-grouped average gives 0.25.
        primes = bnet_text2primes("S, S\nA, A & !S\nB, A\nC, C\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="B", outcome="Some", weight=1.0),
        ]

        preds = get_NAV_prediction(primes, experiments, method="min_traps")
        values = preds[()].average_values

        self.assertEqual(values["S"], 0.5)
        self.assertEqual(values["A"], 0.25)
        self.assertEqual(values["B"], 0.25)
        self.assertEqual(values["C"], 0.5)
        # All sources fixed: single group, same as averaging the two traps.
        self.assertEqual(
            _nav_from_min_traps(primes, {"S": 1, "C": 0}),
            {"S": 1.0, "A": 0.0, "B": 0.0, "C": 0.0},
        )

    def test_avg_observed_does_not_inject_synthetic_keys(self):
        primes = bnet_text2primes("A, A\nA_2, A_2\nA_3, A_3\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A_avg", outcome="Some", weight=1.0),
        ]
        preds = get_NAV_prediction(primes, experiments, method="ldoi")
        values = preds[()].average_values
        self.assertEqual(values, {"A": 0.5, "A_2": 0.5, "A_3": 0.5})
        self.assertNotIn("A_avg", values)


class TestLdoiEnhanced(unittest.TestCase):

    def test_enhanced_nav_from_forcers_mass_allocation(self):
        self.assertEqual(_enhanced_nav_from_forcers({}), 0.5)
        # Both states of one size-1 ancestor: full measure 50/50.
        self.assertEqual(
            _enhanced_nav_from_forcers({("A",): {(0,): 0, (1,): 1}}),
            0.5,
        )
        # One-sided: half measure at forced value, half unknown → 0.75.
        self.assertEqual(
            _enhanced_nav_from_forcers({("A",): {(1,): 1}}),
            0.75,
        )
        # A=0 ∪ B=0 covers 3/4 at 0, remainder 1/4 at 0.5 → 0.125.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 0}, ("B",): {(0,): 0}}
            ),
            0.125,
        )

    def test_enhanced_nav_from_forcers_size_two_mass(self):
        # 1 of 4 states forcing 1: mass 1/4 at 1, remainder 0.5 → 0.625.
        self.assertEqual(
            _enhanced_nav_from_forcers({("A", "B"): {(0, 1): 1}}),
            0.625,
        )
        # 3 of 4 states forcing 0: mass 3/4 at 0, remainder 0.5 → 0.125.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {
                    ("A", "B"): {
                        (0, 0): 0,
                        (0, 1): 0,
                        (1, 0): 0,
                    }
                }
            ),
            0.125,
        )
        # All 4 states: average of forced values.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {
                    ("A", "B"): {
                        (0, 0): 0,
                        (0, 1): 0,
                        (1, 0): 1,
                        (1, 1): 1,
                    }
                }
            ),
            0.5,
        )

    def test_enhanced_nav_from_forcers_shared_node_exact(self):
        # A=0 (half) and (A,B)=(1,1) (quarter) are disjoint; union is 3/4 at 1.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 1}, ("A", "B"): {(1, 1): 1}}
            ),
            0.875,
        )

    def test_enhanced_nav_from_forcers_conflict(self):
        with self.assertRaises(ValueError):
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 0}, ("B",): {(0,): 1}}
            )

    def test_prune_subsumed_forcers_drops_extensions(self):
        # A=1 already covers (A,B)=(1,*); keep only the size-1 forcer.
        pruned = _prune_subsumed_forcers(
            {
                ("A",): {(1,): 1},
                ("A", "B"): {(1, 0): 1, (1, 1): 1},
            }
        )
        self.assertEqual(pruned, {("A",): {(1,): 1}})

    def test_prune_subsumed_forcers_keeps_incomparable(self):
        # A=0 and (A,B)=(1,1) do not nest — both kept.
        forcers = {("A",): {(0,): 1}, ("A", "B"): {(1, 1): 1}}
        self.assertEqual(_prune_subsumed_forcers(forcers), forcers)

    def test_prune_subsumed_forcers_nested_conflict_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _prune_subsumed_forcers(
                {("A",): {(1,): 1}, ("A", "B"): {(1, 0): 0}}
            )
        self.assertIn("conflicting nested forcers", str(ctx.exception))

    def test_enhanced_nav_from_forcers_subsumed_pair_same_nav(self):
        # Redundant (A,B) extensions must not change NAV vs A=1 alone.
        alone = _enhanced_nav_from_forcers({("A",): {(1,): 1}})
        with_pairs = _enhanced_nav_from_forcers(
            {
                ("A",): {(1,): 1},
                ("A", "B"): {(1, 0): 1, (1, 1): 1},
                ("A", "C"): {(1, 1): 1},
            }
        )
        self.assertEqual(alone, 0.75)
        self.assertEqual(with_pairs, alone)

    def test_enhanced_nav_from_forcers_impossible_renormalize(self):
        # A=0 forces 0; A=1 impossible → full mass at 0.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 0}},
                {("A",): {(1,)}},
            ),
            0.0,
        )
        # A=0 forces 1; A=1 impossible → full mass at 1.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 1}},
                {("A",): {(1,)}},
            ),
            1.0,
        )
        # All assignments impossible → fallback 0.5.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 0}},
                {("A",): {(0,), (1,)}},
            ),
            0.5,
        )
        # Unrelated impossible components must not enlarge the measure.
        self.assertEqual(
            _enhanced_nav_from_forcers(
                {("A",): {(0,): 0}},
                {("Z",): {(1,)}, ("X", "Y"): {(0, 1)}},
            ),
            0.25,
        )

    def test_nav_from_ldoi_enhanced_one_sided_forcer(self):
        # A=0 self-sustains (and forces C=0); A=1 does not without B.
        # B=0 also forces C=0; B sustains both ways → B NAV stays 0.5.
        primes = bnet_text2primes("A, A & B\nB, B\nC, A\n")

        plain = _nav_from_ldoi(primes, {})
        enhanced = _nav_from_ldoi_enhanced(primes, {})

        self.assertEqual(plain, {"A": 0.5, "B": 0.5, "C": 0.5})
        self.assertEqual(enhanced["B"], 0.5)
        # Forcers for C: A=0 and B=0 → expected 0.125
        self.assertEqual(enhanced["C"], 0.125)
        self.assertEqual(enhanced["A"], 0.125)

    def test_nav_from_ldoi_enhanced_self_negating_complement(self):
        # A=0 self-sustains and forces C=0; A=1 self-negates (B=!A → B=0
        # covers an A→0 implicant). Full mass on A=0 → C=A=0.
        primes = bnet_text2primes("A, A & B\nB, !A\nC, A\n")
        enhanced = _nav_from_ldoi_enhanced(primes, {})
        self.assertEqual(enhanced["A"], 0.0)
        self.assertEqual(enhanced["C"], 0.0)
        self.assertEqual(enhanced["B"], 1.0)

    def test_nav_from_ldoi_enhanced_both_sustain_and_negate_raises(self):
        # Malformed primes: A=0 covers implicants for both values.
        primes = {"A": [[{"A": 0}], [{"A": 0}]]}
        with self.assertRaises(ValueError) as ctx:
            _nav_from_ldoi_enhanced(primes, {})
        self.assertIn("both self-sustains and self-negates", str(ctx.exception))

    def test_get_NAV_prediction_ldoi_enhanced(self):
        primes = bnet_text2primes("A, A & B\nB, B\nC, A\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="C", outcome="Some", weight=1.0),
        ]
        preds = get_NAV_prediction(primes, experiments, method="ldoi_enhanced")
        self.assertEqual(preds[()].average_values["C"], 0.125)

    def test_sc_subsets_nested_two_cycle(self):
        # A↔B nested in the larger SCC {A,B,C}.
        primes = bnet_text2primes("A, B\nB, A | C\nC, A\n")
        subsets = _sc_subsets_by_size(primes, cutoff=2)
        self.assertEqual(subsets[1], [])
        self.assertIn(("A", "B"), subsets[2])

    def test_nav_from_ldoi_enhanced_cutoff_two_mutual_cycle(self):
        # A↔B needs free Z for ON, so only (A,B)=(0,0) self-sustains and
        # forces C=0. No self-loops → cutoff=1 matches plain LDOI.
        primes = bnet_text2primes("A, B & Z\nB, A & Z\nC, A\nZ, !X\nX, !Z\n")
        plain = _nav_from_ldoi(primes, {})
        cut1 = _nav_from_ldoi_enhanced(primes, {}, cutoff=1)
        cut2 = _nav_from_ldoi_enhanced(primes, {}, cutoff=2)

        self.assertEqual(cut1, plain)
        self.assertEqual(cut1["C"], 0.5)
        self.assertLess(cut2["C"], cut1["C"])
        self.assertEqual(cut2["C"], cut2["A"])

    def test_ldoi_enhanced_cutoff_validation(self):
        primes = bnet_text2primes("A, A\n")
        with self.assertRaises(ValueError):
            _nav_from_ldoi_enhanced(primes, {}, cutoff=0)
        with self.assertRaises(ValueError):
            _nav_from_ldoi_enhanced(primes, {}, cutoff=4)
        with self.assertRaises(ValueError):
            _sc_subsets_by_size(primes, cutoff=0)
        with self.assertRaises(ValueError):
            _sc_subsets_by_size(primes, cutoff=4)

    def test_nav_from_ldoi_enhanced_joint_self_loops_cutoff_two(self):
        # Two independent self-loops; C=A&B. Size-1 forcers A=0 / B=0 move C
        # toward 0 (NAV 0.125). Joint of sustaining states also records
        # (A,B)=(1,1)→C=1, raising C. No A↔B edge → SC cutoff 2 == cutoff 1.
        primes = bnet_text2primes("A, A\nB, B\nC, A & B\n")
        cut2 = _nav_from_ldoi_enhanced(primes, {}, cutoff=2)
        stats: dict = {}
        joint = _nav_from_ldoi_enhanced(
            primes,
            {},
            cutoff=2,
            joint_self_loops=True,
            joint_stats=stats,
        )

        self.assertEqual(cut2["C"], 0.125)
        self.assertEqual(joint["C"], 0.25)
        self.assertGreater(stats["n_sustaining_size1"], 0)
        self.assertGreater(stats["n_joint_probed"], 0)
        self.assertGreater(stats["n_joint_kept"], 0)
        # Joint probes specific sustaining-state pairs, not all 2^2 per node pair.
        # 2 nodes × 2 values = 4 size-1 sustaining states → C(4,2) with same-node
        # pairs dropped → 4 cross-node pairs (not 4 node-pair × 4 assignments).
        self.assertEqual(stats["n_joint_probed"], 4)

    def test_nav_from_ldoi_enhanced_joint_self_loops_cutoff_three(self):
        # Self-loop A + mutual B↔C; T=A&B. SC probes alone give T NAV 0.125
        # (A=0 and (B,C)=(0,0)). Joint A=1 with (B,C)=(1,1) adds a T=1 forcer.
        primes = bnet_text2primes("A, A\nB, C\nC, B\nT, A & B\n")
        sc_only = _nav_from_ldoi_enhanced(primes, {}, cutoff=3)
        stats: dict = {}
        joint = _nav_from_ldoi_enhanced(
            primes,
            {},
            cutoff=3,
            joint_self_loops=True,
            joint_stats=stats,
        )
        self.assertEqual(sc_only["T"], 0.125)
        self.assertEqual(joint["T"], 0.25)
        self.assertGreaterEqual(stats["n_sustaining_size2"], 1)
        self.assertGreater(stats["n_joint_probed"], 0)

    def test_nav_from_ldoi_enhanced_joint_default_matches_sc_only(self):
        primes = bnet_text2primes("A, A\nB, B\nC, A & B\n")
        a = _nav_from_ldoi_enhanced(primes, {}, cutoff=2)
        b = _nav_from_ldoi_enhanced(
            primes, {}, cutoff=2, joint_self_loops=False
        )
        self.assertEqual(a, b)

    def test_get_NAV_prediction_ldoi_enhanced_joint_self_loops(self):
        primes = bnet_text2primes("A, A\nB, B\nC, A & B\n")
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="C", outcome="Some", weight=1.0),
        ]
        preds = get_NAV_prediction(
            primes,
            experiments,
            method="ldoi_enhanced",
            cutoff=2,
            joint_self_loops=True,
        )
        self.assertEqual(preds[()].average_values["C"], 0.25)


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
