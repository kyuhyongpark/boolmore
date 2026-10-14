"""LDOI optimistic upper-bound scoring and min_traps prune."""

from functools import partial

from pyboolnet.external.bnet2primes import bnet_text2primes

from boolmore.evaluation.agreement import get_agreements
from boolmore.evaluation.nav_score import get_NAV_scores
from boolmore.evaluation.score import Evaluator, get_model_score
from boolmore.experiment import NAVExperiment
from boolmore.inference.prediction import (
    NAVPrediction,
    _nav_from_ldoi_partial,
    get_NAV_upper_bound_prediction,
)
from boolmore.model import Model


EMPTY_CONSTRAINTS = {
    "fixed": [],
    "regulate": {},
    "necessary": {},
    "group": {},
    "possible_constant": [],
}


def _tiny_primes():
    # A is free without intervention; fixing A forces B.
    return bnet_text2primes("A, !A\nB, A\n")


def test_ldoi_partial_marks_free_nodes_none():
    primes = _tiny_primes()
    assert _nav_from_ldoi_partial(primes, {}) == {"A": None, "B": None}
    assert _nav_from_ldoi_partial(primes, {"A": 1}) == {"A": 1.0, "B": 1.0}
    assert _nav_from_ldoi_partial(primes, {"A": 0}) == {"A": 0.0, "B": 0.0}


def test_upper_bound_prediction_uses_none_for_free():
    primes = _tiny_primes()
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
        NAVExperiment(id=2, fixes=(("A", 1),), observed="B", outcome="ON", weight=1.0),
    ]
    preds = get_NAV_upper_bound_prediction(primes, experiments)
    assert preds[()].average_values == {"A": None, "B": None}
    assert preds[(("A", 1),)].average_values == {"A": 1.0, "B": 1.0}


def test_none_predict_gives_full_agreement_for_all_outcomes():
    outcomes = ("ON", "Some/ON", "Some", "OFF/Some", "OFF")
    for outcome in outcomes:
        experiments = [
            NAVExperiment(id=1, fixes=(), observed="A", outcome=outcome, weight=2.0),
        ]
        predictions = {
            (): NAVPrediction(fixes=(), average_values={"A": None}),
        }
        agreements = get_agreements(experiments, predictions)
        _id, weight, _out, predict_value, agreement = agreements["A"][()][0]
        assert predict_value is None
        assert agreement == 1.0
        assert weight == 2.0


def test_forced_mismatch_lowers_upper_bound():
    primes = _tiny_primes()
    # LDOI forces B=0 when A=0, but experiment expects ON → agreement 0.
    experiments = [
        NAVExperiment(id=1, fixes=(("A", 0),), observed="B", outcome="ON", weight=1.0),
    ]
    preds = get_NAV_upper_bound_prediction(primes, experiments)
    items = get_NAV_scores(experiments, preds, default_sources={}, hierarchy=False)
    max_score, score = get_model_score(items)
    assert max_score == 1.0
    assert score == 0.0


def test_free_nodes_give_optimistic_max_when_unforced():
    primes = _tiny_primes()
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="ON", weight=1.0),
        NAVExperiment(id=2, fixes=(), observed="B", outcome="OFF", weight=1.0),
    ]
    preds = get_NAV_upper_bound_prediction(primes, experiments)
    items = get_NAV_scores(experiments, preds, default_sources={}, hierarchy=False)
    max_score, score = get_model_score(items)
    assert max_score == 2.0
    assert score == 2.0


def test_evaluate_prunes_when_upper_bound_below_threshold():
    primes = _tiny_primes()
    model = Model.from_primes(primes)
    experiments = [
        NAVExperiment(id=1, fixes=(("A", 0),), observed="B", outcome="ON", weight=1.0),
    ]

    def boom_prediction(_primes, _experiments):
        raise AssertionError("full min_traps prediction should be skipped")

    evaluator = Evaluator(
        base=model,
        experiments=experiments,
        prediction_fn=boom_prediction,
        score_fn=partial(get_NAV_scores, default_sources={}, hierarchy=False),
        constraints=EMPTY_CONSTRAINTS,
        run_type="NAV",
        hierarchy=False,
        nav_method="min_traps",
        ldoi_prune=True,
    )

    result = evaluator.evaluate(model, model_id=7, score_threshold=0.5)
    assert result.is_upper_bound is True
    assert result.score == 0.0
    assert result.model_id == 7


def test_evaluate_runs_full_when_upper_bound_meets_threshold():
    primes = _tiny_primes()
    model = Model.from_primes(primes)
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="ON", weight=1.0),
    ]
    full_called = {"n": 0}

    def fake_full(primes_arg, experiments_arg):
        full_called["n"] += 1
        return {
            (): NAVPrediction(fixes=(), average_values={"A": 0.5, "B": 0.5}),
        }

    evaluator = Evaluator(
        base=model,
        experiments=experiments,
        prediction_fn=fake_full,
        score_fn=partial(get_NAV_scores, default_sources={}, hierarchy=False),
        constraints=EMPTY_CONSTRAINTS,
        run_type="NAV",
        hierarchy=False,
        nav_method="min_traps",
        ldoi_prune=True,
    )

    # Optimistic UB is 1.0 (free A → ?); threshold 0.5 → do not prune.
    result = evaluator.evaluate(model, model_id=3, score_threshold=0.5)
    assert result.is_upper_bound is False
    assert full_called["n"] == 1
    assert result.score == 0.5  # agreement for ON at predict 0.5


def test_evaluate_skips_prune_without_threshold():
    primes = _tiny_primes()
    model = Model.from_primes(primes)
    experiments = [
        NAVExperiment(id=1, fixes=(("A", 0),), observed="B", outcome="ON", weight=1.0),
    ]
    full_called = {"n": 0}

    def fake_full(primes_arg, experiments_arg):
        full_called["n"] += 1
        return {
            (("A", 0),): NAVPrediction(
                fixes=(("A", 0),), average_values={"A": 0.0, "B": 0.0}
            ),
        }

    evaluator = Evaluator(
        base=model,
        experiments=experiments,
        prediction_fn=fake_full,
        score_fn=partial(get_NAV_scores, default_sources={}, hierarchy=False),
        constraints=EMPTY_CONSTRAINTS,
        run_type="NAV",
        hierarchy=False,
        nav_method="min_traps",
        ldoi_prune=True,
    )

    result = evaluator.evaluate(model)
    assert result.is_upper_bound is False
    assert full_called["n"] == 1


def test_evaluate_skips_prune_when_ldoi_method():
    primes = _tiny_primes()
    model = Model.from_primes(primes)
    experiments = [
        NAVExperiment(id=1, fixes=(("A", 0),), observed="B", outcome="ON", weight=1.0),
    ]
    full_called = {"n": 0}

    def fake_full(primes_arg, experiments_arg):
        full_called["n"] += 1
        return {
            (("A", 0),): NAVPrediction(
                fixes=(("A", 0),), average_values={"A": 0.0, "B": 0.0}
            ),
        }

    evaluator = Evaluator(
        base=model,
        experiments=experiments,
        prediction_fn=fake_full,
        score_fn=partial(get_NAV_scores, default_sources={}, hierarchy=False),
        constraints=EMPTY_CONSTRAINTS,
        run_type="NAV",
        hierarchy=False,
        nav_method="ldoi",
        ldoi_prune=True,
    )

    result = evaluator.evaluate(model, score_threshold=99.0)
    assert result.is_upper_bound is False
    assert full_called["n"] == 1
