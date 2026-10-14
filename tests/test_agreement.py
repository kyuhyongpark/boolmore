"""Characterization tests for NAV outcome–prediction agreement curves."""

import pytest

from boolmore.evaluation.agreement import _agreement_score, get_agreements, line
from boolmore.experiment import NAVExperiment
from boolmore.inference.prediction import NAVPrediction
from boolmore.outcome import ComparativeOutcome


# --- line ---


def test_line_interpolates_midpoint():
    assert line(0.5, (0.0, 0.0), (1.0, 1.0)) == pytest.approx(0.5)


def test_line_returns_endpoints():
    assert line(0.0, (0.0, 2.0), (1.0, 4.0)) == pytest.approx(2.0)
    assert line(1.0, (0.0, 2.0), (1.0, 4.0)) == pytest.approx(4.0)


def test_line_outside_segment_is_zero():
    assert line(-0.1, (0.0, 0.0), (1.0, 1.0)) == 0.0
    assert line(1.1, (0.0, 0.0), (1.0, 1.0)) == 0.0


def test_line_decreasing_segment():
    assert line(0.75, (0.5, 1.0), (1.0, 0.0)) == pytest.approx(0.5)


# --- _agreement_score: None / unknown prediction ---


@pytest.mark.parametrize(
    "outcome",
    ["ON", "Some/ON", "Some", "OFF/Some", "OFF", (0.3, 0.4), 0.5],
)
def test_none_predict_scores_full_agreement(outcome):
    assert _agreement_score(outcome, None) == 1.0


# --- _agreement_score: ON  (agreement ≈ predict in [0, 1]) ---


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ],
)
def test_agreement_on(predict, expected):
    assert _agreement_score("ON", predict) == pytest.approx(expected)


# --- _agreement_score: Some/ON  (ramps 0→1 over [0, 0.5], then flat) ---


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.25, 0.5),
        (0.5, 1.0),
        (0.75, 1.0),
        (1.0, 1.0),
    ],
)
def test_agreement_some_on(predict, expected):
    assert _agreement_score("Some/ON", predict) == pytest.approx(expected)


# --- _agreement_score: Some  (peak plateau on [0.25, 0.75]) ---


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.125, 0.5),
        (0.25, 1.0),
        (0.5, 1.0),
        (0.75, 1.0),
        (0.875, 0.5),
        (1.0, 0.0),
    ],
)
def test_agreement_some(predict, expected):
    assert _agreement_score("Some", predict) == pytest.approx(expected)


# --- _agreement_score: OFF/Some  (flat 1 on [0, 0.5], then fall to 0) ---


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 1.0),
        (0.25, 1.0),
        (0.5, 1.0),
        (0.75, 0.5),
        (1.0, 0.0),
    ],
)
def test_agreement_off_some(predict, expected):
    assert _agreement_score("OFF/Some", predict) == pytest.approx(expected)


# --- _agreement_score: OFF  (agreement ≈ 1 - predict) ---


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 1.0),
        (0.5, 0.5),
        (1.0, 0.0),
    ],
)
def test_agreement_off(predict, expected):
    assert _agreement_score("OFF", predict) == pytest.approx(expected)


# --- _agreement_score: clamp FP noise at [0, 1] ---


@pytest.mark.parametrize(
    "outcome, predict, expected",
    [
        ("ON", 1.0 + 1e-12, 1.0),
        ("Some/ON", 1.0 + 1e-12, 1.0),
        ("OFF", -1e-12, 1.0),
        ("OFF/Some", -1e-12, 1.0),
        ("Some", 1.0 + 1e-12, 0.0),
        ("Some", -1e-12, 0.0),
    ],
)
def test_agreement_clamps_fp_overshoot(outcome, predict, expected):
    assert _agreement_score(outcome, predict) == pytest.approx(expected)


def test_agreement_unknown_outcome_raises():
    with pytest.raises(ValueError, match="Unexpected experiment outcome"):
        _agreement_score("MAYBE", 0.5)


# --- _agreement_score: tuple ranges equivalent to string labels ---


@pytest.mark.parametrize(
    "label, rng",
    [
        ("ON", (1.0, 1.0)),
        ("Some/ON", (0.5, 1.0)),
        ("Some", (0.25, 0.75)),
        ("OFF/Some", (0.0, 0.5)),
        ("OFF", (0.0, 0.0)),
    ],
)
@pytest.mark.parametrize("predict", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_tuple_range_matches_label(label, rng, predict):
    assert _agreement_score(rng, predict) == pytest.approx(
        _agreement_score(label, predict)
    )


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.15, 0.5),  # midpoint of left shoulder [0, 0.3]
        (0.3, 1.0),
        (0.35, 1.0),
        (0.4, 1.0),
        (0.7, 0.5),  # midpoint of right shoulder [0.4, 1.0]
        (1.0, 0.0),
    ],
)
def test_agreement_custom_range(predict, expected):
    assert _agreement_score((0.3, 0.4), predict) == pytest.approx(expected)


@pytest.mark.parametrize(
    "outcome",
    [
        (0.6, 0.4),
        (-0.1, 0.5),
        (0.0, 1.1),
        (1.5, 2.0),
    ],
)
def test_agreement_invalid_range_raises(outcome):
    with pytest.raises(ValueError, match="Outcome range must satisfy"):
        _agreement_score(outcome, 0.5)


# --- _agreement_score: scalar float outcomes (±0.1 band) ---


@pytest.mark.parametrize("predict", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_float_zero_matches_off(predict):
    assert _agreement_score(0.0, predict) == pytest.approx(
        _agreement_score("OFF", predict)
    )


@pytest.mark.parametrize("predict", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_float_one_matches_on(predict):
    assert _agreement_score(1.0, predict) == pytest.approx(
        _agreement_score("ON", predict)
    )


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.2, 0.5),  # midpoint of left shoulder [0, 0.4]
        (0.4, 1.0),
        (0.5, 1.0),
        (0.6, 1.0),
        (0.8, 0.5),  # midpoint of right shoulder [0.6, 1.0]
        (1.0, 0.0),
    ],
)
def test_agreement_float_mid(predict, expected):
    assert _agreement_score(0.5, predict) == pytest.approx(expected)


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 1.0),
        (0.075, 1.0),
        (0.15, 1.0),
        (0.575, 0.5),  # midpoint of right shoulder [0.15, 1.0]
        (1.0, 0.0),
    ],
)
def test_agreement_float_near_zero_clamps(predict, expected):
    assert _agreement_score(0.05, predict) == pytest.approx(expected)


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.425, 0.5),  # midpoint of left shoulder [0, 0.85]
        (0.85, 1.0),
        (0.95, 1.0),
        (1.0, 1.0),
    ],
)
def test_agreement_float_near_one_clamps(predict, expected):
    assert _agreement_score(0.95, predict) == pytest.approx(expected)


@pytest.mark.parametrize("outcome", [-0.1, 1.1, 2.0])
def test_agreement_invalid_float_raises(outcome):
    with pytest.raises(ValueError, match="Outcome float must satisfy"):
        _agreement_score(outcome, 0.5)


# --- get_agreements ---


def _prediction(fixes, values):
    return {fixes: NAVPrediction(fixes=fixes, average_values=values)}


def test_get_agreements_groups_by_observed_node():
    fixes_a = (("S", 0),)
    fixes_b = (("S", 1),)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome="ON", weight=2.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="B", outcome="OFF", weight=3.0),
        NAVExperiment(id=3, fixes=(), observed="A", outcome="Some", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 1.0, "B": 0.0}),
        **_prediction(fixes_b, {"A": 0.0, "B": 0.0}),
        **_prediction((), {"A": 0.5, "B": 0.5}),
    }

    agreements = get_agreements(experiments, predictions)

    assert set(agreements) == {"A", "B"}
    assert set(agreements["A"]) == {fixes_a, ()}
    assert set(agreements["B"]) == {fixes_b}

    exp_id, weight, outcome, predict, agreement = agreements["A"][fixes_a]
    assert (exp_id, weight, outcome, predict, agreement) == (1, 2.0, "ON", 1.0, 1.0)

    exp_id, weight, outcome, predict, agreement = agreements["B"][fixes_b]
    assert (exp_id, weight, outcome, predict, agreement) == (2, 3.0, "OFF", 0.0, 1.0)

    exp_id, weight, outcome, predict, agreement = agreements["A"][()]
    assert (exp_id, weight, outcome, predict) == (3, 1.0, "Some", 0.5)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_none_predict_is_optimistic():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction((), {"A": None})

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][()]
    assert predict is None
    assert agreement == 1.0


def test_get_agreements_duplicate_fixes_raises():
    fixes = (("X", 1),)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome="ON", weight=1.0),
        NAVExperiment(id=2, fixes=fixes, observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction(fixes, {"A": 0.5})

    with pytest.raises(ValueError, match="Duplicate experimental entry"):
        get_agreements(experiments, predictions)


def test_get_agreements_same_fixes_different_observed_ok():
    fixes = (("X", 1),)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome="ON", weight=1.0),
        NAVExperiment(id=2, fixes=fixes, observed="B", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction(fixes, {"A": 1.0, "B": 0.0})

    agreements = get_agreements(experiments, predictions)
    assert agreements["A"][fixes][0] == 1
    assert agreements["B"][fixes][0] == 2


def test_get_agreements_unexpected_outcome_raises():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="MAYBE", weight=1.0),
    ]
    predictions = _prediction((), {"A": 0.5})

    with pytest.raises(ValueError, match="Unexpected experiment outcome"):
        get_agreements(experiments, predictions)


def test_get_agreements_empty():
    assert get_agreements([], {}) == {}


def test_get_agreements_tuple_outcome():
    experiments = [
        NAVExperiment(
            id=1, fixes=(), observed="A", outcome=(0.5, 1.0), weight=1.0
        ),
    ]
    predictions = _prediction((), {"A": 0.75})

    agreements = get_agreements(experiments, predictions)
    exp_id, weight, outcome, predict, agreement = agreements["A"][()]
    assert (exp_id, weight, outcome, predict) == (1, 1.0, (0.5, 1.0), 0.75)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_avg_observed_uses_stored_key():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A_avg", outcome="Some", weight=1.0),
    ]
    predictions = _prediction(
        (),
        {"A": 0.0, "A_2": 1.0, "A_avg": 0.5},
    )

    agreements = get_agreements(experiments, predictions)
    exp_id, weight, outcome, predict, agreement = agreements["A_avg"][()]
    assert (exp_id, weight, outcome, predict) == (1, 1.0, "Some", 0.5)
    assert agreement == pytest.approx(1.0)


# --- get_agreements: comparative outcomes ---


def test_get_agreements_higher_than():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="higher", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 1.0}),
        **_prediction(fixes_b, {"A": 0.0}),
    }

    agreements = get_agreements(experiments, predictions)
    exp_id, weight, out, predict, agreement = agreements["A"][fixes_a]
    assert exp_id == 1
    assert weight == 1.0
    assert out == outcome
    assert predict == pytest.approx(1.0)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_lower_than():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="lower", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="A", outcome="ON", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 0.0}),
        **_prediction(fixes_b, {"A": 1.0}),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][fixes_a]
    assert predict == pytest.approx(1.0)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_higher_than_wrong_direction_scores_zero():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="higher", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="A", outcome="ON", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 0.0}),
        **_prediction(fixes_b, {"A": 1.0}),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][fixes_a]
    assert predict == pytest.approx(-1.0)
    assert agreement == pytest.approx(0.0)


def test_get_agreements_same_as():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="same", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="A", outcome="Some", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 0.5}),
        **_prediction(fixes_b, {"A": 0.5}),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][fixes_a]
    assert predict == pytest.approx(0.0)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_same_as_large_diff_scores_zero():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="same", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 1.0}),
        **_prediction(fixes_b, {"A": 0.0}),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][fixes_a]
    assert predict == pytest.approx(1.0)
    assert agreement == pytest.approx(0.0)


def test_get_agreements_comparative_none_is_optimistic():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="higher", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": None}),
        **_prediction(fixes_b, {"A": 0.0}),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][fixes_a]
    assert predict is None
    assert agreement == 1.0


def test_get_agreements_comparative_missing_id_raises():
    experiments = [
        NAVExperiment(
            id=1,
            fixes=(),
            observed="A",
            outcome=ComparativeOutcome(relation="higher", other_id=99),
            weight=1.0,
        ),
    ]
    predictions = _prediction((), {"A": 1.0})

    with pytest.raises(ValueError, match="missing id 99"):
        get_agreements(experiments, predictions)


def test_get_agreements_comparative_uses_each_observed():
    fixes_a = (("S", 1),)
    fixes_b = ()
    outcome = ComparativeOutcome(relation="higher", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="B", outcome="OFF", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes_a, {"A": 0.75, "B": 0.0}),
        **_prediction(fixes_b, {"A": 0.0, "B": 0.25}),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][fixes_a]
    assert predict == pytest.approx(0.5)
    assert agreement == pytest.approx(0.5)
