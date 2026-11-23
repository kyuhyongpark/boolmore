"""Characterization tests for NAV outcome–prediction agreement curves."""

import pytest

from boolmore.evaluation.agreement import (
    _agreement_score,
    _average_related_values,
    _related_node_names,
    build_related_nodes,
    entity_max_level,
    get_agreements,
    line,
)
from boolmore.experiment import NAVExperiment
from boolmore.inference.prediction import NAVPrediction
from boolmore.outcome import (
    ComparativeOutcome,
    comparative_higher_lower_threshold,
    resolve_outcome_range,
)


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


# --- _agreement_score: scalar float outcomes (±0.25 band) ---


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
        (0.125, 0.5),  # midpoint of left shoulder [0, 0.25]
        (0.25, 1.0),
        (0.5, 1.0),
        (0.75, 1.0),
        (0.875, 0.5),  # midpoint of right shoulder [0.75, 1.0]
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
        (0.65, 0.5),  # midpoint of right shoulder [0.3, 1.0]
        (1.0, 0.0),
    ],
)
def test_agreement_float_near_zero_clamps(predict, expected):
    assert _agreement_score(0.05, predict) == pytest.approx(expected)


@pytest.mark.parametrize(
    "predict, expected",
    [
        (0.0, 0.0),
        (0.35, 0.5),  # midpoint of left shoulder [0, 0.7]
        (0.7, 1.0),
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
    assert set(agreements["A"]) == {()}
    assert set(agreements["A"][()]) == {fixes_a, ()}
    assert set(agreements["B"]) == {()}
    assert set(agreements["B"][()]) == {fixes_b}

    exp_id, weight, outcome, predict, agreement = agreements["A"][()][fixes_a][0]
    assert (exp_id, weight, outcome, predict, agreement) == (1, 2.0, "ON", 1.0, 1.0)

    exp_id, weight, outcome, predict, agreement = agreements["B"][()][fixes_b][0]
    assert (exp_id, weight, outcome, predict, agreement) == (2, 3.0, "OFF", 0.0, 1.0)

    exp_id, weight, outcome, predict, agreement = agreements["A"][()][()][0]
    assert (exp_id, weight, outcome, predict) == (3, 1.0, "Some", 0.5)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_none_predict_is_optimistic():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction((), {"A": None})

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["A"][()][()][0]
    assert predict is None
    assert agreement == 1.0


def test_get_agreements_duplicate_absolute_raises():
    fixes = (("X", 1),)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome="ON", weight=1.0),
        NAVExperiment(id=2, fixes=fixes, observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction(fixes, {"A": 0.5})

    with pytest.raises(ValueError, match="non-comparative outcomes"):
        get_agreements(experiments, predictions)


def test_get_agreements_validate_co_located_false_allows_duplicate_absolute():
    fixes = (("X", 1),)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome="ON", weight=1.0),
        NAVExperiment(id=2, fixes=fixes, observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction(fixes, {"A": 0.5})

    agreements = get_agreements(
        experiments, predictions, validate_co_located=False
    )
    entries = agreements["A"][()][fixes]
    assert len(entries) == 2
    assert {entry[0] for entry in entries} == {1, 2}


def test_get_agreements_duplicate_comparative_other_id_raises():
    fixes = (("X", 1),)
    outcome = ComparativeOutcome(relation="higher", other_id=3)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=2, fixes=fixes, observed="A", outcome=outcome, weight=1.0),
        NAVExperiment(id=3, fixes=(), observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes, {"A": 1.0}),
        **_prediction((), {"A": 0.0}),
    }

    with pytest.raises(ValueError, match="comparative other_id=3"):
        get_agreements(experiments, predictions)


def test_get_agreements_absolute_plus_comparative_ok():
    fixes = (("X", 1),)
    comparative = ComparativeOutcome(relation="higher", other_id=2)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome="ON", weight=1.0),
        NAVExperiment(
            id=3, fixes=fixes, observed="A", outcome=comparative, weight=1.0
        ),
        NAVExperiment(id=2, fixes=(), observed="A", outcome="OFF", weight=1.0),
    ]
    predictions = {
        **_prediction(fixes, {"A": 1.0}),
        **_prediction((), {"A": 0.0}),
    }

    agreements = get_agreements(experiments, predictions)
    entries = agreements["A"][()][fixes]
    assert len(entries) == 2
    assert {e[0] for e in entries} == {1, 3}


def test_get_agreements_two_comparatives_different_other_ids_ok():
    fixes = (("X", 1),)
    experiments = [
        NAVExperiment(
            id=1,
            fixes=fixes,
            observed="A",
            outcome=ComparativeOutcome(relation="higher", other_id=3),
            weight=1.0,
        ),
        NAVExperiment(
            id=2,
            fixes=fixes,
            observed="A",
            outcome=ComparativeOutcome(relation="same", other_id=4),
            weight=1.0,
        ),
        NAVExperiment(id=3, fixes=(), observed="A", outcome="OFF", weight=1.0),
        NAVExperiment(
            id=4, fixes=(("Y", 1),), observed="A", outcome="Some", weight=1.0
        ),
    ]
    predictions = {
        **_prediction(fixes, {"A": 1.0}),
        **_prediction((), {"A": 0.0}),
        **_prediction((("Y", 1),), {"A": 1.0}),
    }

    agreements = get_agreements(experiments, predictions)
    assert {e[0] for e in agreements["A"][()][fixes]} == {1, 2}


def test_get_agreements_same_fixes_different_observed_ok():
    fixes = (("X", 1),)
    experiments = [
        NAVExperiment(id=1, fixes=fixes, observed="A", outcome="ON", weight=1.0),
        NAVExperiment(id=2, fixes=fixes, observed="B", outcome="OFF", weight=1.0),
    ]
    predictions = _prediction(fixes, {"A": 1.0, "B": 0.0})

    agreements = get_agreements(experiments, predictions)
    assert agreements["A"][()][fixes][0][0] == 1
    assert agreements["B"][()][fixes][0][0] == 2


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
    exp_id, weight, outcome, predict, agreement = agreements["A"][()][()][0]
    assert (exp_id, weight, outcome, predict) == (1, 1.0, (0.5, 1.0), 0.75)
    assert agreement == pytest.approx(1.0)


def test_get_agreements_observed_means_siblings():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
    ]
    predictions = _prediction((), {"A": 0.0, "A_2": 1.0})

    agreements = get_agreements(experiments, predictions)
    exp_id, weight, outcome, predict, agreement = agreements["A"][()][()][0]
    assert (exp_id, weight, outcome, predict) == (1, 1.0, "Some", 0.5)
    assert agreement == pytest.approx(1.0)


def test_related_node_names_includes_int_siblings_excludes_unrelated():
    values = {
        "A": 0.0,
        "A_2": 1.0,
        "A_3": 0.5,
        "A_X": 0.0,  # letter suffix is not a sibling
        "A_T": 0.9,  # distinct entity (transcript), not a sibling
        "A_iT": 0.25,
        "A_2b": 0.75,
        "AB": 1.0,
    }
    assert sorted(_related_node_names("A", values)) == ["A", "A_2", "A_3"]
    assert _average_related_values(values, "A") == pytest.approx(0.5)


def test_build_related_nodes_matches_scan():
    node_names = ["A", "A_2", "A_3", "B", "A_X"]
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
        NAVExperiment(id=2, fixes=(), observed="B", outcome="OFF", weight=1.0),
    ]
    related = build_related_nodes(node_names, experiments)
    assert sorted(related["A"]) == sorted(_related_node_names("A", node_names))
    assert sorted(related["B"]) == sorted(_related_node_names("B", node_names))


def test_get_agreements_with_related_nodes_matches_uncached():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
    ]
    predictions = _prediction((), {"A": 0.0, "A_2": 1.0})
    related = build_related_nodes(["A", "A_2"], experiments)

    uncached = get_agreements(experiments, predictions)
    cached = get_agreements(experiments, predictions, related_nodes=related)
    assert cached == uncached
    assert _average_related_values(
        predictions[()].average_values, "A", related_nodes=related
    ) == pytest.approx(0.5)


def test_get_agreements_observed_none_when_any_sibling_none():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
    ]
    predictions = _prediction((), {"A": None, "A_2": None})

    agreements = get_agreements(experiments, predictions)
    _, _, _, predict, agreement = agreements["A"][()][()][0]
    assert predict is None
    assert agreement == pytest.approx(1.0)


def test_get_agreements_observed_no_siblings_raises():
    experiments = [
        NAVExperiment(id=1, fixes=(), observed="A", outcome="Some", weight=1.0),
    ]
    predictions = _prediction((), {"B": 0.5})

    with pytest.raises(ValueError, match="No related nodes"):
        get_agreements(experiments, predictions)


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
    exp_id, weight, out, predict, agreement = agreements["A"][()][fixes_a][0]
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
    _id, _w, _out, predict, agreement = agreements["A"][()][fixes_a][0]
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
    _id, _w, _out, predict, agreement = agreements["A"][()][fixes_a][0]
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
    _id, _w, _out, predict, agreement = agreements["A"][()][fixes_a][0]
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
    _id, _w, _out, predict, agreement = agreements["A"][()][fixes_a][0]
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
    _id, _w, _out, predict, agreement = agreements["A"][()][fixes_a][0]
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
    _id, _w, _out, predict, agreement = agreements["A"][()][fixes_a][0]
    assert predict == pytest.approx(0.5)
    assert agreement == pytest.approx(1.0)


@pytest.mark.parametrize(
    "related, expected_k",
    [
        (["A"], 1),
        (["A_1", "A_2"], 2),
        (["A", "A_1", "A_2", "A_3"], 3),
        (["Proliferation_1", "Proliferation_2", "Proliferation_3", "Proliferation_4"], 4),
    ],
)
def test_entity_max_level(related, expected_k):
    base = "Proliferation" if related[0].startswith("Proliferation") else "A"
    assert entity_max_level(base, related) == expected_k


@pytest.mark.parametrize(
    "max_level, expected",
    [(1, 0.5), (2, 0.25), (3, 1.0 / 6.0), (4, 0.125)],
)
def test_comparative_higher_lower_threshold(max_level, expected):
    assert comparative_higher_lower_threshold(max_level) == pytest.approx(expected)
    assert resolve_outcome_range(
        ComparativeOutcome(relation="higher", other_id=2),
        max_level=max_level,
    ) == pytest.approx((expected, 1.0))


def test_agreement_score_higher_uses_max_level():
    outcome = ComparativeOutcome(relation="higher", other_id=2)
    # diff 0.25: Boolean k=1 → half credit; k=4 → full credit
    assert _agreement_score(outcome, 0.25, max_level=1) == pytest.approx(0.5)
    assert _agreement_score(outcome, 0.25, max_level=4) == pytest.approx(1.0)
    assert _agreement_score(outcome, 0.0625, max_level=4) == pytest.approx(0.5)


def test_get_agreements_higher_than_multi_level_threshold():
    """Five-level entity (P_1..P_4): k=4 ⇒ threshold 1/8; Δ=0.25 is full agree."""
    fixes_a = (("S", 1),)
    fixes_b = ()
    higher = ComparativeOutcome(relation="higher", other_id=2)
    lower = ComparativeOutcome(relation="lower", other_id=1)
    experiments = [
        NAVExperiment(id=1, fixes=fixes_a, observed="P", outcome=higher, weight=1.0),
        NAVExperiment(id=2, fixes=fixes_b, observed="P", outcome=lower, weight=1.0),
    ]
    preds_a = {"P_1": 1.0, "P_2": 1.0, "P_3": 0.0, "P_4": 0.0}  # mean 0.5
    preds_b = {"P_1": 1.0, "P_2": 0.0, "P_3": 0.0, "P_4": 0.0}  # mean 0.25
    predictions = {
        **_prediction(fixes_a, preds_a),
        **_prediction(fixes_b, preds_b),
    }

    agreements = get_agreements(experiments, predictions)
    _id, _w, _out, predict, agreement = agreements["P"][()][fixes_a][0]
    assert predict == pytest.approx(0.25)
    assert agreement == pytest.approx(1.0)
