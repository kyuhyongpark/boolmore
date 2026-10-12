from boolmore.evaluation.nav_score import (
    _is_intervention_subset,
    get_NAV_scores,
    get_flat_scores,
    get_hierarchy_scores,
    intervention_hierarchy,
)
from boolmore.experiment import NAVExperiment


# --- intervention_hierarchy ---


def test_intervention_hierarchy_empty():
    assert intervention_hierarchy((), {"S": 0}) == 0


def test_intervention_hierarchy_defaults_ignored():
    fixes = (("S", 0), ("T", 1))
    default_sources = {"S": 0, "T": 1}
    assert intervention_hierarchy(fixes, default_sources) == 0


def test_intervention_hierarchy_counts_non_defaults():
    fixes = (("S", 0), ("T", 1), ("X", 1))
    default_sources = {"S": 0, "T": 0}
    # ("S", 0) ignored, ("T", 1) counted, ("X", 1) counted
    assert intervention_hierarchy(fixes, default_sources) == 2


def test_intervention_hierarchy_non_source_always_counts():
    fixes = (("X", 0), ("Y", 1))
    assert intervention_hierarchy(fixes, {"S": 0}) == 2


# --- _is_intervention_subset ---


def test_is_intervention_subset_empty_candidate():
    assert _is_intervention_subset({}, {"S": 0, "X": 1}, {"S": 0}) is True


def test_is_intervention_subset_defaults_ignored():
    candidate = {"S": 0}
    reference = {"S": 0, "X": 1}
    assert _is_intervention_subset(candidate, reference, {"S": 0}) is True


def test_is_intervention_subset_matching_non_default():
    candidate = {"S": 1, "X": 1}
    reference = {"S": 1, "X": 1, "Y": 0}
    assert _is_intervention_subset(candidate, reference, {"S": 0}) is True


def test_is_intervention_subset_missing_non_default():
    candidate = {"X": 1}
    reference = {"S": 0}
    assert _is_intervention_subset(candidate, reference, {"S": 0}) is False


def test_is_intervention_subset_value_mismatch():
    candidate = {"X": 1}
    reference = {"X": 0}
    assert _is_intervention_subset(candidate, reference, {"S": 0}) is False


def test_is_intervention_subset_non_default_source_mismatch():
    candidate = {"S": 1}
    reference = {"S": 0}
    assert _is_intervention_subset(candidate, reference, {"S": 0}) is False


# --- get_flat_scores ---


def test_get_flat_scores_basic():
    fixes = (("S", 0), ("X", 1))
    agreements = {
        "A": {
            fixes: (1, 2.0, "ON", 1.0, 0.5),
        }
    }
    scores = get_flat_scores(agreements, {"S": 0})

    assert len(scores) == 1
    item = scores[0]
    assert item.id == 1
    assert item.weight == 2.0
    assert item.agreement == 0.5
    assert item.score == 1.0
    assert item.hierarchy == 1
    assert item.fixes == fixes
    assert item.observed == "A"
    assert item.outcome == "ON"
    assert item.predict_value == 1.0


def test_get_flat_scores_empty_default_sources():
    fixes = (("X", 1),)
    agreements = {
        "A": {
            fixes: (1, 1.0, "OFF", 0.0, 1.0),
        }
    }
    scores = get_flat_scores(agreements)

    assert scores[0].score == 1.0
    assert scores[0].hierarchy == 1


def test_get_flat_scores_empty_agreements():
    assert get_flat_scores({}) == []


# --- get_hierarchy_scores ---


def test_get_hierarchy_scores_propagates_general_disagreement():
    """More specific experiments get zero score if a hierarchical subset disagrees."""
    default_sources = {"S": 0}
    general = (("S", 0),)
    specific = (("S", 0), ("X", 1))
    agreements = {
        "A": {
            general: (1, 1.0, "ON", 0.0, 0.0),
            specific: (2, 2.0, "ON", 1.0, 1.0),
        }
    }

    scores = {item.id: item for item in get_hierarchy_scores(agreements, default_sources)}

    assert scores[1].score == 0.0
    assert scores[1].hierarchy == 0
    # weight * general_agreement * specific_agreement = 2.0 * 0.0 * 1.0
    assert scores[2].score == 0.0
    assert scores[2].hierarchy == 1


def test_get_hierarchy_scores_multiplies_subset_agreements():
    default_sources = {"S": 0}
    general = (("S", 0),)
    specific = (("S", 0), ("X", 1))
    agreements = {
        "A": {
            general: (1, 1.0, "ON", 1.0, 1.0),
            specific: (2, 2.0, "ON", 0.5, 0.5),
        }
    }

    scores = {item.id: item for item in get_hierarchy_scores(agreements, default_sources)}

    assert scores[1].score == 1.0
    # weight * general * specific = 2.0 * 1.0 * 0.5
    assert scores[2].score == 1.0


def test_get_hierarchy_scores_independent_observed_nodes():
    """Hierarchy subsets are scoped per observed node."""
    default_sources = {"S": 0}
    shared_fixes = (("S", 0),)
    agreements = {
        "A": {
            shared_fixes: (1, 1.0, "ON", 1.0, 1.0),
        },
        "B": {
            shared_fixes: (2, 3.0, "OFF", 0.0, 1.0),
        },
    }

    scores = get_hierarchy_scores(agreements, default_sources)
    by_id = {item.id: item for item in scores}

    assert by_id[1].score == 1.0
    assert by_id[2].score == 3.0
    assert by_id[1].observed == "A"
    assert by_id[2].observed == "B"


# --- get_NAV_scores ---


def test_get_NAV_scores_flat():
    experiments = [
        NAVExperiment(
            id=1,
            fixes=(("S", 0),),
            observed="A",
            outcome="ON",
            weight=2.0,
        ),
    ]
    predictions = {
        (("S", 0),): {"A": 1.0},
    }

    scores = get_NAV_scores(
        experiments,
        predictions,
        default_sources={"S": 0},
        hierarchy=False,
    )

    assert len(scores) == 1
    assert scores[0].agreement == 1.0
    assert scores[0].score == 2.0
    assert scores[0].hierarchy == 0


def test_get_NAV_scores_hierarchy_default():
    experiments = [
        NAVExperiment(
            id=1,
            fixes=(("S", 0),),
            observed="A",
            outcome="ON",
            weight=1.0,
        ),
        NAVExperiment(
            id=2,
            fixes=(("S", 0), ("X", 1)),
            observed="A",
            outcome="ON",
            weight=2.0,
        ),
    ]
    predictions = {
        (("S", 0),): {"A": 0.0},
        (("S", 0), ("X", 1)): {"A": 1.0},
    }

    scores = get_NAV_scores(
        experiments,
        predictions,
        default_sources={"S": 0},
    )
    by_id = {item.id: item for item in scores}

    assert by_id[1].agreement == 0.0
    assert by_id[1].score == 0.0
    assert by_id[2].agreement == 1.0
    # hierarchy multiplies in the general disagreement
    assert by_id[2].score == 0.0


def test_get_NAV_scores_hierarchy_all_agree():
    experiments = [
        NAVExperiment(
            id=1,
            fixes=(("S", 0),),
            observed="A",
            outcome="OFF",
            weight=1.0,
        ),
        NAVExperiment(
            id=2,
            fixes=(("S", 0), ("X", 1)),
            observed="A",
            outcome="ON",
            weight=2.0,
        ),
    ]
    predictions = {
        (("S", 0),): {"A": 0.0},
        (("S", 0), ("X", 1)): {"A": 1.0},
    }

    scores = get_NAV_scores(
        experiments,
        predictions,
        default_sources={"S": 0},
        hierarchy=True,
    )
    by_id = {item.id: item for item in scores}

    assert by_id[1].score == 1.0
    assert by_id[2].score == 2.0
