import pytest

from boolmore.model import Model


@pytest.fixture
def base_primes():
    return {
        "A": [[{"A": 0}], [{"A": 1}]],
        "B": [[{"A": 1}, {"B": 0}], [{"A": 0, "B": 1}]],
    }


def test_from_primes_creates_base_model(base_primes):

    model = Model.from_primes(base_primes)

    assert model.base is model
    assert model.edge_pool == []
    assert model.primes == base_primes
    assert model.regulators_dict == {"A": ("A",), "B": ("A","B")}
    assert model.signs_dict == {"A": "1", "B": "01"}
    assert model.rr_dict == {"A": "10", "B": "1000"}
    assert model.extra_edges == []
    assert model.n_extra_edges == 0


def test_from_primes_uses_base_model(base_primes):
    edge_pool = [("B", "A", "1")]
    base = Model.from_primes(base_primes, edge_pool=edge_pool)

    primes = {
        "A": [[{"A": 0}, {"B": 0}],[{"A": 1, "B": 1}]],
        "B": [[{"A": 1}, {"B": 0}],[{"A": 0, "B": 1}]],
    }

    model = Model.from_primes(primes,base=base)

    assert model.base is base
    assert model.edge_pool == base.edge_pool
    assert model.primes == primes
    assert model.regulators_dict == {"A": ("A","B"), "B": ("A","B")}
    assert model.signs_dict == {"A": "11", "B": "01"}
    assert model.rr_dict == {"A": "1000", "B": "1000"}
    assert model.extra_edges == edge_pool
    assert model.n_extra_edges == len(edge_pool)


def test_from_primes_rejects_sign_mismatch(base_primes):
    edge_pool = [("B", "A", "0")]

    base = Model.from_primes(base_primes, edge_pool=edge_pool)

    primes = {
        "A": [[{"A": 0}, {"B": 0}],[{"A": 1, "B": 1}]],
        "B": [[{"A": 1}, {"B": 0}],[{"A": 0, "B": 1}]],
    }

    with pytest.raises(ValueError):
        Model.from_primes(primes, base=base)


def test_from_primes_ignores_irrelevant_edge(base_primes):
    edge_pool = [("B", "A", "1")]

    model = Model.from_primes(base_primes, edge_pool=edge_pool)

    assert model.extra_edges == []
    assert model.n_extra_edges == 0


def test_validate_edge_pool_rejects_invalid_edge(base_primes):
    model = Model.from_primes(base_primes)
    model.edge_pool = [("C", "A", "2")]

    with pytest.raises(ValueError):
        model.validate_edge_pool()


def test_validate_edge_pool_rejects_duplicate_regulator_target(base_primes):
    model = Model.from_primes(base_primes)
    model.edge_pool = [
        ("C", "A", "1"),
        ("C", "A", "0"),
    ]

    with pytest.raises(ValueError):
        model.validate_edge_pool()


def test_validate_edge_pool_rejects_existing_base_edge(base_primes):
    model = Model.from_primes(base_primes)
    model.edge_pool = [("A", "B", "1")]

    with pytest.raises(ValueError):
        model.validate_edge_pool()


def test_from_primes_allows_removed_base_regulator(base_primes):
    base = Model.from_primes(base_primes)

    primes = {
        "A": base_primes["A"],
        "B": [[{"B": 0}], [{"B": 1}]],
    }

    model = Model.from_primes(primes, base=base)

    assert model.regulators_dict["B"] == ("A", "B")


def test_construct_edges(base_primes):
    base = Model.from_primes(
        base_primes,
        edge_pool=[("C", "B", "1")],
    )

    model = Model.from_primes(
        {
            "A": base_primes["A"],
            "B": [[{"A": 1}, {"C": 0}], [{"A": 0, "C": 1}]],
        },
        base=base,
    )

    edges = model._construct_edges()

    assert edges == [
        {
            "regulator": "A",
            "target": "A",
            "sign": "1",
            "source": "base",
            "effective": True,
        },
        {
            "regulator": "A",
            "target": "B",
            "sign": "0",
            "source": "base",
            "effective": True,
        },
        {
            "regulator": "B",
            "target": "B",
            "sign": "1",
            "source": "base",
            "effective": False,
        },
        {
            "regulator": "C",
            "target": "B",
            "sign": "1",
            "source": "edge_pool",
            "effective": True,
        },
    ]

def test_construct_edges_BASE(base_primes):
    base = Model.from_primes(
        base_primes,
        edge_pool=[("C", "B", "1")],
    )

    model = Model.from_primes(
        base_primes,
        base=base,
    )

    edges = model._construct_edges()

    assert edges == [
        {
            "regulator": "A",
            "target": "A",
            "sign": "1",
            "source": "base",
            "effective": True,
        },
        {
            "regulator": "A",
            "target": "B",
            "sign": "0",
            "source": "base",
            "effective": True,
        },
        {
            "regulator": "B",
            "target": "B",
            "sign": "1",
            "source": "base",
            "effective": True,
        },
    ]