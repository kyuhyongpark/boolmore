from boolmore.model import Model
from boolmore.evaluation.complexity import get_model_complexity


def test_get_model_complexity():
    primes = {
        "A": [
                [{"B":0, "C":1}],
                [{"B": 1},{"C": 0}]
             ],
        "B": [
                [{"A": 0}, {"B": 1}],
                [{"A": 1, "B": 0},]
             ],
        "C": [
                [{"A": 0, "D": 0}],
                [{"A": 1}, {"D": 1}]
             ],
        "D": [
                [{"D": 0}], 
                [{"D": 1}]
             ],
    }

    model = Model.from_primes(primes)

    complexity = get_model_complexity(model)

    assert complexity == {
        "n_edges": 7,
        "n_self_edges": 2,
        "n_prime_implicants": 7,
        "n_extra_edges": 0
    }