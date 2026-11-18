"""
Lock the external rr convention A&B | A | B | 1.

Legacy oracles copy the pre-refactor index formulas so production helpers
must keep identical external I/O while migrating to internal mask order.
"""

from __future__ import annotations

import itertools
import random
from unittest.mock import patch

import pytest

import boolmore.boolean_functions as bf
from boolmore.boolean_functions import (
    get_max_irr,
    get_uni_rr,
    prime2rr,
    rr2bnet,
    rr2prime,
)
from boolmore.evaluation.constraint import (
    check_necessary,
    check_regulate,
    check_source,
)
from boolmore.genetic.operators.constraint_enforcement import (
    group_bi2bi,
    group_rr2rr,
    impose_necessary,
    rr2group_rr,
)
from boolmore.genetic.operators.mutation import add_regulator, delete_regulator


# ---------------------------------------------------------------------------
# Legacy conversion oracles (external A&B|A|B|1 index formulas)
# ---------------------------------------------------------------------------


def legacy_prime2rr(prime, regulators=None, signs=None):
    if regulators is None:
        nodes = set()
        for dct in prime[1]:
            nodes = nodes.union(set(dct.keys()))
        nodes = list(nodes)
        nodes.sort()
        regulators = tuple(nodes)

    if len(regulators) == 0:
        signs = ""
        rr = "1" if bool(prime[1]) else "0"
        return regulators, rr, signs

    rr = ["0"] * (2 ** len(regulators))
    if signs is None:
        signs_list = ["*"] * len(regulators)
        for implicant in prime[1]:
            for i, node in enumerate(regulators):
                if node in implicant:
                    if implicant[node] == 0 and signs_list[i] == "*":
                        signs_list[i] = "0"
                    elif implicant[node] == 1 and signs_list[i] == "*":
                        signs_list[i] = "1"
                    else:
                        if implicant[node] != int(signs_list[i]):
                            pass
        signs = "".join(signs_list)

    for implicant in prime[1]:
        bi = ""
        for node in regulators:
            bi += "1" if node in implicant else "0"
        rr[-int(bi, 2) - 1] = "1"

    return regulators, "".join(rr), signs


def legacy_rr2prime(regulators, rr, signs, inverted=False):
    k = len(regulators)
    inv = int(inverted)
    min_rr = get_uni_rr(rr, max=False)
    max_irr = get_max_irr(rr)
    min_irr = get_uni_rr(max_irr, max=False)

    prime: list[list[dict]] = [[], []]
    for i in range(len(rr)):
        if min_rr[i] != "1":
            continue
        bi = format(2**k - i - 1, "0" + str(k) + "b")
        implicant = {}
        for j, num in enumerate(bi):
            if num == "1":
                implicant[regulators[j]] = (int(signs[j]) - inv) % 2
        prime[1 - inv].append(implicant)

    for i in range(len(rr)):
        if min_irr[i] != "1":
            continue
        bi = format(2**k - i - 1, "0" + str(k) + "b")
        implicant = {}
        for j, num in enumerate(bi):
            if num == "1":
                implicant[regulators[j]] = (1 - int(signs[j]) - inv) % 2
        prime[0 - inv].append(implicant)

    return prime


def legacy_rr2bnet(regulators, rr, signs):
    nodes = list(regulators)
    k = len(nodes)
    for i, num in enumerate(signs):
        if num == "0":
            nodes[i] = "!" + nodes[i]

    bnet = "("
    for i, num in enumerate(rr):
        bnet += str(num)
        bi = format(2**k - i - 1, "0" + str(k) + "b")
        for j, node in enumerate(nodes):
            if bi[j] == "1":
                bnet += " & " + node
        bnet += ") | ("
    return bnet.removesuffix(" | (")


def legacy_get_max_irr(rr: str) -> str:
    max_rr = get_uni_rr(rr, max=True)
    max_rr = max_rr.replace("0", "X")
    max_rr = max_rr.replace("1", "0")
    max_rr = max_rr.replace("X", "1")
    return max_rr[::-1]


def legacy_check_source(regulators, rr, node, min_rr=None):
    if node not in regulators:
        return False
    k = len(regulators)
    n = regulators.index(node)
    if min_rr is None:
        min_rr = get_uni_rr(rr, max=False)
    add = sum(int(min_rr[i]) for i in range(2**k))
    return add == 1 and min_rr[2**k - 2 ** (k - n - 1) - 1] == "1"


def legacy_check_regulate(regulators, rr, reg, is_min=False):
    if reg not in regulators:
        return False
    k = len(regulators)
    n = regulators.index(reg)
    bi = rr if is_min else get_uni_rr(rr, max=False)
    add = 0
    for i in range(2**k):
        if i % 2 ** (k - n) <= 2 ** (k - 1 - n) - 1:
            add += int(bi[i])
    return add != 0


def legacy_check_necessary(regulators, rr, nec, min_rr=None):
    if nec not in regulators:
        return False
    if min_rr is not None:
        regulates = legacy_check_regulate(regulators, min_rr, nec, is_min=True)
    else:
        regulates = legacy_check_regulate(regulators, rr, nec)
    if not regulates:
        return False
    k = len(regulators)
    n = regulators.index(nec)
    add = 0
    for i in range(2**k):
        if i % 2 ** (k - n) > 2 ** (k - 1 - n) - 1:
            add += int(rr[i])
    return add == 0


def legacy_impose_necessary(regulators, rr, nec):
    k = len(regulators)
    n = regulators.index(nec)
    bi = list(rr)
    for i in range(2**k):
        if i % 2 ** (k - n) > 2 ** (k - 1 - n) - 1:
            bi[i] = "0"
    return "".join(bi)


def legacy_rr2group_rr(regulators, rr, groups):
    group_regulators = []
    individual_regulators = set(regulators)
    for group in groups:
        group_regulators.append(tuple(group))
        individual_regulators -= set(group)
    group_regulators.extend(sorted(list(individual_regulators)))
    group_regulators = tuple(group_regulators)

    m = len(group_regulators)
    group_rr = ["0"] * (2**m)
    for i in range(2**m):
        group_bi = format(2**m - 1 - i, "0" + str(m) + "b")
        bi = group_bi2bi(group_bi, group_regulators)
        if rr[-int(bi, 2) - 1] == "1":
            group_rr[i] = "1"
    return "".join(group_rr), group_regulators


def legacy_group_rr2rr(regulators, group_rr, group_regulators):
    m = len(group_regulators)
    k = len(regulators)
    rr = ["0"] * (2**k)
    for i in range(2**m):
        group_bi = format(2**m - 1 - i, "0" + str(m) + "b")
        bi = group_bi2bi(group_bi, group_regulators)
        if group_rr[i] == "1":
            rr[-int(bi, 2) - 1] = "1"
    return "".join(rr)


def legacy_add_regulator(regulators, rr, signs, new_regulator, new_sign, bias=0.5):
    nodes = list(regulators)
    nodes.append(new_regulator)
    added_regulators = tuple(nodes)
    added_signs = signs + new_sign

    rnd = random.random()
    if rnd < bias:
        rr = bf.get_max_irr(rr)

    added_rr = ""
    for bi in rr:
        added_rr += str(random.randint(0, 1))
        added_rr += bi

    if rnd < bias:
        added_rr = bf.get_max_irr(added_rr)

    return added_regulators, added_rr, added_signs


def legacy_delete_regulator(regulators, rr, signs, target_regulator, bias=0.5):
    k = len(regulators)
    n = regulators.index(target_regulator)

    nodes = list(regulators)
    del nodes[n]
    deleted_regulators = tuple(nodes)
    deleted_signs = signs[:n] + signs[n + 1 :]

    deleted_rr = ""
    rnd = random.random()
    if rnd < bias:
        rr = bf.get_max_irr(rr)

    bi = list(rr)
    for i in range(2**k):
        if i % 2 ** (k - n) > 2 ** (k - 1 - n) - 1:
            deleted_rr += bi[i]

    if rnd < bias:
        deleted_rr = bf.get_max_irr(deleted_rr)

    return deleted_regulators, deleted_rr, deleted_signs


# ---------------------------------------------------------------------------
# Hardcoded external convention: regulators (A, B) → A&B | A | B | 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,prime,expected_rr,expected_bnet",
    [
        (
            "A&B",
            [[], [{"A": 1, "B": 1}]],
            "1000",
            "(1 & A & B) | (0 & A) | (0 & B) | (0)",
        ),
        (
            "A",
            [[], [{"A": 1}]],
            "0100",
            "(0 & A & B) | (1 & A) | (0 & B) | (0)",
        ),
        (
            "B",
            [[], [{"B": 1}]],
            "0010",
            "(0 & A & B) | (0 & A) | (1 & B) | (0)",
        ),
        (
            "1",
            [[], [{}]],
            "0001",
            "(0 & A & B) | (0 & A) | (0 & B) | (1)",
        ),
    ],
)
def test_external_order_prime_rr_bnet(name, prime, expected_rr, expected_bnet):
    regulators = ("A", "B")
    signs = "11"
    _, rr, out_signs = prime2rr(prime, regulators=regulators, signs=signs)
    assert out_signs == signs
    assert rr == expected_rr
    assert rr2bnet(regulators, rr, signs) == expected_bnet


@pytest.mark.parametrize(
    "rr,expected_activation",
    [
        ("1000", [{"A": 1, "B": 1}]),
        ("0100", [{"A": 1}]),
        ("0010", [{"B": 1}]),
        ("0001", [{}]),
    ],
)
def test_rr2prime_activation_terms(rr, expected_activation):
    primes = rr2prime(("A", "B"), rr, "11")
    assert primes[1] == expected_activation


@pytest.mark.parametrize(
    "rr,expected_max_irr",
    [
        ("1000", "1110"),
        ("0100", "1100"),
        ("0010", "1010"),
        ("0001", "0000"),
    ],
)
def test_get_max_irr_hardcoded(rr, expected_max_irr):
    assert get_max_irr(rr) == expected_max_irr


@pytest.mark.parametrize("k", range(0, 4))
def test_conversions_match_legacy(k):
    regulators = tuple(f"R{i}" for i in range(k))
    signs = "1" * k
    n = 2**k
    for bits in itertools.product("01", repeat=n):
        rr = "".join(bits)
        assert rr2bnet(regulators, rr, signs) == legacy_rr2bnet(regulators, rr, signs)
        assert get_max_irr(rr) == legacy_get_max_irr(rr)
        assert rr2prime(regulators, rr, signs) == legacy_rr2prime(regulators, rr, signs)


@pytest.mark.parametrize("k", range(0, 4))
def test_prime2rr_matches_legacy_single_implicants(k):
    regulators = tuple(f"R{i}" for i in range(k))
    signs = "1" * k
    # empty activation and each singleton/full mask via presence bits
    for mask in range(2**k):
        implicant = {}
        for i, node in enumerate(regulators):
            if mask & (1 << (k - i - 1)):
                implicant[node] = 1
        prime = [[], [implicant]]
        assert prime2rr(prime, regulators=regulators, signs=signs) == legacy_prime2rr(
            prime, regulators=regulators, signs=signs
        )


# ---------------------------------------------------------------------------
# Constraint predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regulators,rr,node,expected",
    [
        (("N",), "10", "N", True),
        (("N",), "01", "N", False),
        (("N", "A"), "0100", "N", True),
        (("N", "A"), "1000", "N", False),
        (("A", "N"), "0010", "N", True),
        (("A", "N"), "0100", "N", False),
        (("A", "N", "B"), "00000100", "N", True),
        (("A", "N", "B"), "01000000", "N", False),
    ],
)
def test_check_source_hardcoded(regulators, rr, node, expected):
    assert check_source(regulators, rr, node) is expected


@pytest.mark.parametrize(
    "regulators,rr,reg,expected",
    [
        (("A", "B"), "0100", "A", True),
        (("A", "B"), "0010", "A", False),
        (("A", "B"), "0010", "B", True),
        (("A", "B"), "0100", "B", False),
        (("A", "B", "C"), "00010000", "A", True),
        (("A", "B", "C"), "00000100", "B", True),
        (("A", "B", "C"), "00000010", "C", True),
        (("A", "B", "C"), "00000001", "A", False),
    ],
)
def test_check_regulate_hardcoded(regulators, rr, reg, expected):
    assert check_regulate(regulators, rr, reg) is expected


@pytest.mark.parametrize(
    "regulators,rr,nec,expected",
    [
        (("A", "B"), "0100", "A", True),
        (("A", "B"), "1100", "A", True),
        (("A", "B"), "0110", "A", False),  # B alone also on
        (("A", "B"), "0010", "A", False),
        (("A", "B"), "0010", "B", True),
        (("A", "B"), "1010", "B", True),
        (("A", "B"), "0110", "B", False),
        (("A", "N", "B"), "00010000", "N", False),  # A alone; N absent
        (("A", "N", "B"), "01000000", "N", True),  # A&N only
        (("A", "N", "B"), "01010000", "N", False),  # A without N is on
    ],
)
def test_check_necessary_hardcoded(regulators, rr, nec, expected):
    assert check_necessary(regulators, rr, nec) is expected


@pytest.mark.parametrize("k", range(1, 4))
def test_constraint_predicates_match_legacy(k):
    regulators = tuple(f"R{i}" for i in range(k))
    n = 2**k
    for bits in itertools.product("01", repeat=n):
        rr = "".join(bits)
        for reg in regulators:
            assert check_source(regulators, rr, reg) == legacy_check_source(
                regulators, rr, reg
            )
            assert check_regulate(regulators, rr, reg) == legacy_check_regulate(
                regulators, rr, reg
            )
            assert check_necessary(regulators, rr, reg) == legacy_check_necessary(
                regulators, rr, reg
            )


# ---------------------------------------------------------------------------
# Enforcement + regulator edit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regulators,rr,nec,expected",
    [
        (("A", "B"), "1111", "A", "1100"),
        (("A", "B"), "1111", "B", "1010"),
        (("A", "N", "B"), "11111111", "N", "11001100"),
        (("A", "B"), "0100", "A", "0100"),
        (("A", "B"), "0110", "A", "0100"),
    ],
)
def test_impose_necessary_hardcoded(regulators, rr, nec, expected):
    assert impose_necessary(regulators, rr, nec) == expected


@pytest.mark.parametrize("k", range(1, 4))
def test_impose_necessary_matches_legacy(k):
    regulators = tuple(f"R{i}" for i in range(k))
    n = 2**k
    for bits in itertools.product("01", repeat=n):
        rr = "".join(bits)
        for nec in regulators:
            assert impose_necessary(regulators, rr, nec) == legacy_impose_necessary(
                regulators, rr, nec
            )


def test_group_rr_hardcoded_and_roundtrip_shape():
    regulators = ("A", "B", "C")
    rr = "10101010"
    group_rr, group_regulators = rr2group_rr(regulators, rr, [["B", "C"]])
    assert group_regulators == (("B", "C"), "A")
    assert group_rr == "1100"
    assert group_rr2rr(regulators, group_rr, group_regulators) == "10001000"

    legacy_g, legacy_regs = legacy_rr2group_rr(regulators, rr, [["B", "C"]])
    assert (group_rr, group_regulators) == (legacy_g, legacy_regs)
    assert group_rr2rr(regulators, group_rr, group_regulators) == legacy_group_rr2rr(
        regulators, group_rr, group_regulators
    )


@pytest.mark.parametrize("k", [2, 3])
def test_group_rr_matches_legacy_exhaustive(k):
    regulators = tuple(chr(ord("A") + i) for i in range(k))
    groups = [[regulators[0], regulators[1]]] if k >= 2 else []
    n = 2**k
    samples = (
        itertools.product("01", repeat=n)
        if k == 2
        else ("".join(random.Random(i).choice("01") for _ in range(n)) for i in range(40))
    )
    for bits in samples:
        rr = "".join(bits) if isinstance(bits, tuple) else bits
        g1, r1 = rr2group_rr(regulators, rr, groups)
        g2, r2 = legacy_rr2group_rr(regulators, rr, groups)
        assert (g1, r1) == (g2, r2)
        assert group_rr2rr(regulators, g1, r1) == legacy_group_rr2rr(regulators, g1, r1)


def test_add_regulator_bias0_deterministic():
    class Seq:
        def __init__(self, vals):
            self.vals = list(vals)
            self.i = 0

        def randint(self, a, b):
            v = self.vals[self.i]
            self.i += 1
            return v

    with patch("boolmore.genetic.operators.mutation.random") as m:
        m.random.return_value = 0.9
        m.randint.side_effect = Seq([0, 1, 0, 1]).randint
        regs, rr, signs = add_regulator(("A", "B"), "1000", "11", "C", "1", bias=0.0)
    assert regs == ("A", "B", "C")
    assert signs == "111"
    assert rr == "01100010"


def test_add_regulator_bias1_deterministic():
    class Seq:
        def __init__(self, vals):
            self.vals = list(vals)
            self.i = 0

        def randint(self, a, b):
            v = self.vals[self.i]
            self.i += 1
            return v

    with patch("boolmore.genetic.operators.mutation.random") as m:
        m.random.return_value = 0.0
        m.randint.side_effect = Seq([0, 0, 0, 0]).randint
        regs, rr, signs = add_regulator(("A", "B"), "1000", "11", "C", "1", bias=1.0)
    assert regs == ("A", "B", "C")
    assert signs == "111"
    assert rr == "11000000"


@pytest.mark.parametrize(
    "target,rr,expected",
    [
        ("B", "1110", "10"),
        ("A", "1110", "10"),
    ],
)
def test_delete_regulator_bias0(target, rr, expected):
    regs, out, signs = delete_regulator(("A", "B"), rr, "11", target, bias=0.0)
    assert out == expected
    assert len(regs) == 1
    assert len(signs) == 1


def test_add_delete_match_legacy_seeded():
    for seed in range(20):
        rng_state = random.getstate()
        random.seed(seed)
        prod = add_regulator(("A", "B"), "1010", "11", "C", "0", bias=0.5)
        random.setstate(rng_state)
        random.seed(seed)
        legacy = legacy_add_regulator(("A", "B"), "1010", "11", "C", "0", bias=0.5)
        assert prod == legacy

        random.seed(seed + 100)
        prod = delete_regulator(("A", "B", "C"), "10101010", "111", "B", bias=0.5)
        random.seed(seed + 100)
        legacy = legacy_delete_regulator(("A", "B", "C"), "10101010", "111", "B", bias=0.5)
        assert prod == legacy
