import itertools
import pytest
import math
import itertools as it

from boolmore.boolean_functions import get_uni_rr


@pytest.mark.parametrize(
    "rr, max_mode, expected",
    [
        ("0", True, "0"),
        ("1", True, "1"),
        ("0", False, "0"),
        ("1", False, "1"),
        ("1100", True, "1111"),
        ("1100", False, "1000"),
    ],
)
def test_get_uni_rr_base_cases(rr, max_mode, expected):
    """Mask-order base cases for get_uni_rr."""
    assert get_uni_rr(rr, max=max_mode) == expected


@pytest.mark.parametrize("max_mode", [True, False])
@pytest.mark.parametrize("k", range(0, 5))
def test_get_uni_rr_properties(k, max_mode):
    """
    Exhaustively test all Boolean rules with 2^k entries for k <= 4.

    Verify that:
    - the output has the same length,
    - the output is still a binary string,
    - the operation is idempotent.
    """
    n = 2**k

    for bits in itertools.product("01", repeat=n):
        rr = "".join(bits)

        out = get_uni_rr(rr, max=max_mode)

        assert len(out) == n
        assert set(out) <= {"0", "1"}
        assert get_uni_rr(out, max=max_mode) == out


def get_uni_rr_reference1(rr:str, max:bool=True) -> str:
    """
    Mask-order reference: index i is implicant mask i.
    """
    n = len(rr)
    k = int(math.log2(n)) if n > 1 else 0

    if k == 0:
        return rr

    uni_rr = list(rr)
    modified = []

    for i in range(n):
        if uni_rr[i] != '1':
            continue
        bi = format(i, '0' + str(k) + 'b')
        if bi in modified:
            continue

        lst = []
        for num in bi:
            if num == '0':
                lst.append(['0', '1'])
            else:
                lst.append(['1'])
        positions = [''.join(p) for p in it.product(*lst)]
        positions.remove(bi)
        modified.extend(positions)
        for position in positions:
            j = int(position, 2)
            uni_rr[j] = str(int(max))

    return ''.join(uni_rr)


def get_uni_rr_reference2(rr: str, max: bool = True) -> str:
    """
    Mask-order bitwise reference: index == mask.
    """
    n = len(rr)

    if n <= 1:
        return rr

    uni_rr = list(rr)
    modified = set()
    fill_value = '1' if max else '0'

    for i in range(n):
        if uni_rr[i] != '1':
            continue
        if i in modified:
            continue

        for position in range(i + 1, n):
            if (position & i) == i:
                modified.add(position)
                uni_rr[position] = fill_value

    return ''.join(uni_rr)


@pytest.mark.parametrize("max_mode", [True, False])
@pytest.mark.parametrize("k", range(0, 5))  # exhaustive up to 2^4 = 16 length strings
def test_get_uni_rr_matches_reference(k, max_mode):
    """
    Exhaustively compare optimized vs reference implementation.

    This guarantees refactoring does not change behavior.
    """
    n = 2**k

    for bits in itertools.product("01", repeat=n):
        rr = "".join(bits)

        expected1 = get_uni_rr_reference1(rr, max=max_mode)
        expected2 = get_uni_rr_reference2(rr, max=max_mode)

        assert expected1 == expected2

        actual = get_uni_rr(rr, max=max_mode)

        assert actual == expected1
