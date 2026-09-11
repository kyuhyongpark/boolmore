import itertools
import pytest
import math
import itertools as it

from boolmore.core.conversions import get_uni_rr


@pytest.mark.parametrize(
    "rr, max_mode, expected",
    [
        ("0", True, "0"),
        ("1", True, "1"),
        ("0", False, "0"),
        ("1", False, "1"),
        ("0011", True, "1111"),
        ("0011", False, "0001"),
    ],
)
def test_get_uni_rr_base_cases(rr, max_mode, expected):
    """Rules of length 1 should be returned unchanged."""
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
    Returns the unique binary representation (max or min) of a rule
    when given any binary representation of a rule.

    Parameters
    ----------
    rr  - representation of the rule            :length 2^k binary str
    max - if True, get maximal representation   :bool
          if False, get minimal representation

    Returns
    -------
    uni_rr - unique representation of the rules                 :length 2^k binary str
             if maximal, it is equivalent to the truth table
             if minimal, it is in a Blake canonical form

    """
    # k = number of regulators
    n = len(rr)
    k = int(math.log2(n))

    # already minimal if the node is a fixed node
    if k == 0:
        return rr

    uni_rr = list(rr)

    modified = []

    # iterate through the string in reverse
    for i in range(n):
        # get the regulator values if there is '1'
        if uni_rr[-i-1] == '1':
            # bi represents the regulator values in binary string
            bi = format(i, '0' + str(k) + 'b')
        else:
            continue
        # find all the positions that need to be changed to 0
        if bi in modified:
            continue

        lst = []
        for num in bi:
            if num == '0':
                lst.append(['0', '1'])
            else:
                lst.append(['1'])
        positions = [''.join(p) for p in it.product(*lst)]
        # do not change the original position
        positions.remove(bi)
        modified.extend(positions)
        # change to '0' if min or '1' if max in the gained positions
        for position in positions:
            j = n - int(position,2) - 1
            uni_rr[j] = str(int(max))

    uni_rr = ''.join(uni_rr)

    return uni_rr

def get_uni_rr_reference2(rr: str, max: bool = True) -> str:
    """
    Returns the unique canonical representation (maximal truth table or 
    minimal Blake canonical form) of a Boolean rule.
    
    This function processes a binary string of length 2^k (where k is the 
    number of regulators) by identifying and filling out sub-cubes of a 
    hypercube using fast bitwise integer logic.

    Parameters
    ----------
    rr  - representation of the rule            : length 2^k binary str
    max - if True, get maximal representation   : bool
          if False, get minimal representation

    Returns
    -------
    uni_rr - unique representation of the rule  : length 2^k binary str
    """
    n = len(rr)
    
    # Base case: if the string length is 1 or less (k = 0 regulators),
    # the rule represents a fixed node configuration and cannot be minimized further.
    if n <= 1:
        return rr

    # Convert the string to a list of characters to allow fast, 
    # in-place mutations at specific index positions.
    uni_rr = list(rr)
    
    # Store already visited sub-cube positions in a set to ensure 
    # instant O(1) lookup speeds and prevent redundant evaluations.
    modified = set()
    fill_value = '1' if max else '0'

    # Iterate through all binary rule configurations in reverse order (from n-1 down to 0).
    # Processing in reverse ensures that higher-order implicants are evaluated first.
    for i in range(n):
        # Calculate the actual array index corresponding to the reversed loop variable.
        rev_idx = n - i - 1
        
        # We only evaluate active rule states ('1'). Inactive states ('0') 
        # do not trigger prime implicant/sub-cube expansions.
        if uni_rr[rev_idx] != '1':
            continue
            
        # Skip this position if it has already been covered and filled 
        # by a previously processed sub-cube.
        if i in modified:
            continue

        # Scan the entire state space to locate sub-cube coordinates.
        # Instead of slow string generation, we use low-level bitwise operations.
        for position in range(n):
            if position == i:
                continue
                
            # BITWISE LOGIC: (position & i) == i
            # Checks if 'position' contains a 1 at every single binary slot where 'i' has a 1.
            # If True, 'position' is mathematically verified to be a sub-cube coordinate
            # covered by the root implicant 'i'.
            if (position & i) == i:
                modified.add(position)
                
                # Maps the integer position back to its correct index in the output array.
                target_idx = n - position - 1
                uni_rr[target_idx] = fill_value

    # Recombine the character array into the final canonical binary string representation.
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