def generate_source_masks(source_order, sources_partial):
    """
    Generate all masks consistent with a partial assignment of sources.

    Each source corresponds to one bit in the output mask. The order of bits
    follows `source_order`, with the first source mapped to the leftmost bit.

    Parameters
    ----------
    source_order : sequence
        Ordered list of source names.
    sources_partial : dict
        Mapping from source name to a fixed value (0 or 1). Sources not
        present in this dictionary are treated as free variables.

    Yields
    ------
    int
        Integer masks satisfying the partial assignment. Masks are generated
        in binary-counting order over the free variables, from left to right.
    """
    n = len(source_order)

    free_pos = []
    base_bits = []

    # Build the fixed portion of the mask and record the positions of
    # unconstrained (free) sources.
    for i, s in enumerate(source_order):
        if s in sources_partial:
            base_bits.append(str(sources_partial[s]))
        else:
            free_pos.append(i)
            base_bits.append('0')

    base = int("".join(base_bits), 2)


    # Enumerate all assignments to the free sources.
    m = len(free_pos)
    for x in range(2**m):
        mask = base

        # Interpret x as a bit string aligned with free_pos:
        # the leftmost bit of x corresponds to free_pos[0],
        # the next bit to free_pos[1], and so on.
        for j in range(m):
            # working from the leftmost bit of x.
            if x & (1 << m-1-j):
                # add into the leftmost bit of the free position.
                mask |= (1 << (n-1-free_pos[j]))

        yield mask

def mask_to_sources(mask, source_order):
    """
    Convert an integer bitmask into a full assignment dictionary.

    Each source in `source_order` corresponds to one bit:
    - source_order[0] → leftmost (highest) bit
    - source_order[-1] → rightmost (lowest) bit

    Parameters
    ----------
    mask : int
        Integer bitmask.
    source_order : list[str]
        Ordered list of sources.

    Returns
    -------
    dict
        Mapping {source: 0 or 1} for every source (no partials).
    """
    n = len(source_order)

    result = {}
    for i, s in enumerate(source_order):
        bit_pos = n - 1 - i
        result[s] = (mask >> bit_pos) & 1

    return result
