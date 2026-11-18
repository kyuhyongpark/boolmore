import boolmore.boolean_functions as bf


def try_add_regulator(
    regulators: tuple[str, ...],
    rr: str,
    reg: str,
    max_rr: str | None = None,
    max_irr: str | None = None,
    necessary: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, str | None, str | None, str]:
    """
    Heuristic attempt to make ``reg`` appear by flipping one related bit.

    Always tries maximal (forward) space first: flip the **rightmost**
    (largest-index) mask that includes ``reg`` (and all ``necessary``
    regulators when given) and is currently ``'0'``. If no such mask
    exists, fall back to max-irr with only the ``reg`` bit required,
    again flipping the rightmost zero.
    This does **not** guarantee that ``reg`` appears in the minimal form;
    callers must re-check with ``check_regulate``.

    Because ``max_rr`` / ``max_irr`` are upward-closed, a zero at the
    exact required mask is necessary and sufficient for any candidate
    to exist; search then walks supersets largest-first.

    ``max_rr`` / ``max_irr`` are optional caches for the repair base. When
    provided they are reused; newly computed forms are returned so the
    caller can keep caching across retries on the same base.

    Parameters
    ----------
    regulators - the regulating nodes           :length k tuple[str]
    rr         - rule in mask order             :length 2^k binary str
                 Used only when ``max_rr`` is not supplied.
    reg        - regulator to try to introduce  :str
    max_rr     - optional cached maximal form of the repair base
    max_irr    - optional cached max-inverted form of the repair base
    necessary  - regulators that must appear in forward flipped masks

    Returns
    -------
    repaired_rr - possibly modified rule in mask order  :str
                  Unchanged ``rr`` when neither space has a candidate bit.
    max_rr      - maximal form used/computed for this base (cacheable)
    max_irr     - max-inverted form if computed, else the input cache
    path        - ``"forward"``, ``"irr"``, or ``"noop"``
    """
    assert reg in regulators, "The node should be one of the regulators"

    k = len(regulators)
    n = regulators.index(reg)
    bit = 1 << (k - n - 1)

    need_mask = 0
    if necessary:
        for nec in necessary:
            assert nec in regulators, "Necessary node should be one of the regulators"
            need_mask |= 1 << (k - regulators.index(nec) - 1)

    def _rightmost_zero(base: str, require: int) -> int | None:
        # Upward-closed: if require itself is 1, every superset is 1.
        if base[require] == "1":
            return None
        # Walk supersets of require from largest to smallest.
        free = ((1 << k) - 1) ^ require
        s = free
        while True:
            idx = s | require
            if base[idx] == "0":
                return idx
            if s == 0:
                return None
            s = (s - 1) & free

    def _flip(base: str, idx: int) -> str:
        return base[:idx] + "1" + base[idx + 1 :]

    # Reuse caller caches when present; build only as needed.
    def _max() -> str:
        nonlocal max_rr
        if max_rr is None:
            max_rr = bf.get_uni_rr(rr, max=True)
        return max_rr

    def _irr() -> str:
        nonlocal max_irr
        if max_irr is None:
            max_irr = bf.get_max_irr(rr, max_rr=_max())
        return max_irr

    def _try(base: str, require: int, inverted: bool) -> str | None:
        idx = _rightmost_zero(base, require)
        if idx is None:
            return None
        flipped = _flip(base, idx)
        if inverted:
            return bf.get_max_irr(flipped)
        return flipped

    # Always forward first; irr only if forward has no target.
    forward_require = bit | need_mask
    repaired = _try(_max(), forward_require, inverted=False)
    if repaired is not None:
        return repaired, max_rr, max_irr, "forward"

    repaired = _try(_irr(), bit, inverted=True)
    if repaired is not None:
        return repaired, max_rr, max_irr, "irr"

    return rr, max_rr, max_irr, "noop"


def impose_necessary(regulators:tuple[str, ...], rr:str, nec:str) -> str:
    """
    Impose the necessary condition for the rule.

    ``rr`` uses mask order (``index == mask``); see ``boolmore.boolean_functions``.

    Parameters
    ----------
    regulators - the regulating nodes           :length k tuple[str]
    rr         - rule in mask order             :length 2^k binary str
    nec        - the necessary node             :str

    Returns
    -------
    necessary_rr - modified so that nec is necessary    :str
        

    """
    assert nec in regulators, "The node should be one of the regulators"

    # k = number of regulators
    k = len(regulators)
    # n = nth regulator
    n = regulators.index(nec)
    bit = 1 << (k - n - 1)

    bi = list(rr)
    for i in range(2**k):
        # Zero masks where regulator n's bit is unset.
        if not (i & bit):
            bi[i] = '0'

    return ''.join(bi)


def group_bi2bi(group_bi:str, group_regulators:tuple[str, tuple[str, ...]]) -> str:
    """
    Returns the binary position of the old representation
    when given a binary position of the group representation
    and the group regulators.

    The returned ``bi`` is a mask string interpreted directly as an ``rr``
    index (``int(bi, 2)``) under the public mask-order convention.

    Parameters
    ----------
    group_bi         - binary position of the rule representation with  :length m binary str
                       grouped regulators
                       e.g. 101 with regulators (A, B, (C,D)) means the
                            position in the rr that represents A&(C&D)
    group_regulators - e.g. (A, (B,C), (B,D))                           :length m tuple[str, tuple[str]]

    Returns
    -------
    bi - binary position of the rule representation with the original regulators    :length k binary str
         e.g. 101 with grouped regulators (A, B, (C,D))
              becomes 1011 with regulators (A, B, C, D)
         This transition is not onto, which enforces the group constraint.
         However, this transition is also not one-to-one which does cause bias.

    """
    num = iter(group_bi)
    bi_dict = {}
    for reg in group_regulators:
        if type(reg) == tuple:
            value = next(num)
            for node in reg:
                if node not in bi_dict.keys():
                    bi_dict[node] = value
                elif bi_dict[node] == '0':
                    bi_dict[node] = value
                else:
                    pass
        else:
            bi_dict[reg] = next(num)
    
    # sort by node names, which restores the order of nodes in the original regulators
    bi_dict = dict(sorted(bi_dict.items()))

    bi = ''.join(bi_dict.values())

    return bi


def rr2group_rr(regulators:tuple[str, ...], rr:str, groups:list[list[str]]) -> tuple[str, tuple[str, tuple[str, ...]]]:
    """
    Convert the representation of a rule into a representation in which
    certain nodes are grouped together.

    Input and output ``rr`` strings use mask order (``index == mask``).

    Parameters
    ----------
    regulators - the regulating nodes           :length k tuple[str]
    rr         - rule in mask order             :length 2^k binary str
    groups     - groups of nodes                :list[list[str]]
                 [[node A, node B], [node A, node C], ...]
    
    Returns
    -------
    group_rr         - grouped rule in mask order                   :length 2^m binary str
    group_regulators - the regulating nodes where some are grouped  :length m tuple[str, tuple[str]]
                       (node A, node B, (node C, node D), ...)

    """

    for group in groups:
        for node in group:
            assert node in regulators, f"{node} is not one of the regulators {regulators}"

    group_regulators = []
    individual_regulators = set(regulators)
    for i, group in enumerate(groups):
        group_regulators.append(tuple(group)) # type: ignore
        individual_regulators -= set(group)
    group_regulators.extend((sorted(list(individual_regulators))))
    group_regulators = tuple(group_regulators)

    m = len(group_regulators)
    group_rr = ['0']*(2**m)

    for i in range(2**m):
        group_bi = format(i, '0' + str(m) + 'b')
        bi = group_bi2bi(group_bi, group_regulators) # type: ignore
        if rr[int(bi, 2)] == '1':
            group_rr[i] = '1'

    return ''.join(group_rr), group_regulators # type: ignore


def group_rr2rr(regulators:tuple[str, ...], group_rr:str, group_regulators:tuple[str, tuple[str, ...]]) -> str:
    """
    Convert the representation of a rule with grouped regulators
    into a representation with original regulators.

    Input and output ``rr`` strings use mask order (``index == mask``).

    Parameters
    ----------
    regulators       - the regulating nodes                         :length k tuple[str]
    group_rr         - grouped rule in mask order                   :length 2^m binary str
    group_regulators - the regulating nodes where some are grouped  :length m tuple[str, tuple[str]]
                       (node A, node B, (node C, node D), ...)

    Returns
    -------
    rr - rule in mask order     :length 2^k binary str

    """

    m = len(group_regulators)
    k = len(regulators)
    rr = ['0']*(2**k)

    for i in range(2**m):
        group_bi = format(i, '0' + str(m) + 'b')
        bi = group_bi2bi(group_bi, group_regulators)
        if group_rr[i] == '1':
            rr[int(bi, 2)] = '1'

    return ''.join(rr)
