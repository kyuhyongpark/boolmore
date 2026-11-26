"""Import PyBoolNet-style primes from ``.bnet`` via python-sat Primer-B."""

from __future__ import annotations

import re
from pathlib import Path

from boolmore.types import PrimeType

__all__ = [
    "bnet_text_to_primes",
    "bnet_file_to_primes",
    "canon_primes",
]

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CONSTANT_OFF: PrimeType = [[{}], []]
_CONSTANT_ON: PrimeType = [[], [{}]]


def canon_primes(primes: dict[str, PrimeType]) -> dict[str, PrimeType]:
    """Sort cubes within each polarity for order-independent comparison."""
    out: dict[str, PrimeType] = {}
    for node, (off, on) in primes.items():
        out[node] = [
            sorted(off, key=lambda d: tuple(sorted(d.items()))),
            sorted(on, key=lambda d: tuple(sorted(d.items()))),
        ]
    return out


def _sort_cubes(cubes: list[dict[str, int]]) -> list[dict[str, int]]:
    return sorted(cubes, key=lambda d: tuple(sorted(d.items())))


def _parse_bnet_rules(text: str) -> dict[str, str]:
    """Parse BoolNet text into ``{node: expression}`` (no auto-sources)."""
    rules: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Header may use spaces or tabs: "targets, factors" / "targets,\tfactors".
        lower = "".join(line.lower().split())
        if lower == "targets,factors" or lower.startswith("targets,factors"):
            continue
        if "," not in line:
            raise ValueError(f"invalid bnet line (no comma): {raw!r}")
        name, expr = line.split(",", 1)
        name = name.strip()
        expr = expr.strip()
        if not name:
            raise ValueError(f"invalid bnet line (empty target): {raw!r}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid node name: {name!r}")
        rules[name] = expr
    return rules


def _collect_idents(expr: str) -> set[str]:
    idents: set[str] = set()
    for m in _IDENT_RE.finditer(expr):
        tok = m.group(0)
        if tok.lower() in ("true", "false"):
            continue
        idents.add(tok)
    return idents


def _expr_to_pysat_formula(expr: str):
    """Parse ``!`` / ``&`` / ``|`` expression into a pysat Formula."""
    from pysat.formula import Atom, And, Or, Neg

    text = expr.strip()
    if not text:
        raise ValueError("empty expression")

    n = len(text)
    i = 0

    def _skip_ws() -> None:
        nonlocal i
        while i < n and text[i].isspace():
            i += 1

    def _parse_or():
        nonlocal i
        terms = [_parse_and()]
        while True:
            _skip_ws()
            if i < n and text[i] == "|":
                i += 1
                terms.append(_parse_and())
            else:
                break
        return terms[0] if len(terms) == 1 else Or(*terms)

    def _parse_and():
        nonlocal i
        terms = [_parse_unary()]
        while True:
            _skip_ws()
            if i < n and text[i] == "&":
                i += 1
                terms.append(_parse_unary())
            else:
                break
        return terms[0] if len(terms) == 1 else And(*terms)

    def _parse_unary():
        nonlocal i
        _skip_ws()
        if i < n and text[i] in ("!", "~"):
            i += 1
            return Neg(_parse_unary())
        return _parse_atom()

    def _parse_atom():
        nonlocal i
        _skip_ws()
        if i >= n:
            raise ValueError(f"unexpected end of expression near {text!r}")
        if text[i] == "(":
            i += 1
            inner = _parse_or()
            _skip_ws()
            if i >= n or text[i] != ")":
                raise ValueError(f"missing ')' in {text!r}")
            i += 1
            return inner
        if text[i] in ("0", "1") and (
            i + 1 >= n or not (text[i + 1].isalnum() or text[i + 1] == "_")
        ):
            bit = text[i]
            i += 1
            return Atom(True) if bit == "1" else Atom(False)
        m = _IDENT_RE.match(text, i)
        if not m:
            raise ValueError(f"unexpected token at {text[i:]!r} in {text!r}")
        name = m.group(0)
        i = m.end()
        low = name.lower()
        if low == "true":
            return Atom(True)
        if low == "false":
            return Atom(False)
        return Atom(name)

    formula = _parse_or()
    _skip_ws()
    if i != n:
        raise ValueError(f"trailing junk {text[i:]!r} in {text!r}")
    return formula


def _constant_primes(expr: str) -> PrimeType | None:
    raw = expr.strip()
    if raw in ("0", "False", "false"):
        return [list(_CONSTANT_OFF[0]), list(_CONSTANT_OFF[1])]
    if raw in ("1", "True", "true"):
        return [list(_CONSTANT_ON[0]), list(_CONSTANT_ON[1])]
    return None


def _signed_atom(form) -> tuple[str, int] | None:
    """Reduce nested ``Neg`` wrappers to ``(name, 0|1)``; skip bool constants."""
    from pysat.formula import Atom, Neg

    polarity = 1
    cur = form
    while isinstance(cur, Neg):
        polarity ^= 1
        cur = cur.subformula
    if not isinstance(cur, Atom):
        raise ValueError(f"unexpected formula in prime: {form!r}")
    obj = cur.object
    if isinstance(obj, bool):
        return None
    return str(obj), polarity


def _forms_to_cube(forms) -> dict[str, int]:
    cube: dict[str, int] = {}
    for form in forms:
        signed = _signed_atom(form)
        if signed is None:
            continue
        name, bit = signed
        cube[name] = bit
    return cube


def _primer_polarity_primes(formula) -> list[dict[str, int]]:
    from pysat.examples.primer import Primer
    from pysat.formula import Formula

    out: list[dict[str, int]] = []
    with Primer(formula, implicates=False) as primer:
        for lits in primer.enumerate():
            if not lits:
                out.append({})
                continue
            # atoms_only=True drops Neg(Neg(Atom)) (no leaf Atom); keep wrappers.
            forms = Formula.formulas(lits, atoms_only=False)
            out.append(_forms_to_cube(forms))
    return out


def _expr_to_primes(expr: str) -> PrimeType:
    const = _constant_primes(expr)
    if const is not None:
        return const

    from pysat.formula import Formula, Neg

    try:
        f = _expr_to_pysat_formula(expr)
        on_p = _primer_polarity_primes(f)
        off_p = _primer_polarity_primes(Neg(f))
        return [_sort_cubes(off_p), _sort_cubes(on_p)]
    finally:
        Formula.cleanup()


def _source_primes(name: str) -> PrimeType:
    return [[{name: 0}], [{name: 1}]]


def bnet_text_to_primes(text: str) -> dict[str, PrimeType]:
    """Convert BoolNet text to PyBoolNet primes via Primer-B.

    Regulators that appear in expressions but have no update rule are added
    as sources (self-loops), matching BNetToPrime.
    """
    rules = _parse_bnet_rules(text)
    referenced: set[str] = set()
    for expr in rules.values():
        referenced |= _collect_idents(expr)

    primes: dict[str, PrimeType] = {}
    for name, expr in rules.items():
        primes[name] = _expr_to_primes(expr)

    for name in sorted(referenced - set(rules)):
        primes[name] = _source_primes(name)

    return primes


def bnet_file_to_primes(path: str | Path) -> dict[str, PrimeType]:
    """Load a ``.bnet`` file and convert to PyBoolNet primes via Primer-B."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return bnet_text_to_primes(text)
