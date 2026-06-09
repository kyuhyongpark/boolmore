import csv
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

from boolmore.core.experiment import Experiment

Assignment = tuple[tuple[str, int], ...]
ExpType = tuple[int, float, Assignment, str, str]
Signature = tuple[Assignment, Assignment, Assignment]  # sources, perturbation, phenotype


def comment_removal(line:str) -> bool:
    return not line.startswith("#") and not line.isspace()

def import_exps(location:str) -> tuple[list[ExpType], list[Assignment]]:
    """
    Reads a tsv file and returns experiments and interventions.
    
    Note:
    All source nodes that have a default source value specified in `default_sources`
    should be specified for every experiment.

    By default, `default_sources` has all source nodes set to 0,
    and hence all source nodes should be specified.
    If `default_sources` is given manually,
    source nodes that are not in the `default_sources` can be omitted in the experiements.
    
    The tsv file should have 6 columns
    ID    - e.g. 1
    SCORE - e.g. 1.0
    SOURCE - e.g. A=1
    PERT - e.g. B KO, C KO, D CA
    NODE  - the observed node
    VALUE - one of OFF, OFF/Some, Some, Some/ON, ON

    Parameters
    ----------
    location - data location    :str

    Returns
    -------
    experiments - list of exp                           :list[ExpType]
        exp     - info of a single experiment           :ExpType = tuple[int, float, FixesType, str, str]
            exp[0] - id of the experiment               :int
            exp[1] - max_score for the experiment       :float
            exp[2] - fixes                              :FixesType = tuple[tuple[str, int]]
                     ((node A, value1), (node B, value2), ...)
            exp[3] - observed_node                      :str
            exp[4] - outcome_value                      :str
                     one of OFF, OFF/Some, Some, Some/ON, ON

    interventions - summarized list of fixes for convenience    :list[FixesType]
        fixes     - ((node A, value1), (node B, value2), ...)   :FixesType = tuple[tuple[str, int]]

    """
    ID, SCORE, SOURCE, PERT, NODE, VALUE = 0, 1, 2, 3, 4, 5
    
    file = open(location, "r")
    lines = filter(comment_removal, file)
    data = csv.reader(lines, delimiter="\t")

    # skip the first row
    next(data)

    experiments = []
    interventions = []
    for row in data:
        exp = [int(row[ID]), float(row[SCORE])]

        fixes_list = []
        if row[SOURCE] != "":
            # add source node values to the fixes
            source_str = row[SOURCE].split(",")
            for sth in source_str:
                node, value = sth.strip().split("=")
                fix = tuple([node, int(value)])
                fixes_list.append(fix)
        
        if row[PERT] != "":
            # add other perturbations to the fixes
            pert_str = row[PERT].split(",")
            for sth in pert_str:
                node, value_str = sth.strip().split(" ")
                if value_str == "KO":
                    value = 0
                elif value_str == "CA":
                    value = 1
                else:
                    raise Exception("Perturbation should be KO or CA")
                fix = tuple([node, int(value)])
                fixes_list.append(fix)
        # fixes should be sorted so that they do not depend on the order of user input
        fixes = tuple(sorted(fixes_list, key= lambda x:x[0]))
        exp.append(fixes)

        exp.append(row[NODE])
        exp.append(row[VALUE])

        if fixes not in interventions:
            interventions.append(fixes)
        else:
            for experiment in experiments:
                if fixes in experiment:
                    assert exp[3] != experiment[3], f"{experiment[0]} and {exp[0]} are duplicates" 

        # add the entry
        experiments.append(tuple(exp))

    return experiments, interventions


# -------------------------
# Error container
# -------------------------

@dataclass
class ParseError:
    line: int
    id: Optional[str]
    message: str
    row: dict


class CSVParseException(Exception):
    def __init__(self, errors: list[ParseError]):
        self.errors = errors
        super().__init__(self._format())

    def _format(self) -> str:
        lines = ["Multiple errors occurred while parsing CSV:\n"]
        for e in self.errors:
            lines.append(
                f"- line {e.line}, id={e.id}: {e.message}\n"
                f"  row={e.row}\n"
            )
        return "\n".join(lines)


# -------------------------
# Pre-filter
# -------------------------

def valid_lines(file):
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        yield line


# -------------------------
# Parsers (now return errors instead of raising)
# -------------------------

def parse_id(x: str, line_num: int, errors: list[ParseError], row: dict) -> Optional[int]:
    if x is None or x.strip() == "":
        errors.append(ParseError(line_num, None, "Missing id", row))
        return None

    try:
        return int(x)
    except ValueError:
        errors.append(ParseError(line_num, x, f"Invalid id '{x}'", row))
        return None


def parse_bool(x: str, line_num: int, errors: list[ParseError], id: Optional[str], row: dict) -> Optional[bool]:
    if x is None:
        errors.append(ParseError(line_num, id, "Missing expected_exists", row))
        return None

    x = x.strip().lower()

    if x == "true":
        return True
    if x == "false":
        return False

    errors.append(ParseError(line_num, id, f"Invalid expected_exists '{x}'", row))
    return None


def parse_float(x: str, field: str, line_num: int, errors: list[ParseError], id: Optional[str], row: dict) -> Optional[float]:
    if x is None or x.strip() == "":
        errors.append(ParseError(line_num, id, f"Missing field '{field}'", row))
        return None

    try:
        return float(x)
    except ValueError:
        errors.append(ParseError(line_num, id, f"Invalid float '{x}' in '{field}'", row))
        return None


def parse_assignment_block(block: str, field: str, line_num: int, errors: list[ParseError], id: Optional[str], row: dict) -> Assignment:
    if not block or block.strip() == "":
        return tuple()

    items: list[tuple[str, int]] = []

    for part in block.split(";"):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            errors.append(ParseError(line_num, id, f"Invalid assignment '{part}' (missing '=')", row))
            continue

        node, value = part.split("=", 1)
        node = node.strip()
        value = value.strip()

        if not node:
            errors.append(ParseError(line_num, id, f"Empty node in '{field}'", row))
            continue

        if value not in {"0", "1"}:
            errors.append(ParseError(line_num, id, f"Invalid value '{value}' for node '{node}'", row))
            continue

        items.append((node, int(value)))

    return tuple(sorted(items, key=lambda x: x[0]))


# -------------------------
# Main loader
# -------------------------

def import_phenotypes(location: str) -> list[Experiment]:
    """
    Import phenotype-based experiments from a csv file.

    Each row in the file corresponds to a single Experiment. All structured
    fields are encoded as strings and must follow a strict assignment format.

    The function parses each row and converts it into an Experiment object.

    ------------------------------------------------------------------------
    REQUIRED COLUMN HEADERS
    ------------------------------------------------------------------------

    id : str or int
        Unique identifier of the experiment.

    sources : str
        Assignment of source node states.
        Format:
            "node1=value1; node2=value2;..."
        Example:
            "A=1; B=0; C=1"

    perturbation : str
        Assignment of perturbation conditions.
        Same format as sources

    phenotype : str
        Observed phenotype assignment.
        Same format as sources

    expected_exists : int or bool
        Whether the phenotype is expected to exist under the given conditions.
        Allowed values:
            true / false

    weight : float
        Importance weight of this experiment in scoring/benchmarking.
        Must be a valid floating point number (e.g. 1.0, 0.5, 2.3).

    ------------------------------------------------------------------------
    CELL ENCODING RULES
    ------------------------------------------------------------------------

    - Node names are strings without commas or parentheses.
    - Whitespace is ignored
    - # comments are ignored
    - Order of nodes does NOT matter; internally they are normalized.

    ------------------------------------------------------------------------
    RETURNS
    ------------------------------------------------------------------------

    list[Experiment]
        Parsed experiments as immutable Experiment dataclass instances.

    ------------------------------------------------------------------------
    ERRORS
    ------------------------------------------------------------------------

    Raises:
        ValueError:
            - Missing required columns
            - Malformed assignment strings
            - invalid values in fields
            - Duplicate experiment ids
            - Duplicate signatures (sources, perturbation, phenotype)
    """
    experiments: list[Experiment] = []
    errors: list[ParseError] = []

    signatures: dict[Signature, list[int]] = defaultdict(list)
    ids_seen: dict[int, int] = defaultdict(int)

    with open(location, newline="") as file:
        reader = csv.DictReader(valid_lines(file))

        for line_num, row in enumerate(reader, start=2):

            exp_id = parse_id(row.get("id"), line_num, errors, row)

            sources = parse_assignment_block(
                row.get("sources", ""), "sources", line_num, errors, exp_id, row
            )

            perturbation = parse_assignment_block(
                row.get("perturbation", ""), "perturbation", line_num, errors, exp_id, row
            )

            phenotype = parse_assignment_block(
                row.get("phenotype", ""), "phenotype", line_num, errors, exp_id, row
            )

            weight = parse_float(
                row.get("weight"), "weight", line_num, errors, exp_id, row
            )

            expected_exists = parse_bool(
                row.get("expected_exists"), line_num, errors, exp_id, row
            )

            # Only construct Experiment if core fields are valid
            if None not in (exp_id, weight, expected_exists):
                exp = Experiment(
                    id=exp_id,
                    weight=weight,
                    sources=sources,
                    perturbation=perturbation,
                    phenotype=phenotype,
                    expected_exists=expected_exists,
                )

                experiments.append(exp)

                # -------------------------
                # Track duplicates (id)
                # -------------------------
                ids_seen[exp_id] += 1

                # -------------------------
                # Track structural signature
                # -------------------------
                sig = (sources, perturbation, phenotype)
                signatures[sig].append(exp_id)

    # -------------------------
    # Post-pass validation
    # -------------------------

    # 1. Duplicate IDs
    for exp_id, count in ids_seen.items():
        if count > 1:
            errors.append(
                ParseError(
                    line=0,
                    id=str(exp_id),
                    message=f"Duplicate id {exp_id} appears {count} times",
                    row={}
                )
            )

    # 2. Duplicate structural signatures
    for sig, ids in signatures.items():
        if len(ids) > 1:
            errors.append(
                ParseError(
                    line=0,
                    id=",".join(map(str, ids)),
                    message="Duplicate experiment signature (sources, perturbation, phenotype)",
                    row={
                        "sources": sig[0],
                        "perturbation": sig[1],
                        "phenotype": sig[2],
                    }
                )
            )

    # -------------------------
    # Final check
    # -------------------------
    if errors:
        raise CSVParseException(errors)

    return experiments

def check_phenotypes(primes: dict, experiments: list[Experiment]) -> None:
    """
    Verify that every node appearing in the experiments exists in `primes`.

    Checks the nodes appearing in
    - sources
    - perturbation
    - phenotype

    Collects all errors before raising a CSVParseException.
    """
    errors: list[ParseError] = []

    valid_nodes = set(primes.keys())

    for exp in experiments:
        for field_name, assignment in (
            ("sources", exp.sources),
            ("perturbation", exp.perturbation),
            ("phenotype", exp.phenotype),
        ):
            for node, _ in assignment:
                if node not in valid_nodes:
                    errors.append(
                        ParseError(
                            line=0,
                            id=str(exp.id),
                            message=f"Unknown node '{node}' in {field_name}",
                            row={
                                "sources": exp.sources,
                                "perturbation": exp.perturbation,
                                "phenotype": exp.phenotype,
                            },
                        )
                    )

    if errors:
        raise CSVParseException(errors)