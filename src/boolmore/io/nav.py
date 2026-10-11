"""Import NAV-style experiments from TSV files."""

import csv

from boolmore.experiment import NAVExperiment


def import_NAV_exps(location: str) -> list[NAVExperiment]:
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
    experiments - list[NAVExperiment]

    """
    ID, SCORE, SOURCE, PERT, NODE, VALUE = 0, 1, 2, 3, 4, 5
    
    file = open(location, "r")
    lines = (line for line in file if not line.startswith("#") and not line.isspace())
    data = csv.reader(lines, delimiter="\t")

    # skip the first row
    next(data)

    exps = []
    interventions = []
    for row in data:

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

        exp = NAVExperiment(id =int(row[ID]),
                            fixes = fixes,
                            observed = row[NODE],
                            outcome = row[VALUE],
                            weight = float(row[SCORE]))

        if fixes not in interventions:
            interventions.append(fixes)
        else:
            for other_exp in exps:
                if fixes == other_exp.fixes:
                    assert exp.observed != other_exp.observed, f"{other_exp.id} and {exp.id} are duplicates" 

        # add the entry
        exps.append(exp)

    return exps
