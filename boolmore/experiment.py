import csv

FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]

def comment_removal(line:str) -> bool:
    return not line.startswith("#") and not line.isspace()

def import_exps(location:str) -> tuple[list[ExpType], list[FixesType], float]:
    """
    Reads a tsv file and returns experiments and interventions.

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

    max_score     - possible maximum score              :float

    """
    ID, SCORE, SOURCE, PERT, NODE, VALUE = 0, 1, 2, 3, 4, 5
    
    file = open(location, "r")
    lines = filter(comment_removal, file)
    data = csv.reader(lines, delimiter="\t")

    # skip the first row
    next(data)

    experiments = []
    interventions = []
    max_score = 0.0
    for row in data:
        exp = [int(row[ID]), float(row[SCORE])]
        max_score += float(row[SCORE])

        if row[SOURCE] == "" and row[PERT] == "":
            fixes = tuple()
            exp.append(fixes)
            continue

        fixes = []
        if row[SOURCE] != "":
            # add source node values to the fixes
            source_str = row[SOURCE].split(",")
            for sth in source_str:
                node, value = sth.strip().split("=")
                fix = tuple([node, int(value)])
                fixes.append(fix)
        
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
                fixes.append(fix)
        # fixes should be sorted so that they do not depend on the order of user input
        fixes = tuple(sorted(fixes, key= lambda x:x[0]))
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

    return experiments, interventions, max_score