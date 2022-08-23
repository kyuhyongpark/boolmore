import csv
from mutation import *
from score import *
from constraint import *
import config

def import_exps_new(location):
    """
    Returns exps

    Parameters
    ----------
    location : data location

    Returns
    -------
    experiments : list of tuple
        expset[0] : score (float) representing the score for the experiment set
        expset[1] : exp (dict) representing each perturbation/result pair
            key : fixes (tuple) ((node A, value1), (node B, value2), ...)
            value : result (tuple) (measured_node, values)

    interventions : list of tuples
        each element represents a set of intervention
    """

    file = open(location, "r")
    lines = filter(comment_removal, file)
    data = csv.reader(lines, delimiter='\t')

    experiments = []
    interventions = []
    expset = None
    for row in data:
        # check if it's a new entry
        new_entry = check_score(row)
        if new_entry == True:
            # add the previous entry if it exists
            if expset != None:
                expset.append(exp)
                experiments.append(tuple(expset))
            # start the new entry
            expset = []
            expset.append(float(row[0]))
            exp = {}
            continue

        fixes = []
        row_iter = iter(row)
        for sth in row_iter:
            fix = tuple([sth, next(row_iter)])
            fixes.append(fix)
        # fixes should be sorted so that they do not depend on the order of user input
        fixes = tuple(sorted(fixes, key= lambda x:x[0]))
        result = tuple(next(data))

        exp[fixes] = result

        if fixes not in interventions:
            interventions.append(fixes)

    # add the last entry
    expset.append(exp)
    experiments.append(tuple(expset))

    return experiments, interventions

def comment_removal(line):
    return not line.startswith("#") and not line.isspace()

def check_score(row):
    try:
        float(row[0])
        return True
    except ValueError:
        return False
