import csv

def comment_removal(line):
    return not line.startswith("#") and not line.isspace()

# 20230425 import from tsv files
def import_exps(location):
    """
    Reads a tsv file and returns experiments and interventions

    The tsv file should have 5 columns
    ID - e.g. 1
    SCORE - e.g. 1.0
    INTERVENTION - e.g. A=1,B=0,C=0
    NODE - the observed node
    VALUE - one of OFF, OFF/Some, Some, Some/ON, ON

    Parameters
    ----------
    location : data location

    Returns
    -------
    experiments : list of exp - list[tuple]
        exp[0] : id of the experiment - int
        exp[1] : max_score for the experiment - float
        exp[2] : fixes - tuple(tuple(str, int))
                 ((node A, value1), (node B, value2), ...)
        exp[3] : observed_node - str
        exp[4] : outcome_value - str

    interventions : list of fixes - list[tuple]
        each element represents fixes - tuple(tuple(str, int))
    """
    ID = 0
    SCORE = 1
    INTERVENTION = 2
    NODE = 3
    VALUE = 4
    
    file = open(location, "r")
    lines = filter(comment_removal, file)
    data = csv.reader(lines, delimiter='\t')

    # skip the first row
    next(data)

    experiments = []
    interventions = []
    for row in data:
        exp = [int(row[ID]), float(row[SCORE])]

        fixes = []
        intervention_string = row[INTERVENTION].split(',')
        for sth in intervention_string:
            node_value = sth.split('=')
            fix = tuple([node_value[0], int(node_value[1])])
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
                    assert exp[NODE] != experiment[NODE], f'{experiment[ID]} and {exp[ID]} are duplicates' 

        # add the entry
        experiments.append(tuple(exp))

    return experiments, interventions


# def check_score(row):
#     try:
#         float(row[0])
#         return True
#     except ValueError:
#         return False

# 20220923 grouping expriments, lenient criteria
# def import_exps(location):
#     """
#     Returns experiments and interventions

#     Parameters
#     ----------
#     location : data location

#     Returns
#     -------
#     experiments : list of expset (tuple)
#         expset[0] : score (float) representing the score for the experiment set
#         expset[1] : exp (dict) representing each perturbation/result pair
#             exp key : fixes (tuple) ((node A, value1), (node B, value2), ...)
#             exp value : result (tuple) (measured_node, values)

#     interventions : list of tuples
#         each element represents fixes (tuple)
#     """

#     file = open(location, "r")
#     lines = filter(comment_removal, file)
#     data = csv.reader(lines, delimiter='\t')

#     experiments = []
#     interventions = []
#     expset = None
#     for row in data:
#         # check if it's a new entry
#         new_entry = check_score(row)
#         if new_entry == True:
#             # add the previous entry if it exists
#             if expset != None:
#                 expset.append(exp)
#                 experiments.append(tuple(expset))
#             # start the new entry
#             expset = []
#             expset.append(float(row[0]))
#             exp = {}
#             continue

#         fixes = []
#         row_iter = iter(row)
#         for sth in row_iter:
#             fix = tuple([sth, next(row_iter)])
#             fixes.append(fix)
#         # fixes should be sorted so that they do not depend on the order of user input
#         fixes = tuple(sorted(fixes, key= lambda x:x[0]))
#         result = tuple(next(data))

#         exp[fixes] = result

#         if fixes not in interventions:
#             interventions.append(fixes)

#     # add the last entry
#     expset.append(exp)
#     experiments.append(tuple(expset))

#     return experiments, interventions

# version when experiments are grouped, but lenient scoring criteria is not adopted.
# def import_exps(location):
#     """
#     Returns exps
#
#     Parameters
#     ----------
#     location : data location
#
#     Returns
#     -------
#     experiments : list of tuple
#         expset[0] : score (float) representing the score for the experiment set
#         expset[1] : exp (dict) representing each perturbation/result pair
#             key : fixes (tuple) ((node A, value1), (node B, value2), ...)
#             value : result (tuple) (measured_node, values)
#
#     interventions : list of tuples
#         each element represents a set of intervention
#     """
#
#     file = open(location, "r")
#     lines = filter(comment_removal, file)
#     data = csv.reader(lines, delimiter='\t')
#
#     experiments = []
#     interventions = []
#     expset = None
#     for row in data:
#         # check if it's a new entry
#         new_entry = check_score(row)
#         if new_entry == True:
#             # add the previous entry if it exists
#             if expset != None:
#                 expset.append(exp)
#                 experiments.append(tuple(expset))
#             # start the new entry
#             expset = []
#             expset.append(float(row[0]))
#             exp = {}
#             continue
#
#         fixes = []
#         row_iter = iter(row)
#         for sth in row_iter:
#             fix = tuple([sth, next(row_iter)])
#             fixes.append(fix)
#         # fixes should be sorted so that they do not depend on the order of user input
#         fixes = tuple(sorted(fixes, key= lambda x:x[0]))
#         result = tuple(next(data))
#
#         exp[fixes] = result
#
#         if fixes not in interventions:
#             interventions.append(fixes)
#
#     # add the last entry
#     expset.append(exp)
#     experiments.append(tuple(expset))
#
#     return experiments, interventions

# version when experiments were not grouped
# def import_exps(location):
#     """
#     Returns exps
#
#     Parameters
#     ----------
#     location : data location
#
#     Returns
#     -------
#     exps : dictionary of list of tuples
#         key represents the perturbation
#         value is a list where outcomes are stored
#         each outcome is a tuple (measured_node, 1, 0, 0)
#         1,0,0 denotes constitutive activation
#     """
#
#     file = open(location, "r")
#     data = csv.reader(file, delimiter='\t')
#
#     lst = []
#     for row in data:
#         exp = []
#         # get the fix
#         fix = []
#         row_iter = iter(row)
#         for sth in row_iter:
#             fix1 = tuple([sth, next(row_iter)])
#             fix.append(fix1)
#         fix = tuple(sorted(fix, key= lambda x:x[0]))
#         # get the result
#         result = tuple(next(data))
#
#         exp.append(fix)
#         exp.append(result)
#         lst.append(exp)
#
#     lst = sorted(lst, key = lambda x: x[0])
#
#     exps = {}
#     for exp in lst:
#         if exp[0] not in exps:
#             exps[exp[0]] = []
#         exps[exp[0]].append(exp[1])
#
#     return exps
