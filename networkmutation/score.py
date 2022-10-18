import itertools as it

def line(input, start, end):
    '''
    Returns a straight line from the start to the end
    any value outside the region returns 0
    '''
    if not start[0] <= input <= end[0]:
        return 0

    slope = (end[1]-start[1])/(end[0]-start[0])
    output = start[1] + slope * (input - start[0])
    return output

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

# 20221004 hierarchy scoring
def get_agreement(experiments, predictions):
    """
    Returns attractor agreements
    when given experimental results and model predictions

    Parameters
    ----------
    experiments : list of expset (tuple)
        expset[0] : score (float) representing the score for the experiment set
        expset[1] : exp (dict) representing each perturbation/result pair
            exp key : fixes (tuple) ((node A, value1), (node B, value2), ...)
            exp value : outcome (tuple) (measured_node, values)

    predictions : dict
        keys : fixes (tuple) ((node A, value1), (node B, value2), ...)
        values : dict
            average value of every node in the attractors

    Returns
    -------
    agreements : dict
        keys : Outcome node
        values : dict
            keys : fixes (tuple)
            values : [agreement(float), outcome(tuple), predict value(float)]

    Notes
    -----
    experimental data input

    score: The maximum score the model can get from the experiment
    e.g. [1.0]

    Interventions: The nodes, and the values (0 or 1) separated by tab
    e.g. [ABA	0	CPK3_21	0]

    Outcome: A single node, and the expected value separated by tab.
    The outcome value can be 0 or 0.5 or 1, and multiple values can be listed.
    If multiple values for outcome are listed,
    the model will be checked whether it matches with one of them.
    e.g. [Closure	0	0.5]
    The model is considered to agree with this result if it gets 0 or 0.5.
    Optional arguments: lenient scoring
    'lenient 1' or 'lenient 0' can be added when a single outcome value is listed.
    Scoring method will give half score in that direction.
    e.g. [Closure	0	lenient 1] will give 1 score if 0, and 0.5 score if 0.5

    If multiple intervention/outcome pairs are listed,
    the model should agree with all to get the score.
    """
    agreements = {}
    for expset in experiments:
        exp = expset[1]
        for fix in exp:
            outcome = exp[fix]
            node = outcome[0]
            if node not in agreements:
                agreements[node] = {}
            predict_value = predictions[fix][node]
            # experiment showed (ON)
            if outcome[1:] == ('1',):
                agreement = line(predict_value,(0.7,0),(1,1))
            # experiment showed (ON lenient 0)
            elif outcome[1:] == ('1', 'lenient 0'):
                agreement = line(predict_value,(0,0),(1,1))
            # experiment showed (ON/some ON)
            elif '1' in outcome and '0.5' in outcome:
                agreement = max(line(predict_value,(0,0),(0.3,1)),line(predict_value,(0.3,1),(1,1)))
            # experiment showed (some ON lenient 1)
            elif outcome[1:] == ('0.5', 'lenient 1'):
                agreement = max(line(predict_value,(0,0),(0.3,1)),line(predict_value,(0.3,1),(0.7,1)),line(predict_value,(0.7,1),(1,0.5)))
            # experiment showed (some ON)
            elif outcome[1:] == ('0.5',):
                agreement = max(line(predict_value,(0,0),(0.3,1)),line(predict_value,(0.3,1),(0.7,1)),line(predict_value,(0.7,1),(1,0)))
            # experiment showed (some ON lenient 1)
            elif outcome[1:] == ('0.5', 'lenient 0'):
                agreement = max(line(predict_value,(0,0.5),(0.3,1)),line(predict_value,(0.3,1),(0.7,1)),line(predict_value,(0.7,1),(1,0)))
            # experiment showed (some ON/OFF)
            elif '0.5' in outcome and '0' in outcome:
                agreement = max(line(predict_value,(0,1),(0.7,1)),line(predict_value,(0.7,1),(1,0)))
            # experiment showed (OFF lenient 1)
            elif outcome[1:] == ('0', 'lenient 1'):
                agreement = line(predict_value,(0,1),(1,0))
            # experiment showed (OFF)
            elif outcome[1:] == ('0',):
                agreement = line(predict_value,(0,1),(0.3,0))
            else:
                print("Unexpected input", outcome)
                raise Exception("Unexpected experiment input")

            if fix in agreements[node]:
                print("These fixes are already included", fix)
                raise Exception("Duplicate experimental entry")
            else:
                agreements[node][fix] = []

            agreements[node][fix].append(agreement)
            agreements[node][fix].append(outcome)
            agreements[node][fix].append(predict_value)
        #     print("Perturbation: ", fix)
        #     print("Experiment: ", outcome)
        #     print("Prediction: ", predict_value)
        #     print("agreement: ", agreement)
        # print("- - - - - - - - - -")
    return agreements

# 20221006 hierarchy scoring
def get_hierarchy_score(experiments, agreements, extra_edges):
    """
    Returns model score
    when given experimental results and model attractor agreements

    Parameters
    ----------
    experiments : list of expset (tuple)
        expset[0] : score (float) representing the score for the experiment set
        expset[1] : exp (dict) representing each perturbation/result pair
            exp key : fixes (tuple) ((node A, value1), (node B, value2), ...)
            exp value : outcome (tuple) (measured_node, values)

    agreements : dict
        keys : Outcome node
        values : dict
            keys : fixes (tuple)
            values : [agreement(float), outcome(tuple), predict value(float)]

    extra_edges : list of extra_edge (tuple)
        extra_edge : tuple representing (starting node, )
    Returns
    -------
    total : The model's total score

    """
    total = 0.0
    max_score = 0.0
    cost = len(extra_edges) * 0.5
    total -= cost
    for expset in experiments:
        score = float(expset[0])
        max_score += score
        exp = expset[1]
        for fix in exp:
            outcome = exp[fix]
            node = outcome[0]
            # print("Perturbation: ", fix)
            # print("Experiment: ", outcome)
            # print("Prediction: ", agreements[node][fix][2])
            # print("Agreement: ", agreements[node][fix][0])

            subsets = powerset(fix)
            for subset in subsets:
                # print('sub_intervention: ',subset)
                if subset in agreements[node]:
                    score *= agreements[node][subset][0]
                    # print("Experiment: ", agreements[node][subset][1])
                    # print("Prediction: ", agreements[node][subset][2])
                    # print('agreement: ',agreements[node][subset][0])
                    # print('current score: ',score)

        # print("Adding ", score)
        # print("- - - - - - - - - -")
        total += score
    # print("Total ", total, "/", max_score)

    return total


# 20220923 grouping experiment, lenient criteria
def get_score(exps, predictions, extra_edges):
    '''
    experimental data input

    score: The model will get this score if it agrees with all the results below
    e.g. [1.0]

    Interventions: The nodes, and the values (0 or 1) separated by tab
    e.g. [ABA	0	CPK3_21	0]

    Outcome: A single node, and the expected value separated by tab
    The outcome value can be 0 or 0.5 or 1, and multiple values can be listed
    If multiple values for outcome are listed,
    the model will be checked whether it matches with one of them.
    e.g. [Closure	0	0.5]
    the model is considered to agree with this result if it gets 0 or 0.5
    Optional arguments: lenient scoring
    'lenient 1' or 'lenient 0' can be added when a single outcome value is listed.
    Scoring method will give half score in that direction
    e.g. [Closure	0	lenient 1] will give 1 score if 0, and 0.5 score if 0.5

    If multiple intervention/outcome pairs are listed,
    the model should agree with all to get the score.
    '''
    total = 0.0
    max_score = 0.0
    cost = len(extra_edges) * 0.5
    total -= cost
    for expset in exps:
        score = float(expset[0])
        max_score += score
        exp = expset[1]
        for fix in exp:
            result = exp[fix]
            node = result[0]
            predict_value = predictions[fix][node]
            # experiment showed (ON)
            if result[1:] == ('1',):
                mult = line(predict_value,(0.7,0),(1,1))
            # experiment showed (ON lenient 0)
            elif result[1:] == ('1', 'lenient 0'):
                mult = line(predict_value,(0,0),(1,1))
            # experiment showed (ON/some ON)
            elif '1' in result and '0.5' in result:
                mult = max(line(predict_value,(0,0),(0.3,1)),line(predict_value,(0.3,1),(1,1)))
            # experiment showed (some ON lenient 1)
            elif result[1:] == ('0.5', 'lenient 1'):
                mult = max(line(predict_value,(0,0),(0.3,1)),line(predict_value,(0.3,1),(0.7,1)),line(predict_value,(0.7,1),(1,0.5)))
            # experiment showed (some ON)
            elif result[1:] == ('0.5',):
                mult = max(line(predict_value,(0,0),(0.3,1)),line(predict_value,(0.3,1),(0.7,1)),line(predict_value,(0.7,1),(1,0)))
            # experiment showed (some ON lenient 1)
            elif result[1:] == ('0.5', 'lenient 0'):
                mult = max(line(predict_value,(0,0.5),(0.3,1)),line(predict_value,(0.3,1),(0.7,1)),line(predict_value,(0.7,1),(1,0)))
            # experiment showed (some ON/OFF)
            elif '0.5' in result and '0' in result:
                mult = max(line(predict_value,(0,1),(0.7,1)),line(predict_value,(0.7,1),(1,0)))
            # experiment showed (OFF lenient 1)
            elif result[1:] == ('0', 'lenient 1'):
                mult = line(predict_value,(0,1),(1,0))
            # experiment showed (OFF)
            elif result[1:] == ('0',):
                mult = line(predict_value,(0,1),(0.3,0))
            else:
                print("Unexpected input", result)
                raise Exception("Unexpected experiment input")
            score *= mult

        #     print("Perturbation: ", fix)
        #     print("Experiment: ", result)
        #     print("Prediction: ", predict_value)
        #     print("multiplying ", mult)
        # print("Adding ", score)
        # print("- - - - - - - - - -")
        total += score
    # print("Total ", total, "/", max_score)

    return total

# grouping experiment implemented, but no lenient criteria
# def get_score(exps, predictions, extra_edges):
#     '''
#     To be modified to meet the criteria
#     '''
#     total = 0.0
#     max_score = 0.0
#     cost = len(extra_edges) * 0.5
#     total -= cost
#     for expset in exps:
#         score = float(expset[0])
#         max_score += score
#         exp = expset[1]
#         for fix in exp:
#             result = exp[fix]
#             node = result[0]
#             predict_value = predictions[fix][node]
#             # experiment showed (ON)
#             if '1' in result and '0.5' not in result:
#                 mult = line(predict_value,0.5,1,'inc')
#             # experiment showed (ON/some ON)
#             elif '1' in result and '0.5' in result:
#                 mult = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,1,'flat'))
#             # experiment showed (some ON)
#             elif '0.5' in result and '0' not in result:
#                 mult = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,0.7,'dec'))
#             # experiment showed (some ON/OFF)
#             elif '0.5' in result and '0' in result:
#                 mult = max(line(predict_value,0,0.5,'flat'),line(predict_value,0.5,0.7,'dec'))
#             # experiment showed (OFF)
#             elif '0.5' not in result and '0' in result:
#                 mult = line(predict_value,0,0.5,'dec')
#             else:
#                 raise Exception("Unexpected experiment input")
#             score *= mult
#
#             # print("Perturbation: ", fix)
#             # print("Experiment: ", result)
#             # print("Prediction: ", predict_value)
#             # print("multiplying ", mult)
#         # print("Adding ", score)
#         # print("- - - - - - - - - -")
#         total += score
#     print("Total ", total, "/", max_score)
#
#     return total

# def get_score(exps, predictions, extra_edges):
#     '''
#     To be modified to meet the criteria
#     '''
#     score = 0.0
#     cost = len(extra_edges) * 0.5
#     score -= cost
#
#     counts = [0]*14
#     for fix in exps:
#         for result in exps[fix]:
#             node = result[0]
#             predict_value = predictions[fix][node]
#             # experiment showed (ON)
#             if result[1] == '1' and result[2] == '0':
#                 counts[0] += 2
#                 add = 2*line(predict_value,0.5,1,'inc')
#                 counts[1] += add
#             # experiment showed (ON/some ON)
#             elif result[1] == '1' and result[2] == '1':
#                 counts[2] += 1
#                 add = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,1,'flat'))
#                 counts[3] += add
#             # experiment showed (some ON)
#             elif result[1] == '0' and result[2] == '1' and result[3] == '0':
#                 # find the wildtype
#                 for fix1 in fix:
#                     if fix1[0] == 'ABA':
#                         value = fix1[1]
#                         break
#                 wt = (('ABA', value),)
#                 wt_behaviors = exps[wt]
#                 # print('WT behavior:', wt_behaviors)
#                 wt_behavior = None
#                 for wtb1 in wt_behaviors:
#                     if wtb1[0] == node:
#                         wt_behavior = wtb1
#
#                 if wt_behavior == None or wt_behavior[2] == '1':
#                     counts[4] += 1
#                     add = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,0.7,'dec'))
#                     counts[5] += add
#                 elif wt_behavior[3] == '1':
#                     counts[6] += 1
#                     add = max(line(predict_value,0.3,0.5,'inc'),line(predict_value,0.5,1,'dec'))
#                     counts[7] += add
#                 else: # wt_behavior[1] == '1'
#                     counts[8] += 1
#                     add = max(line(predict_value,0,0.5,'inc'),line(predict_value,0.5,0.7,'dec'))
#                     counts[9] += add
#             # experiment showed (some ON/OFF)
#             elif result[2] == '1' and result[3] == '1':
#                 counts[10] += 1
#                 add = max(line(predict_value,0,0.5,'flat'),line(predict_value,0.5,0.7,'dec'))
#                 counts[11] += add
#             # experiment showed (OFF)
#             elif result[2] == '0' and result[3] == '1':
#                 counts[12] +=1
#                 add = line(predict_value,0,0.5,'dec')
#                 counts[13] += add
#             else:
#                 raise Exception("Unexpected experiment input")
#
#             score += add
#             print("- - - - - - - - - -")
#             print("Perturbation: ", fix)
#             print("Experiment: ", result)
#             print("Prediction: ", predict_value)
#             print("adding ", add)
#             print("Current score ", score)
#     print()
#     print("- - - - - - - - - -")
#     print("total score: ",score, '/', counts[0]+counts[2]+counts[4]+counts[6]+counts[8]+counts[10]+counts[12])
#     print("edge cost: ", cost)
#     print("score got from ON: ", counts[1], '/', counts[0])
#     print("score got from ON/some: ", counts[3], '/', counts[2])
#     print("score got from some: ", counts[5], '/', counts[4])
#     print("score got from increased: ", counts[7], '/', counts[6])
#     print("score got from reduced: ", counts[9], '/', counts[8])
#     print("score got from some/OFF: ", counts[11], '/', counts[10])
#     print("score got from OFF: ", counts[13], '/', counts[12])
#     return score
