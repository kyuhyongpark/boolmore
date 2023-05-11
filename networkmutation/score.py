import itertools as it
import math

def line(input, start, end):
    '''
    Returns a straight line from the start to the end
    any value outside the region returns 0
    - - - - - - -
    Parameters
    input : input value
    start : (x,y)
    end : (x,y)
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

# 20230425 attractor agreement function update
def get_agreement(experiments, predictions):
    """
    Returns attractor agreements
    when given experimental outcomes and model predictions

    Parameters
    ----------
    experiments : list of exp - list[tuple]
        exp[0] : id of the experiment - int
        exp[1] : max_score for the experiment - float
        exp[2] : fixes - tuple(tuple(str, int))
                 ((node A, value1), (node B, value2), ...)
        exp[3] : observed_node - str
        exp[4] : outcome_value - str

    predictions : dict{tuple: dict}
        keys : fixes - tuple(tuple(str, int))
               ((node A, value1), (node B, value2), ...)
        values : average value of every node in the attractors - dict{str: float}
                 {observed_node: predict_value}
    
    Returns
    -------
    agreements : dict{str: dict}
        keys : observed_node
        values : dict{tuple: list}
            keys : fixes - tuple(tuple(str, int))
                   ((node A, value1), (node B, value2), ...)
            values : list[int, float, str, float, float]
                     [id, max_score, outcome_value, predict_value, agreement]

    Notes
    -----
    agreements are categorized by observed node,
    so that it is easier to find all the interventions for that observed node.
    """

    ID = 0
    SCORE = 1
    INTERVENTION = 2
    NODE = 3
    VALUE = 4

    agreements = {}
    for exp in experiments:
        id = exp[ID]
        max_score = exp[SCORE]
        fixes = exp[INTERVENTION]
        observed_node = exp[NODE]
        outcome_value = exp[VALUE]

        predict_value = predictions[fixes][observed_node]

        # experiment showed (ON)
        if outcome_value == 'ON':
            agreement = max(line(predict_value,(0,0),(1,1)),line(predict_value,(1,1),(1.0001,1)))
        # experiment showed (Some/ON)
        elif outcome_value == 'Some/ON':
            agreement = max(line(predict_value,(0,0),(0.5,1)),line(predict_value,(0.5,1),(1,1)),line(predict_value,(1,1),(1.0001,1)))
        # experiment showed (Some)
        elif outcome_value == 'Some':
            agreement = max(line(predict_value,(0,0),(0.25,1)),line(predict_value,(0.25,1),(0.75,1)),line(predict_value,(0.75,1),(1,0)))
        # experiment showed (OFF/Some)
        elif outcome_value == 'OFF/Some':
            agreement = max(line(predict_value,(-0.0001,1),(0,1)),line(predict_value,(0,1),(0.5,1)),line(predict_value,(0.5,1),(1,0)))
        # experiment showed (OFF)
        elif outcome_value == 'OFF':
            agreement = max(line(predict_value,(-0.0001,1),(0,1)),line(predict_value,(0,1),(1,0)))
        else:
            print("Unexpected input", outcome_value)
            raise Exception("Unexpected experiment input")

        if observed_node not in agreements:
            agreements[observed_node] = {}
        elif fixes in agreements[observed_node]:
            print(f'{agreements[observed_node][fixes][0]} and {id} are duplicates')
            raise Exception("Duplicate experimental entry")
        
        agreements[observed_node][fixes] = [id, max_score, outcome_value, predict_value, agreement]

        #     print("Intervention: ", fixes)
        #     print("Observed node: ", observed_node)
        #     print("Outcome: ", outcome_value)
        #     print("Prediction: ", predict_value)
        #     print("agreement: ", agreement)
        # print("- - - - - - - - - -")

    return agreements

# 20230511 multi-dimensional scoring update
def get_hierarchy_score(agreements, default_sources, report = False, file = None):
    """
    Returns model score
    when given attractor agreements

    Parameters
    ----------
    agreements : dict{str: dict}
        keys : observed_node
        values : dict{tuple: list}
            keys : fixes - tuple(tuple(str, int))
                   ((node A, value1), (node B, value2), ...)
            values : list[int, float, str, float, float]
                     [id, max_score, outcome_value, predict_value, agreement]

    default_sources : dict{str: int}
                      Shows what are the default settings for the source nodes.
                      This setting is considered the top of the hierarchy.

    extra_edges : list of extra_edge (tuple)
        extra_edge : tuple representing (regulator, target, sign)

    penalty : penalty for an extra edge (float)
    
    report : if True, make a csv file with detailed report

    file : the location and name of the detailed report file

    Returns
    -------
    total : The model's total score

    """
    ID = 0
    MAX_SCORE = 1
    OUTCOME_VALUE = 2
    PREDICT_VALUE = 3
    AGREEMENT = 4

    if report:
        fp = open(file, 'w')

    total_score = 0.0
    model_max_score = 0.0
    
    for observed_node in agreements:
        for fixes in agreements[observed_node]:
            lst = agreements[observed_node][fixes]

            current_score = 0.0
            current_score += lst[MAX_SCORE]
            model_max_score += lst[MAX_SCORE]

            subsets = powerset(fixes)
            subset_fixes = set()

            for subset in subsets:
                for source in default_sources:
                    if tuple([source, 0]) not in subset and tuple([source, 1]) not in subset:
                        subset_fix = list(subset)
                        subset_fix.append(tuple([source,default_sources[source]]))
                        subset_fix = tuple(sorted(subset_fix, key= lambda x:x[0]))
                    else:
                        subset_fix = subset

                subset_fixes.add(subset_fix)

            for subset_fix in subset_fixes:
                if subset_fix in agreements[observed_node]:
                    current_score *= agreements[observed_node][subset_fix][AGREEMENT]

            total_score += current_score
                
            if report:
                fp.write(str(agreements[observed_node][fixes][ID]) + '\t')
                fp.write(str(int(math.log2(len(subset_fixes)))) + '\t')
                for fix in fixes:
                    fp.write(str(fix[0]) + '=' + str(fix[1]) + ',')
                fp.write('\t')
                fp.write(str(observed_node) + '\t')
                fp.write(str(agreements[observed_node][fixes][OUTCOME_VALUE]) + '\t')
                fp.write(str(round(agreements[observed_node][fixes][PREDICT_VALUE],3)) + '\t')
                fp.write(str(round(agreements[observed_node][fixes][AGREEMENT],3)) + '\t')
                fp.write(str(round(current_score,3)) + '\n')

            # print("Adding ", current_score)
            # print("- - - - - - - - - -")
            
    # print("Total ", total_score, "/", model_max_score)
    if report:
        fp.write('total\t\t\t\t\t\t\t' + str(total_score) + '\n')
        fp.write('out of\t\t\t\t\t\t\t' + str(model_max_score) + '\n')

    return total_score