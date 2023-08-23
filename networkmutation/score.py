# TODO: Allow hierarchy score to have a default setting for default sources

import itertools as it
import math
from collections.abc import Iterable


FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict[str, float]]
AgreeType = dict[str, dict[FixesType, tuple[int, float, str, float, float]]]


def line(input:float, start:tuple[float, float], end:tuple[float, float]) -> float:
    """
    Takes the input as the x value and returns the y value
    on a straight line from the start to the end.
    Any value outside the region returns 0.
    
    Parameters
    ----------
    input   - x value   :float
    start   - (x1,y1)   :tuple[float, float]
    end     - (x2,y2)   :tuple[float, float]

    Returns
    -------
    output  - y value   :float

    """
    if not start[0] <= input <= end[0]:
        return 0

    slope = (end[1]-start[1])/(end[0]-start[0])
    output = start[1] + slope * (input - start[0])

    return output


def powerset(iterable:Iterable) -> it.chain:
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    s = list(iterable)

    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))


# 20230425 attractor agreement function update
def get_agreement(experiments:list[ExpType], predictions:PredictType) -> AgreeType:
    """
    Returns attractor agreements
    when given experimental outcomes and model predictions

    Parameters
    ----------
    experiments - list of exp                           :list[ExpType]
        exp     - info of a single experiment           :ExpType = tuple[int, float, FixesType, str, str]
            exp[0] - id of the experiment               :int
            exp[1] - max_score for the experiment       :float
            exp[2] - fixes                              :FixesType = tuple[tuple[str, int]]
                     ((node A, value1), (node B, value2), ...)
            exp[3] - observed_node                      :str
            exp[4] - outcome_value                      :str
                     one of OFF, OFF/Some, Some, Some/ON, ON

    predictions - average attractor values for all fixes    :PredictType = dict[FixesType, dict]
        keys   - fixes                                      :FixesType = tuple[tuple[str, int]]
        values - average values of nodes                    :dict[str, float]
                 {observed_node: predict_value}
    
    Returns
    -------
    agreements - collection of all data                     :AgreeType = dict[str, dict]
        key   - observed_node                               :str
        value - data for the node                           :dict[FixesType, tuple]
            key   - fixes                                   :FixesType = tuple[tuple[str, int]]
            value - data for given fixes                    :tuple[int, float, str, float, float]
                    (id, max_score, outcome_value, predict_value, agreement)

    Notes
    -----
    agreements are categorized by observed node,
    so that it is easier to find all the interventions for that observed node.

    """

    ID = 0
    SCORE = 1
    FIXES = 2
    NODE = 3
    VALUE = 4

    agreements = {}
    for exp in experiments:
        id = exp[ID]
        max_score = exp[SCORE]
        fixes = exp[FIXES]
        observed_node = exp[NODE]
        outcome_value = exp[VALUE]

        predict_value = predictions[fixes][observed_node]

        # experiment showed (ON)
        if outcome_value == 'ON':
            agreement = max(line(predict_value,(0,0),(1,1)),
                            line(predict_value,(1,1),(1.0001,1)))
        # experiment showed (Some/ON)
        elif outcome_value == 'Some/ON':
            agreement = max(line(predict_value,(0,0),(0.5,1)),
                            line(predict_value,(0.5,1),(1,1)),
                            line(predict_value,(1,1),(1.0001,1)))
        # experiment showed (Some)
        elif outcome_value == 'Some':
            agreement = max(line(predict_value,(0,0),(0.25,1)),
                            line(predict_value,(0.25,1),(0.75,1)),
                            line(predict_value,(0.75,1),(1,0)))
        # experiment showed (OFF/Some)
        elif outcome_value == 'OFF/Some':
            agreement = max(line(predict_value,(-0.0001,1),(0,1)),
                            line(predict_value,(0,1),(0.5,1)),
                            line(predict_value,(0.5,1),(1,0)))
        # experiment showed (OFF)
        elif outcome_value == 'OFF':
            agreement = max(line(predict_value,(-0.0001,1),(0,1)),
                            line(predict_value,(0,1),(1,0)))
        else:
            print("Unexpected input", outcome_value)
            raise Exception("Unexpected experiment input")

        if observed_node not in agreements:
            agreements[observed_node] = {}
        elif fixes in agreements[observed_node]:
            print(f'{agreements[observed_node][fixes][0]} and {id} are duplicates')
            raise Exception("Duplicate experimental entry")
        
        agreements[observed_node][fixes] = id, max_score, outcome_value, predict_value, agreement

        #     print("Fixes: ", fixes)
        #     print("Observed node: ", observed_node)
        #     print("Outcome: ", outcome_value)
        #     print("Prediction: ", predict_value)
        #     print("agreement: ", agreement)
        # print("- - - - - - - - - -")

    return agreements


# 20230511 multi-dimensional scoring update
def get_hierarchy_score(agreements:AgreeType, default_sources:dict[str,int],
                        report:bool=False, file:str='score_report.tsv') -> float:
    """
    Returns model score
    when given attractor agreements

    Parameters
    ----------
    agreements - collection of all data                     :AgreeType = dict[str, dict]
        key   - observed_node                               :str
        value - data for the node                           :dict[FixesType, tuple]
            key   - fixes                                   :FixesType = tuple[tuple[str, int]]
                    ((node A, value1), (node B, value2), ...)
            value - data for given fixes                    :tuple[int, float, str, float, float]
                    (id, max_score, outcome_value, predict_value, agreement)
    
    default_sources - Shows the default settings for the source nodes,      :dict[str, int]
                      which is considered the top of the hierarchy
                      These source nodes must have a defined value in
                      every experiments
    report          - if True, make a csv file with detailed report         :bool
    file            - the location and name of the detailed report file     :str

    Returns
    -------
    score - The model's score   :float

    """
    ID = 0
    MAX_SCORE = 1
    OUTCOME_VALUE = 2
    PREDICT_VALUE = 3
    AGREEMENT = 4

    if report:
        fp = open(file, 'w')
        fp.write('id\thierarchy\tfixes\tobserved node\texperimental outcome\t')
        fp.write('predict value\tattractor agreement\tscore\n')

    score = 0.0
    model_max_score = 0.0
    
    for observed_node in agreements:
        for fixes in agreements[observed_node]:
            for source in default_sources:
                assert tuple([source, 0]) in fixes or tuple([source, 1]) in fixes, "default sources must always be specified"
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

                subset_fixes.add(subset_fix) # type: ignore

            for subset_fix in subset_fixes:
                if subset_fix in agreements[observed_node]:
                    current_score *= agreements[observed_node][subset_fix][AGREEMENT]

            score += current_score
                
            if report:
                fp.write(str(agreements[observed_node][fixes][ID]) + '\t') # type: ignore
                fp.write(str(int(math.log2(len(subset_fixes)))) + '\t') # type: ignore
                for fix in fixes:
                    fp.write(str(fix[0]) + '=' + str(fix[1]) + ',') # type: ignore
                fp.write('\t') # type: ignore
                fp.write(str(observed_node) + '\t') # type: ignore
                fp.write(str(agreements[observed_node][fixes][OUTCOME_VALUE]) + '\t') # type: ignore
                fp.write(str(round(agreements[observed_node][fixes][PREDICT_VALUE],3)) + '\t') # type: ignore
                fp.write(str(round(agreements[observed_node][fixes][AGREEMENT],3)) + '\t') # type: ignore
                fp.write(str(round(current_score,3)) + '\n') # type: ignore

            # print("Adding ", current_score)
            # print("- - - - - - - - - -")
            
    # print("Total ", score, "/", model_max_score)
    if report:
        fp.write('total\t\t\t\t\t\t\t' + str(score) + '\n') # type: ignore
        fp.write('out of\t\t\t\t\t\t\t' + str(model_max_score) + '\n') # type: ignore

    return score