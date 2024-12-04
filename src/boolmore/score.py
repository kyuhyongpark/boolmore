import itertools as it
from collections.abc import Iterable


FixesType = tuple[tuple[str, int],...]
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


def get_agreement(experiments:list[ExpType], predictions:PredictType) -> tuple[AgreeType, float]:
    """
    Returns attractor agreements when given experimental outcomes and model predictions.
    agreements are categorized by observed node,
    so that it is easier to find all the interventions for that observed node.
    Also returns non-hierarchy score.

    Parameters
    ----------
    experiments : list[ExpType]
        exp : ExpType
            info of a single experiment                
            exp[0] : int
                id of the experiment
            exp[1] : float
                max_score for the experiment
            exp[2] : FixesType
                fixes - ((node A, value1), (node B, value2), ...)
            exp[3] : str
                observed_node
            exp[4] : str
                outcome_value - one of OFF, OFF/Some, Some, Some/ON, ON

    predictions : PredictType
        average attractor values for all fixes
        keys : FixesType
        values : dict[str, float]
            average values of nodes - {observed_node: predict_value}
    
    Returns
    -------
    agreements : AgreeType
        collection of all data
        key : str
            observed_node
        value : dict[FixesType, tuple]
            data for the node
            key : FixesType
            value : tuple[int, float, str, float, float]
                data for given fixes - (id, max_score, outcome_value, predict_value, agreement)

    non_hierarchy_score : float
        non-hierarchy score considering all attractor agreements

    """

    ID = 0
    SCORE = 1
    FIXES = 2
    NODE = 3
    VALUE = 4

    non_hierarchy_score = 0

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

        non_hierarchy_score += max_score * agreement

        # print("Fixes: ", fixes)
        # print("Observed node: ", observed_node)
        # print("Outcome: ", outcome_value)
        # print("Prediction: ", predict_value)
        # print("agreement: ", agreement)
        # print("- - - - - - - - - -")

    return agreements, non_hierarchy_score


def get_hierarchy_score(agreements:AgreeType, default_sources:dict[str,int],
                        report:bool=False, file:str='score_report.tsv') -> float:
    """
    Returns model score
    when given attractor agreements

    Parameters
    ----------
    agreements : AgreeType
        collection of all data
        key : str
            observed_node
        value : dict[FixesType, tuple]
            data for the node
            key : FixesType
            value : tuple[int, float, str, float, float]
                data for given fixes - (id, max_score, outcome_value, predict_value, agreement)
    
    default_sources : dict[str, int]
        Shows the default settings for the source nodes, which is considered the top of the hierarchy
        These source nodes must have a defined value in every experiments.

    report : bool
        if True, make a csv file with detailed report
    
    file : str
        the location and name of the detailed report file

    Returns
    -------
    score : float
        how well the model agrees with experimental results
        one point in score means agreement to one perturbation

    """
    ID = 0
    MAX_SCORE = 1
    OUTCOME_VALUE = 2
    PREDICT_VALUE = 3
    AGREEMENT = 4

    if report:
        fp = open(file, 'w')
        fp.write('id\thierarchy\tfixes\tobserved_node\texperimental_outcome\t')
        fp.write('predict_value\tattractor_agreement\tscore\n')

    score = 0.0
    model_max_score = 0.0
    
    for observed_node in agreements:
        for fixes in agreements[observed_node]:
            fixes_dict = {key:value for (key, value) in fixes}

            # get the hierarchy
            hierarchy = 0
            for node in fixes_dict:
                if node in default_sources:
                    # source node value not same as default sources
                    if fixes_dict[node] != default_sources[node]:
                        hierarchy += 1
                # non source nodes
                else:
                    hierarchy += 1

            # for given fixes, find the subset fixes
            subset_fixes_set = set()
            for other_fixes in agreements[observed_node]:
                other_fixes_dict = {key:value for (key, value) in other_fixes}

                is_subset = True
                for node in other_fixes_dict:
                    # source nodes
                    if node in default_sources:
                        # source node same as default sources
                        if other_fixes_dict[node] == default_sources[node]:
                            continue
                        # source node same as in given fixes
                        elif other_fixes_dict[node] == fixes_dict[node]:
                            continue
                        else:
                            is_subset = False
                            break
                    # non source nodes
                    else:
                        # the node is not in given fixes
                        if node not in fixes_dict:
                            is_subset = False
                            break
                        # the node is fixed to same value as in given fixes
                        elif other_fixes_dict[node] == fixes_dict[node]:
                            continue
                        else:
                            is_subset = False
                            break

                if is_subset:
                    subset_fixes_set.add(other_fixes)

            current_score = agreements[observed_node][fixes][MAX_SCORE]
            model_max_score += agreements[observed_node][fixes][MAX_SCORE]

            for subset_fixes in subset_fixes_set:
                current_score *= agreements[observed_node][subset_fixes][AGREEMENT]

            score += current_score
                
            if report:
                fp.write(str(agreements[observed_node][fixes][ID]) + '\t') # type: ignore
                # TODO: fix reporting hierarchy number
                fp.write(str(hierarchy) + '\t') # type: ignore
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
        fp.write('total\t' + str(score) + '\n') # type: ignore
        fp.write('max\t' + str(model_max_score) + '\n') # type: ignore
        fp.write('per\t' + str(score/model_max_score*100) + '%\n') # type: ignore

    return score


if __name__=="__main__":
    from pyboolnet.external.bnet2primes import bnet_file2primes

    from boolmore.experiment import import_exps
    from boolmore.model import Model

    DATA = "boolmore/case_study/data/data_Li_20230926.tsv"
    # MODEL = "boolmore/comparison/gitsbe/ABA_A_20241105_214621/models/ABA_A_network_run_0__G399_M138.bnet"
    MODEL = "boolmore/case_study/baseline_models/ABA_2006_Li.bnet"
    # MODEL = "boolmore/comparison/gitsbe/ABA_GA1_A.bnet"
    NAME = None
    DEFAULT_SOURCES = {"ABA":0}

    if NAME == None:
        NAME = MODEL.split("/")[-1][:-5]

    print("Loading experimental data . . .")
    exps, pert = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading model . . .")
    primes = bnet_file2primes(MODEL)
    n1 = Model.import_model(primes, default_sources=DEFAULT_SOURCES)
    print("Model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps, report=True, file=NAME+'_score.tsv')
    n1.info()