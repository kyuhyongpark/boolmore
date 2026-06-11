import itertools as it
from collections.abc import Iterable

from boolmore.core.experiment import PhenotypeExperiment, NAVExperiment
from boolmore.core.prediction import PhenotypePrediction


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


def get_agreement(exps:list[NAVExperiment], predictions:PredictType) -> tuple[AgreeType, float]:
    """
    Returns attractor agreements when given experimental outcomes and model predictions.
    agreements are categorized by observed node,
    so that it is easier to find all the interventions for that observed node.
    Also returns non-hierarchy score.

    Parameters
    ----------
    exps : list[NAVExperiment]

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

    non_hierarchy_score = 0

    agreements = {}
    for exp in exps:
        id = exp.id
        max_score = exp.weight
        fixes = exp.fixes
        observed_node = exp.observed
        outcome_value = exp.outcome

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


def get_phenotype_score(
    experiments: list[PhenotypeExperiment],
    predictions: list[PhenotypePrediction],
) -> None:
    """
    Update each PhenotypePrediction with its agreement and score.

    agreement is 1.0 if predicted_exists matches the corresponding
    Experiment's expected_exists, and 0.0 otherwise.

    score is weight * agreement.

    Experiments and Predictions are matched by their id.
    """
    exp_by_id = {exp.id: exp for exp in experiments}

    missing = []

    for pred in predictions:
        exp = exp_by_id.get(pred.id)
        if exp is None:
            missing.append(pred.id)
            continue

        pred.agreement = (
            1.0 if pred.predicted_exists == exp.expected_exists else 0.0
        )
        pred.score = exp.weight * pred.agreement

    if missing:
        raise ValueError(
            f"No Experiment found for Prediction id(s): {missing}"
        )


def get_NAV_score(
    exps:list[NAVExperiment],
    predictions,
    default_sources,
    hierarchy:bool=True,
    report:bool=False,
    file:str="score_report.tsv"):
    """
    Assigns self.score when given experiments.
    Requires self.predictions to be calculated beforehand.

    Can be modified to meet the desired criteria.
    
    Assigns
    -------
    self.max_score : float
        max possible score of the model

    self.non_hierarchy_score : float
        how well the model agrees with experimental results, ignoring hierarchy
    
    self.score : float
        how well the model agrees with experimental results
        one point in score means agreement to one perturbation
    
    """
    max_score = 0.0
    for exp in exps:
        max_score += exp.weight
    agreements, non_hierarchy_score = get_agreement(exps, predictions)

    if hierarchy:
        score = get_hierarchy_score(agreements, default_sources, report=report, file=file)
    else:
        score = non_hierarchy_score

    return max_score, score