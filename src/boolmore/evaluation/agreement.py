from boolmore.experiment import NAVExperiment

FixesType = tuple[tuple[str, int],...]
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

def get_agreements(exps:list[NAVExperiment], predictions:PredictType) -> tuple[AgreeType, float]:
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

    """
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

    return agreements