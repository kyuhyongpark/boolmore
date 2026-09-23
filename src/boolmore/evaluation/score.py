from dataclasses import dataclass

from boolmore.model import Model
from boolmore.experiment import PhenotypeExperiment, NAVExperiment
from boolmore.inference.prediction import PhenotypePrediction
from boolmore.evaluation.agreement import get_agreements
from boolmore.evaluation.complexity import get_model_complexity


FixesType = tuple[tuple[str, int],...]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict[str, float]]
AgreeType = dict[str, dict[FixesType, tuple[int, float, str, float, float]]]

@dataclass
class EvalResult:
    model_id: int
    max_score: float
    score: float
    n_edges: int
    n_self_edges: int
    n_prime_implicants: int
    n_extra_edges: int
    details: any


class Evaluator:
    def __init__(self, exps, prediction_fn, score_fn):
        """
        exps : list of experiment dataclasses
        """
        self.exps = exps
        self.prediction_fn = prediction_fn
        self.score_fn = score_fn

    def evaluate(self, model:Model, id:int=-1)->EvalResult:
        predictions = self.prediction_fn(model.primes, self.exps)
        score_items:list[EvaluationItemScore] = self.score_fn(self.exps, predictions)
        max_score, score = get_model_score(score_items)
        complexity = get_model_complexity(model)
        result = EvalResult(model_id=id,
                            max_score=max_score,
                            score=score,
                            n_edges=complexity["n_edges"],
                            n_self_edges=complexity["n_self_edges"],
                            n_prime_implicants=complexity["n_prime_implicants"],
                            n_extra_edges=model.n_extra_edges,
                            details=[predictions, score_items])
        return result


@dataclass(frozen=True)
class EvaluationItemScore:
    id:int
    weight:float
    agreement:float
    score:float


def get_non_hierarchy_scores(
    agreements: AgreeType,
) -> list[EvaluationItemScore]:
    ID = 0
    MAX_SCORE = 1
    AGREEMENT = 4

    results: list[EvaluationItemScore] = []

    for observed_node in agreements:
        for fixes in agreements[observed_node]:

            base_weight = agreements[observed_node][fixes][MAX_SCORE]
            agreement = agreements[observed_node][fixes][AGREEMENT]

            # no hierarchy / subset effects
            score = base_weight * agreement

            results.append(
                EvaluationItemScore(
                    id=agreements[observed_node][fixes][ID],
                    weight=base_weight,
                    agreement=agreement,
                    score=score,
                )
            )

    return results


def get_hierarchy_scores(
    agreements: AgreeType,
    default_sources: dict[str, int],
    report: bool = False,
    file: str = 'score_report.tsv'
) -> list[EvaluationItemScore]:
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

    model_max_score = 0.0
    results: list[EvaluationItemScore] = []

    for observed_node in agreements:
        for fixes in agreements[observed_node]:
            fixes_dict = {key: value for (key, value) in fixes}

            hierarchy = 0
            for node in fixes_dict:
                if node in default_sources:
                    if fixes_dict[node] != default_sources[node]:
                        hierarchy += 1
                else:
                    hierarchy += 1

            subset_fixes_set = set()
            for other_fixes in agreements[observed_node]:
                other_fixes_dict = {key: value for (key, value) in other_fixes}

                is_subset = True
                for node in other_fixes_dict:
                    if node in default_sources:
                        if other_fixes_dict[node] == default_sources[node]:
                            continue
                        elif other_fixes_dict[node] == fixes_dict[node]:
                            continue
                        else:
                            is_subset = False
                            break
                    else:
                        if node not in fixes_dict:
                            is_subset = False
                            break
                        elif other_fixes_dict[node] == fixes_dict[node]:
                            continue
                        else:
                            is_subset = False
                            break

                if is_subset:
                    subset_fixes_set.add(other_fixes)

            base_weight = agreements[observed_node][fixes][MAX_SCORE]

            current_score = base_weight
            for subset_fixes in subset_fixes_set:
                current_score *= agreements[observed_node][subset_fixes][AGREEMENT]

            results.append(
                EvaluationItemScore(
                    id=agreements[observed_node][fixes][ID],
                    weight=base_weight,
                    agreement=agreements[observed_node][fixes][AGREEMENT],
                    score=current_score,
                )
            )

            if report:
                fp.write(str(agreements[observed_node][fixes][ID]) + '\t')
                fp.write(str(hierarchy) + '\t')
                for fix in fixes:
                    fp.write(str(fix[0]) + '=' + str(fix[1]) + ',')
                fp.write('\t')
                fp.write(str(observed_node) + '\t')
                fp.write(str(agreements[observed_node][fixes][OUTCOME_VALUE]) + '\t')
                fp.write(str(round(agreements[observed_node][fixes][PREDICT_VALUE], 3)) + '\t')
                fp.write(str(round(agreements[observed_node][fixes][AGREEMENT], 3)) + '\t')
                fp.write(str(round(current_score, 3)) + '\n')

    if report:
        total_score = sum(r.score for r in results)
        model_max_score = sum(r.weight for r in results)
        fp.write('total\t' + str(total_score) + '\n')
        fp.write('max\t' + str(model_max_score) + '\n')
        fp.write('per\t' + str(total_score / model_max_score * 100) + '%\n')

    return results


def get_NAV_scores(
    exps:list[NAVExperiment],
    predictions,
    default_sources:dict={},
    hierarchy:bool=True,
    report:bool=False,
    file:str="score_report.tsv"):
    """
    Returns score when given experiments.
    Requires predictions to be calculated beforehand.

    Can be modified to meet the desired criteria.
    
    Returns
    -------
    max_score : float
        max possible score of the model

    score : float
        how well the model agrees with experimental results
        one point in score means agreement to one perturbation
    
    """
    agreements = get_agreements(exps, predictions)

    if hierarchy:
        scores = get_hierarchy_scores(agreements, default_sources, report=report, file=file)
    else:
        scores = get_non_hierarchy_scores(agreements)

    return scores


def get_phenotype_scores(
    experiments: list[PhenotypeExperiment],
    predictions: list[PhenotypePrediction],
) -> list[EvaluationItemScore]:
    """
    Update each PhenotypePrediction with its agreement and score.

    agreement is 1.0 if predicted_exists matches the corresponding
    Experiment's expected_exists, and 0.0 otherwise.

    score is weight * agreement.

    Experiments and Predictions are matched by their id.

    Returns
    -------
    max_score : float
        Sum of the experiment weights.

    score : float
        Sum of the individual prediction scores.
    TODO: return or modify input, not do both at the same time.
    """
    exp_by_id = {exp.id: exp for exp in experiments}

    scores: list[EvaluationItemScore] = []

    missing = []

    for pred in predictions:
        exp = exp_by_id.get(pred.id)
        if exp is None:
            missing.append(pred.id)
            continue

        agreement = (
            1.0 if pred.predicted_exists == exp.expected_exists else 0.0
        )

        score = exp.weight * agreement

        scores.append(
            EvaluationItemScore(
                id=pred.id,
                weight=exp.weight,
                agreement=agreement,
                score=score,
            )
        )

    if missing:
        raise ValueError(
            f"No Experiment found for Prediction id(s): {missing}"
        )

    return scores


def get_model_score(
    scores: list[EvaluationItemScore],
) -> tuple[float, float]:

    max_score = sum(s.weight for s in scores)
    total_score = sum(s.score for s in scores)

    return max_score, total_score

