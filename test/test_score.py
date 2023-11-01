import sys
sys.path.insert(0, "../boolmore")

import pystablemotifs as sm
from model import Model
from experiment import *
from score import *

def test_score1():
    """Test basic score curve"""
    exps:list[ExpType] = [(1, 1.0, (("ABA", 0),), "Closure", "OFF"),]
    predictions:PredictType = {(("ABA", 0),):{"Closure":0.0},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score == 1.0

def test_score2():
    """Test basic score curve"""
    exps:list[ExpType] = [(1, 1.0, (("ABA", 0),), "Closure", "OFF"),]
    predictions:PredictType = {(("ABA", 0),):{"Closure":0.2},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score == 0.8

def test_score3():
    """Test hierarchy"""
    exps:list[ExpType] = [(2, 1.0, (("ABA", 1),), "Closure", "ON"),]
    predictions:PredictType = {(("ABA", 1),):{"Closure":0.2},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score == 0.2

def test_score4():
    """Test hierarchy"""
    exps:list[ExpType] = [(1, 1.0, (("ABA", 0),), "Closure", "OFF"),
                          (2, 1.0, (("ABA", 1),), "Closure", "ON"),]
    predictions:PredictType = {(("ABA", 0),):{"Closure":1.0},
                               (("ABA", 1),):{"Closure":1.0},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score == 0.0

def test_score5():
    """Test hierarchy"""
    exps:list[ExpType] = [(1, 1.0, (("ABA", 0),), "Closure", "OFF"),
                          (2, 2.0, (("ABA", 1),), "Closure", "ON"),]
    predictions:PredictType = {(("ABA", 0),):{"Closure":0.2},
                               (("ABA", 1),):{"Closure":1.0},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score - 2.4 < 0.00001

def test_score6():
    """Test hierarchy"""
    exps:list[ExpType] = [(1, 1.0, (("ABA", 0),), "Closure", "OFF"),
                          (2, 1.0, (("ABA", 1),), "Closure", "ON"),
                          (3, 1.0, (("ABA", 0), ("ROS", 1)), "Closure", "ON"),]
    predictions:PredictType = {(("ABA", 0),):{"Closure":0.2},
                               (("ABA", 1),):{"Closure":1.0},
                               (("ABA", 0), ("ROS", 1),):{"Closure":1.0},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score - 2.4 < 0.00001

def test_score7():
    """Test hierarchy"""
    exps:list[ExpType] = [(1, 1.0, (("ABA", 0),), "Closure", "OFF"),
                          (2, 1.0, (("ABA", 1),), "Closure", "ON"),
                          (3, 1.0, (("ABA", 0), ("ROS", 1)), "Closure", "ON"),
                          (4, 1.0, (("ABA", 1), ("ROS", 1)), "Closure", "ON"),]
    predictions:PredictType = {(("ABA", 0),):{"Closure":0.2},
                               (("ABA", 1),):{"Closure":0.8},
                               (("ABA", 0), ("ROS", 1),):{"Closure":1.0},
                               (("ABA", 1), ("ROS", 1),):{"Closure":0.8},}

    agreements = get_agreement(exps, predictions)
    score = get_hierarchy_score(agreements, {"ABA":0})

    assert score - 2.752 < 0.00001

def test_score_ABA():
    exps, fixes_list = import_exps("../case_study/data/data_20230925.tsv")

    base_primes = sm.format.import_primes("../case_study/baseline_models/ABA_GA_base_A.txt")
    base = Model.import_model(base_primes)
    base.get_predictions(fixes_list)
    base.get_model_score(exps)

    assert base.score == 311.71313474579495

test_score1()
test_score2()
test_score3()
test_score4()
test_score5()
test_score6()
test_score7()
test_score_ABA()