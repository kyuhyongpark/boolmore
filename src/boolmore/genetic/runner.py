"""GA orchestration: config bootstrap, logging, and model export."""

import datetime
import os
from pathlib import Path

from boolmore.boolean_functions import prime2bnet
from boolmore.evaluation.metrics import compare_model_functions
from boolmore.evaluation.score import format_score
from boolmore.io.export import export_model
from boolmore.io.load import load_config, load_model
from boolmore.genetic.algorithm import GAConfig, GAState, ga_main
from boolmore.genetic.population import Candidate
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.selection import Selector
from boolmore.run_setup import build_run_setup


def log_candidate(fp, candidate: Candidate, label, path):
    fp.write(f"\n# {label}: {os.path.abspath(path)}\n")
    fp.write(candidate.info() + "\n")
    fp.write("# targets,\tfactors\n")

    bnet = candidate.model.bnet
    for line in bnet.splitlines():
        fp.write("# " + line + "\n")


def export_models(
        states: list[GAState],
        export_name: str,
        export_top: int = 0,
        export_thresh: float = 0.0,
        export_same: bool = False,
        ):
    final = states[-1].population[0]

    # Always export the final best model
    export_model(final.model, file_name=f"{export_name}_{final.id}_gen{final.generation}")

    # Export top models from each generation
    if export_top:
        for state in states[1:]:
            for candidate in state.population[:export_top]:
                if candidate.eval_result.score > export_thresh:
                    export_model(candidate.model, file_name=f"{export_name}_{candidate.id}_gen{candidate.generation}")

    # Export all models tied with the best model in the final generation
    if export_same:
        for candidate in states[-1].population:
            if candidate.eval_result.score == final.eval_result.score:
                export_model(candidate.model, filename=f"{export_name}_{candidate.id}_gen{candidate.generation}")


def run_ga(
        json_file:str|None=None,
        run_type:str|None=None,
        data_file:str|None=None,
        base_file:str|None=None,
        start_file:str|None=None,
        stop_if_max:bool|None=None,
        hierarchy:bool|None=None,
        seed:int|None=None,
        run_name:str|None=None,
        export_top:int=0,
        export_thresh:float=0.0,
        export_name:str|None=None,
        export_same:bool=False,
        core:int=1,
        )-> tuple[Candidate, Candidate, list[GAState]]:
    """
    Imports parameters, experiments, base model in the json file.
    Runs genetic algorithm and exports refined models.
    
    Instead of json file, locations of the data and base model can be used.
    In this case, constraints or extra edges are inapplicable.

    Parameters
    ----------
    json_file : str | None
        location of the json file containing parameters
    
    Optional
    --------
    run_type : str | None
        "Phenotype" or "NAV"
    data_file : str | None
        location of the data file that overrides the json file.
        Must be given if json_file is not given.
    base_file : str | None
        location of the base file that overrides the json file.
        Must be given if json_file is not given.
    start_file : str | None
        location of the bnet file of the starting model
        if None, base_model is the starting model
    stop_if_max : bool
        if True, stop when the max score is reached
    hierarchy : bool
        if True, hierarchy scoring is used

    seed : int | None
        random seed for reproducibility
    run_name : str | None
        models are exported as (run_name)_id_gen.bnet
        log is exported as (run_name)_log.txt
        if None, takes the start_model name
    export_top : int
        export top models from each generation
    export_thresh : float
        export models with scores higher than this
    export_name : str | None
        name of the exported models
    export_same : bool
        if True, export models with the same score for the final generation
    core : int
        if larger than 1, model evaluation is done in parallel

    Returns
    -------
    base : Candidate
        the base model and its evaluation
    start : Candidate
        the starting model and its evaluation
    states : list[GAState]
        the states of the genetic algorithm

    """
    # ---------- Load run configuration ----------
    if json_file is not None:
        if any(x is not None for x in [run_type, data_file, base_file, start_file, stop_if_max, hierarchy]):
            raise ValueError("If json file is given, other parameters should not be given.")

        config_path = Path(json_file).resolve()
        config_dict = load_config(config_path)
        config_dir = config_path.parent
    else:
        if any(x is None for x in [run_type, data_file, base_file]):
            raise ValueError("If json file is not given, run_type, data_file and base_file must be given.")

        config_path = None
        config_dir = Path.cwd()
        config_dict = {
            "run_type": run_type,
            "data": data_file,
            "base": base_file,
            "start": start_file if start_file is not None else base_file,
            "hierarchy": hierarchy if hierarchy is not None else True,
            "default_sources": {},
            "generate_defaults": True,
            "constraints": {
                "fixed": [],
                "regulate": {},
                "necessary" : {},
                "group": {},
                "possible_constant": []
                },
            "edge_pool": [],
            "parameters" : {
                "stop_if_max": stop_if_max if stop_if_max is not None else True,
                "starting_gen": 0,
                "total_iter": 10,
                "per_iter" : 10,
                "keep" : 2,
                "mix" : 0,
                "prob" : 0.1,
                "edge_prob" : 0.5,
                "order_by" : ["n_extra_edges", "n_prime_implicants"]
                }
            }

    setup = build_run_setup(
        config_dict,
        config_dir=config_dir,
        config_path=config_path,
        verbose=True,
    )

    parameters = (setup.parameters or config_dict["parameters"]).copy()
    STARTING_GEN = parameters.pop("starting_gen")
    config = GAConfig(**parameters, seed=seed, core=core)

    if run_name is None:
        run_name = setup.start_path.stem

    LOG = run_name + "_log.txt"

    if export_name is None:
        export_name = run_name

    # ---------- Load starting model ----------
    if setup.start_path.resolve() == setup.base_path.resolve():
        start_model = setup.base_model
    else:
        print(f"Loading starting model from {setup.start_path}")
        start_model = load_model(setup.start_path, base=setup.base_model)
        print("Starting model loaded.")

    # ---------- Evaluate base and start models ----------
    start_single = datetime.datetime.now()
    _, base_result = setup.evaluate(setup.base_model)
    base = Candidate(model=setup.base_model, eval_result=base_result)
    end_single = datetime.datetime.now()
    print(base.model.info())
    print(f"{format_score(base.eval_result)}\n")
    print(
        f"\tElapsed time for single evaluation: {end_single-start_single}\n"
        f"\tEstimated GA evaluation time: {(end_single-start_single)*config.total_iter*config.per_iter/core}\n"
        )

    _, start_result = setup.evaluate(start_model, id=0)
    start = Candidate(start_model, 0, STARTING_GEN, start_result)
    print(start.model.info())
    print(format_score(start.eval_result))
    print()

    # ---------- Run genetic algorithm ----------
    start_time = datetime.datetime.now()

    selector = Selector(keep=config.keep, order_by=config.order_by)
    reproducer = Reproducer(selector=selector, constraints=setup.constraints)
    states = ga_main(start, setup.evaluator, selector, reproducer, config)

    end_time = datetime.datetime.now()

    print()
    export_models(states, export_name, export_top, export_thresh, export_same)

    final = states[-1].population[0]

    # ---------- Write log ----------
    with open(LOG, "w") as fp:
        fp.write(f"# RUN_TYPE={setup.run_type!r}\n")
        fp.write(f"# DATA: {setup.data_path}\n")
        fp.write(f"# HIERARCHY={setup.hierarchy}\n")
        fp.write(f"# DEFAULT_SOURCES={setup.default_sources}\n")
        fp.write(f"# GENERATE_DEFAULTS={setup.generate_defaults}\n")
        fp.write(f"# CONSTRAINTS={setup.constraints}\n")
        fp.write(f"# EDGE_POOL={setup.edge_pool}\n\n")

        fp.write(f"# seed: {config.seed}\n")
        fp.write(f"# stop if max: {config.stop_if_max}\n")
        fp.write(f"# starting gen: {STARTING_GEN}\n")
        fp.write(f"# total iterations: {config.total_iter}\n")
        fp.write(f"# per iteration: {config.per_iter}\n")
        fp.write(f"# keep: {config.keep}\n")
        fp.write(f"# mix: {config.mix}\n")
        fp.write(f"# prob: {config.prob}\n")
        fp.write(f"# edge prob: {config.edge_prob}\n")
        fp.write(f"# order by: {config.order_by}\n\n")

        fp.write(f"# {core=}\n\n")

        log_candidate(fp, base, "BASE", str(setup.base_path))
        log_candidate(fp, start, "START", str(setup.start_path))

        fp.write(f"\n# {start_time=}\n")
        fp.write(f"# {end_time=}\n")
        fp.write(f"# elapsed time: {end_time-start_time}\n\n")

        for state in states[1:]:
            candidate = state.population[0]
            fp.write(candidate.summary(config.order_by) + "\n")
        log_candidate(fp, final, "FINAL", f"{export_name}_{final.id}_gen{final.generation}.bnet")

    # ---------- Analyze and report results ----------
    differences = compare_model_functions(start.model, final.model)

    print(f"""
        The algorithm ran for {len(states)-1} iterations,
        generating {states[-1].generated} models.
        Mutated {len(differences)} functions, 
        and increased score from {round(start.eval_result.score,2)} / {start.eval_result.max_score} ({round(start.eval_result.score/start.eval_result.max_score*100,1)}%)
        to {round(final.eval_result.score,2)} / {final.eval_result.max_score} ({round(final.eval_result.score/final.eval_result.max_score*100,1)}%).\n
        Total elapsed time: {end_time-start_time}""")
    print()

    print("-----modified functions-----")
    for node, (prime1, prime2) in differences.items():
        print("start:" + prime2bnet(node, prime1))
        print("final:" + prime2bnet(node, prime2))

    return base, start, states
