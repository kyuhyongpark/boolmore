"""GA orchestration: config bootstrap, logging, and model export."""

import datetime
import os

from boolmore.boolean_functions import prime2bnet
from boolmore.evaluation.metrics import compare_model_functions
from boolmore.evaluation.score import format_score
from boolmore.io.export import export_model
from boolmore.io.paths import project_root
from boolmore.genetic.algorithm import GAState, ga_main
from boolmore.genetic.population import Candidate
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.selection import Selector
from boolmore.run_setup import (
    build_evaluator,
    build_ga_config,
    build_run_config,
    load_start_model,
)


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
    config_dict = build_run_config(
        json_file=json_file,
        run_type=run_type,
        data_file=data_file,
        base_file=base_file,
        start_file=start_file,
        stop_if_max=stop_if_max,
        hierarchy=hierarchy,
    )

    evaluator = build_evaluator(config_dict, verbose=True)
    ga = build_ga_config(config_dict, seed=seed, core=core)

    data_path = (project_root() / config_dict["data"]).resolve()
    base_path = (project_root() / config_dict["base"]).resolve()
    start_path = (project_root() / config_dict["start"]).resolve()

    if run_name is None:
        run_name = start_path.stem

    LOG = run_name + "_log.txt"

    if export_name is None:
        export_name = run_name

    # ---------- Load starting model ----------
    start_model = load_start_model(
        evaluator,
        config_dict["start"],
        base=config_dict["base"],
        verbose=True,
    )

    # ---------- Evaluate base and start models ----------
    start_single = datetime.datetime.now()
    base_result = evaluator.evaluate(evaluator.base)
    base = Candidate(model=evaluator.base, eval_result=base_result)
    end_single = datetime.datetime.now()
    print(base.model.info())
    print(f"{format_score(base.eval_result)}\n")
    print(
        f"\tElapsed time for single evaluation: {end_single-start_single}\n"
        f"\tEstimated GA evaluation time: {(end_single-start_single)*ga.total_iter*ga.per_iter/core}\n"
        )

    start_result = evaluator.evaluate(start_model, id=0)
    start = Candidate(start_model, 0, ga.starting_gen, start_result)
    print(start.model.info())
    print(format_score(start.eval_result))
    print()

    # ---------- Run genetic algorithm ----------
    start_time = datetime.datetime.now()

    selector = Selector(keep=ga.keep, order_by=ga.order_by)
    reproducer = Reproducer(selector=selector, constraints=evaluator.constraints)
    states = ga_main(start, evaluator, selector, reproducer, ga)

    end_time = datetime.datetime.now()

    print()
    export_models(states, export_name, export_top, export_thresh, export_same)

    final = states[-1].population[0]

    # ---------- Write log ----------
    with open(LOG, "w") as fp:
        fp.write(f"# RUN_TYPE={evaluator.run_type!r}\n")
        fp.write(f"# DATA: {data_path}\n")
        fp.write(f"# HIERARCHY={evaluator.hierarchy}\n")
        fp.write(f"# DEFAULT_SOURCES={evaluator.default_sources}\n")
        fp.write(f"# GENERATE_DEFAULTS={evaluator.generate_defaults}\n")
        fp.write(f"# CONSTRAINTS={evaluator.constraints}\n")
        fp.write(f"# EDGE_POOL={evaluator.base.edge_pool}\n\n")

        fp.write(f"# seed: {ga.seed}\n")
        fp.write(f"# stop if max: {ga.stop_if_max}\n")
        fp.write(f"# starting gen: {ga.starting_gen}\n")
        fp.write(f"# total iterations: {ga.total_iter}\n")
        fp.write(f"# per iteration: {ga.per_iter}\n")
        fp.write(f"# keep: {ga.keep}\n")
        fp.write(f"# mix: {ga.mix}\n")
        fp.write(f"# prob: {ga.prob}\n")
        fp.write(f"# edge prob: {ga.edge_prob}\n")
        fp.write(f"# order by: {ga.order_by}\n\n")

        fp.write(f"# {core=}\n\n")

        log_candidate(fp, base, "BASE", str(base_path))
        log_candidate(fp, start, "START", str(start_path))

        fp.write(f"\n# {start_time=}\n")
        fp.write(f"# {end_time=}\n")
        fp.write(f"# elapsed time: {end_time-start_time}\n\n")

        for state in states[1:]:
            candidate = state.population[0]
            fp.write(candidate.summary(ga.order_by) + "\n")
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
