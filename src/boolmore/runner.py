"""GA orchestration: config bootstrap, logging, and model export."""

from __future__ import annotations

import csv
import datetime
import os
import sys
from collections.abc import Sequence
from typing import TextIO

from boolmore.boolean_functions import prime2bnet
from boolmore.evaluation.metrics import compare_model_functions
from boolmore.io.export import export_candidate
from boolmore.io.paths import project_root
from boolmore.genetic.algorithm import GAState, ga_main
from boolmore.genetic.population import Candidate
from boolmore.genetic.operators.reproduction import Reproducer
from boolmore.genetic.selection import Selector
from boolmore.io.load import load_model
from boolmore.run_setup import (
    build_evaluator,
    build_ga_config,
    build_run_config,
)

_EVAL_CSV_FIELDS = (
    "iteration",
    "id",
    "generation",
    "score",
    "max_score",
    "is_upper_bound",
    "n_prime_implicants",
    "n_eff_edges",
    "n_eff_self_edges",
    "n_extra_edges",
    "satisfies_constraints",
)


class RunLogger:
    """Mirror every stdout write to a log file."""

    def __init__(self, path: str):
        self.path = path
        self._fp: TextIO = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self  # type: ignore[assignment]

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._fp.write(data)
        self._fp.flush()
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._fp.flush()

    def isatty(self) -> bool:
        return self._stdout.isatty()

    def close(self) -> None:
        if sys.stdout is self:
            sys.stdout = self._stdout
        self._fp.close()


def log_candidate(candidate: Candidate, label: str, path: str) -> None:
    """Print candidate path, eval/model summary, and bnet rules."""
    print(f"\n{label}: {os.path.abspath(path)}")
    print(candidate.info())
    print("targets,\tfactors")
    print(candidate.model.bnet)


def _rule_rhs(node: str, primes) -> str:
    """Boolean rule body for ``node`` (without the ``node,\\t`` prefix)."""
    line = prime2bnet(node, primes)
    _, _, rhs = line.partition(",\t")
    return rhs if rhs else line


def _print_function_block(node: str, rows: list[tuple[str, str]]) -> None:
    """Print one node's compared rules with aligned labels."""
    print(f"\n  {node}")
    width = max(len(label) for label, _ in rows) + 1  # include ':'
    for label, text in rows:
        print(f"    {(label + ':'):<{width}} {text}")


def init_eval_csv(path: str) -> None:
    """Create/overwrite the per-model eval CSV with a header row."""
    with open(path, "w", newline="", encoding="utf-8") as fp:
        csv.DictWriter(fp, fieldnames=_EVAL_CSV_FIELDS).writeheader()


def append_generation_evals(path: str, state: GAState) -> None:
    """Append one summary row per model in ``state.population`` (no item_scores)."""
    rows = []
    for candidate in state.population:
        result = candidate.eval_result
        if result is None:
            continue
        rows.append({
            "iteration": state.iteration,
            "id": candidate.id,
            "generation": candidate.generation,
            "score": result.score,
            "max_score": result.max_score,
            "is_upper_bound": result.is_upper_bound,
            "n_prime_implicants": result.n_prime_implicants,
            "n_eff_edges": result.n_eff_edges,
            "n_eff_self_edges": result.n_eff_self_edges,
            "n_extra_edges": result.n_extra_edges,
            "satisfies_constraints": result.satisfies_constraints,
        })
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as fp:
        csv.DictWriter(fp, fieldnames=_EVAL_CSV_FIELDS).writerows(rows)


def export_models(
        states: list[GAState],
        run_name: str,
        export_top: int = 0,
        export_thresh: float = 0.0,
        export_same: bool = False,
        ):
    final = states[-1].population[0]

    # Always export the final best model
    export_candidate(final, run_name)

    # Export top models from each generation
    if export_top:
        for state in states[1:]:
            for candidate in state.population[:export_top]:
                if candidate.eval_result.score > export_thresh:
                    export_candidate(candidate, run_name)

    # Export all models tied with the best model in the final generation
    if export_same:
        for candidate in states[-1].population:
            if candidate.eval_result.score == final.eval_result.score:
                export_candidate(candidate, run_name)


def run_ga(
        json_file:str|None=None,
        run_type:str|None=None,
        data_file:str|None=None,
        base_file:str|None=None,
        start_file:str|None=None,
        stop_if_max:bool|None=None,
        hierarchy:bool|None=None,
        nav_method:str|None=None,
        seed:int|None=None,
        run_name:str|None=None,
        export_top:int=0,
        export_thresh:float=0.0,
        export_same:bool=False,
        core:int=1,
        id_col: str | Sequence[str] | None = None,
        weight_col: str | Sequence[str] | None = None,
        observed_col: str | Sequence[str] | None = None,
        outcome_col: str | Sequence[str] | None = None,
        fixes_cols: Sequence[str] | None = None,
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
    nav_method : str | None
        NAV prediction method ("min_traps" or "ldoi"); default min_traps

    seed : int | None
        random seed for reproducibility
    run_name : str | None
        Prefix for all run outputs: ``(run_name)_log.txt``,
        ``(run_name)_evals.csv``, and exported ``.bnet`` / ``.pkl`` /
        ``_score.csv`` files. If None, defaults to
        ``{start_stem}_{YYYYMMDD_HHMMSS}`` so successive runs do not
        overwrite each other. Pass an explicit name (e.g. ``"demo"``)
        to overwrite a fixed set of files.
    export_top : int
        export top models from each generation
    export_thresh : float
        export models with scores higher than this
    export_same : bool
        if True, export models with the same score for the final generation
    core : int
        if larger than 1, model evaluation is done in parallel
    id_col, weight_col, observed_col, outcome_col, fixes_cols
        Optional NAV CSV column names forwarded to ``build_evaluator`` /
        ``import_NAV_exps``. Leave as ``None`` to use import defaults.

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
        nav_method=nav_method,
    )

    data_path = (project_root() / config_dict["data"]).resolve()
    base_path = (project_root() / config_dict["base"]).resolve()
    start_path = (project_root() / config_dict["start"]).resolve()

    if run_name is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{start_path.stem}_{stamp}"

    LOG = run_name + "_log.txt"
    EVAL_CSV = run_name + "_evals.csv"

    logger = RunLogger(LOG)
    try:
        ga = build_ga_config(config_dict, seed=seed, core=core)
        init_eval_csv(EVAL_CSV)

        # ---------- General setup (before any loading) ----------
        print(f"RUN_TYPE={config_dict['run_type']!r}")
        print(f"DATA: {data_path}")
        print(f"BASE: {base_path}")
        print(f"START: {start_path}")
        print(f"HIERARCHY={config_dict['hierarchy']}")
        print(f"NAV_METHOD={config_dict.get('nav_method', 'min_traps')!r}")
        print(f"LDOI_PRUNE={config_dict.get('ldoi_prune', True)}")
        print(f"DEFAULT_SOURCES={config_dict.get('default_sources', {})}")
        print(f"GENERATE_DEFAULTS={config_dict.get('generate_defaults', True)}")
        print(f"CONSTRAINTS={config_dict.get('constraints')}")
        print(f"EDGE_POOL={config_dict.get('edge_pool')}")
        print()
        print(f"seed: {ga.seed}")
        print(f"stop if max: {ga.stop_if_max}")
        print(f"starting gen: {ga.starting_gen}")
        print(f"total iterations: {ga.total_iter}")
        print(f"per iteration: {ga.per_iter}")
        print(f"keep: {ga.keep}")
        print(f"mix: {ga.mix}")
        print(f"prob: {ga.prob}")
        print(f"edge prob: {ga.edge_prob}")
        print(f"order by: {ga.order_by}")
        print(f"{core=}")
        print()

        # Loads base then experimental data (prints load messages + data stats)
        evaluator = build_evaluator(
            config_dict,
            verbose=True,
            id_col=id_col,
            weight_col=weight_col,
            observed_col=observed_col,
            outcome_col=outcome_col,
            fixes_cols=fixes_cols,
        )

        # ---------- Evaluate and print base (same detail as START/FINAL) ----------
        start_single = datetime.datetime.now()
        base_result = evaluator.evaluate(evaluator.base)
        base = Candidate(model=evaluator.base, eval_result=base_result)
        end_single = datetime.datetime.now()
        log_candidate(base, "BASE", str(base_path))
        print(
            f"Elapsed time for single evaluation: {end_single-start_single}\n"
            f"Estimated GA evaluation time: "
            f"{(end_single-start_single)*ga.total_iter*ga.per_iter/core}\n"
        )

        # ---------- Load and evaluate starting model ----------
        if start_path == base_path:
            print("Starting model is the base model.")
            start_model = evaluator.base
        else:
            print(f"Loading starting model from {start_path}")
            start_model = load_model(start_path, base=evaluator.base)
            print("Starting model loaded.")

        start_result = evaluator.evaluate(start_model, model_id=0)
        start = Candidate(start_model, 0, ga.starting_gen, start_result)
        log_candidate(start, "START", str(start_path))
        print()

        # ---------- Run genetic algorithm ----------
        start_time = datetime.datetime.now()
        print(f"start_time: {start_time}\n")

        def on_generation(state: GAState) -> None:
            append_generation_evals(EVAL_CSV, state)

        selector = Selector(keep=ga.keep, order_by=ga.order_by)
        reproducer = Reproducer(selector=selector, constraints=evaluator.constraints)
        states = ga_main(
            start, evaluator, selector, reproducer, ga, on_generation=on_generation
        )
        print(f"Eval summaries: {os.path.abspath(EVAL_CSV)}")

        end_time = datetime.datetime.now()

        print()
        export_models(states, run_name, export_top, export_thresh, export_same)

        final = states[-1].population[0]
        log_candidate(
            final,
            "FINAL",
            f"{run_name}_{final.id}_gen{final.generation}.bnet",
        )

        # ---------- Analyze and report results ----------
        start_vs_final = compare_model_functions(start.model, final.model)

        print(
            f"Ran {len(states)-1} iterations, generated {states[-1].generated} models, "
            f"mutated {len(start_vs_final)} functions. "
            f"Elapsed: {end_time-start_time}"
        )
        print()

        print("-----modified functions-----")
        if start_path == base_path:
            for node, (start_prime, final_prime) in start_vs_final.items():
                _print_function_block(node, [
                    ("start", _rule_rhs(node, start_prime)),
                    ("final", _rule_rhs(node, final_prime)),
                ])
        else:
            base_vs_start = compare_model_functions(base.model, start.model)
            base_vs_final = compare_model_functions(base.model, final.model)
            differing_nodes = set(base_vs_start) | set(start_vs_final) | set(base_vs_final)
            for node in base.model.primes:
                if node not in differing_nodes:
                    continue
                if node not in base_vs_start:
                    start_text = "(same as base)"
                else:
                    start_text = _rule_rhs(node, start.model.primes[node])
                if node not in base_vs_final:
                    final_text = "(same as base)"
                elif node not in start_vs_final:
                    final_text = "(same as start)"
                else:
                    final_text = _rule_rhs(node, final.model.primes[node])
                _print_function_block(node, [
                    ("base", _rule_rhs(node, base.model.primes[node])),
                    ("start", start_text),
                    ("final", final_text),
                ])
            print(
                f"\n{len(differing_nodes)} out of {len(base.model.primes)} "
                f"functions differ among base, start, and final"
            )

        return base, start, states
    finally:
        logger.close()
