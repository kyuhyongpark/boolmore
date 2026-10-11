"""Config → Evaluator / GAConfig wiring shared by ``run_ga`` and eval scripts."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from boolmore.evaluation.score import Evaluator, get_NAV_scores, get_phenotype_scores
from boolmore.genetic.algorithm import GAConfig
from boolmore.inference.prediction import get_NAV_prediction, get_phenotype_prediction
from boolmore.io.load import import_NAV_exps, import_phenotypes, load_json_config, load_model
from boolmore.io.paths import project_root
from boolmore.model import Model


def build_run_config(
    *,
    json_file: str | None = None,
    run_type: str | None = None,
    data_file: str | None = None,
    base_file: str | None = None,
    start_file: str | None = None,
    stop_if_max: bool | None = None,
    hierarchy: bool | None = None,
) -> dict:
    """Build a run config dict from JSON or inline args."""
    if json_file is not None:
        if any(
            x is not None
            for x in [run_type, data_file, base_file, start_file, stop_if_max, hierarchy]
        ):
            raise ValueError("If json file is given, other parameters should not be given.")

        return load_json_config((project_root() / json_file).resolve())

    if any(x is None for x in [run_type, data_file, base_file]):
        raise ValueError(
            "If json file is not given, run_type, data_file and base_file must be given."
        )

    return {
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
            "necessary": {},
            "group": {},
            "possible_constant": [],
        },
        "edge_pool": [],
        "parameters": {
            "stop_if_max": stop_if_max if stop_if_max is not None else True,
            "starting_gen": 0,
            "total_iter": 10,
            "per_iter": 10,
            "keep": 2,
            "mix": 0,
            "prob": 0.1,
            "edge_prob": 0.5,
            "order_by": ["n_extra_edges", "n_prime_implicants"],
        },
    }


def build_evaluator(
    config: dict | str | Path,
    *,
    verbose: bool = False,
) -> Evaluator:
    """Build a ready ``Evaluator`` from a config dict or JSON path."""
    if not isinstance(config, dict):
        config = load_json_config((project_root() / config).resolve())

    run_type = config["run_type"]
    data_path = (project_root() / config["data"]).resolve()
    base_path = (project_root() / config["base"]).resolve()
    hierarchy = config["hierarchy"]
    default_sources = dict(config["default_sources"])
    generate_defaults = config["generate_defaults"]
    constraints = config["constraints"]
    edge_pool = config["edge_pool"]

    if verbose:
        print(f"Loading base model from {base_path}")
    base_model = load_model(base_path, edge_pool=edge_pool)
    if verbose:
        print("Base model loaded.")

    if verbose:
        print(f"Loading experimental data from {data_path}")
    if run_type == "NAV":
        if generate_defaults:
            for node, primes in base_model.primes.items():
                if primes == [[{node: 0}], [{node: 1}]]:
                    default_sources[node] = 0

        exps = import_NAV_exps(str(data_path))
        prediction_fn = get_NAV_prediction
        score_fn = partial(
            get_NAV_scores,
            default_sources=default_sources,
            hierarchy=hierarchy,
        )
    elif run_type == "Phenotype":
        exps = import_phenotypes(str(data_path))
        prediction_fn = get_phenotype_prediction
        score_fn = get_phenotype_scores
    else:
        raise ValueError(f"Unsupported run type: {run_type}")

    if verbose:
        print("Experimental data loaded.\n")

    return Evaluator(
        base=base_model,
        exps=exps,
        prediction_fn=prediction_fn,
        score_fn=score_fn,
        constraints=constraints,
        run_type=run_type,
        hierarchy=hierarchy,
        default_sources=default_sources,
        generate_defaults=generate_defaults,
    )


def build_ga_config(
    config: dict,
    *,
    seed: int | None = None,
    core: int = 1,
) -> GAConfig:
    """Build ``GAConfig`` from the ``parameters`` section of a run config."""
    parameters = dict(config["parameters"])
    return GAConfig(**parameters, seed=seed, core=core)


def load_start_model(
    evaluator: Evaluator,
    start: str | Path,
    *,
    base: str | Path | None = None,
    verbose: bool = False,
) -> Model:
    """Load the GA starting model, reusing ``evaluator.base`` when paths match."""
    start_path = (project_root() / start).resolve()
    if base is not None and start_path == (project_root() / base).resolve():
        return evaluator.base

    if verbose:
        print(f"Loading starting model from {start_path}")
    start_model = load_model(start_path, base=evaluator.base)
    if verbose:
        print("Starting model loaded.")
    return start_model
