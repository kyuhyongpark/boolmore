"""Config → base model + Evaluator wiring shared by ``run_ga`` and eval scripts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from boolmore.evaluation.score import (
    Evaluator,
    EvalResult,
    get_NAV_scores,
    get_phenotype_scores,
)
from boolmore.inference.prediction import get_NAV_prediction, get_phenotype_prediction
from boolmore.io.load import import_NAV_exps, import_phenotypes, load_config, load_model
from boolmore.io.paths import resolve_under
from boolmore.model import Model


@dataclass
class RunSetup:
    """Models, experiments, and evaluator for one config-driven run."""

    run_type: str
    hierarchy: bool
    generate_defaults: bool
    default_sources: dict[str, int]
    constraints: dict
    edge_pool: list
    data_path: Path
    base_path: Path
    start_path: Path
    base_model: Model
    evaluator: Evaluator
    config_dir: Path
    config_path: Path | None = None
    parameters: dict | None = None

    def evaluate(
        self,
        model: Model | str | Path,
        id: int = -1,
    ) -> tuple[Model, EvalResult]:
        """Evaluate a model (or model file) with this setup's Evaluator."""
        if not isinstance(model, Model):
            model = load_model(model, base=self.base_model)
        return model, self.evaluator.evaluate(model, id)

    def write_nav_report(self, result: EvalResult, file: str | Path) -> Path:
        """Write a per-experiment NAV score TSV using predictions from ``result``."""
        if self.run_type != "NAV":
            raise ValueError("Detailed report is only supported for NAV run_type.")

        report_path = Path(file)
        predictions = result.details[0]
        get_NAV_scores(
            self.evaluator.exps,
            predictions,
            default_sources=self.default_sources,
            hierarchy=self.hierarchy,
            report=True,
            file=str(report_path),
        )
        return report_path.resolve()


def build_run_setup(
    config: dict,
    *,
    config_dir: str | Path,
    config_path: str | Path | None = None,
    verbose: bool = False,
) -> RunSetup:
    """Build Evaluator + base model from a config dict (same wiring as ``run_ga``)."""
    config_dir = Path(config_dir).resolve()
    config_path = Path(config_path).resolve() if config_path is not None else None

    run_type = config["run_type"]
    data_path = resolve_under(config_dir, config["data"])
    base_path = resolve_under(config_dir, config["base"])
    start_path = resolve_under(config_dir, config["start"])
    hierarchy = config["hierarchy"]
    default_sources = dict(config["default_sources"])
    generate_defaults = config["generate_defaults"]
    constraints = config["constraints"]
    edge_pool = config["edge_pool"]
    parameters = config.get("parameters")

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

    evaluator = Evaluator(
        exps=exps,
        prediction_fn=prediction_fn,
        score_fn=score_fn,
        constraints=constraints,
    )
    if verbose:
        print("Experimental data loaded.\n")

    return RunSetup(
        run_type=run_type,
        hierarchy=hierarchy,
        generate_defaults=generate_defaults,
        default_sources=default_sources,
        constraints=constraints,
        edge_pool=edge_pool,
        data_path=data_path,
        base_path=base_path,
        start_path=start_path,
        base_model=base_model,
        evaluator=evaluator,
        config_dir=config_dir,
        config_path=config_path,
        parameters=dict(parameters) if parameters is not None else None,
    )


def load_run_setup(config_path: str | Path, *, verbose: bool = False) -> RunSetup:
    """Load a JSON config file and build a ``RunSetup``."""
    config_path = Path(config_path).resolve()
    return build_run_setup(
        load_config(config_path),
        config_dir=config_path.parent,
        config_path=config_path,
        verbose=verbose,
    )
