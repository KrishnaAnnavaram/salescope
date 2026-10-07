"""Hyperparameter search inside grouped cross-validation, then one score on an untouched holdout.

The default backend is a seeded random search over a small space (scikit-learn only). The ``optuna``
backend (``pip install salescope[tune]``) uses the same spaces and the same objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import ParameterSampler

from salescope.data import split_xy
from salescope.evaluate import cross_validate
from salescope.metrics import regression_report
from salescope.models import build_model
from salescope.splits import holdout_split

SEARCH_SPACES: dict[str, dict[str, list]] = {
    "ridge": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "min_samples_leaf": [2, 5, 10, 20, 40],
        "max_features": [0.3, 0.5, 0.8, 1.0],
        "max_depth": [None, 8, 12, 16],
    },
    "hist_gbm": {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_iter": [150, 300, 500],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [10, 20, 40, 80],
        "l2_regularization": [0.0, 0.5, 1.0, 5.0],
    },
    "lightgbm": {
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "min_child_samples": [10, 20, 50],
    },
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "iterations": [300, 600]},
}


@dataclass
class TuneResult:
    model: str
    scheme: str
    best_params: dict
    best_cv_rmse: float
    trials: list[dict] = field(default_factory=list)
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    default_holdout_metrics: dict[str, float] = field(default_factory=dict)


def _objective(dev: pd.DataFrame, model: str, scheme: str, params: dict, n_splits: int, seed: int,
               reference_year: int) -> float:
    res = cross_validate(dev, model, scheme, n_splits, seed, reference_year=reference_year, params=params)
    return res.oof_metrics["rmse"]


def tune(
    df: pd.DataFrame,
    model: str,
    scheme: str = "new_outlet",
    n_iter: int = 10,
    n_splits: int = 5,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    backend: str = "random",
    reference_year: int = 2013,
) -> TuneResult:
    if model not in SEARCH_SPACES:
        raise KeyError(f"no search space for {model!r}. Known: {sorted(SEARCH_SPACES)}")
    space = SEARCH_SPACES[model]
    X, _ = split_xy(df)
    dev_pos, hold_pos = holdout_split(X, scheme, holdout_fraction, seed)
    dev, hold = df.iloc[dev_pos], df.iloc[hold_pos]

    trials: list[dict] = []
    if backend == "random":
        for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
            score = _objective(dev, model, scheme, params, n_splits, seed, reference_year)
            trials.append({"params": dict(params), "cv_rmse": score})
    elif backend == "optuna":
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("optuna is not installed. Run: pip install 'salescope[tune]'") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {k: trial.suggest_categorical(k, v) for k, v in space.items()}
            return _objective(dev, model, scheme, params, n_splits, seed, reference_year)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iter)
        trials = [{"params": t.params, "cv_rmse": float(t.value)} for t in study.trials]
    else:
        raise ValueError(f"unknown backend {backend!r}: use 'random' or 'optuna'")

    best = min(trials, key=lambda t: t["cv_rmse"])
    X_dev, y_dev = split_xy(dev)
    X_hold, y_hold = split_xy(hold)
    tuned = build_model(model, seed=seed, reference_year=reference_year, params=best["params"]).fit(X_dev, y_dev)
    default = build_model(model, seed=seed, reference_year=reference_year).fit(X_dev, y_dev)
    return TuneResult(
        model=model,
        scheme=scheme,
        best_params=best["params"],
        best_cv_rmse=float(best["cv_rmse"]),
        trials=trials,
        holdout_metrics=regression_report(y_hold, tuned.predict(X_hold)),
        default_holdout_metrics=regression_report(y_hold, default.predict(X_hold)),
    )

