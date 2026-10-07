"""Leakage-free stacking with out-of-fold (OOF) predictions.

Procedure:

1. Split the rows into a development part and a holdout part, with the groups of the CV scheme.
2. On the development part, make OOF predictions for each base model (each row is predicted by a
   model that did not see it).
3. Fit the meta-model on the OOF predictions only. The predictions are columns of one DataFrame
   with the row index, so each meta row combines predictions of the same item-outlet pair.
4. Refit each base model on the full development part and score the stack once on the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from salescope.data import split_xy
from salescope.metrics import regression_report
from salescope.models import build_model
from salescope.splits import holdout_split, iter_folds


class AlignmentError(ValueError):
    """Base-model predictions do not refer to the same rows."""


@dataclass
class StackResult:
    scheme: str
    base_models: list[str]
    weights: dict[str, float]
    intercept: float
    holdout_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    n_dev: int = 0
    n_holdout: int = 0
    meta_train_index: pd.Index | None = None
    holdout_index: pd.Index | None = None

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.holdout_metrics).T[["rmse", "rmsle", "mae", "r2"]]


def oof_predictions(X: pd.DataFrame, y: pd.Series, models: dict[str, object], scheme: str,
                    n_splits: int, seed: int) -> pd.DataFrame:
    """Return one OOF prediction column for each model, indexed like ``X``."""
    out = pd.DataFrame(index=X.index, columns=list(models), dtype=float)
    for train_idx, test_idx in iter_folds(X, scheme, n_splits, seed):
        for name, template in models.items():
            est = clone(template).fit(X.iloc[train_idx], y.iloc[train_idx])
            out.iloc[test_idx, out.columns.get_loc(name)] = est.predict(X.iloc[test_idx])
    if out.isna().any().any():  # pragma: no cover - guarded by iter_folds
        raise RuntimeError("missing OOF predictions")
    return out


def align_predictions(preds: dict[str, pd.Series]) -> pd.DataFrame:
    """Join prediction Series by index. Refuses Series that do not cover the same rows."""
    names = list(preds)
    if not names:
        raise ValueError("no predictions to align")
    ref = preds[names[0]].index
    for name in names[1:]:
        idx = preds[name].index
        if len(idx) != len(ref) or not idx.sort_values().equals(ref.sort_values()):
            raise AlignmentError(f"predictions of {name!r} cover other rows than {names[0]!r}")
        if idx.has_duplicates:
            raise AlignmentError(f"predictions of {name!r} have duplicate row labels")
    return pd.DataFrame({n: preds[n].reindex(ref) for n in names})


def stack_evaluate(
    df: pd.DataFrame,
    base_models: list[str],
    scheme: str = "random",
    n_splits: int = 5,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    reference_year: int = 2013,
) -> StackResult:
    if len(base_models) < 2:
        raise ValueError("stacking needs at least 2 base models")
    X, y = split_xy(df)
    dev, hold = holdout_split(X, scheme, holdout_fraction, seed)
    X_dev, y_dev, X_hold, y_hold = X.iloc[dev], y.iloc[dev], X.iloc[hold], y.iloc[hold]
    templates = {m: build_model(m, seed=seed, reference_year=reference_year) for m in base_models}

    meta_X = oof_predictions(X_dev, y_dev, templates, scheme, n_splits, seed)
    meta = LinearRegression(positive=True).fit(meta_X, y_dev.loc[meta_X.index])

    hold_preds: dict[str, pd.Series] = {}
    for name, template in templates.items():
        est = clone(template).fit(X_dev, y_dev)
        hold_preds[name] = pd.Series(est.predict(X_hold), index=X_hold.index)
    hold_X = align_predictions(hold_preds)[meta_X.columns]
    stack_pred = meta.predict(hold_X)

    metrics = {name: regression_report(y_hold, hold_X[name]) for name in base_models}
    metrics["stack"] = regression_report(y_hold, stack_pred)
    return StackResult(
        scheme=scheme,
        base_models=list(base_models),
        weights={n: float(w) for n, w in zip(meta_X.columns, meta.coef_)},
        intercept=float(meta.intercept_),
        holdout_metrics=metrics,
        n_dev=len(dev),
        n_holdout=len(hold),
        meta_train_index=meta_X.index,
        holdout_index=X_hold.index,
    )
