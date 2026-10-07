"""Cross-validated evaluation. Every reported test number comes from rows that the model did not see."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from salescope.data import split_xy
from salescope.metrics import METRIC_NAMES, bootstrap_ci, regression_report
from salescope.models import build_model
from salescope.splits import effective_splits, iter_folds


@dataclass
class CVResult:
    model: str
    scheme: str
    n_splits: int
    oof: pd.Series  # out-of-fold prediction for each row, same index as the input
    fold_test: list[dict[str, float]] = field(default_factory=list)
    fold_train: list[dict[str, float]] = field(default_factory=list)
    oof_metrics: dict[str, float] = field(default_factory=dict)
    rmse_ci: tuple[float, float] = (float("nan"), float("nan"))

    def summary_row(self) -> dict[str, float | str | int]:
        row: dict[str, float | str | int] = {"model": self.model, "scheme": self.scheme, "folds": self.n_splits}
        test = pd.DataFrame(self.fold_test)
        train = pd.DataFrame(self.fold_train)
        for m in METRIC_NAMES:
            row[f"{m}_mean"] = float(test[m].mean())
            row[f"{m}_std"] = float(test[m].std(ddof=1)) if len(test) > 1 else 0.0
        row["train_rmse_mean"] = float(train["rmse"].mean())
        for m in METRIC_NAMES:
            row[f"oof_{m}"] = self.oof_metrics[m]
        row["oof_rmse_ci_low"], row["oof_rmse_ci_high"] = self.rmse_ci
        return row


def cross_validate(
    df: pd.DataFrame,
    model: str,
    scheme: str = "random",
    n_splits: int = 5,
    seed: int = 42,
    log_target: bool = False,
    reference_year: int = 2013,
    params: dict | None = None,
) -> CVResult:
    """Fit a fresh pipeline in each fold and predict the held-out rows.

    The preprocessing (imputation maps, encoders, scaler) is part of the pipeline, so it is fitted on
    the training rows of each fold only. Train-fold metrics are kept apart and labelled ``train``.
    """
    X, y = split_xy(df)
    template = build_model(model, seed=seed, reference_year=reference_year, log_target=log_target, params=params)
    oof = pd.Series(np.nan, index=df.index, dtype=float)
    result = CVResult(model=model, scheme=scheme, n_splits=effective_splits(X, scheme, n_splits), oof=oof)
    for train_idx, test_idx in iter_folds(X, scheme, n_splits, seed):
        est = clone(template)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_test = est.predict(X.iloc[test_idx])
        pred_train = est.predict(X.iloc[train_idx])
        oof.iloc[test_idx] = pred_test
        result.fold_test.append(regression_report(y.iloc[test_idx], pred_test))
        result.fold_train.append(regression_report(y.iloc[train_idx], pred_train))
    if oof.isna().any():  # pragma: no cover - guarded by iter_folds
        raise RuntimeError("some rows have no out-of-fold prediction")
    result.oof_metrics = regression_report(y, oof)
    result.rmse_ci = bootstrap_ci(y, oof, "rmse", seed=seed)
    return result


def compare(
    df: pd.DataFrame,
    models: list[str],
    schemes: list[str],
    n_splits: int = 5,
    seed: int = 42,
    log_target: bool = False,
    reference_year: int = 2013,
) -> pd.DataFrame:
    """Cross-validate each model under each scheme and return one summary row for each pair."""
    rows = [
        cross_validate(df, m, s, n_splits, seed, log_target, reference_year).summary_row()
        for s in schemes
        for m in models
    ]
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame) -> str:
    cols = ["scheme", "model", "folds", "rmse_mean", "rmse_std", "oof_rmse", "oof_rmse_ci_low",
            "oof_rmse_ci_high", "oof_rmsle", "oof_mae", "oof_r2", "train_rmse_mean"]
    shown = table[cols].copy()
    for c in cols[3:]:
        shown[c] = shown[c].map(lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}")
    return shown.to_string(index=False)
