"""Model registry. Each entry builds a full pipeline: features -> encoding -> regressor.

The core models need only scikit-learn. LightGBM and CatBoost are optional (``pip install
salescope[boost]``) and are imported only when you build them.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from salescope.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, BigMartFeatures


class OutletTypeMRPBaseline(BaseEstimator, RegressorMixin):
    """Domain baseline: sales = MRP x median(sales / MRP) of the outlet type.

    It needs no tuning and shows how much the learned models add beyond price and store format.
    """

    def fit(self, X: pd.DataFrame, y):
        ratio = np.asarray(y, dtype=float) / X["Item_MRP"].to_numpy(dtype=float)
        frame = pd.DataFrame({"type": X["Outlet_Type"].to_numpy(), "ratio": ratio})
        self.ratio_by_type_ = frame.groupby("type")["ratio"].median().to_dict()
        self.global_ratio_ = float(np.median(ratio))
        return self

    def predict(self, X: pd.DataFrame):
        ratio = X["Outlet_Type"].map(self.ratio_by_type_).fillna(self.global_ratio_).to_numpy(dtype=float)
        return X["Item_MRP"].to_numpy(dtype=float) * ratio


class NonNegative(BaseEstimator, RegressorMixin):
    """Clip the predictions of the wrapped regressor at 0, because sales cannot be negative."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.regressor_.predict(X), 0.0, None)


def _onehot_encoder() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _ordinal_encoder() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _categorical_mask() -> list[bool]:
    return [True] * len(CATEGORICAL_FEATURES) + [False] * len(NUMERIC_FEATURES)


def _mean(seed: int, params: dict) -> list[tuple[str, Any]]:
    return [("model", DummyRegressor(strategy="mean"))]


def _mrp(seed: int, params: dict) -> list[tuple[str, Any]]:
    return [("model", OutletTypeMRPBaseline())]


def _ridge(seed: int, params: dict) -> list[tuple[str, Any]]:
    return [("encode", _onehot_encoder()), ("model", Ridge(**{"alpha": 3.0, **params}))]


def _random_forest(seed: int, params: dict) -> list[tuple[str, Any]]:
    defaults = {"n_estimators": 200, "min_samples_leaf": 10, "max_features": 0.5, "n_jobs": 1}
    return [("encode", _ordinal_encoder()),
            ("model", RandomForestRegressor(random_state=seed, **{**defaults, **params}))]


def _hist_gbm(seed: int, params: dict) -> list[tuple[str, Any]]:
    defaults = {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 8, "min_samples_leaf": 30,
                "l2_regularization": 1.0}
    model = HistGradientBoostingRegressor(random_state=seed, categorical_features=_categorical_mask(),
                                          **{**defaults, **params})
    return [("encode", _ordinal_encoder()), ("model", model)]


def _lightgbm(seed: int, params: dict) -> list[tuple[str, Any]]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError("lightgbm is not installed. Run: pip install 'salescope[boost]'") from exc
    defaults = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 20,
                "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}
    return [("encode", _ordinal_encoder()), ("model", LGBMRegressor(random_state=seed, **{**defaults, **params}))]


def _catboost(seed: int, params: dict) -> list[tuple[str, Any]]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError("catboost is not installed. Run: pip install 'salescope[boost]'") from exc
    defaults = {"iterations": 600, "learning_rate": 0.05, "depth": 6, "verbose": False}
    model = CatBoostRegressor(random_seed=seed, cat_features=CATEGORICAL_FEATURES, **{**defaults, **params})
    return [("model", model)]


REGISTRY: dict[str, Callable[[int, dict], list[tuple[str, Any]]]] = {
    "mean": _mean,
    "mrp_baseline": _mrp,
    "ridge": _ridge,
    "random_forest": _random_forest,
    "hist_gbm": _hist_gbm,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
}
CORE_MODELS = ("mean", "mrp_baseline", "ridge", "random_forest", "hist_gbm")
OPTIONAL_MODELS = ("lightgbm", "catboost")


def build_model(
    name: str,
    seed: int = 42,
    reference_year: int = 2013,
    log_target: bool = False,
    params: dict | None = None,
):
    """Return an unfitted estimator for ``name``. Raises KeyError for an unknown name.

    The result is ``NonNegative(Pipeline([features, (encode), model]))``. With ``log_target`` the
    pipeline is fitted on ``log1p(sales)`` and the predictions are changed back with ``expm1``.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}. Known: {sorted(REGISTRY)}")
    steps = [("features", BigMartFeatures(reference_year=reference_year))]
    steps += REGISTRY[name](seed, dict(params or {}))
    pipe = Pipeline(steps)
    if log_target:
        return NonNegative(TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1))
    return NonNegative(pipe)


def parse_model_list(text: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown model(s) {unknown}. Known: {sorted(REGISTRY)}")
    if not names:
        raise ValueError("no model names given")
    return names
