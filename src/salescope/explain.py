"""Explainability and error analysis on held-out rows.

``permutation_importance_table`` needs only scikit-learn. ``shap_summary`` needs the ``explain``
extra (``pip install salescope[explain]``) and works for the tree models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from salescope.data import split_xy
from salescope.metrics import mae, rmse
from salescope.models import build_model
from salescope.splits import holdout_split

# Raw columns that the importance procedure permutes. The identifiers are excluded: the models do
# not use them as features, so a permutation would only change the lookup statistics.
EXPLAINED_COLUMNS = [
    "Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type", "Item_MRP",
    "Outlet_Establishment_Year", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
]


@dataclass
class ExplainResult:
    model: str
    scheme: str
    importance: pd.DataFrame
    errors_by_outlet_type: pd.DataFrame
    errors_by_item_category: pd.DataFrame


def permutation_importance_table(model, X: pd.DataFrame, y: pd.Series, seed: int = 42,
                                 n_repeats: int = 5) -> pd.DataFrame:
    """Increase in RMSE when one raw column is shuffled (mean and std over repeats)."""
    cols = [c for c in EXPLAINED_COLUMNS if c in X.columns]
    rng = np.random.default_rng(seed)
    base = rmse(y, model.predict(X))
    rows = []
    for col in cols:
        scores = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            scores.append(rmse(y, model.predict(shuffled)) - base)
        rows.append({"column": col, "rmse_increase": float(np.mean(scores)), "std": float(np.std(scores))})
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def error_by_group(y: pd.Series, pred, groups: pd.Series) -> pd.DataFrame:
    """RMSE, MAE and mean bias (prediction - actual) for each group value."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float), "g": np.asarray(groups)})
    rows = []
    for g, part in frame.groupby("g"):
        rows.append({"group": g, "rows": len(part), "rmse": rmse(part.y, part.p), "mae": mae(part.y, part.p),
                     "bias": float((part.p - part.y).mean())})
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def explain(df: pd.DataFrame, model: str = "hist_gbm", scheme: str = "new_outlet", seed: int = 42,
            holdout_fraction: float = 0.2, reference_year: int = 2013) -> ExplainResult:
    X, y = split_xy(df)
    dev, hold = holdout_split(X, scheme, holdout_fraction, seed)
    est = build_model(model, seed=seed, reference_year=reference_year).fit(X.iloc[dev], y.iloc[dev])
    X_hold, y_hold = X.iloc[hold], y.iloc[hold]
    pred = est.predict(X_hold)
    category = X_hold["Item_Identifier"].str[:2].map({"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"})
    return ExplainResult(
        model=model,
        scheme=scheme,
        importance=permutation_importance_table(est, X_hold, y_hold, seed=seed),
        errors_by_outlet_type=error_by_group(y_hold, pred, X_hold["Outlet_Type"]),
        errors_by_item_category=error_by_group(y_hold, pred, category.fillna("Other")),
    )


def shap_summary(pipeline, X: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """Mean absolute SHAP value of each model feature. Needs ``shap`` and a fitted tree pipeline."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is not installed. Run: pip install 'salescope[explain]'") from exc
    while hasattr(pipeline, "regressor_"):  # unwrap NonNegative / TransformedTargetRegressor
        pipeline = pipeline.regressor_
    steps = dict(pipeline.named_steps)
    feats = steps["features"].transform(X.iloc[:max_rows])
    encoded = steps["encode"].transform(feats) if "encode" in steps else feats
    explainer = shap.TreeExplainer(steps["model"])
    values = explainer.shap_values(encoded)
    names = list(getattr(encoded, "columns", range(np.asarray(encoded).shape[1])))
    return (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
