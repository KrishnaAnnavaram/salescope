"""BigMart feature engineering as a scikit-learn transformer.

Every statistic (item weight, item visibility, outlet size by type) is learned in ``fit`` from the
training rows only, so the transformer is safe inside cross-validation. The identifiers are used
to look up these statistics, but they never become model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from salescope.schema import ID_COL, OUTLET_COL, TARGET

CATEGORY_BY_PREFIX = {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}

NUMERIC_FEATURES = [
    "Item_Weight",
    "Item_Visibility",
    "Visibility_Ratio",
    "Item_MRP",
    "Outlet_Age",
    "Visibility_Was_Zero",
    "Weight_Was_Missing",
    "Outlet_Size_Was_Missing",
]
CATEGORICAL_FEATURES = [
    "Item_Fat_Content",
    "Item_Type",
    "Item_Category",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]


class LeakageError(ValueError):
    """The target column reached the feature transformer."""


class BigMartFeatures(BaseEstimator, TransformerMixin):
    """Fix the known BigMart data problems and derive domain features.

    Parameters
    ----------
    reference_year:
        The year that the sales were recorded. ``Outlet_Age = reference_year - establishment year``.
    include_outlet_id:
        Keep ``Outlet_Identifier`` as a categorical feature. Off by default because it cannot
        generalise to a new outlet.
    """

    def __init__(self, reference_year: int = 2013, include_outlet_id: bool = False):
        self.reference_year = reference_year
        self.include_outlet_id = include_outlet_id

    # -- fit -------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        self._guard(X)
        weight = X.groupby(ID_COL)["Item_Weight"].median()
        self.item_weight_ = weight.dropna().to_dict()
        self.global_weight_ = float(X["Item_Weight"].median())
        visible = X[X["Item_Visibility"] > 0]
        self.item_visibility_ = visible.groupby(ID_COL)["Item_Visibility"].mean().to_dict()
        self.global_visibility_ = float(visible["Item_Visibility"].mean()) if len(visible) else 0.05
        sizes = X.dropna(subset=["Outlet_Size"])
        self.size_by_type_ = (
            sizes.groupby("Outlet_Type")["Outlet_Size"].agg(lambda s: s.mode().iloc[0]).to_dict()
        )
        self.global_size_ = sizes["Outlet_Size"].mode().iloc[0] if len(sizes) else "Medium"
        self.feature_names_out_ = list(NUMERIC_FEATURES) + self.categorical_features
        return self

    @property
    def categorical_features(self) -> list[str]:
        return list(CATEGORICAL_FEATURES) + ([OUTLET_COL] if self.include_outlet_id else [])

    # -- transform -------------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._guard(X)
        if not hasattr(self, "item_weight_"):
            raise RuntimeError("BigMartFeatures is not fitted")
        ids = X[ID_COL].astype(str)
        out = pd.DataFrame(index=X.index)

        weight = X["Item_Weight"].astype(float)
        out["Weight_Was_Missing"] = weight.isna().astype(float)
        weight = weight.fillna(ids.map(self.item_weight_)).fillna(self.global_weight_)
        out["Item_Weight"] = weight

        vis = X["Item_Visibility"].astype(float)
        item_mean = ids.map(self.item_visibility_).astype(float).fillna(self.global_visibility_)
        zero = vis <= 0
        out["Visibility_Was_Zero"] = zero.astype(float)
        vis = vis.where(~zero, item_mean)
        out["Item_Visibility"] = vis
        out["Visibility_Ratio"] = vis / item_mean.replace(0, np.nan).fillna(self.global_visibility_)

        out["Item_MRP"] = X["Item_MRP"].astype(float)
        out["Outlet_Age"] = (self.reference_year - X["Outlet_Establishment_Year"].astype(float)).clip(lower=0)

        size = X["Outlet_Size"]
        out["Outlet_Size_Was_Missing"] = size.isna().astype(float)
        fallback = X["Outlet_Type"].map(self.size_by_type_).fillna(self.global_size_)
        out["Outlet_Size"] = size.where(size.notna(), fallback).astype(str)

        category = ids.str[:2].map(CATEGORY_BY_PREFIX).fillna("Other")
        out["Item_Category"] = category
        out["Item_Fat_Content"] = X["Item_Fat_Content"].astype(str).where(category != "Non-Consumable", "Non-Edible")
        out["Item_Type"] = X["Item_Type"].astype(str)
        out["Outlet_Location_Type"] = X["Outlet_Location_Type"].astype(str)
        out["Outlet_Type"] = X["Outlet_Type"].astype(str)
        if self.include_outlet_id:
            out[OUTLET_COL] = X[OUTLET_COL].astype(str)
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    @staticmethod
    def _guard(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BigMartFeatures needs a pandas DataFrame")
        if TARGET in X.columns:
            raise LeakageError(f"{TARGET} is in the features. Call data.split_xy first.")
